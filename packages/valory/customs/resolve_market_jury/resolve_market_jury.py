# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
"""Multi-model jury tool for resolving prediction markets.

Fans out a market question to N independent AI voters (each with native web search),
then an Anthropic Claude judge synthesizes the final verdict.

Drop-in replacement for resolve_market_reasoning — same input kwargs, same output
tuple shape, same JSON result schema ({is_valid, is_determinable, has_occurred}).
"""

import functools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import anthropic
import openai

# ---------------------------------------------------------------------------
# Types (mech tool contract)
# ---------------------------------------------------------------------------

MechResponseWithKeys = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]
MaxCostResponse = float

DEFAULT_DELIVERY_RATE = 100


# ---------------------------------------------------------------------------
# Voter / Judge configuration
# ---------------------------------------------------------------------------

OPENAI_MODEL = "gpt-4.1-2025-04-14"
GROK_MODEL = "x-ai/grok-4.1-fast"
GEMINI_MODEL = "google/gemini-2.5-flash"
JUDGE_MODEL = "claude-sonnet-4-20250514"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

N_VOTER_CALLS = 3  # default number of voters
N_JUDGE_CALLS = 1
N_MODEL_CALLS = N_VOTER_CALLS + N_JUDGE_CALLS

VOTER_TIMEOUT = 120  # seconds per voter API call
JUDGE_TIMEOUT = 120
JUDGE_MAX_RETRIES = 3
JUDGE_RETRY_DELAY = 5  # seconds

ALLOWED_TOOLS = [
    "resolve-market-jury-v1",
]

TOOL_TO_ENGINE = {
    "resolve-market-jury-v1": JUDGE_MODEL,
}


# ---------------------------------------------------------------------------
# Shared voter prompt
# ---------------------------------------------------------------------------

VOTER_PROMPT = """You are an expert fact-checker resolving a prediction market question.
The question specifies a deadline date that has already passed. Your task is to determine whether the event actually happened before that deadline.

INSTRUCTIONS:
* Read the question carefully. Pay close attention to EVERY qualifier, condition, and constraint.
* The question will contain specific conditions that ALL must be met for the answer to be "yes". These may include:
  - A specific deadline date (the event must have occurred ON OR BEFORE this date)
  - A specific actor (who must perform the action — e.g., a government, company, institution)
  - A specific action (what exactly must happen — e.g., "publicly announce", "officially confirm")
  - Specific thresholds (e.g., "at least 10%", "exceed $4.00", "affecting 50%")
  - A specific source requirement (e.g., "confirmed by a scientific institution", "as reported by AAA")
* Do NOT conflate related events with the exact event described. For example:
  - Private citizens finding meteorite fragments is NOT the same as "confirmed by a scientific institution"
  - Widespread blackouts are NOT the same as "a nationwide rationing schedule"
  - A price of $3.95 does NOT meet a threshold of "$4.00 or more"
  - An existing bill reaching a vote is NOT the same as a "new bill being introduced"
* Search the web for evidence. Use reliable, verifiable sources.

DETERMINING YOUR ANSWER:
* If the evidence clearly shows ALL conditions were met before the deadline: has_occurred = true.
* If the evidence clearly shows any condition was NOT met: has_occurred = false.
* IMPORTANT — when to set is_determinable = false:
  - You found conflicting evidence and cannot confidently decide
  - The event is too recent or niche and reliable sources do not cover it yet
  - You are less than 80% confident in your answer
  - Do NOT default to true/false when uncertain — returning is_determinable = false is the correct and expected response when evidence is insufficient
* When to set is_valid = false:
  - The question is ambiguous, self-contradictory, or has no clear resolution criteria
  - The question uses relative dates ("in 6 months") instead of absolute dates
  - The question asks about subjective opinions rather than verifiable facts

Question: "{question}"

CRITICAL: Respond with ONLY valid JSON. No markdown, no text before or after the JSON.
{{
    "is_valid": true or false,
    "is_determinable": true or false,
    "has_occurred": true or false or null,
    "confidence": "high" or "medium" or "low",
    "reasoning": "200 words max. State which specific conditions you checked and whether each was met.",
    "sources": ["url1", "url2"]
}}"""

JUDGE_PROMPT = """You are the final judge resolving a prediction market question.
Your role is to synthesize the independent assessments below into a single verdict.
You do NOT search the web — you evaluate the evidence the voters already gathered.

Question: "{question}"

Voter assessments:
{votes}

Rules:
1. MAJORITY RULE: If a clear majority of voters agree (e.g., 2 out of 3), you MUST follow the majority UNLESS the minority voter provides specific, verifiable evidence (named sources, URLs, concrete facts) that directly contradicts the majority AND the majority's reasoning has a clear factual error or logical flaw. Simply having a different opinion is NOT enough to override the majority.
2. If voters UNANIMOUSLY agree, follow their consensus.
3. If any voter says the question is invalid with sound reasoning, mark as invalid.
4. If voters are split with no clear majority, or if multiple voters report low confidence, set is_determinable = false.
5. If voters disagree on whether the event occurred but there IS a majority, weigh:
   - Does the minority voter cite specific verifiable sources the majority missed?
   - Does the minority voter identify a factual error in the majority's reasoning?
   - Only override the majority if the answer to BOTH questions is yes.

Respond in JSON only (no markdown fences, no text before or after the JSON object):
{{
    "is_valid": true or false,
    "is_determinable": true or false,
    "has_occurred": true or false or null,
    "judge_reasoning": "explanation. If overriding majority, explicitly state what factual error you found."
}}"""


# ---------------------------------------------------------------------------
# VoterResult
# ---------------------------------------------------------------------------


@dataclass
class VoterResult:
    """Uniform output from every voter."""

    voter: str
    model: str
    is_valid: Optional[bool] = None
    is_determinable: Optional[bool] = None
    has_occurred: Optional[bool] = None
    confidence: str = ""
    reasoning: str = ""
    sources: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Voter registry
# ---------------------------------------------------------------------------


@dataclass
class VoterConfig:
    """Configuration for a single voter."""

    adapter: str  # "_adapter_openai" or "_adapter_openrouter"
    model: str  # model name / slug
    key_name: str  # KeyChain service name
    search_cost: float  # estimated per-call search cost in USD


VOTER_REGISTRY: Dict[str, VoterConfig] = {
    "openai": VoterConfig(
        adapter="_adapter_openai",
        model=OPENAI_MODEL,
        key_name="openai",
        search_cost=0.01,
    ),
    "grok": VoterConfig(
        adapter="_adapter_openrouter",
        model=GROK_MODEL,
        key_name="openrouter",
        search_cost=0.005,
    ),
    "gemini": VoterConfig(
        adapter="_adapter_openrouter",
        model=GEMINI_MODEL,
        key_name="openrouter",
        search_cost=0.014,
    ),
}

DEFAULT_VOTERS = ["openai", "grok", "gemini"]


# ---------------------------------------------------------------------------
# JSON parsing (shared across all adapters)
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from a response that may contain markdown fences or extra text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown fences
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    # Find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _parse_vote(raw: str, voter: str, model: str) -> VoterResult:
    """Parse raw LLM text into a VoterResult."""
    data = _extract_json(raw)
    if data is None:
        return VoterResult(
            voter=voter,
            model=model,
            error=f"Unparseable JSON: {raw[:200]}",
        )
    return VoterResult(
        voter=voter,
        model=model,
        is_valid=data.get("is_valid"),
        is_determinable=data.get("is_determinable"),
        has_occurred=data.get("has_occurred"),
        confidence=data.get("confidence", ""),
        reasoning=data.get("reasoning", ""),
        sources=data.get("sources", []),
    )


# ---------------------------------------------------------------------------
# Adapter: OpenAI (native web_search tool)
# ---------------------------------------------------------------------------


def _adapter_openai(
    voter_name: str,
    model: str,
    prompt: str,
    api_key: str,
) -> VoterResult:
    """Run OpenAI voter with native web_search tool via Responses API."""
    client = openai.OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search"}],
        input=prompt,
        timeout=VOTER_TIMEOUT,
    )
    # Extract text from response output items
    text_parts = []
    for item in response.output:
        if hasattr(item, "text"):
            text_parts.append(item.text)
        elif hasattr(item, "content"):
            for block in item.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
    raw = "\n".join(text_parts)
    return _parse_vote(raw, voter_name, model)


# ---------------------------------------------------------------------------
# Adapter: OpenRouter (any model with :online search)
# ---------------------------------------------------------------------------


def _adapter_openrouter(
    voter_name: str,
    model: str,
    prompt: str,
    api_key: str,
) -> VoterResult:
    """Run OpenRouter voter with :online search plugin."""
    client = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    model_online = model if ":online" in model else f"{model}:online"
    response = client.chat.completions.create(
        model=model_online,
        messages=[{"role": "user", "content": prompt}],
        timeout=VOTER_TIMEOUT,
    )
    raw = response.choices[0].message.content or ""
    return _parse_vote(raw, voter_name, model_online)


_ADAPTERS: Dict[str, Callable] = {
    "_adapter_openai": _adapter_openai,
    "_adapter_openrouter": _adapter_openrouter,
}


# ---------------------------------------------------------------------------
# Voting facade
# ---------------------------------------------------------------------------


def cast_vote(
    voter_name: str,
    question: str,
    api_keys: Any,
) -> VoterResult:
    """Uniform entry point — delegates to provider-specific adapter."""
    config = VOTER_REGISTRY[voter_name]
    api_key = api_keys[config.key_name]
    prompt = VOTER_PROMPT.format(question=question)
    adapter_fn = _ADAPTERS[config.adapter]
    return adapter_fn(
        voter_name=voter_name,
        model=config.model,
        prompt=prompt,
        api_key=api_key,
    )


def collect_votes(
    question: str,
    voter_names: List[str],
    api_keys: Any,
) -> List[VoterResult]:
    """Fan out to all voters in parallel, collect results."""
    results: List[VoterResult] = []
    with ThreadPoolExecutor(max_workers=len(voter_names)) as pool:
        futures = {
            pool.submit(cast_vote, name, question, api_keys): name
            for name in voter_names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result(timeout=VOTER_TIMEOUT + 30)
                results.append(result)
                print(
                    f"  Voter [{name}]: "
                    f"has_occurred={result.has_occurred}, "
                    f"is_determinable={result.is_determinable}, "
                    f"confidence={result.confidence}"
                )
            except Exception as e:  # pylint: disable=broad-except
                print(f"  Voter [{name}] failed: {e}")
    return results


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def _run_judge(
    question: str,
    votes: List[VoterResult],
    api_key: str,
) -> dict:
    """Anthropic Claude judge — synthesizes voter results into final verdict."""
    client = anthropic.Anthropic(api_key=api_key)

    formatted_votes = ""
    for i, v in enumerate(votes, 1):
        vote_data = {
            "is_valid": v.is_valid,
            "is_determinable": v.is_determinable,
            "has_occurred": v.has_occurred,
            "confidence": v.confidence,
            "reasoning": v.reasoning,
            "sources": v.sources,
        }
        formatted_votes += f"\nVoter {i}:\n{json.dumps(vote_data, indent=2)}\n"

    prompt = JUDGE_PROMPT.format(question=question, votes=formatted_votes)

    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            response = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except anthropic._exceptions.OverloadedError:
            if attempt < JUDGE_MAX_RETRIES - 1:
                print(
                    f"  Judge overloaded, retrying in {JUDGE_RETRY_DELAY}s "
                    f"(attempt {attempt + 1}/{JUDGE_MAX_RETRIES})..."
                )
                time.sleep(JUDGE_RETRY_DELAY)
            else:
                raise

    raw = response.content[0].text
    data = _extract_json(raw)
    if data is None:
        return {
            "is_valid": False,
            "is_determinable": False,
            "has_occurred": None,
            "judge_reasoning": f"Unparseable judge response: {raw[:200]}",
        }
    return data


# ---------------------------------------------------------------------------
# Consensus helpers
# ---------------------------------------------------------------------------


def _decided_votes(votes: List[VoterResult]) -> List[VoterResult]:
    """Filter to votes that are valid and determinable."""
    return [
        v
        for v in votes
        if v.is_determinable is not False
        and v.is_valid is not False
        and v.error is None
    ]


def _all_agree(votes: List[VoterResult]) -> bool:
    """Check if all decided votes unanimously agree on has_occurred."""
    decided = _decided_votes(votes)
    if len(decided) < 2:
        return False
    return all(v.has_occurred == decided[0].has_occurred for v in decided)


def _build_consensus_result(votes: List[VoterResult]) -> dict:
    """Build result from unanimous votes (skip judge)."""
    decided = _decided_votes(votes)
    return {
        "is_valid": True,
        "is_determinable": True,
        "has_occurred": decided[0].has_occurred,
        "votes": [asdict(v) for v in votes],
        "judge_reasoning": "Unanimous voter consensus — judge skipped.",
        "agreement_ratio": 1.0,
        "n_voters": len(votes),
        "n_successful": len(decided),
    }


def _compute_agreement(votes: List[VoterResult]) -> float:
    """Compute agreement ratio among decided votes."""
    decided = _decided_votes(votes)
    if not decided:
        return 0.0
    yes_count = sum(1 for v in decided if v.has_occurred is True)
    no_count = sum(1 for v in decided if v.has_occurred is False)
    return max(yes_count, no_count) / len(decided)


# ---------------------------------------------------------------------------
# @with_key_rotation decorator
# ---------------------------------------------------------------------------


def with_key_rotation(func: Callable) -> Callable:
    """Decorator that retries a function with API key rotation on failure."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MechResponseWithKeys:
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponseWithKeys:
            try:
                result: MechResponse = func(*args, **kwargs)
                return result + (api_keys,)
            except anthropic.RateLimitError:
                service = "anthropic"
                if retries_left.get(service, 0) <= 0:
                    raise
                retries_left[service] -= 1
                api_keys.rotate(service)
                return execute()
            except openai.RateLimitError:
                for service in ("openai", "openrouter"):
                    if retries_left.get(service, 0) > 0:
                        retries_left[service] -= 1
                        api_keys.rotate(service)
                else:
                    raise
                return execute()
            except Exception as e:  # pylint: disable=broad-except
                print(f"Unexpected error in run(): {e}")
                return str(e), "", None, None, api_keys

        return execute()

    return wrapper


# ---------------------------------------------------------------------------
# run() — mech tool entry point
# ---------------------------------------------------------------------------


@with_key_rotation
def run(**kwargs: Any) -> Union[MaxCostResponse, MechResponse]:
    """Run the resolve_market_jury tool.

    :param kwargs: keyword arguments including prompt, tool, api_keys,
        delivery_rate, and counter_callback.
    :return: max cost float (if delivery_rate==0) or MechResponse tuple.
    """
    tool = kwargs["tool"]
    delivery_rate = int(kwargs.get("delivery_rate", DEFAULT_DELIVERY_RATE))
    counter_callback: Optional[Callable] = kwargs.get("counter_callback", None)
    api_keys = kwargs["api_keys"]
    prompt = kwargs["prompt"]

    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported. Allowed: {ALLOWED_TOOLS}")

    # Cost calculation mode
    if delivery_rate == 0:
        if not counter_callback:
            raise ValueError(
                "A delivery rate of `0` was passed, but no counter callback was given."
            )
        max_cost = counter_callback(
            max_cost=True,
            models_calls=(OPENAI_MODEL,) * N_VOTER_CALLS + (JUDGE_MODEL,),
        )
        return max_cost

    selected_voters = DEFAULT_VOTERS

    # 1. Fan out to voters (parallel)
    print(f"Collecting votes from {selected_voters}...")
    votes = collect_votes(prompt, selected_voters, api_keys)

    # 2. Early exit: no successful votes
    if not votes:
        result: Dict[str, Any] = {
            "is_valid": False,
            "is_determinable": False,
            "has_occurred": None,
            "votes": [],
            "judge_reasoning": "All voters failed.",
            "agreement_ratio": 0.0,
            "n_voters": len(selected_voters),
            "n_successful": 0,
        }
        return json.dumps(result), "All voters failed", None, counter_callback

    # 3. Unanimous early exit (cost saving — skip judge)
    if _all_agree(votes):
        print("  Unanimous consensus — skipping judge.")
        result = _build_consensus_result(votes)
        reasoning = "\n---\n".join(v.reasoning for v in votes if v.reasoning)
        return json.dumps(result), reasoning, None, counter_callback

    # 4. Judge synthesizes (only when voters disagree or partial)
    print("  Voters disagree — running judge...")
    verdict = _run_judge(prompt, votes, api_keys["anthropic"])

    # 5. Build result with vote metadata
    result = {
        "is_valid": verdict.get("is_valid", True),
        "is_determinable": verdict.get("is_determinable", True),
        "has_occurred": verdict.get("has_occurred"),
        "votes": [asdict(v) for v in votes],
        "judge_reasoning": verdict.get("judge_reasoning", ""),
        "agreement_ratio": _compute_agreement(votes),
        "n_voters": len(selected_voters),
        "n_successful": len(votes),
    }

    reasoning = "\n---\n".join(v.reasoning for v in votes if v.reasoning)
    return json.dumps(result), reasoning, None, counter_callback
