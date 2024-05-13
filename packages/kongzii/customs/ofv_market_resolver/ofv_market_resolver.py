from factcheck import FactCheck
from factcheck.utils.multimodal import modal_normalization
import json
from langchain_openai import ChatOpenAI
from typing import Annotated, Any, Dict, Optional, Tuple
from pydantic import BaseModel, BeforeValidator

DEFAULT_OPENAI_MODEL = "gpt-4-0125-preview"

Factuality = Annotated[
    bool | None,
    BeforeValidator(lambda v: None if v in ("Nothing to check.", "non-factual") else v),
]


class FactCheckClaimDetails(BaseModel):
    claim: str
    factuality: Factuality
    correction: str | None
    reference_url: str


class FactCheckResult(BaseModel):
    factuality: Factuality
    claims_details: list[FactCheckClaimDetails] | None


def factcheck(
    statement: str,
    model: str = DEFAULT_OPENAI_MODEL,
    openai_api_key: str | None = None,
    serper_api_key: str | None = None,
) -> FactCheckResult:
    api_config = {
        "OPENAI_API_KEY": openai_api_key,
        "SERPER_API_KEY": serper_api_key,
    }
    factcheck = FactCheck(
        default_model=model,
        api_config=api_config,
        retriever="serper",
        num_seed_retries=5,
    )
    content = modal_normalization("string", statement)
    res = factcheck.check_response(content)

    return FactCheckResult.model_validate(res)


def rewrite_as_sentence(
    question: str,
    model: str = DEFAULT_OPENAI_MODEL,
    openai_api_key: str | None = None,
) -> str:
    """
    Rewrites the question into a sentence, example:

    `Will former Trump Organization CFO Allen Weisselberg be sentenced to jail by 15 April 2024?`
    ->
    `Former Trump Organization CFO Allen Weisselberg was sentenced to jail by 15 April 2024.`
    """
    llm = ChatOpenAI(
        model=model,
        temperature=0.0,
        api_key=openai_api_key,
    )

    prompt = f"""
Rewrite the question into a simple annoucment sentence stating a fact or prediction like it is already known.  
Make future tense into past tense.
For future questions that ask if something will happen "by" some date, rewrite it to "before" that date or any time sooner.
For future questions that ask if something will happen "on" some date, rewrite it to "on" that date.
If the question is both "on" and "by" some date, rewrite it as "before or any time sooner than" that date.
If the question is about exact date, keep it exact. 
If the question is about a date range, keep it a range.
Always keep the same meaning.                          
Never negate the sentence into opposite meaning of the question.                  

Question: {question}
Sentence:                                         
"""
    completion = str(llm.invoke(prompt, max_tokens=512).content)

    return completion


# TODO: This could be imported from prediction-market-agent-tooling, but given the conflict in the langchain versions,
# it would require changes in other mechs of this repository.
def is_predictable_binary(
    question: str,
    model: str = DEFAULT_OPENAI_MODEL,
    openai_api_key: str | None = None,
) -> str:
    """
    Evaluate if the question is actually answerable.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=0.0,
        api_key=openai_api_key,
    )

    prompt = f"""Main signs about a fully qualified question (sometimes referred to as a "market"):
- The market's question needs to be specific, without use of pronouns.
- The market's question needs to have a clear future event.
- The market's question needs to have a clear time frame.
- The event in the market's question doesn't have to be ultra-specific, it will be decided by a crowd later on.
- If the market's question contains date, but without an year, it's okay.
- If the market's question contains year, but without an exact date, it's okay.
- The market's question can not be about itself or refer to itself.
- The answer is probably Google-able, after the event happened.
- The potential asnwer can be only "Yes" or "No".

Follow a chain of thought to evaluate if the question is fully qualified:

First, write the parts of the following question:

"{question}"

Then, write down what is the future event of the question, what it refers to and when that event will happen if the question contains it.

Then, explain why do you think it is or isn't fully qualified.

Finally, write your final decision, write `decision: ` followed by either "yes it is fully qualified" or "no it isn't fully qualified" about the question. Don't write anything else after that. You must include "yes" or "no".
"""
    completion = str(llm.invoke(prompt, max_tokens=512).content)

    try:
        decision = completion.lower().rsplit("decision", 1)[1]
    except IndexError as e:
        raise ValueError(
            f"Invalid completion in is_predictable for `{question}`: {completion}"
        ) from e

    if "yes" in decision:
        is_predictable = True
    elif "no" in decision:
        is_predictable = False
    else:
        raise ValueError(
            f"Invalid completion in is_predictable for `{question}`: {completion}"
        )

    return is_predictable


def build_run_result(
    has_occurred: bool | None, is_determinable: bool | None
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    return (
        json.dumps(
            {
                "has_occurred": has_occurred,
                "is_determinable": is_determinable,
            }
        ),
        "",
        None,
        None,
    )


def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """
    Run the prediction market resolver based on Open Fact Verifier.

    Returns:
        - None if can't decide
        - True if the answer for the question is "Yes"
        - False if the answer for the question is "No"
    """

    market_question = kwargs["prompt"]
    api_keys = kwargs.get("api_keys", {})
    openai_api_key = api_keys["openai"]
    serper_api_key = api_keys["serperapi"]

    # Check if the question is reasonable to look for an answer.
    is_answerable = is_predictable_binary(
        market_question, openai_api_key=openai_api_key
    )
    if not is_answerable:
        print(
            f"Question `{market_question}` is not answerable, skipping fact checking."
        )
        return build_run_result(has_occurred=None, is_determinable=is_answerable)

    # Rewrite the question (which was about a future) into a sentence (which is about the past).
    market_sentence = rewrite_as_sentence(
        market_question, openai_api_key=openai_api_key
    )
    print(f"Question `{market_question}` rewritten into `{market_sentence}`.")
    # Fact-check the sentence.
    factresult = factcheck(
        market_sentence, openai_api_key=openai_api_key, serper_api_key=serper_api_key
    )
    print(
        f"Fact check result for `{market_sentence}` is `{factresult.factuality}`, because {factresult.claims_details}."
    )

    return build_run_result(
        has_occurred=factresult.factuality, is_determinable=is_answerable
    )
