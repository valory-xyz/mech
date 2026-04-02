# Benchmark Test Log — Superforcaster Cached Prompt Replay

## Problem

The superforcaster tool is the highest-volume prediction tool (15,712 predictions in the benchmark) and contributes the most total Brier error (4,111 out of ~10,000 across all tools). Analysis of the benchmark reports (Mar 31 and Apr 2 CI runs) revealed:

- **Severe overconfidence:** 4,304 predictions had p_yes >= 0.90, but only 28.5% of those resolved Yes. The 0.9-1.0 bucket alone accounts for **69% of superforcaster's total Brier error**.
- **Clustering at specific values:** The model repeatedly outputs p_yes=0.93 (1,491 times) and 0.97 (1,358 times) — suspiciously round numbers suggesting pattern-matching rather than evidence-based reasoning.
- **No base-rate awareness:** The prompt never mentions that ~85% of prediction market questions resolve "No". The model treats each question in isolation, anchoring on "this sounds plausible" rather than starting from a base rate.
- **No tail discipline:** Nothing prevents the model from outputting p_yes=0.97 on a "Will X publicly announce by DATE" question where there's no evidence X has already announced anything.
- **Stale knowledge cutoff:** The prompt says "Your pretraining knowledge cutoff: October 2023" but the model is gpt-4.1-2025-04-14. This may cause the model to discount information it actually has.
- **Contradictory output format:** The prompt has a 7-step XML reasoning chain (`<facts>`, `<yes>`, `<no>`, `<thinking>`, `<answer>`) but then says "Output only the JSON object. Do not include any other contents." These are mutually exclusive instructions.

## What we changed

We surgically edited the stored production prompts (fetched from IPFS deliveries) to add calibration rules. The question, sources, and reasoning structure remain identical — only the instructions change.

### Edit 1: Removed stale knowledge cutoff
```
- Your pretraining knowledge cutoff: October 2023
```
Deleted. Wrong for gpt-4.1.

### Edit 2: Base-rate anchoring (injected after step 4, before step 5)
```
BASE-RATE ANCHORING (mandatory before outputting any probability):
- Identify the event category (e.g. regulatory action, product launch, weather event).
- Before adjusting based on evidence, consider: what is the base rate for this
  type of event resolving "Yes"? Use this as your starting point.
- Only move away from the base rate when you have specific, concrete evidence
  -- not general plausibility or "sounds likely" reasoning.
```
Forces the model to consider base rates before outputting any number.

### Edit 3: Evidence-aware tail discipline (injected after step 6, before step 7)
```
TAIL DISCIPLINE (mandatory before final answer):
- First, review your findings from steps 1-4. If the sources confirm the event
  has ALREADY occurred or been officially completed (e.g. agreement signed,
  contract awarded, data published, official statement issued), then deadline
  skepticism does not apply -- maintain high probability based on the evidence.
- If the event has NOT yet occurred according to the sources:
  - p_yes above 0.90 requires evidence of a specific verifiable institutional
    commitment (signed agreement, published schedule, awarded contract).
  - p_yes above 0.80 requires strong, specific, verifiable evidence -- not just
    "this seems likely" or "this company often does this."
  - When the question includes a specific deadline ("on or before [date]"),
    consider whether the deadline is realistic given the current evidence stage.
    Intentions, plans, and proposals are NOT the same as completed actions.
  - Absence of evidence that the event has occurred IS evidence against it.
  - General plausibility, company reputation, or past patterns alone do NOT
    justify probabilities above 0.80.
```
The key design choice: the tail discipline first checks whether the evidence shows the event already happened (in which case high confidence is correct), before applying skepticism on unresolved events. This prevents the rules from overriding genuine evidence while still catching baseless overconfidence.

## How we tested

1. Fetched `production_log.jsonl` from Github Artefacts.
2. For 20 superforcaster rows, fetched the stored `prediction_prompt` from IPFS via `deliver_id` -> subgraph IPFS hash -> IPFS gateway
3. Applied the edits above to each stored prompt
4. Replayed 20 rows sequentially through gpt-4.1-2025-04-14 (temperature=0, max_tokens=4000)
5. Compared new p_yes against known outcomes

## Results (20 rows)

**Model:** gpt-4.1-2025-04-14, temperature=0, max_tokens=4000

| # | Question (truncated) | Orig p_yes | New p_yes | Outcome | Orig Brier | New Brier | Delta |
|---|---|---|---|---|---|---|---|
| 1 | NWS confirm Marquette snowfall | 0.99 | 0.99 | No | 0.9801 | 0.9801 | +0.0000 |
| 2 | BTS album Platinum | 0.98 | 0.97 | No | 0.9604 | 0.9409 | +0.0195 |
| 3 | Cause of death Perez-Jimenez | 0.13 | 0.10 | No | 0.0169 | 0.0100 | +0.0069 |
| 4 | USPS surcharge approval | 0.97 | 0.10 | No | 0.9409 | 0.0100 | +0.9309 |
| 5 | NVIDIA above $170 Mar 31 | 0.72 | 0.68 | No | 0.5184 | 0.4624 | +0.0560 |
| 6 | UK-Nigeria treaty | 0.97 | 0.88 | Yes | 0.0009 | 0.0144 | -0.0135 |
| 7 | Trump "Fake News" | 0.32 | 0.35 | Yes | 0.4624 | 0.4225 | +0.0399 |
| 8 | Amazon product sale | 0.93 | 0.03 | No | 0.8649 | 0.0009 | +0.8640 |
| 9 | Riverside County Sheriff | 0.09 | 0.13 | No | 0.0081 | 0.0169 | -0.0088 |
| 10 | Qatar Energy LNG | 0.01 | 0.01 | No | 0.0001 | 0.0001 | +0.0000 |
| 11 | WHO announce Mar 22 | 0.04 | 0.01 | No | 0.0016 | 0.0001 | +0.0015 |
| 12 | NASA major contract | 0.97 | 0.05 | Yes | 0.0009 | 0.9025 | -0.9016 |
| 13 | USPS surcharge (dup) | 0.97 | 0.12 | No | 0.9409 | 0.0144 | +0.9265 |
| 14 | BYD/Geely announce | 0.93 | 0.07 | No | 0.8649 | 0.0049 | +0.8600 |
| 15 | NWS Marquette (dup) | 0.99 | 0.99 | No | 0.9801 | 0.9801 | +0.0000 |
| 16 | Yangtze Memory announce | 0.13 | 0.13 | No | 0.0169 | 0.0169 | +0.0000 |
| 17 | NVIDIA above $170 end Mar | 0.23 | 0.23 | No | 0.0529 | 0.0529 | +0.0000 |
| 18 | Google feature rollout | 0.93 | 0.20 | No | 0.8649 | 0.0400 | +0.8249 |
| 19 | Google Michigan data center | 0.98 | 0.97 | No | 0.9604 | 0.9409 | +0.0195 |
| 20 | Bureau of Engraving announce | 0.93 | 0.25 | No | 0.8649 | 0.0625 | +0.8024 |

### Summary

| Metric | Value |
|--------|-------|
| **Avg original Brier** | **0.5151** |
| **Avg new Brier** | **0.2937** |
| **Avg improvement** | **+0.2214 (43% reduction)** |
| Improved | 12/20 |
| Worsened | 3/20 |
| Unchanged | 5/20 |

### Analysis

**What worked:** The calibration rules dramatically fixed overconfident-wrong predictions where the model had no evidence the event occurred. Cases like USPS surcharge (0.97->0.10), Amazon product (0.93->0.03), BYD/Geely (0.93->0.07), Google feature (0.93->0.20), Bureau of Engraving (0.93->0.25) all saw massive Brier improvements (0.8-0.9 points each). The model correctly recognized "no evidence this already happened" = low probability.

**What worked correctly on already-occurred events:** Row 6 (UK-Nigeria treaty) — the sources explicitly say the agreement was already signed ("signed on the sidelines of Tinubu's state visit", Reuters confirms). The model correctly maintained high probability (0.88) because the evidence-aware tail discipline says: if sources confirm the event already occurred, deadline skepticism doesn't apply. Brier worsened by only 0.0135 vs original.

**One remaining hard case:** Row 12 (NASA contract, 0.97->0.05, outcome=Yes) — the sources describe NASA's $20 billion moon base plan and RFIs/RFPs, but don't explicitly confirm a "$1 billion contract was awarded." The model correctly applies skepticism — plans and proposals are not completed actions. The outcome was Yes, so this hurts Brier, but the model's reasoning is defensible given the evidence it had.

**Low-confidence predictions preserved:** Rows where the original prediction was already good (Qatar Energy 0.01, Riverside 0.09, Yangtze 0.13, NVIDIA 0.23) were mostly unchanged — the edits didn't introduce regression on well-calibrated predictions.

---

## Next steps

### 1. Run on larger dataset (needs OpenAI API keys with sufficient quota)

The 20-row test was intentionally skewed toward overconfident failures. To validate the prompt edits don't regress on the broader distribution, we need to run on a larger, representative sample:

- Use the full 50-row dataset already in `sf_prompt_replay.jsonl` (balanced sample: overconfident-wrong, correct-low, mid-range, correct-high)
- Ideally run on 200+ rows for statistically significant Brier comparison
- The `production_log_with_ids.jsonl` has 28,169 superforcaster rows with `deliver_id` — IPFS prompt fetching scales linearly
- Estimated cost: ~$0.50-1.00 per 50 rows (gpt-4.1, ~9k tokens input + ~4k output per request)

### 2. Run tournament mode to test candidate against baseline

Once the cached replay pipeline and tournament mode are available:

- Register the prompt edits as a candidate configuration
- Run tournament: baseline (original PREDICTION_PROMPT) vs candidate (edited prompts) on the same set of questions with identical source content
- Tournament mode ensures deterministic comparison — same evidence, same model, different prompts
- This is the gold standard test before deploying any prompt changes to production

### 3. Analyse cost impact of longer prompts

The edits add ~500 tokens to each prompt (~8,100 -> ~8,600 chars). At scale this affects:

- **Per-request cost:** Additional input tokens billed by OpenAI. Need to measure the actual token increase across the dataset and calculate the cost delta at current gpt-4.1 pricing.
- **Latency:** Longer prompts mean slightly longer time-to-first-token. With the 240s task deadline, we need to confirm this doesn't push borderline requests into timeouts.
- **Token budget for reasoning:** The model now generates a longer reasoning chain (base-rate analysis + tail discipline reflection) before outputting JSON. The max_tokens=4000 was sufficient in testing but may need adjustment if some prompts produce truncated output at scale.

### 4. Apply edits to the actual PREDICTION_PROMPT in superforcaster.py

After validation on larger dataset + tournament:
- Port the base-rate anchoring and evidence-aware tail discipline into the tool's `PREDICTION_PROMPT` constant
- Remove stale knowledge cutoff date
- Run full CI checks before PR
