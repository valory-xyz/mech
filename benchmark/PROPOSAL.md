# Prediction Tool Benchmark & Continuous Improvement System

## Problem

The old benchmark (olas-predict-benchmark, archived) used the Autocast dataset — academic questions from 2022 with pre-scraped source links. It measured binary accuracy only, had no prompt variation support, and couldn't drive iterative improvement. Production accuracy is measured after the fact via trade outcomes on [olas.network](https://olas.network/agent-economies/predict#tool-accuracy), but there is no feedback loop from production back into tool development.

We need a closed-loop system: **benchmark locally → validate → deploy → observe production → feed back → improve → repeat**.

## Design Goals

1. **Production data flywheel** — production predictions and outcomes continuously feed back into the benchmark dataset
2. **Automated tool improvement** — systematically search for better prompts, models, tool configs, and tool code
3. **Real prediction market questions** — Polymarket/Omen questions, not academic datasets
4. **Temporal integrity** — never let a tool "predict" something it can Google the answer to
5. **Edge over market** — measure whether predictions beat market consensus, not just standalone accuracy
6. **Local pre-validation** — catch regressions and validate improvements before they hit production
7. **Continuous loop** — the system runs perpetually: improve → benchmark → promote → observe → learn → improve

## The Closed Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│   │  HARVEST  │───▶│BENCHMARK │───▶│  SEARCH  │───▶│ PROMOTE  │     │
│   │          │    │          │    │          │    │          │     │
│   │ Collect   │    │ Score    │    │ Generate │    │ Canary   │     │
│   │ resolved  │    │ tools on │    │ & test   │    │ deploy   │     │
│   │ markets + │    │ dataset  │    │ improved │    │ winner,  │     │
│   │ prod data │    │          │    │ variants │    │ validate │     │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│        ▲                                               │            │
│        │           ┌──────────┐                        │            │
│        │           │TOURNAMENT│──(continuous)──────┐    │            │
│        │           │          │                    │    │            │
│        │           │ Predict  │    ┌──────────┐   │    │            │
│        │           │ on open  │───▶│  RESOLVE │   │    │            │
│        │           │ markets  │    │ & score  │───┘    │            │
│        │           └──────────┘    └──────────┘        │            │
│        │                                               │            │
│        │           ┌──────────┐                        │            │
│        └───────────│PRODUCTION│◀───────────────────────┘            │
│                    │          │                                      │
│                    │ Trader   │                                      │
│                    │ trades,  │                                      │
│                    │ markets  │                                      │
│                    │ resolve  │                                      │
│                    └──────────┘                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Architecture

```
benchmark/
├── datasets/
│   ├── polymarket_resolved.jsonl   # Resolved Polymarket questions with outcomes
│   ├── omen_resolved.jsonl         # Resolved Omen questions with outcomes
│   ├── production_log.jsonl        # Our own predictions + actual outcomes
│   ├── open_markets.jsonl          # Currently open markets (for tournament)
│   ├── snapshots/                  # Cached web content per question per date
│   ├── fetch_resolved.py           # Pull recently resolved markets from APIs
│   ├── fetch_open.py               # Pull currently open markets for tournament
│   └── fetch_production.py         # Pull our production predictions + outcomes
├── runner.py                       # Core benchmark runner (replay mode)
├── tournament.py                   # Forward-looking tournament runner
├── scorer.py                       # Scoring: Brier, edge, calibration
├── sweep.py                        # Parameter sweep (prompts, models, temps)
├── search.py                       # Automated tool improvement search
├── compare.py                      # Compare two benchmark runs
├── promote.py                      # Promote winning config to production
├── results/                        # Timestamped run results (gitignored)
└── prompts/
    └── templates.py                # Prompt template variants to test
```

---

## Part 1: Temporal Integrity — The Core Constraint

**The single most important property of a prediction benchmark is that the tool cannot see the answer.**

If we ask a tool "Will the Fed cut rates in January 2026?" and the market resolved on Jan 31, 2026, any tool running *after* that date will Google the outcome and trivially return p_yes=0.99 or p_no=0.99. This measures web search quality, not forecasting ability. The old Autocast benchmark handled this by pre-scraping source links. Our approach is more principled.

### Three Evaluation Modes

Each mode trades off temporal purity against iteration speed:

#### Mode 1: Production Replay (Gold Standard)

Use actual production predictions that were made *before* the market resolved. No temporal contamination possible — the prediction was already made and recorded on-chain.

```
Timeline:  ──────[prediction made]──────────[market resolves]──────[we score]──▶
                       ↑                          ↑                     ↑
                  tool ran here            outcome known          we compare
                  (no future info)
```

- **Source:** `fetch_production.py` — indexes on-chain Request/Deliver events + IPFS data
- **Pros:** Perfect temporal integrity, represents actual production distribution
- **Cons:** Limited to questions we've already seen in production, selection bias from trader
- **Use for:** Final accuracy measurement, baseline establishment, production monitoring

#### Mode 2: Forward-Looking Tournament (Out-of-Sample)

Make predictions on currently *open* markets. Store them. Wait for resolution. Score later.

```
Timeline:  ──[market opens]──[we predict NOW]──────[market resolves]──[we score]──▶
                                    ↑                     ↑                ↑
                              no future info        outcome known    compare prediction
```

- **Source:** `fetch_open.py` pulls currently open Polymarket/Omen markets
- **Runner:** `tournament.py` runs all tools on open markets, stores predictions with timestamps
- **Scoring:** A separate scheduled job matches stored predictions against resolutions as they come in
- **Pros:** No temporal contamination, tests on never-before-seen questions, tests new tools on questions they haven't seen
- **Cons:** Latency — you must wait for markets to resolve (days to months)
- **Use for:** Evaluating new tools/prompts before production deployment, building the eval dataset

#### Mode 3: Cached Content Replay (Fast Iteration)

For rapid local iteration (prompt sweeps, parameter search), we need instant results. We can't wait for markets to resolve. Instead, we replay *cached web content* from a point in time before resolution.

```
Timeline:  ──[market opens]──[content snapshot T₀]──────[market resolves T₁]──▶
                                      ↑
                          we cache search results + page content here
                          tools replay against this cache, not live web
```

**How it works:**
1. For each resolved market, we store a *content snapshot*: the web search results and scraped pages that were available at some point before resolution.
2. Several tools already accept `source_links` as a kwarg — we extend this pattern so tools can be fed pre-fetched content instead of doing live search.
3. During benchmarking, the runner injects cached content, bypassing live web search.

```python
# datasets/snapshots/ structure
# One directory per question, containing search results and page content
# captured at a specific timestamp before market resolution
snapshots/
  polymarket_abc123/
    metadata.json        # {"question": "...", "snapshot_at": "2026-01-15", "resolved_at": "2026-02-20"}
    search_results.json  # Google/Serper results as returned at snapshot time
    pages/               # Scraped HTML/text of each URL
      0.txt
      1.txt
      ...
```

Building snapshots:
- **From production:** When the mech runs a prediction in production, cache the search results and scraped content alongside the prediction. This gives us snapshots at the exact time of prediction — perfect temporal alignment.
- **From open markets:** When `tournament.py` runs predictions on open markets, it also caches the web content. When the market resolves, we have both a prediction and its content snapshot.
- **Retroactive (lossy):** For resolved markets where we don't have cached content, we can snapshot today. This introduces temporal contamination for recently resolved markets but is acceptable for markets that resolved > 6 months ago (the web has moved on, outcome-reporting articles are buried).

- **Pros:** Instant results, hermetic (deterministic), can compare tools on identical information
- **Cons:** Doesn't test the tool's search query formulation (queries are fixed), snapshot may not perfectly represent what was available
- **Use for:** Prompt sweeps, parameter search, regression testing in CI

### Which Mode When

| Activity | Mode | Why |
|----------|------|-----|
| Establish production baseline | Production Replay | Ground truth, no contamination |
| Evaluate new tool before deploy | Tournament | Out-of-sample, temporally clean |
| Prompt sweep (50+ variants) | Cached Replay | Need instant results for iteration |
| Parameter grid search | Cached Replay | Same |
| CI regression check on PR | Cached Replay | Must complete in minutes |
| Final validation before promotion | Tournament (if available) or Production Replay | Must be temporally clean |
| Monthly accuracy report | Production Replay | Reflects actual system performance |

---

## Part 2: Datasets — Where Ground Truth Comes From

### 2a. Production Predictions (The Flywheel)

This is the primary dataset. Our mech already makes predictions in production — we capture them and match against outcomes.

**How it works today:** The mech receives Request events on-chain, runs a tool, and emits Deliver events with the prediction (stored on IPFS). The trader then trades based on these predictions. Markets eventually resolve.

**What we add:** `fetch_production.py` reconstructs the production prediction log:

1. **Index on-chain events** — Read Request and Deliver events from the mech marketplace contract. Each Request contains the prompt. Each Deliver contains the prediction result (p_yes, p_no) on IPFS.

2. **Record prediction timing** — Crucially, the block timestamp of the Deliver event tells us *when* the prediction was made. This enables temporal analysis.

3. **Match predictions to outcomes** — For each prediction, look up whether the underlying market has resolved. For Omen: query the subgraph. For Polymarket: query their API. Match by question text (fuzzy) or market ID if available.

4. **Record market context at prediction time** — Where possible, capture the market probability at the time of prediction (from Polymarket/Omen APIs or cached data). This enables edge-over-market measurement.

```jsonl
{"id": "prod_001", "question": "Will X happen?", "tool": "prediction-online", "model": "gpt-4.1-2025-04-14", "p_yes": 0.72, "predicted_at": "2026-01-10T14:23:00Z", "market_prob_at_prediction": 0.65, "resolution": true, "resolved_at": "2026-02-15", "source": "production", "chain": "gnosis", "request_tx": "0xabc...", "time_horizon_days": 36}
```

**Selection bias caveat:** Production data is not a random sample of all markets — the trader selects markets where it expects edge. This means production data is biased toward "tradeable" markets. The benchmark should track this and supplement with broader market data (see 2b) to avoid overfitting to the trader's selection criteria.

**Cadence:** Run `fetch_production.py` weekly. Incremental — only fetches newly resolved markets.

### 2b. Resolved Prediction Markets (Cold Start + Breadth)

Pull recently resolved markets from Polymarket and Omen for breadth beyond what the trader selects.

```jsonl
{"id": "polymarket_abc123", "question": "Will X happen by Y?", "resolution": true, "resolved_at": "2026-02-15", "source": "polymarket", "category": "politics", "final_market_prob": 0.92, "open_date": "2025-11-01", "volume_usd": 450000}
```

**Polymarket** — via their public API (`/markets` endpoint), filter by `resolved=true`.

**Omen** — via the Gnosis conditional tokens subgraph. Query resolved conditions with payouts.

**Important:** These markets can only be used with Cached Content Replay mode (Mode 3) or as tournament inputs *before* they resolve. Never run tools with live web search on already-resolved markets.

### 2c. Open Markets (Tournament Feed)

```jsonl
{"id": "polymarket_xyz789", "question": "Will Y happen by Z?", "source": "polymarket", "category": "crypto", "current_prob": 0.45, "close_date": "2026-06-01", "volume_usd": 120000, "fetched_at": "2026-03-15T10:00:00Z"}
```

`fetch_open.py` pulls currently open markets. These feed into the tournament (Mode 2). When they resolve, they become ground truth data with temporally clean predictions.

### 2d. Dataset Splits

- **eval set** (~100+ questions) — never used during search/sweep, only for final scoring. Must be temporally clean (production replay or tournament predictions only).
- **dev set** (~200+ questions) — used for sweep/search iterations. Can use cached replay.
- **hard set** (~50 questions) — questions where current tools perform worst, to focus improvement.
- **stratified** — all sets should be balanced across categories and time horizons (see Part 3).

The hard set is automatically refreshed: after each full benchmark run, the bottom 20% by Brier score becomes the new hard set.

**Eval set rotation:** The eval set must be refreshed periodically (quarterly) with new tournament results to prevent implicit overfitting. Old eval questions move to the dev set.

---

## Part 3: Scoring — What We Measure

### Primary Metrics

| Metric | Formula | Why it matters |
|--------|---------|----------------|
| **Brier score** | `mean((p_yes - outcome)²)` | Gold standard for probabilistic forecasting. Lower is better. Random = 0.25, perfect = 0.0. |
| **Edge over market** | `mean((market_prob - outcome)² - (p_yes - outcome)²)` | Positive means tool beats market consensus. This is what generates trading profit. |
| **Calibration error (ECE)** | Bin predictions by decile, compare mean prediction to actual frequency | Detects systematic overconfidence/underconfidence. A tool predicting 0.7 should be right ~70% of the time. |
| **Resolution rate** | `valid_json_results / total_questions` | Tool reliability — a tool that crashes 20% of the time is unusable regardless of accuracy. |

**Edge over market is the most important metric for tool selection.** A tool with Brier score 0.20 that always agrees with the market (edge ≈ 0) generates zero trading profit. A tool with Brier score 0.22 that systematically disagrees with the market *in the right direction* (edge > 0) is far more valuable. The trader profits from the *difference* between the tool's prediction and the market price, not from standalone accuracy.

### Secondary Metrics

| Metric | What it measures |
|--------|-----------------|
| **Discrimination (AUC-ROC)** | Can the tool rank questions by likelihood? Separable from calibration — a tool can rank well but be poorly calibrated, and vice versa. |
| **Log loss** | `-mean(outcome * log(p_yes) + (1-outcome) * log(p_no))`. Heavily penalizes confident wrong predictions (p_yes=0.99 when outcome=false). |
| **Sharpness** | `mean(abs(p_yes - 0.5))`. How far from 50/50 are predictions? Higher sharpness with good calibration is ideal. Low sharpness means the tool hedges too much. |
| **Cost per question** | USD per prediction (from token counts). |
| **Latency p50/p95** | Response time distribution. |
| **Simulated PnL** | Simulate Kelly criterion or fixed-fraction betting against market odds using tool predictions. Closest proxy to actual trading value. |

### Stratified Analysis

Aggregate metrics hide important patterns. Always break down by:

**Time horizon** — days between prediction and market resolution:
- Short (< 7 days): tool should rely on current news, events likely already unfolding
- Medium (7-30 days): mix of current evidence and forecasting
- Long (> 30 days): base rates and structural analysis matter more than news

**Category** — politics, crypto, sports, science, geopolitics, economics, etc.

**Market efficiency** — bucket by market volume/liquidity:
- High-volume markets (> $100k) are more efficient (harder to beat)
- Low-volume markets are noisier but offer more edge opportunity

**Prediction timing** — how far before resolution was the prediction made:
- Predictions made 1 day before resolution are much easier than 30 days before
- Normalize by prediction lead time for fair comparison

**Difficulty** — bucket by how close the final market price was to 50%:
- Markets at 90/10 are "easy" (strong prior)
- Markets at 55/45 are "hard" (genuinely uncertain)
- Tool value is highest on hard markets where the market itself is uncertain

```
$ python benchmark/scorer.py results/run.json --stratify

Overall:
  Brier: 0.198  Edge: +0.033  Calibration: 0.045  Resolution: 96.8%

By time horizon:
  Short (<7d):    Brier: 0.142  Edge: +0.051  n=45
  Medium (7-30d): Brier: 0.201  Edge: +0.028  n=89
  Long (>30d):    Brier: 0.247  Edge: +0.019  n=66

By category:
  crypto:     Brier: 0.189  Edge: +0.041  n=52
  politics:   Brier: 0.211  Edge: +0.022  n=73
  sports:     Brier: 0.178  Edge: +0.038  n=38
  other:      Brier: 0.215  Edge: +0.030  n=37

By difficulty (market price distance from 0.5):
  Easy (>0.3):    Brier: 0.098  Edge: +0.012  n=78
  Medium (0.15-0.3): Brier: 0.209  Edge: +0.039  n=68
  Hard (<0.15):   Brier: 0.301  Edge: +0.048  n=54
```

This reveals *where* a tool adds value. A tool might be great at crypto but terrible at politics, or strong on short-horizon but useless on long-horizon questions.

---

## Part 4: Statistical Methodology

### Significance Testing

When comparing two tools or configurations, we need to know if the difference is real or noise.

**Paired bootstrap test** (preferred over t-test):
1. For each question, compute the Brier score difference: `d_i = brier_A(i) - brier_B(i)`
2. Resample `d` with replacement 10,000 times
3. Compute 95% confidence interval on `mean(d)`
4. If the CI excludes zero, the difference is significant

Why paired bootstrap over t-test: prediction errors are not normally distributed (they're bounded in [0,1] and often skewed), and questions are not equally difficult. The bootstrap makes no distributional assumptions.

**Multiple comparison correction:** When testing N prompt variants in a sweep, the probability of at least one false positive is `1 - (1 - α)^N`. With 100 variants at α=0.05, there's a 99.4% chance of at least one spurious "improvement."

Solutions (in order of preference):
1. **Hold-out validation:** Run the sweep on the dev set. Only the single best candidate gets tested on the eval set. This is one comparison, no correction needed.
2. **Bonferroni correction:** Divide α by N. With 100 variants, require p < 0.0005.
3. **False discovery rate (Benjamini-Hochberg):** Less conservative than Bonferroni, controls the expected proportion of false positives among rejections.

### Minimum Detectable Effect

With N questions in the eval set and a given variance in Brier scores, there's a minimum improvement we can reliably detect. For typical prediction markets:

- **N=50 questions**: Can detect Brier improvements > ~0.05 (large)
- **N=100 questions**: Can detect Brier improvements > ~0.035 (medium)
- **N=200 questions**: Can detect Brier improvements > ~0.025 (small)
- **N=500 questions**: Can detect Brier improvements > ~0.015 (fine-grained)

This means: don't waste time on subtle prompt tweaks if your eval set is only 50 questions. Either get more data or focus on changes likely to produce large improvements.

### Overfitting During Search

The prompt evolution loop (Level 2) runs many generations on the dev set. This risks overfitting to dev set quirks.

**Guardrails:**
- Track the **generalization gap**: `dev_brier - eval_brier`. If the gap grows across search iterations, you're overfitting.
- **Early stopping:** If dev performance improves but eval performance plateaus for K generations, stop.
- **Regularize by diversity:** Don't just keep the best-scoring prompts; keep a diverse population (e.g., measure prompt embedding distance, penalize too-similar prompts).
- **Fresh eval set:** Rotate the eval set quarterly using new tournament predictions so it can't be implicitly memorized.

### Correlation Between Questions

Questions from the same time period or category may be correlated (e.g., all crypto questions move together during a market crash). This violates the independence assumption of most tests.

**Mitigation:**
- **Cluster bootstrap:** Resample by time-period clusters rather than individual questions.
- **Stratified eval set:** Ensure the eval set is balanced across categories and time periods, so no single cluster dominates.
- **Report per-stratum results:** If a tool "improves" overall but only because it got better at one category, that's less convincing than broad improvement.

---

## Part 5: Benchmark Runner

### Runner: Direct Tool Invocation

```python
def run_benchmark(
    tools: list[str],
    dataset: str,
    models: list[str],
    prompt_template: str = "default",
    mode: str = "cached_replay",  # "cached_replay", "production_replay", "live" (tournament only)
    sample_n: int | None = None,
    parallel: int = 4,
) -> BenchmarkResult:
    questions = load_dataset(dataset)
    results = []
    for q in questions:
        prompt = render_prompt(prompt_template, q["question"])
        for tool, model in product(tools, models):
            kwargs = build_kwargs(tool, model, prompt, q, mode)
            # In cached_replay mode, kwargs includes source_links from snapshot
            # In production_replay mode, we use the stored prediction (no re-running)
            # In live mode (tournament), tools hit live web search

            t0 = time.time()
            result = run_tool(**kwargs)
            results.append({
                "question_id": q["id"],
                "tool": tool, "model": model,
                "p_yes": result["p_yes"],
                "ground_truth": q["resolution"],
                "market_prob": q.get("market_prob_at_prediction"),
                "predicted_at": datetime.utcnow().isoformat(),
                "time_horizon_days": q.get("time_horizon_days"),
                "category": q.get("category"),
                "latency_s": time.time() - t0,
                "tokens_in": counter.input_tokens,
                "tokens_out": counter.output_tokens,
                "cost_usd": counter.total_cost,
            })
    return BenchmarkResult(results)
```

Reuses the exact `run()` functions and `KeyChain` from the existing codebase — no mocking, no abstraction layer.

### Tournament Runner

```python
# tournament.py — forward-looking predictions on open markets
def run_tournament(
    tools: list[str],
    models: list[str],
    prompt_template: str = "default",
) -> None:
    open_markets = load_dataset("open_markets.jsonl")
    predictions = []

    for q in open_markets:
        prompt = render_prompt(prompt_template, q["question"])
        for tool, model in product(tools, models):
            result, cached_content = run_tool_with_cache(tool, model, prompt)
            predictions.append({
                "question_id": q["id"],
                "tool": tool, "model": model,
                "prompt_template": prompt_template,
                "p_yes": result["p_yes"],
                "predicted_at": datetime.utcnow().isoformat(),
                "market_prob_at_prediction": q["current_prob"],
                "close_date": q["close_date"],
            })
            # Also save content snapshot for future cached replay
            save_snapshot(q["id"], cached_content)

    save_tournament_predictions(predictions)
    # Later: score_tournament.py matches these against resolutions
```

The tournament produces both temporally clean predictions *and* content snapshots that can be used for fast cached replay later. This means every tournament run contributes to both the eval set (via predictions) and the dev set (via cached content for replay).

### Compare: Regression Detection

```
$ python benchmark/compare.py results/baseline.json results/candidate.json

                        baseline  candidate  delta      95% CI          p-value
Brier score             0.231     0.198     -0.033 ↓   [-0.052, -0.014] 0.003 *
Edge over market        0.018     0.044     +0.026 ↑   [+0.008, +0.044] 0.008 *
Calibration (ECE)       0.089     0.064     -0.025 ↓   [-0.041, -0.009] 0.011 *
Discrimination (AUC)    0.72      0.76      +0.04  ↑
Resolution rate         94.1%     96.8%     +2.7%  ↑
Sharpness               0.18      0.22      +0.04  ↑
Avg cost/question       $0.042    $0.038    -$0.004
Avg latency (s)         12.3      11.8      -0.5

Per-tool breakdown (Brier / Edge):
  prediction-online     0.245/+0.012  0.211/+0.038  -0.034/-  +0.026 ↑
  superforcaster        0.218/+0.021  0.189/+0.049  -0.029/-  +0.028 ↑
  prediction-rag        0.229/+0.020  0.194/+0.045  -0.035/-  +0.025 ↑

By time horizon:
  Short (<7d):          0.178     0.149     -0.029
  Medium (7-30d):       0.234     0.201     -0.033
  Long (>30d):          0.278     0.243     -0.035

Generalization check:
  Dev set Brier:        0.191     (gap from eval: 0.007 — OK)

N=200 questions, paired bootstrap 10k resamples, * = p<0.05
```

---

## Part 6: Automated Tool Improvement (The Search)

Instead of manually tweaking tools, we systematically search the space of possible improvements. Each level of search is increasingly powerful but also more expensive.

### Level 1: Parameter Sweep

Exhaustive or random search over hyperparameters that tools already support.

```python
sweep_config = {
    "tools": ["prediction-online", "prediction-request-rag", "superforcaster"],
    "models": ["gpt-4.1-2025-04-14", "claude-sonnet-4-6", "claude-opus-4-6"],
    "prompt_templates": ["default", "superforecaster_v2", "cot_explicit", "base_rate_first"],
    "temperatures": [0.0, 0.2, 0.5],
    "num_urls": [3, 5, 10],
    "num_queries": [1, 2, 3],
}

results = run_sweep(sweep_config, dataset="dev_set.jsonl", sample_n=50, mode="cached_replay")
results.leaderboard()
```

**Output:**

```
Rank  Tool                    Model              Prompt             Temp  Brier  Edge    Cost
1     prediction-request-rag  claude-sonnet-4-6  base_rate_first    0.0   0.178  +0.048  $0.05
2     superforcaster          gpt-4.1            superforecaster_v2 0.2   0.185  +0.041  $0.04
3     prediction-online       claude-sonnet-4-6  cot_explicit       0.0   0.191  +0.035  $0.03
...
```

Run on the dev set using cached replay. The top candidate is then validated on the eval set — this is the single comparison that matters, no multiple comparison correction needed.

### Level 2: Prompt Evolution

Go beyond hand-written templates. Use an LLM to generate and evolve prompts.

```python
def evolve_prompts(
    base_prompts: list[str],
    dataset: str,
    generations: int = 5,
    population: int = 10,
    sample_n: int = 30,
) -> list[PromptResult]:
    population = base_prompts + generate_variants(base_prompts, count=population)

    for gen in range(generations):
        # Score each prompt on a DIFFERENT random sample from dev set each generation
        # (reduces overfitting to specific questions)
        sample = random_sample(dataset, sample_n)
        scores = [(p, benchmark(p, sample, mode="cached_replay")) for p in population]
        scores.sort(key=lambda x: x[1].brier_score)

        # Track generalization gap
        if gen % 2 == 0:
            eval_score = benchmark(scores[0][0], eval_holdout, mode="cached_replay")
            log_generalization_gap(gen, scores[0][1].brier_score, eval_score.brier_score)

        # Keep top half, generate new variants
        survivors = [p for p, _ in scores[:len(scores)//2]]
        new_variants = llm_generate_variants(
            survivors,
            # Feed failure analysis to the LLM so it can learn from mistakes
            failure_examples=get_worst_predictions(survivors[0], sample),
        )
        population = survivors + new_variants

    return scores[0]
```

**What gets varied in prompts:**
- Reasoning structure (CoT steps, role-play, structured analysis)
- Calibration instructions ("consider base rates", "avoid anchoring")
- Information gathering strategy ("focus on recent news", "look for disconfirming evidence")
- Output format instructions (how to express uncertainty)
- Domain-specific priors per question category
- Time-horizon awareness ("this event is X days away, consider...")

### Level 3: Tool Code Modification

Beyond prompt engineering — modify tool code.

```python
def search_tool_variants(
    base_tool: str,
    dataset: str,
    sample_n: int = 30,
) -> ToolVariant:
    source = read_tool_source(base_tool)

    # Analyze failures WITH stratification
    failures = get_worst_predictions(base_tool, dataset)
    failure_analysis = llm_analyze_failures(source, failures)
    # e.g., "Tool overestimates on long-horizon questions (>30d). Web search
    #         retrieves recent news but misses base rates. Also poorly calibrated
    #         on crypto — predicts too extreme (sharpness=0.38 but ECE=0.12)."

    variants = llm_propose_modifications(source, failure_analysis)
    # e.g., "Add time-horizon-aware confidence shrinkage",
    #        "Add base rate lookup before web search",
    #        "Add post-hoc calibration: shrink extreme predictions toward 0.5"

    for variant in variants:
        write_variant_to_sandbox(variant)
        score = benchmark(variant, dataset, sample_n, mode="cached_replay")
        if score.brier < baseline.brier and passes_bootstrap_test(score, baseline):
            return variant

    return None
```

**Types of code modifications to search over:**
- **Search strategy** — query formulation, source filtering, number of queries
- **Reasoning pipeline** — add/remove/reorder reasoning steps
- **Post-processing** — calibration adjustments (Platt scaling, isotonic regression on dev set, shrinkage toward base rates)
- **Time-horizon adaptation** — different strategies for short vs long horizon questions
- **Information extraction** — how web results are summarized and fed to the LLM
- **Category-specific behavior** — different strategies for politics vs crypto vs sports

### Level 4: Tool Composition & Ensembles

Combine existing tools. Ensembles are a known way to improve forecasting — even simple averages of independent forecasters typically outperform individuals.

```python
ensemble_configs = [
    # Simple average (usually strong baseline)
    {"type": "mean", "tools": ["prediction-online", "superforcaster", "prediction-request-rag"]},

    # Weighted by inverse Brier score on dev set
    {"type": "weighted", "tools": ["prediction-online", "superforcaster"], "weight_by": "inverse_brier"},

    # Median (robust to outlier tools giving extreme predictions)
    {"type": "median", "tools": ["prediction-online", "prediction-request-rag", "superforcaster"]},

    # Extremize: average then push away from 0.5 (corrects for shared information)
    {"type": "extremized_mean", "tools": ["prediction-online", "superforcaster"], "extremization": 1.5},

    # Cascade: cheap tool first, expensive tool only if confidence is low
    {"type": "cascade", "tools": ["prediction-offline", "prediction-request-rag"], "threshold": 0.3},

    # Category-specific routing based on stratified performance
    {"type": "router", "rules": {
        "crypto": "prediction-request-rag",
        "politics": "superforcaster",
        "default": "prediction-online",
    }},
]
```

Note on **extremization**: When multiple tools share the same information sources (same web search, same LLM), averaging washes out signal. Extremizing (pushing the average away from 0.5) corrects for this. The optimal extremization factor can be tuned on the dev set.

---

## Part 7: Promotion Pipeline — Local Win to Production

### Step 1: Local Validation (Automated)

```bash
# Run candidate against eval set (held-out, temporally clean)
python benchmark/runner.py --config candidate.yaml --dataset eval_set.jsonl --mode cached_replay

# Compare against current production baseline
python benchmark/compare.py results/prod_baseline.json results/candidate_eval.json

# Promotion gates:
#   1. Edge over market must be positive (tool adds value beyond market consensus)
#   2. Brier improvement over baseline must be significant (bootstrap p < 0.05)
#   3. Resolution rate must not regress
#   4. Cost must not increase by more than 2x
#   5. No stratum regression: tool must not get worse in ANY category/horizon bucket by > 1 SD
```

Gate 5 is important: a tool that improves overall by 0.03 Brier but gets catastrophically worse on crypto questions (because the search compensated by overfitting to politics) is not a safe promotion.

### Step 2: Tournament Validation (if time allows)

Before canary deployment, enter the candidate in the forward-looking tournament for 2-4 weeks. This gives temporally clean out-of-sample results on never-before-seen questions.

### Step 3: Canary Deployment

The mech supports multiple tools via `TOOLS_TO_PACKAGE_HASH`. To canary:

1. **Register the improved tool** as a new IPFS package (e.g., `prediction-online-v2`)
2. **Update `TOOLS_TO_PACKAGE_HASH`** to include both old and new tool
3. **Configure the trader** to route a percentage (e.g., 10-20%) of predictions to the new tool
4. **Monitor production metrics** via `fetch_production.py` for 2-4 weeks

### Step 4: Production Monitoring

```
Tool: prediction-online-v2 (canary, 2 weeks)
  Predictions made:    47
  Markets resolved:    31  (minimum 30 for statistical power)
  Brier score:         0.189  (baseline: 0.224, p=0.04)
  Edge over market:    +0.041 (baseline: +0.018)
  Binary accuracy:     74.2%  (baseline: 69.4%)
  Avg cost:            $0.041
  Trader PnL impact:   +$142 vs counterfactual (estimated)

  By category:
    crypto:    +0.052 edge (n=12)
    politics:  +0.031 edge (n=11)
    other:     +0.039 edge (n=8)

  ⚠ Warning: Only 31 resolved markets. CI on Brier: [0.142, 0.236].
             Consider extending canary for higher confidence.
```

### Step 5: Full Rollout or Rollback

If the canary shows improvement after sufficient resolved markets:
- **Promote:** Update production config to use new version
- **Archive:** Old version's results become the new baseline
- **Feed back:** Production data from the new tool feeds into the next benchmark cycle

If the canary underperforms:
- **Rollback:** Remove from routing
- **Analyze:** Why did benchmark improvement not transfer to production?
  - Distribution shift (different question types in production vs benchmark?)
  - Temporal effects (tool relied on patterns that shifted?)
  - Selection bias (benchmark had different difficulty distribution?)
- **Update benchmark:** Add failing cases to the hard set, adjust dataset composition

---

## Part 8: The Full Cadence

```
Continuous:
  - Tournament runs on newly opened markets (automated)
  - score_tournament.py matches predictions to resolutions as markets close

Weekly:
  - fetch_production.py pulls newly resolved production predictions
  - Canary metrics reviewed (if active canary)

Monthly:
  - fetch_resolved.py refreshes Polymarket/Omen datasets
  - Full benchmark run against all tools (production replay mode)
  - Parameter sweep on dev set (Level 1)
  - Update hard set from worst performers
  - Publish accuracy report with stratified analysis

Quarterly:
  - Prompt evolution search (Level 2)
  - Tool code modification search (Level 3)
  - Ensemble exploration (Level 4)
  - Refresh eval set with recent tournament predictions
  - Promote best candidate through validation → tournament → canary pipeline
```

---

## Part 9: Usage

### Quick Start

```bash
# 1. Refresh datasets
python benchmark/fetch_resolved.py --source polymarket --days 90
python benchmark/fetch_production.py --since 2026-01-01

# 2. Establish baseline (production replay — temporally clean)
python benchmark/runner.py --tools all --dataset production_log.jsonl --mode production_replay --save baseline

# 3. Run parameter sweep (cached replay — fast iteration)
python benchmark/sweep.py --config sweep_config.yaml --dataset dev_set.jsonl --mode cached_replay --sample 50

# 4. Run prompt evolution
python benchmark/search.py --mode prompts --generations 5 --dataset dev_set.jsonl

# 5. Validate winner against eval set
python benchmark/runner.py --config best_candidate.yaml --dataset eval_set.jsonl --mode cached_replay
python benchmark/compare.py results/baseline.json results/best_candidate.json

# 6. Enter tournament for out-of-sample validation
python benchmark/tournament.py --config best_candidate.yaml

# 7. After tournament results look good, promote to canary
python benchmark/promote.py --candidate best_candidate.yaml --canary-pct 10
```

### CI Integration

```yaml
# .github/workflows/benchmark.yml — runs on PRs that touch tools
- name: Quick benchmark (cached replay, no temporal contamination)
  run: |
    python benchmark/runner.py --tools changed --dataset eval_set.jsonl --mode cached_replay --sample 20
    python benchmark/compare.py results/prod_baseline.json results/pr_run.json --fail-on-regression
```

---

## What Makes This SOTA vs the Old Benchmark

| Old (olas-predict-benchmark) | New |
|---|---|
| Autocast academic dataset (2022) | Real Polymarket/Omen markets + production data |
| No temporal controls | Three modes: production replay, tournament, cached replay |
| Binary accuracy only | Brier, edge over market, calibration, discrimination, log loss, sharpness, PnL |
| No market price comparison | Edge over market as primary metric |
| Single prompt template | Automated prompt evolution via LLM |
| No parameter variation | Grid search over models, temps, num_urls, queries |
| No tool improvement | Automated code modification + ensemble search |
| Live web search on resolved markets | Temporally controlled: cached content or pre-resolution predictions |
| No stratified analysis | Breakdown by time horizon, category, difficulty, market efficiency |
| No statistical rigor | Paired bootstrap, multiple comparison correction, minimum detectable effect |
| One-shot run | Continuous tournament + improvement loop with production feedback |
| No deployment pipeline | Tournament validation → canary deployment → statistical monitoring |
| Separate repo with git submodule | In-repo, directly imports tools |
| No production feedback | Production outcomes feed back into benchmark dataset |

## Implementation Plan

1. **Phase 1 — Foundation** (~3 days)
   - `fetch_resolved.py` for Polymarket API + Omen subgraph
   - `runner.py` with cached replay mode
   - `scorer.py` with Brier score, edge over market, calibration, stratified analysis
   - `compare.py` with paired bootstrap significance testing

2. **Phase 2 — Production Flywheel** (~3 days)
   - `fetch_production.py` — index on-chain Request/Deliver events, match to resolved markets, record market prices at prediction time
   - Production replay mode in runner
   - Dataset split management (eval/dev/hard) with stratified balancing

3. **Phase 3 — Tournament** (~2 days)
   - `fetch_open.py` for currently open markets
   - `tournament.py` — run predictions on open markets, cache content snapshots
   - `score_tournament.py` — match stored predictions to resolutions

4. **Phase 4 — Automated Search** (~3 days)
   - `sweep.py` — parameter grid search with dev/eval separation
   - `search.py` — prompt evolution with generalization gap tracking
   - Ensemble testing framework

5. **Phase 5 — Promotion Pipeline** (~2 days)
   - `promote.py` — generate new tool package, update configs
   - Production monitoring integration
   - Canary rollback logic, stratum regression checks

6. **Phase 6 — Tool Code Search** (ongoing)
   - Level 3 code modification search
   - Failure analysis with stratified breakdown
   - Automated PR generation for winning variants
