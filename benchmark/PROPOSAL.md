# Prediction Tool Benchmark & Continuous Improvement System

## Problem

The old benchmark (olas-predict-benchmark, archived) used the Autocast dataset — academic questions from 2022 with pre-scraped source links. It measured binary accuracy only, had no prompt variation support, and couldn't drive iterative improvement. Production accuracy is measured after the fact via trade outcomes on [olas.network](https://olas.network/agent-economies/predict#tool-accuracy), but there is no feedback loop from production back into tool development.

We need a closed-loop system: **benchmark locally → validate → deploy → observe production → feed back → improve → repeat**.

## Core Principles

1. **Temporal integrity is non-negotiable** — never let a tool "predict" something it can Google the answer to.
2. **Production feedback must continuously refresh evaluation data** — production predictions and outcomes feed back into the benchmark dataset.
3. **Market-edge at prediction time is the primary selection objective** — measure whether predictions beat market consensus, not just standalone accuracy.
4. **Reliability and data quality are hard gates, not secondary diagnostics** — a tool that crashes or returns malformed output is unusable regardless of accuracy.
5. **Every reported metric must disclose eligibility denominator and missingness** — no metric without context on what was included and excluded.
6. **No major change is promotable without ablation tests and human review** — improvements must be understood, not just measured.
7. **Promotion decisions require statistical evidence and canary discipline** — no shortcuts from local benchmark to full production.
8. **Platform-aware evaluation** — Omen and Polymarket have different characteristics and must be tracked separately.

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
├── publish.py                      # Upload tool accuracy hashes to IPFS for traders
├── review.py                       # Generate human review sets (wins/losses/outliers)
├── ablation.py                     # Component ablation testing
├── results/                        # Timestamped run results (gitignored)
└── prompts/
    └── templates.py                # Prompt template variants to test
```

---

## Part 1: Data Schema — What We Log

### Versioned Row Schema

Each benchmark row must follow a versioned schema with strict validation at ingest and before scoring. This addresses a current gap: production data logging is insufficient for rigorous evaluation. The schema must be explicit so that missing fields are visible, not silently ignored.

**Required fields per row:**

| Field | Type | Description |
|-------|------|-------------|
| `row_id` | str | Unique identifier |
| `schema_version` | str | Schema version (e.g., `"1.0"`) |
| `mode` | enum | `production_replay` \| `tournament` \| `cached_replay` |
| `question_id` | str | Unique question identifier |
| `market_id` | str | Platform-specific market ID |
| `platform` | enum | `polymarket` \| `omen` |
| `question_text` | str | Full question text |
| `tool_name` | str | Tool identifier |
| `tool_version` | str | Tool version / config hash |
| `model` | str | LLM model used |
| `prompt_template` | str | Prompt template identifier |
| `config_hash` | str | Hash of full tool configuration |
| `predicted_at` | datetime | When the prediction was made |
| `prediction_lead_time_days` | float | Days between prediction and resolution |
| `p_yes` | float | Predicted probability of yes |
| `p_no` | float | Predicted probability of no |
| `prediction_parse_status` | enum | `valid` \| `invalid` \| `malformed` \| `timeout` \| `error` |
| `market_prob_at_prediction` | float? | Market probability when prediction was made |
| `market_prob_source` | str? | Where market prob came from (API, subgraph, etc.) |
| `market_prob_type` | enum? | `mid` \| `last` \| `bid` \| `ask` |
| `market_liquidity_at_prediction` | float? | Market liquidity/volume at prediction time |
| `execution_at` | datetime? | When trade was executed (nullable if no trade) |
| `execution_price` | float? | Trade execution price |
| `execution_size` | float? | Trade size |
| `market_close_at` | datetime? | Market close date |
| `resolved_at` | datetime? | When market resolved |
| `final_outcome` | bool? | Market resolution |
| `latency_ms` | int | Tool execution time |
| `input_tokens` | int | LLM input tokens |
| `output_tokens` | int | LLM output tokens |
| `cost_usd` | float | Total cost |
| `source_snapshot_id` | str? | Reference to cached content snapshot |
| `snapshot_captured_at` | datetime? | When snapshot was taken |
| `snapshot_origin` | enum? | `contemporaneous` \| `retroactive` |
| `match_confidence` | float? | Confidence of market/outcome matching |

**Action item — Trader-side request enrichment:** The trader must include market metadata (liquidity, current probability, spread, market ID, platform) as additional fields in the on-chain mech request. Currently, requests contain only a prompt with the market title. By embedding this context in the request payload:

1. **It lands on IPFS automatically** — the request is stored on-chain with an IPFS hash, so all metadata is preserved at exactly the moment the prediction was requested. No after-the-fact reconstruction needed.
2. **Tools can ignore it** — the metadata is available in `kwargs` but tools that don't need it simply don't read it. No tool changes required.
3. **Benchmark gets ground truth for free** — `fetch_production.py` reads the request from IPFS and has contemporaneous market state without needing to query historical API data (which may not be available).
4. **Future tools can use it** — tools that want to incorporate market price or liquidity into their reasoning can do so without separate API calls.

This is a change to the trader's request construction, not to the mech or tools. The extra fields should include at minimum: `market_id`, `platform`, `market_prob_at_request` (with `market_prob_type`: mid/last/bid/ask), `market_liquidity`, and `market_volume`.

### Completeness & Provenance Flags

Per row, include boolean completeness flags:

| Flag | Meaning |
|------|---------|
| `has_exact_prediction_timestamp` | Prediction time is known precisely (not reconstructed) |
| `has_market_prob_same_timestamp` | Market probability captured at prediction time (not later) |
| `has_evidence_snapshot` | Cached web content available for this prediction |
| `has_execution_data_if_traded` | Trade data present when a trade was made |
| `has_high_confidence_market_match` | Market/outcome matching confidence > threshold |

Per row, include a **provenance grade**:

| Grade | Criteria |
|-------|----------|
| **A** | Contemporaneous capture, high-confidence match, full timing context |
| **B** | Temporally clean but partially missing market/execution context |
| **C** | Reconstructed/retroactive snapshot or weak context |

All reports must stratify by provenance grade. Rows within the same mode are not equally strong by default — provenance flags make this explicit.

### Eligibility Matrix

Every metric report must include: `n_total`, `n_eligible`, `n_excluded`, exclusion reason counts, and row IDs for exclusions.

| Metric | Eligibility Requirements |
|--------|------------------------|
| Forecasting (Brier, log loss, calibration) | Valid prediction + resolved outcome |
| Edge over market | Valid prediction + resolved outcome + market probability at prediction timestamp |
| Execution PnL | Execution timestamp/price/size + mapped market state |
| Reliability | All attempted runs (no exclusions) |

---

## Part 2: Temporal Integrity — The Core Constraint

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
- **Cons:** Limited to questions we've already seen in production, selection bias from trader. **Cannot be used to evaluate new tools** — only tools that already ran in production have stored predictions. To compare a new tool against the production baseline on the same questions, use cached replay (with production-captured snapshots) or tournament mode.
- **Use for:** Measuring accuracy of *current* production tools, establishing baselines, production monitoring, monthly accuracy reports.
- **Not for:** Evaluating candidate tools or comparing alternatives — use cached replay or tournament for that.
- **Primary truth source for published performance.**

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
- **Mandatory out-of-sample validation path for promotion candidates when available.**

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
2. Several tools already accept `source_content` as a kwarg — we extend this pattern so tools can be fed pre-fetched content instead of doing live search.
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

**Prerequisite — all tools must support `source_content`:** All prediction tools now accept a `source_content` kwarg for pre-fetched web content injection. This enables cached replay so that comparisons across tools use the same content snapshot rather than hitting live web, preserving temporal integrity.

The `source_content` format is a structured dict with named keys, extensible without invalidating existing datasets:
- **Web-fetching tools** (prediction_request, prediction_request_sme, prediction_request_rag, prediction_request_reasoning, prediction_url_cot): `{"pages": {url: raw_html, ...}, "pdfs": {url: extracted_text, ...}}`
- **Superforcaster**: `{"serper_response": <full Serper API JSON response>}`

Tools capture `source_content` into `used_params` when `return_source_content` is set to `"true"` in `API_KEYS` (same pattern as `search_provider`). Pages store raw HTML (re-extracted during replay); PDFs store extracted text (since `extract_text_from_pdf` re-downloads from the URL, which can't be replayed).

**Cached replay policy:**
- All tools participating in cached replay must accept `source_content` for evidence injection.
- Rows must carry snapshot provenance and capture timestamp.
- Retroactive snapshots (`snapshot_origin: retroactive`) cannot be mixed silently with contemporaneous snapshots (`snapshot_origin: contemporaneous`) — reports must stratify or filter by snapshot origin.
- **Never the sole basis for production promotion.** Dev/CI/search use only.

**Known limitations of cached replay:**

1. **Retrieval performance is not tested.** Cached replay bypasses the tool's search query formulation — queries are fixed, and the tool receives pre-fetched results. Improvements to how a tool *searches* (query phrasing, source selection, filtering) cannot be evaluated this way. Retrieval improvements require tournament mode. This also means Level 3 code modifications targeting search strategy (query formulation, source filtering) **cannot be evaluated via cached replay** — they must go through the tournament pipeline.

2. **Tools with built-in LLM search cannot be cached.** Some LLM APIs (e.g., models with native web search like Perplexity, or future OpenAI/Anthropic search features) perform retrieval internally. There is no way to inject cached content into these APIs. Such tools can only be evaluated via production replay or tournament mode. This is not a current blocker (no production tools use built-in LLM search today) but must be accounted for as the tool portfolio evolves.

### Which Mode When

| Activity | Mode | Why |
|----------|------|-----|
| Establish production baseline | Production Replay | Ground truth, no contamination |
| Evaluate new tool before deploy | Tournament | Out-of-sample, temporally clean |
| Prompt sweep (50+ variants) | Cached Replay | Need instant results for iteration |
| Parameter grid search | Cached Replay | Same |
| CI regression check on PR | Cached Replay | Must complete in minutes |
| Final validation before promotion | Tournament (if available) | Must be temporally clean + out-of-sample |
| Monthly accuracy report | Production Replay | Reflects actual system performance |
| Compare new tool vs production baseline | Cached Replay (with prod snapshots) or Tournament | Production replay can't score tools that haven't run in production |
| Test retrieval improvements | Tournament | Cached replay can't test search quality |

---

## Part 3: Datasets — Where Ground Truth Comes From

**Scope: binary markets only.** The benchmark supports binary outcome markets (yes/no) only. All tools output `p_yes` / `p_no` that sum to 1. Multi-outcome markets (e.g., "Who will win?" with 5+ candidates) from Polymarket must be filtered out during dataset construction. The scoring framework (Brier score, edge over market, calibration) assumes binary outcomes.

### 3a. Production Predictions (The Flywheel)

This is the primary dataset. Our mech already makes predictions in production — we capture them and match against outcomes.

**How it works today:** The mech receives Request events on-chain, runs a tool, and emits Deliver events with the prediction (stored on IPFS). The trader then trades based on these predictions. Markets eventually resolve.

**What we add:** `fetch_production.py` reconstructs the production prediction log:

1. **Index on-chain events** — Read Request and Deliver events from the mech marketplace contract. Each Request contains the prompt. Each Deliver contains the prediction result (p_yes, p_no) on IPFS.

2. **Record prediction timing** — Crucially, the block timestamp of the Deliver event tells us *when* the prediction was made. This enables temporal analysis.

3. **Match predictions to outcomes** — For each prediction, look up whether the underlying market has resolved. For Omen: query the subgraph. For Polymarket: query their API. Match by question text (fuzzy) or market ID if available. Record `match_confidence` to flag weak matches.

4. **Record market context at prediction time** — Capture market probability, liquidity, and spread at the time of prediction (from Polymarket/Omen APIs or cached data). This enables edge-over-market measurement and sizing analysis.

```jsonl
{"row_id": "prod_001", "schema_version": "1.0", "mode": "production_replay", "question_id": "q_001", "market_id": "poly_abc", "platform": "polymarket", "question_text": "Will X happen?", "tool_name": "prediction-online", "tool_version": "v1.2", "model": "gpt-4.1-2025-04-14", "config_hash": "abc123", "predicted_at": "2026-01-10T14:23:00Z", "prediction_lead_time_days": 36, "p_yes": 0.72, "p_no": 0.28, "prediction_parse_status": "valid", "market_prob_at_prediction": 0.65, "market_prob_source": "polymarket_api", "market_prob_type": "mid", "market_liquidity_at_prediction": 450000, "resolved_at": "2026-02-15", "final_outcome": true, "latency_ms": 12300, "input_tokens": 4200, "output_tokens": 850, "cost_usd": 0.042, "match_confidence": 0.95, "has_exact_prediction_timestamp": true, "has_market_prob_same_timestamp": true, "has_evidence_snapshot": false, "has_high_confidence_market_match": true, "provenance_grade": "A"}
```

**Selection bias caveat:** Production data is not a random sample of all markets — the trader selects markets where it expects edge. This means production data is biased toward "tradeable" markets. The benchmark should track this and supplement with broader market data (see 3b) to avoid overfitting to the trader's selection criteria.

**Cadence:** Run `fetch_production.py` weekly. Incremental — only fetches newly resolved markets.

### 3b. Resolved Prediction Markets (Cold Start + Breadth)

Pull recently resolved markets from Polymarket and Omen for breadth beyond what the trader selects.

```jsonl
{"id": "polymarket_abc123", "question": "Will X happen by Y?", "resolution": true, "resolved_at": "2026-02-15", "source": "polymarket", "category": "politics", "final_market_prob": 0.92, "open_date": "2025-11-01", "volume_usd": 450000}
```

**Polymarket** — via their public API (`/markets` endpoint), filter by `resolved=true`.

**Omen** — via the Gnosis conditional tokens subgraph. Query resolved conditions with payouts.

**Important:** These markets can only be used with Cached Content Replay mode (Mode 3) or as tournament inputs *before* they resolve. Never run tools with live web search on already-resolved markets.

### 3c. Open Markets (Tournament Feed)

```jsonl
{"id": "polymarket_xyz789", "question": "Will Y happen by Z?", "source": "polymarket", "category": "crypto", "current_prob": 0.45, "close_date": "2026-06-01", "volume_usd": 120000, "fetched_at": "2026-03-15T10:00:00Z"}
```

`fetch_open.py` pulls currently open markets. These feed into the tournament (Mode 2). When they resolve, they become ground truth data with temporally clean predictions.

### 3d. Platform-Specific Considerations

Omen and Polymarket markets have structurally different characteristics:

- **Question format:** Omen questions often come from the trader's selection and may be phrased differently than Polymarket questions.
- **Liquidity:** Polymarket markets are typically higher volume, meaning market prices are more efficient (harder to beat).
- **Resolution mechanics:** Different oracles and resolution processes may introduce different kinds of ambiguity.
- **Tool performance may differ systematically by platform** — a tool tuned for Polymarket's question style may underperform on Omen, and vice versa.

**Policy:** All metrics must be reported per-platform in addition to aggregate. If tool performance diverges significantly between platforms, maintain separate evaluation pipelines and consider platform-specific tool configurations. The `platform` field in the row schema enables this stratification.

### 3e. Dataset Splits

- **eval set** (~100+ questions) — never used during search/sweep, only for final scoring. Must be temporally clean (production replay or tournament predictions only).
- **dev set** (~200+ questions) — used for sweep/search iterations. Can use cached replay.
- **hard set** (~50 questions) — questions where current tools perform worst, to focus improvement.
- **stratified** — all sets should be balanced across categories, time horizons, and platforms (see Part 4).

**Refresh cadence is driven by data velocity, not fixed calendar intervals.** At expected throughput (2k-10k mech requests/day), fixed quarterly refreshes risk making the benchmark stale relative to current production behavior and market distribution. Refresh timing is tied to data volume:

- **Dev set and production-monitoring datasets:** Refresh continuously or daily. These are working datasets for iteration and monitoring — staleness directly hinders development.
- **Hard set:** Frozen within a given optimization/promotion cycle (so improvement on hard questions can be tracked), but refreshed on a shorter cadence — weekly or monthly depending on data volume, not quarterly. After each refresh cycle, the bottom 20% by Brier score becomes the new hard set, and the previous hard set is archived with a version tag (e.g., `hard_set_2026_w12.jsonl`).
- **Eval set:** Refreshed monthly (or more frequently if tournament throughput supports it) with new tournament results to prevent implicit overfitting. Old eval questions move to the dev set.

All dataset versions are tagged so that results remain comparable across refresh cycles. The key invariant: hard and eval sets are frozen *within* an optimization cycle but refreshed *between* cycles at a cadence matched to data velocity.

---

## Part 4: Scoring — What We Measure

### Analysis Pipeline (Staged with Gates)

Metrics are not a flat list — they form a staged pipeline where earlier stages gate later ones. This prevents spending time on forecasting quality analysis for a tool that can't reliably produce output.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. RELIABILITY  │────▶│  2. MARKET EDGE   │────▶│    3. BRIER     │────▶│ 4. CALIBRATION  │────▶│    5. PnL       │
│     GATE         │     │                  │     │                 │     │                 │     │                 │
│ valid outputs /  │     │ Primary selection│     │ Primary absolute│     │ Reliability     │     │ Tier 1: simul.  │
│ attempted runs   │     │ metric for       │     │ forecasting     │     │ diagram + ECE   │     │ Tier 2: realized│
│                  │     │ trading value    │     │ quality metric  │     │ (if n sufficient)│    │                 │
│ GATE: <80% →     │     │                  │     │                 │     │                 │     │                 │
│ tool is UNRELI-  │     │                  │     │                 │     │                 │     │                 │
│ ABLE, excluded   │     │                  │     │                 │     │                 │     │                 │
│ from ranking     │     │                  │     │                 │     │                 │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Stage 1 — Reliability Gate:**
- Metric: `valid_structured_outputs / attempted_runs`
- Rule: if < 80%, tool is marked unreliable and excluded from comparative ranking
- Includes all `prediction_parse_status` values: `valid`, `invalid`, `malformed`, `timeout`, `error`
- **Timeouts count as failures.** The benchmark runner must enforce the same `TASK_DEADLINE` (default 240s) as production. A tool that takes 10 minutes per question would time out in production, so it must time out in the benchmark too. Without matching timeouts, reliability numbers are meaningless.
- This is computed on all attempted runs — no exclusions

**Stage 2 — Market-Edge Analysis:**
- Primary selection metric for trading value
- Paired comparison on identical eligible rows only
- Requires: valid prediction + resolved outcome + market probability at prediction timestamp

**Stage 3 — Brier Score Analysis:**
- Primary absolute forecasting quality metric
- Paired comparison on identical eligible rows only

**Stage 4 — Calibration Analysis:**
- Reliability diagram + ECE with minimum 20 samples per bin. With fewer than 200 total questions, use 5 bins instead of 10 deciles. Bins with fewer than 20 samples are reported but excluded from ECE calculation and flagged in output.
- Stratified calibration (by category/time-horizon/platform) requires even larger datasets — only report stratified calibration when each stratum has sufficient samples for at least 5 bins of 20+. Otherwise, report only aggregate calibration and note the limitation.

**Stage 5 — PnL Layer:**
- Tier 1: Execution-aware simulated PnL with spread/slippage assumptions
- Tier 2: Realized execution PnL (when execution data is available)

### Metric Definitions

| Metric | Formula | Why it matters |
|--------|---------|----------------|
| **Reliability** | `valid_outputs / attempted_runs` | Hard gate. A tool that crashes 20% of the time is unusable. |
| **Brier score** | `mean((p_yes - outcome)²)` | Gold standard for probabilistic forecasting. Lower is better. Random = 0.25, perfect = 0.0. |
| **Edge over market** | `mean((market_prob - outcome)² - (p_yes - outcome)²)` | Positive means tool beats market consensus. This is what generates trading profit. |
| **Calibration error (ECE)** | Bin predictions by decile, compare mean prediction to actual frequency | Detects systematic overconfidence/underconfidence. |
| **Resolution rate** | `valid_json_results / total_questions` | Tool reliability — subset of the reliability gate metric. |

**Edge over market is the most important metric for tool selection.** A tool with Brier score 0.20 that always agrees with the market (edge ≈ 0) generates zero trading profit. A tool with Brier score 0.22 that systematically disagrees with the market *in the right direction* (edge > 0) is far more valuable. The trader profits from the *difference* between the tool's prediction and the market price, not from standalone accuracy.

### Secondary Metrics

| Metric | What it measures |
|--------|-----------------|
| **Discrimination (AUC-ROC)** | Can the tool rank questions by likelihood? |
| **Log loss** | `-mean(outcome * log(p_yes) + (1-outcome) * log(p_no))`. Heavily penalizes confident wrong predictions. |
| **Sharpness** | `mean(abs(p_yes - 0.5))`. How far from 50/50 are predictions? Higher sharpness with good calibration is ideal. |
| **Cost per question** | USD per prediction (from token counts). |
| **Latency p50/p95** | Response time distribution. |
| **Simulated PnL** | Simulate Kelly criterion or fixed-fraction betting against market odds. Closest proxy to actual trading value. |

### Stratified Analysis

Aggregate metrics hide important patterns. Always break down by:

**Platform** — Polymarket vs Omen:
- Different market efficiency, liquidity profiles, question styles
- Tool performance may diverge systematically

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

**Provenance grade** — A/B/C (see Part 1):
- Ensures results aren't driven by low-quality data rows

```
$ python benchmark/scorer.py results/run.json --stratify

Overall (n=200, eligible=194, excluded=6):
  Reliability: 97.0%  Brier: 0.198  Edge: +0.033  Calibration: 0.045

By platform:
  polymarket: Brier: 0.191  Edge: +0.038  n=128
  omen:       Brier: 0.209  Edge: +0.025  n=66

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

By provenance:
  A: Brier: 0.195  Edge: +0.035  n=142
  B: Brier: 0.204  Edge: +0.029  n=41
  C: Brier: 0.210  Edge: +0.025  n=11
```

---

## Part 5: Statistical Methodology

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
- **Stratified eval set:** Ensure the eval set is balanced across categories, time periods, and platforms, so no single cluster dominates.
- **Report per-stratum results:** If a tool "improves" overall but only because it got better at one category, that's less convincing than broad improvement.

---

## Part 6: Benchmark Runner

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
            # In cached_replay mode, kwargs includes source_content from snapshot
            # In production_replay mode, we score stored predictions (no re-running)
            #   — this means only tools that ran in production can be scored in this mode
            # In live mode (tournament), tools hit live web search

            t0 = time.time()
            result = run_tool(**kwargs)
            latency_ms = int((time.time() - t0) * 1000)

            # Classify parse status
            parse_status = classify_parse_status(result)

            results.append({
                "row_id": generate_row_id(),
                "schema_version": "1.0",
                "mode": mode,
                "question_id": q["id"],
                "market_id": q.get("market_id"),
                "platform": q.get("platform"),
                "question_text": q["question"],
                "tool_name": tool,
                "model": model,
                "prompt_template": prompt_template,
                "p_yes": result.get("p_yes"),
                "p_no": result.get("p_no"),
                "prediction_parse_status": parse_status,
                "final_outcome": q.get("resolution"),
                "market_prob_at_prediction": q.get("market_prob_at_prediction"),
                "predicted_at": datetime.utcnow().isoformat(),
                "prediction_lead_time_days": q.get("prediction_lead_time_days"),
                "category": q.get("category"),
                "latency_ms": latency_ms,
                "input_tokens": counter.input_tokens,
                "output_tokens": counter.output_tokens,
                "cost_usd": counter.total_cost,
            })
    return BenchmarkResult(results)
```

Reuses the exact `run()` functions and `KeyChain` from the existing codebase — no mocking, no abstraction layer.

**Tool compatibility note:** All prediction tools must support `source_content` for evidence injection (see cached replay prerequisite in Part 2). The runner should validate this at startup and fail fast if a tool lacks support, rather than silently skipping it.

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
                "tool_ipfs_hash": get_tool_hash(tool),  # Pin exact tool code version
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

**Tool versioning:** Tournament predictions store `tool_ipfs_hash` — the IPFS hash of the exact tool code that ran. This is important because local tool code and IPFS-deployed code can diverge. When scoring tournament results weeks later, the hash provides an audit trail of exactly which code version produced the prediction.

### Compare: Regression Detection

```
$ python benchmark/compare.py results/baseline.json results/candidate.json

                        baseline  candidate  delta      95% CI          p-value
Reliability             94.1%     96.8%     +2.7%  ↑
Brier score             0.231     0.198     -0.033 ↓   [-0.052, -0.014] 0.003 *
Edge over market        0.018     0.044     +0.026 ↑   [+0.008, +0.044] 0.008 *
Calibration (ECE)       0.089     0.064     -0.025 ↓   [-0.041, -0.009] 0.011 *
Discrimination (AUC)    0.72      0.76      +0.04  ↑
Sharpness               0.18      0.22      +0.04  ↑
Avg cost/question       $0.042    $0.038    -$0.004
Avg latency (s)         12.3      11.8      -0.5

Per-tool breakdown (Brier / Edge):
  prediction-online     0.245/+0.012  0.211/+0.038  -0.034/-  +0.026 ↑
  superforcaster        0.218/+0.021  0.189/+0.049  -0.029/-  +0.028 ↑
  prediction-rag        0.229/+0.020  0.194/+0.045  -0.035/-  +0.025 ↑

By platform:
  polymarket            0.221     0.189     -0.032
  omen                  0.245     0.212     -0.033

By time horizon:
  Short (<7d):          0.178     0.149     -0.029
  Medium (7-30d):       0.234     0.201     -0.033
  Long (>30d):          0.278     0.243     -0.035

Eligibility: n_total=200, n_eligible=194, n_excluded=6
  excluded: 3 timeout, 2 malformed, 1 no_market_prob
Generalization check:
  Dev set Brier:        0.191     (gap from eval: 0.007 — OK)

N=200 questions, paired bootstrap 10k resamples, * = p<0.05
```

---

## Part 7: Automated Tool Improvement (The Search)

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
- **Search strategy** — query formulation, source filtering, number of queries. **Note: search strategy changes cannot be evaluated via cached replay** (cached replay bypasses the search pipeline entirely). These modifications must be validated through tournament mode.
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

**Cost-performance tradeoffs:** Ensembles and cascades multiply cost. A 3-tool ensemble costs ~3x per question. The benchmark must report cost-adjusted metrics alongside raw metrics to make fair comparisons:

| Metric | Formula | Use |
|--------|---------|-----|
| **Edge per dollar** | `edge / cost_per_question` | Is the ensemble's edge gain worth the cost? |
| **Brier improvement per dollar** | `(baseline_brier - candidate_brier) / cost_per_question` | Cost-normalized quality gain |
| **Break-even trade size** | `cost_per_question / edge` | Minimum trade size where the tool pays for itself |

A $0.15/question ensemble with +0.048 edge may be worse than a $0.05/question single tool with +0.035 edge, depending on trade size. The cascade config partially addresses this (cheap tool first, expensive tool only if uncertain), but the leaderboard should always include cost-adjusted columns so the tradeoff is visible.

---

## Part 8: Ablation Policy

For every promotion candidate, ablation testing is mandatory. The purpose is to understand *why* a tool performs better — not just *that* it does. Without ablation, we risk promoting complexity that adds no value or masks regressions in specific components.

### Required Ablations

Run the full candidate system and each ablation on the same data slice:

| Ablation | What it tests |
|----------|--------------|
| No retrieval/live search | Is the tool's value coming from web search or from LLM reasoning? |
| No base-rate module | Does base-rate awareness help? |
| No calibration layer | Does post-hoc calibration add value? |
| No post-processing/shrinkage | Is confidence shrinkage contributing? |
| No ensemble/routing | Is the ensemble better than its best individual tool? |
| No category-specific rules | Are category-specific strategies helping or overfitting? |

### Ablation Report

For each ablation, report delta vs full candidate across:
- Reliability
- Edge over market
- Brier score
- Calibration (ECE)
- PnL (both tiers)

### Promotion Rules Based on Ablation

- If removing a component does **not hurt** any primary metric → component is non-essential complexity. Remove it.
- If removing a component **improves** primary metrics → component is harmful. Remove it.
- No major architecture change is trusted without ablation survival.

---

## Part 9: Human-in-the-Loop Review

Automated metrics are necessary but not sufficient. Before any promotion, mandatory human review must be completed.

### Review Set (Generated by `review.py`)

After each benchmark run or canary cycle, generate a structured review set:

- Top 10 losses (highest Brier score per question)
- Top 10 wins (lowest Brier score per question where tool disagreed with market)
- Largest market disagreements (where tool and market diverged most)
- All invalid/malformed outputs
- All low-confidence market/outcome matches (`match_confidence` below threshold)

### Cause Taxonomy

Each reviewed case must be labeled with a root cause:

| Cause | Description |
|-------|-------------|
| `reasoning_error` | Tool had correct information but drew wrong conclusion |
| `missing_bad_evidence` | Tool lacked or had incorrect source material |
| `market_snapshot_error` | Market probability at prediction time was wrong or stale |
| `resolution_ambiguity` | Market resolution was ambiguous or disputed |
| `formatting_parse_failure` | Tool produced output that couldn't be parsed |
| `execution_timing_issue` | Prediction timing relative to market was problematic |
| `leakage_matching_issue` | Temporal leakage or incorrect question-market matching |

### Feedback Loop

Review artifacts feed back into:
- **Hard set refresh** — cases with `reasoning_error` or `missing_bad_evidence` become hard set candidates
- **Data quality fixes** — cases with `market_snapshot_error` or `leakage_matching_issue` trigger data pipeline fixes
- **Search constraints** — failure patterns inform what the automated search should target
- **Ablation priorities** — which components to ablate based on observed failure modes

### Review Outcomes

Each reviewed tool/configuration is classified as one of:
- **Good** — passes all checks, suitable for production use
- **Good for specific tests** — passes for certain categories/horizons but not others
- **Needs fix** — reliability or data quality issues must be addressed first
- **Promotion candidate** — passes all gates, ready for canary deployment

---

## Part 10: Promotion Pipeline — Local Win to Production

### Step 1: Local Validation (Automated)

```bash
# Run candidate against eval set (held-out, temporally clean)
python benchmark/runner.py --config candidate.yaml --dataset eval_set.jsonl --mode cached_replay

# Compare against current production baseline
python benchmark/compare.py results/prod_baseline.json results/candidate_eval.json

# Promotion gates:
#   1. Reliability gate: ≥80% valid outputs (hard gate)
#   2. Edge over market must be positive (tool adds value beyond market consensus)
#   3. Brier improvement over baseline must be significant (bootstrap p < 0.05)
#   4. Resolution rate must not regress
#   5. Cost must not increase by more than 2x
#   6. No stratum regression: tool must not get worse in ANY category/horizon/platform
#      bucket by > 1 SD
```

Gate 6 is important: a tool that improves overall by 0.03 Brier but gets catastrophically worse on crypto questions (because the search compensated by overfitting to politics) is not a safe promotion.

### Step 2: Ablation Validation (Mandatory)

Run ablation tests (see Part 8). All components of the candidate must demonstrate positive contribution. Non-essential complexity must be removed before promotion.

### Step 3: Human Review (Mandatory)

Complete the human review process (see Part 9). Review must be documented with cause taxonomy labels. Tool must be classified as "Promotion candidate" before proceeding.

### Step 4: Tournament Validation (if time allows)

Before canary deployment, enter the candidate in the forward-looking tournament for 2-4 weeks. This gives temporally clean out-of-sample results on never-before-seen questions.

### Step 5: Canary Deployment

The mech supports multiple tools via `TOOLS_TO_PACKAGE_HASH`. To canary:

1. **Register the improved tool** as a new IPFS package (e.g., `prediction-online-v2`)
2. **Update `TOOLS_TO_PACKAGE_HASH`** to include both old and new tool
3. **Configure the trader** to route a percentage (e.g., 10-20%) of predictions to the new tool
4. **Canary routing must be randomized** by request/market unit — not sequential or by category
5. **Predefine minimum resolved sample** before any promotion decision (minimum 30 resolved markets)
6. **Monitor production metrics** via `fetch_production.py` for 2-4 weeks

### Step 6: Production Monitoring

```
Tool: prediction-online-v2 (canary, 2 weeks)
  Predictions made:    47
  Markets resolved:    31  (minimum 30 for statistical power)
  Brier score:         0.189  (baseline: 0.224, p=0.04)
  Edge over market:    +0.041 (baseline: +0.018)
  Binary accuracy:     74.2%  (baseline: 69.4%)
  Avg cost:            $0.041
  Trader PnL impact:   +$142 vs counterfactual (estimated)

  By platform:
    polymarket:  +0.045 edge (n=19)
    omen:        +0.035 edge (n=12)

  By category:
    crypto:    +0.052 edge (n=12)
    politics:  +0.031 edge (n=11)
    other:     +0.039 edge (n=8)

  ⚠ Warning: Only 31 resolved markets. CI on Brier: [0.142, 0.236].
             Consider extending canary for higher confidence.
```

### Step 7: Full Rollout or Rollback

**Fixed decision rules (defined ex-ante):**
- Edge > 0 and better than baseline with CI support
- No reliability regression
- No severe stratum collapse (any platform/category/horizon bucket)
- Bounded cost increase (< 2x)

**If canary passes:**
- **Promote:** Update production config to use new version
- **Archive:** Old version's results become the new baseline, config hash frozen as rollback target
- **Publish accuracy:** Upload tool accuracy hash to IPFS for traders (see Part 12)
- **Feed back:** Production data from the new tool feeds into the next benchmark cycle

**If canary underperforms:**
- **Rollback:** Remove from routing, revert to old version (kept with frozen config hash)
- **Analyze:** Why did benchmark improvement not transfer to production?
  - Distribution shift (different question types in production vs benchmark?)
  - Temporal effects (tool relied on patterns that shifted?)
  - Selection bias (benchmark had different difficulty distribution?)
  - Platform effects (canary ran on different Omen/Polymarket mix than benchmark?)
- **Update benchmark:** Add failing cases to the hard set, adjust dataset composition

**Automatic rollback conditions (defined ex-ante):**
- Reliability drops below 80%
- Edge turns negative with >20 resolved markets
- Any single stratum collapses by >2 SD vs baseline

---

## Part 11: The Full Cadence

```
Continuous / Daily:
  - Tournament runs on newly opened markets (automated)
  - score_tournament.py matches predictions to resolutions as markets close
  - Dev set and production-monitoring datasets refresh with new data
  - fetch_production.py pulls newly resolved production predictions (daily at expected volume)

Weekly:
  - Canary metrics reviewed (if active canary)
  - Hard set refresh (if within an optimization cycle boundary)
  - Quick benchmark regression check against production baseline

Monthly:
  - fetch_resolved.py refreshes Polymarket/Omen datasets
  - Full benchmark run against all tools (production replay mode)
  - Parameter sweep on dev set (Level 1)
  - Eval set rotation with recent tournament predictions
  - Publish accuracy report with stratified analysis (per-platform, per-category, etc.)

Per optimization cycle (frequency driven by improvement velocity, typically 1-3 months):
  - Prompt evolution search (Level 2)
  - Tool code modification search (Level 3)
  - Ensemble exploration (Level 4)
  - Promote best candidate through validation → ablation → human review → tournament → canary
  - Hard set frozen during cycle, refreshed at cycle boundary
```

---

## Part 12: Additional Capabilities

### Resolve Market Reasoning Tool

The benchmark system should also be used to evaluate and improve the **resolve market reasoning tool** — the tool that determines whether a market has resolved and what the outcome is. Incorrect resolution reasoning leads to wrong ground truth labels, which corrupt the entire benchmark.

**Approach:**
- Maintain a curated set of markets with known, human-verified resolutions
- Benchmark the resolve tool against this set separately
- Track resolution accuracy, ambiguous-case handling, and edge cases
- Improvements to the resolve tool should go through the same promotion pipeline

### IPFS Accuracy Publication

Traders need access to tool accuracy metrics to make informed routing decisions. After each promotion or monthly accuracy report:

1. `publish.py` generates a structured accuracy report (JSON) with per-tool, per-platform, per-category metrics
2. Upload to IPFS and record the hash
3. Update the tool's metadata to reference the accuracy hash
4. Traders can fetch and verify tool accuracy before routing requests

```bash
# Publish accuracy report to IPFS
python benchmark/publish.py --results results/monthly_report.json --output-hash
# Returns: QmXyz... (IPFS hash for trader consumption)
```

---

## Part 13: Usage

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

# 6. Run ablation tests
python benchmark/ablation.py --candidate best_candidate.yaml --dataset eval_set.jsonl

# 7. Generate human review set
python benchmark/review.py --results results/best_candidate.json --output review_set/

# 8. Enter tournament for out-of-sample validation
python benchmark/tournament.py --config best_candidate.yaml

# 9. After tournament results look good + human review complete, promote to canary
python benchmark/promote.py --candidate best_candidate.yaml --canary-pct 10

# 10. Publish accuracy to IPFS for traders
python benchmark/publish.py --results results/monthly_report.json
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
| Binary accuracy only | Staged pipeline: reliability → edge → Brier → calibration → PnL |
| No market price comparison | Edge over market as primary metric |
| Single prompt template | Automated prompt evolution via LLM |
| No parameter variation | Grid search over models, temps, num_urls, queries |
| No tool improvement | Automated code modification + ensemble search |
| Live web search on resolved markets | Temporally controlled: cached content or pre-resolution predictions |
| No stratified analysis | Breakdown by platform, time horizon, category, difficulty, provenance |
| No statistical rigor | Paired bootstrap, multiple comparison correction, minimum detectable effect |
| One-shot run | Continuous tournament + improvement loop with production feedback |
| No deployment pipeline | Ablation → human review → tournament → canary → statistical monitoring |
| Separate repo with git submodule | In-repo, directly imports tools |
| No production feedback | Production outcomes feed back into benchmark dataset |
| No data quality tracking | Versioned schema, completeness flags, provenance grades, eligibility matrix |
| No ablation testing | Mandatory component ablation before promotion |
| No human review | Structured review with cause taxonomy feeding back into system |
| No accuracy publication | IPFS-published accuracy hashes for trader consumption |
| No platform-specific analysis | Per-platform (Omen vs Polymarket) metrics and evaluation |

## Implementation Plan

### First Sprint — Smallest End-to-End Slice

The first sprint delivers a minimal but complete pipeline: real production data in, scored baseline report out. This validates the schema, data path, and scoring logic before building anything else.

1. **`schema.py` + validation**
   - Implement the benchmark row schema in code
   - Enforce `prediction_parse_status`, completeness flags, provenance grade, and eligibility-related fields
   - Add fixture rows and validator tests

2. **`fetch_production.py` on a narrow slice**
   - Ingest a limited time window of real mech Request/Deliver data
   - Normalize into benchmark rows conforming to the schema
   - Preserve missingness explicitly instead of silently dropping rows
   - Output a first `production_log.jsonl`

3. **`scorer.py` with the gated baseline metrics**
   - Reliability gate first
   - Then Brier score
   - Then edge-over-market where eligible
   - Include `n_total`, `n_eligible`, `n_excluded`, exclusion reasons, and timing

4. **Baseline reporting**
   - Produce one human-readable and one machine-readable baseline report from real production rows
   - Answer: how much data is valid, how much is eligible, and what is missing most often

**In parallel with the first sprint:** ~~Audit all prediction tools for `source_content` support and retrofit those that lack it (superforcaster, gemini-prediction).~~ Done — all online prediction tools now support `source_content` with structured capture. This unblocks cached replay without waiting for the sprint to complete.

### Phased Rollout

0. **Phase 0 — Prerequisites** (before benchmark work begins)
   - Coordinate trader-side request enrichment (market metadata in request payload)

1. **Phase 1 — Foundation** (~3 days)
   - Row schema definition with validation
   - `fetch_resolved.py` for Polymarket API + Omen subgraph (binary markets only)
   - `runner.py` with cached replay mode, production timeout enforcement (`TASK_DEADLINE`)
   - `scorer.py` with staged pipeline: reliability gate (80%) → edge → Brier → calibration (min 20/bin) → stratified analysis (per-platform)
   - `compare.py` with paired bootstrap significance testing, eligibility reporting, cost-adjusted metrics

2. **Phase 2 — Production Flywheel** (~3 days)
   - `fetch_production.py` — index on-chain Request/Deliver events, match to resolved markets, record market prices + liquidity at prediction time, assign provenance grades
   - Production replay mode in runner
   - Dataset split management (eval/dev/hard) with stratified balancing across categories and platforms

3. **Phase 3 — Tournament** (~2 days)
   - `fetch_open.py` for currently open markets
   - `tournament.py` — run predictions on open markets, cache content snapshots
   - `score_tournament.py` — match stored predictions to resolutions

4. **Phase 4 — Automated Search** (~3 days)
   - `sweep.py` — parameter grid search with dev/eval separation
   - `search.py` — prompt evolution with generalization gap tracking
   - Ensemble testing framework

5. **Phase 5 — Validation & Promotion Pipeline** (~3 days)
   - `ablation.py` — component ablation testing framework
   - `review.py` — human review set generation with cause taxonomy
   - `promote.py` — generate new tool package, update configs, with all gates
   - `publish.py` — IPFS accuracy hash publication for traders
   - Production monitoring integration with automatic rollback conditions
   - Canary rollback logic, stratum regression checks

6. **Phase 6 — Tool Code Search & Resolve Tool** (ongoing)
   - Level 3 code modification search
   - Failure analysis with stratified breakdown
   - Automated PR generation for winning variants
   - Resolve market reasoning tool benchmark pipeline

### Prerequisites (Blocking)

1. **Retrofit all tools with `source_content` support.** All online prediction tools now support `source_content` with structured capture via the `return_source_content` flag. Offline tools (e.g. gemini-prediction) don't fetch web content and don't need this. This blocks Phase 1.
2. **Trader-side request enrichment.** The trader must embed market metadata (market ID, platform, probability, liquidity, volume, spread) in the mech request payload. This is a trader repo change. The data lands on IPFS as part of the request and is available for benchmark analysis without reconstruction. This is a prerequisite for edge and PnL analysis on production data.

### Open Questions

1. **Built-in LLM search tools:** If/when tools using models with native web search are added, they cannot participate in cached replay. The benchmark must gracefully handle mixed-mode evaluation where some tools run in tournament mode and others in cached replay.
2. **Fuzzy matching risk for production data:** `fetch_production.py` must match production predictions (prompt strings) to market resolutions. Prompts are not always identical to market titles — the trader may rephrase, add context, or combine questions. A bad fuzzy match silently assigns wrong ground truth. The `match_confidence` field flags weak matches, but the threshold for exclusion and the validation process for matching quality need to be defined during implementation.
