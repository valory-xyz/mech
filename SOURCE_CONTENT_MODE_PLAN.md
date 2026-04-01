# Plan: Store Cleaned Text by Default in `source_content`

## Context

Currently `source_content` stores raw HTML for web pages, which is expensive for on-chain IPFS storage (~50-200KB per page). Tools only use the extracted/cleaned text (~1-5KB) for predictions — the raw HTML is redundant in production.

This plan introduces a `source_content_mode` flag so that production stores only cleaned text (reducing storage cost), while tournament/benchmark can opt into raw HTML when full pipeline replay (including extraction logic testing) is needed.

## Design

**Two flags** in `API_KEYS`:

| Flag | Values | Default | Purpose |
|------|--------|---------|---------|
| `return_source_content` | `"true"` / `"false"` | `"false"` | Whether to include source_content in output (unchanged) |
| `source_content_mode` | `"cleaned"` / `"raw"` | `"cleaned"` | What format to store pages content |

**Stored content includes a `mode` marker** so the replay path knows what format it received:

```python
{"mode": "cleaned", "pages": {url: "extracted text", ...}, "pdfs": {url: "extracted text", ...}}
# or
{"mode": "raw", "pages": {url: "<html>...</html>", ...}, "pdfs": {url: "extracted text", ...}}
```

No existing snapshots exist, so default is `"cleaned"` everywhere (no backward compat concern).

## Config Examples

**Production** (`.1env`) — no change needed, defaults apply:
```json
"return_source_content": ["true"]
```

**Tournament** (benchmark config) — opt into raw:
```json
"return_source_content": ["true"], "source_content_mode": ["raw"]
```

## Changes Per Tool

All 6 tools need 3 changes each. **We will implement in `prediction_request` first and measure storage cost impact before rolling out to the remaining 5 tools.**

### Tools in scope

| # | Tool | File | Has PDFs |
|---|------|------|----------|
| 1 | prediction_request | `packages/valory/customs/prediction_request/prediction_request.py` | Yes |
| 2 | prediction_request_sme | `packages/nickcom007/customs/prediction_request_sme/prediction_request_sme.py` | No |
| 3 | prediction_request_rag | `packages/napthaai/customs/prediction_request_rag/prediction_request_rag.py` | Yes |
| 4 | prediction_request_reasoning | `packages/napthaai/customs/prediction_request_reasoning/prediction_request_reasoning.py` | Yes |
| 5 | prediction_url_cot | `packages/napthaai/customs/prediction_url_cot/prediction_url_cot.py` | Yes |
| 6 | superforcaster | `packages/valory/customs/superforcaster/superforcaster.py` | No (uses Serper JSON) |

### Change 1: Read the new flag (in `run()`)

Add `source_content_mode` read next to existing `return_source_content` read.

| Tool | Line |
|------|------|
| prediction_request | 1125 |
| superforcaster | 391 |
| prediction_request_reasoning | 1237 |
| prediction_url_cot | 940 |
| prediction_request_rag | 1018 |
| prediction_request_sme | 625 |

```python
# Add after existing return_source_content line:
source_content_mode = api_keys.get("source_content_mode", "cleaned")
```

Thread `source_content_mode` through to `extract_texts()` and `fetch_additional_information()`.

### Change 2: Capture path — store cleaned text when mode is `"cleaned"` (in `extract_texts()`)

Currently stores `result.text` (raw HTML) for pages. When mode is `"cleaned"`, store extracted text instead.

| Tool | File | Line | Current code |
|------|------|------|-------------|
| prediction_request | prediction_request.py | 766 | `raw_source_content["pages"][url] = result.text` |
| prediction_request_reasoning | prediction_request_reasoning.py | 707 | same |
| prediction_url_cot | prediction_url_cot.py | 649 | same |
| prediction_request_rag | prediction_request_rag.py | 690 | same |
| prediction_request_sme | prediction_request_sme.py | 401 | same (no PDFs) |
| superforcaster | superforcaster.py | N/A | stores Serper JSON — **no change to content** |

**New logic for 5 web-fetching tools:**

```python
# Add source_content_mode param to extract_texts()
def extract_texts(urls, num_words, source_content_mode="cleaned"):
    raw_source_content = {"mode": source_content_mode, "pages": {}, "pdfs": {}}
    ...
    # For pages:
    doc = extract_text(html=result.text, num_words=num_words)
    if source_content_mode == "raw":
        raw_source_content["pages"][url] = result.text
    else:
        raw_source_content["pages"][url] = doc.text if doc else ""
    ...
```

**Superforcaster:** Stores structured Serper JSON, not HTML. No raw-vs-cleaned distinction. Just add `"mode"` tag for consistency:

```python
captured_source_content = {"mode": source_content_mode, "serper_response": sources_data}
```

### Change 3: Replay path — handle both modes

Currently the replay path always calls `extract_text(html=html)` on cached pages. With `"cleaned"` mode, skip extraction and use directly.

| Tool | File | Lines | Current replay logic |
|------|------|-------|---------------------|
| prediction_request | prediction_request.py | 968-969 | `extract_text(html=html, num_words=num_words)` |
| prediction_request_reasoning | prediction_request_reasoning.py | 1091-1092 | `extract_text(html=html)` |
| prediction_url_cot | prediction_url_cot.py | 836-837 | `extract_text(html=html)` |
| prediction_request_rag | prediction_request_rag.py | 869-870 | `extract_text(html=html)` |
| prediction_request_sme | prediction_request_sme.py | 479-481 | `extract_text(html=html, num_words=num_words)` |
| superforcaster | superforcaster.py | 404-410 | uses Serper JSON directly — **no change** |

**New replay logic for 5 web-fetching tools:**

```python
mode = source_content.get("mode", "cleaned")
for url, content in source_content.get("pages", {}).items():
    if mode == "raw":
        doc = extract_text(html=content, num_words=num_words)
    else:
        doc = ExtendedDocument(text=content, url=url)
```

PDF replay is unchanged — PDFs already store extracted text in both modes.

## Tradeoffs for Cached Replay

| | `mode="raw"` (tournament) | `mode="cleaned"` (production) |
|---|---|---|
| **Storage cost** | High (~50-200KB/page) | Low (estimate TBD — measure after first tool) |
| **Test extraction logic changes** | Yes | No — frozen at capture-time extraction |
| **Test `num_words` variations** | Yes | No — truncation baked in at capture time |
| **Test prompt/model changes** | Yes | Yes |
| **Replay fidelity** | Full pipeline | LLM-only (prompt + model) |

Cached replay with `"cleaned"` mode becomes strictly a **prompt/model evaluation tool**. For retrieval or extraction improvements, use tournament mode with `"raw"`.

## Test Updates

Each tool's existing tests (`test_flag_on_includes_source_content`, `test_flag_off_excludes_source_content`, `test_flag_missing_defaults_off`) remain. Add:

- `test_source_content_mode_cleaned_stores_extracted_text`
- `test_source_content_mode_raw_stores_html`
- `test_source_content_mode_defaults_to_cleaned`
- `test_replay_cleaned_skips_extraction`
- `test_replay_raw_re_extracts`

## Implementation Order

1. Implement in `prediction_request` (tool #1) first
2. Measure storage cost difference (cleaned vs raw) using `measure_source_content_size.py`
3. Implement in remaining 5 tools
4. Update `.example.env` to document `source_content_mode`
5. Update `PROPOSAL.md` benchmark docs
