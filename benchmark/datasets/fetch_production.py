"""
Fetch production prediction data from on-chain subgraphs.

Data flow:
  1. Bulk fetch all recent deliveries from marketplace subgraphs
     → question title, tool, model, p_yes, p_no
  2. Bulk fetch all resolved bets from prediction subgraphs
     → question title, outcome
  3. Match deliveries ↔ resolved markets by question title in memory
  4. Output → production_log.jsonl (append-only)

Usage:
    python benchmark/datasets/fetch_production.py
    python benchmark/datasets/fetch_production.py --lookback-days 30
    python benchmark/datasets/fetch_production.py --output path/to/log.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUESTION_DATA_SEPARATOR = "\u241f"

DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_BATCH_SIZE = 1000
HTTP_TIMEOUT = 30

# Subgraph endpoints — read from env with defaults
PREDICT_OMEN_SUBGRAPH_URL = os.environ.get(
    "PREDICT_OMEN_SUBGRAPH_URL",
    "https://predict-agents.subgraph.autonolas.tech",
)
PREDICT_POLYMARKET_SUBGRAPH_URL = os.environ.get(
    "PREDICT_POLYMARKET_SUBGRAPH_URL",
    "https://predict-polymarket-agents.subgraph.autonolas.tech",
)
MECH_MARKETPLACE_GNOSIS_URL = os.environ.get(
    "MECH_MARKETPLACE_GNOSIS_URL",
    "https://api.subgraph.autonolas.tech/api/proxy/marketplace-gnosis",
)
MECH_MARKETPLACE_POLYGON_URL = os.environ.get(
    "MECH_MARKETPLACE_POLYGON_URL",
    "https://api.subgraph.autonolas.tech/api/proxy/marketplace-polygon",
)

# Category classification keywords
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "crypto": [
        "bitcoin", "btc", "eth", "ethereum", "crypto", "token",
        "defi", "blockchain", "solana", "sol",
    ],
    "politics": [
        "president", "election", "vote", "congress", "senate",
        "parliament", "minister", "governor", "democrat", "republican",
    ],
    "sports": [
        "championship", "league", "cup", "match", "tournament",
        "team", "nba", "nfl", "fifa", "premier league",
    ],
    "economics": [
        "gdp", "inflation", "fed", "interest rate", "unemployment",
        "recession", "federal reserve", "central bank",
    ],
    "tech": [
        "openai", "google", "apple", "microsoft", "launch", "release",
        "software", "chip", "semiconductor",
    ],
    "geopolitics": [
        "war", "conflict", "sanctions", "treaty", "nato", "invasion",
        "military", "ceasefire",
    ],
}

# ---------------------------------------------------------------------------
# GraphQL queries
# ---------------------------------------------------------------------------

# Marketplace: bulk fetch all recent deliveries with prediction data
DELIVERS_QUERY = """
{
  delivers(
    first: %(first)s
    skip: %(skip)s
    orderBy: blockTimestamp
    orderDirection: desc
    where: { blockTimestamp_gt: %(timestamp_gt)s }
  ) {
    id
    blockTimestamp
    model
    toolResponse
    request {
      id
      parsedRequest {
        questionTitle
        tool
      }
    }
  }
}
"""

# Omen: bulk fetch resolved bets (outcome comes with the bet)
OMEN_BETS_QUERY = """
{
  bets(
    first: %(first)s
    skip: %(skip)s
    orderBy: timestamp
    orderDirection: desc
    where: {
      fixedProductMarketMaker_: { currentAnswer_not: null }
      timestamp_gt: %(timestamp_gt)s
    }
  ) {
    id
    timestamp
    outcomeIndex
    fixedProductMarketMaker {
      currentAnswer
      currentAnswerTimestamp
      question
    }
  }
}
"""

# Polymarket: bulk fetch bets, post-filter for resolved
POLYMARKET_BETS_QUERY = """
{
  bets(
    first: %(first)s
    skip: %(skip)s
    orderBy: blockTimestamp
    orderDirection: desc
    where: { blockTimestamp_gt: %(timestamp_gt)s }
  ) {
    id
    blockTimestamp
    outcomeIndex
    question {
      id
      metadata {
        title
        outcomes
      }
      resolution {
        winningIndex
        blockTimestamp
      }
    }
  }
}
"""

# ---------------------------------------------------------------------------
# Subgraph helpers
# ---------------------------------------------------------------------------


def _post_graphql(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Post a GraphQL query and return the JSON response data."""
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    body = resp.json()
    if "errors" in body:
        raise RuntimeError(f"GraphQL errors from {url}: {body['errors']}")
    return body.get("data", {})


def _paginated_fetch(
    url: str,
    query_template: str,
    entity_key: str,
    timestamp_gt: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Fetch all records from a subgraph using pagination."""
    all_records: list[dict[str, Any]] = []
    skip = 0

    while True:
        query = query_template % {
            "first": batch_size,
            "skip": skip,
            "timestamp_gt": timestamp_gt,
        }
        data = _post_graphql(url, {"query": query})
        batch = data.get(entity_key, [])
        if not batch:
            break
        all_records.extend(batch)
        log.info("  fetched %d %s (total %d)", len(batch), entity_key, len(all_records))
        if len(batch) < batch_size:
            break
        skip += batch_size

    return all_records


# ---------------------------------------------------------------------------
# Fetch deliveries (marketplace subgraphs)
# ---------------------------------------------------------------------------


def fetch_deliveries(
    marketplace_url: str,
    timestamp_gt: int,
) -> list[dict[str, Any]]:
    """Bulk fetch all recent deliveries with prediction data.

    Returns list of dicts with: question_title, tool, model, toolResponse, timestamp, deliver_id.
    Skips deliveries with null parsedRequest (IPFS failures on subgraph side).
    """
    raw = _paginated_fetch(
        marketplace_url, DELIVERS_QUERY, "delivers", timestamp_gt,
    )

    deliveries = []
    skipped = 0
    for d in raw:
        request = d.get("request") or {}
        parsed = request.get("parsedRequest")
        if not parsed:
            skipped += 1
            continue

        question_title = _extract_question_title(parsed.get("questionTitle", ""))
        if not question_title:
            skipped += 1
            continue

        deliveries.append({
            "deliver_id": d["id"],
            "timestamp": int(d["blockTimestamp"]),
            "model": d.get("model"),
            "tool_response": d.get("toolResponse"),
            "tool": parsed.get("tool") or "unknown",
            "question_title": question_title,
        })

    if skipped:
        log.info("  skipped %d deliveries with null parsedRequest", skipped)

    return deliveries


# ---------------------------------------------------------------------------
# Fetch resolved markets (prediction subgraphs)
# ---------------------------------------------------------------------------


def fetch_omen_resolved(timestamp_gt: int) -> dict[str, dict[str, Any]]:
    """Bulk fetch resolved Omen markets.

    Returns dict keyed by lowercase question title → {outcome, resolved_at_ts}.
    """
    raw = _paginated_fetch(
        PREDICT_OMEN_SUBGRAPH_URL, OMEN_BETS_QUERY, "bets", timestamp_gt,
    )

    # Deduplicate: multiple bets can reference the same market.
    # We just need the market outcome, keyed by question title.
    markets: dict[str, dict[str, Any]] = {}
    for bet in raw:
        fpmm = bet.get("fixedProductMarketMaker") or {}
        current_answer = fpmm.get("currentAnswer")
        if current_answer is None:
            continue

        try:
            outcome = int(current_answer, 16)
        except (ValueError, TypeError):
            continue

        question_raw = fpmm.get("question", "")
        title = _extract_question_title(question_raw).lower()
        if not title:
            continue

        resolved_at_ts = fpmm.get("currentAnswerTimestamp")
        markets[title] = {
            "outcome": outcome == 1,  # 1 = Yes
            "resolved_at_ts": int(resolved_at_ts) if resolved_at_ts else None,
        }

    return markets


def fetch_polymarket_resolved(timestamp_gt: int) -> dict[str, dict[str, Any]]:
    """Bulk fetch resolved Polymarket markets.

    Returns dict keyed by lowercase question title → {outcome, resolved_at_ts}.
    Post-filters bets to only those with resolved questions.
    """
    raw = _paginated_fetch(
        PREDICT_POLYMARKET_SUBGRAPH_URL, POLYMARKET_BETS_QUERY, "bets", timestamp_gt,
    )

    markets: dict[str, dict[str, Any]] = {}
    for bet in raw:
        question = bet.get("question") or {}
        resolution = question.get("resolution")
        if resolution is None:
            continue

        metadata = question.get("metadata") or {}
        title = (metadata.get("title") or "").strip().lower()
        if not title:
            continue

        winning_index = int(resolution["winningIndex"])
        resolved_at_ts = resolution.get("blockTimestamp")
        markets[title] = {
            "outcome": winning_index == 1,  # 1 = Yes
            "resolved_at_ts": int(resolved_at_ts) if resolved_at_ts else None,
        }

    return markets


# ---------------------------------------------------------------------------
# Question matching
# ---------------------------------------------------------------------------


def _extract_question_title(question: str) -> str:
    """Extract question title using the separator from production code."""
    if not question:
        return ""
    return question.split(QUESTION_DATA_SEPARATOR)[0].strip()


def _match_title(delivery_title: str, markets: dict[str, dict[str, Any]]) -> tuple[Optional[dict[str, Any]], float]:
    """Match a delivery's question title to a resolved market.

    Returns (market_data, match_confidence).
    """
    key = delivery_title.lower()

    # Exact match
    if key in markets:
        return markets[key], 1.0

    # Prefix match (min 20 chars to avoid false positives)
    if len(key) >= 20:
        for market_title, market_data in markets.items():
            if len(market_title) >= 20 and (
                key.startswith(market_title) or market_title.startswith(key)
            ):
                return market_data, 0.8

    return None, 0.0


# ---------------------------------------------------------------------------
# Tool response parsing
# ---------------------------------------------------------------------------


def parse_tool_response(tool_response: Optional[str]) -> dict[str, Any]:
    """Parse a toolResponse JSON string into p_yes, p_no, and parse status."""
    if not tool_response:
        return {
            "p_yes": None,
            "p_no": None,
            "confidence": None,
            "prediction_parse_status": "missing_fields",
        }

    # Check for known IPFS retrieval error messages (only short non-JSON responses)
    if len(tool_response) < 300 and tool_response.lstrip()[:1] != "{":
        lower = tool_response.lower()
        if "could not be retrieved" in lower or "failed to download" in lower:
            return {
                "p_yes": None,
                "p_no": None,
                "confidence": None,
                "prediction_parse_status": "error",
            }

    # Strategy 1: Direct JSON parse
    try:
        data = json.loads(tool_response)
        if isinstance(data, dict):
            p_yes = data.get("p_yes")
            p_no = data.get("p_no")

            if p_yes is not None and p_no is not None:
                p_yes = float(p_yes)
                p_no = float(p_no)

                if 0.0 <= p_yes <= 1.0 and 0.0 <= p_no <= 1.0:
                    return {
                        "p_yes": p_yes,
                        "p_no": p_no,
                        "confidence": float(data["confidence"]) if data.get("confidence") is not None else None,
                        "prediction_parse_status": "valid",
                    }

            return {
                "p_yes": None,
                "p_no": None,
                "confidence": None,
                "prediction_parse_status": "malformed",
            }
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Strategy 2: Regex extraction
    p_yes_match = re.search(r'"p_yes"\s*:\s*([\d.]+)', tool_response)
    p_no_match = re.search(r'"p_no"\s*:\s*([\d.]+)', tool_response)

    if p_yes_match and p_no_match:
        try:
            p_yes = float(p_yes_match.group(1))
            p_no = float(p_no_match.group(1))
            if 0.0 <= p_yes <= 1.0 and 0.0 <= p_no <= 1.0:
                return {
                    "p_yes": p_yes,
                    "p_no": p_no,
                    "confidence": None,
                    "prediction_parse_status": "valid",
                }
        except ValueError:
            pass

    return {
        "p_yes": None,
        "p_no": None,
        "confidence": None,
        "prediction_parse_status": "malformed",
    }


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------


def classify_category(question_text: str) -> str:
    """Classify a question into a category using word-boundary keyword matching."""
    text_lower = question_text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                return category
    return "other"


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def _ts_to_iso(ts: Optional[int]) -> Optional[str]:
    """Convert a unix timestamp to ISO 8601 UTC string."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_row_id(platform: str, deliver_id: str) -> str:
    """Generate a deterministic row ID from platform + deliver_id."""
    h = hashlib.sha256(f"{platform}:{deliver_id}".encode()).hexdigest()[:12]
    return f"prod_{platform}_{h}"


def build_row(
    delivery: dict[str, Any],
    market: dict[str, Any],
    match_confidence: float,
    platform: str,
) -> dict[str, Any]:
    """Build a production_log row from a delivery matched to a resolved market."""
    question_text = delivery["question_title"]
    parsed = parse_tool_response(delivery["tool_response"])
    predicted_at_ts = delivery["timestamp"]
    resolved_at_ts = market["resolved_at_ts"]

    prediction_lead_time_days: Optional[float] = None
    if predicted_at_ts and resolved_at_ts and resolved_at_ts > predicted_at_ts:
        prediction_lead_time_days = round(
            (resolved_at_ts - predicted_at_ts) / 86400, 1
        )

    return {
        "row_id": _make_row_id(platform, delivery["deliver_id"]),
        "platform": platform,
        "question_text": question_text,
        "tool_name": delivery["tool"],
        "model": delivery["model"],
        "p_yes": parsed["p_yes"],
        "p_no": parsed["p_no"],
        "prediction_parse_status": parsed["prediction_parse_status"],
        "final_outcome": market["outcome"],
        "predicted_at": _ts_to_iso(predicted_at_ts),
        "resolved_at": _ts_to_iso(resolved_at_ts),
        "prediction_lead_time_days": prediction_lead_time_days,
        "category": classify_category(question_text),
        "match_confidence": match_confidence,
    }


# ---------------------------------------------------------------------------
# Incremental state & deduplication
# ---------------------------------------------------------------------------


def load_fetch_state(state_path: Path) -> dict[str, Any]:
    """Load incremental fetch state from disk."""
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Could not read fetch state from %s, starting fresh", state_path)
    return {}


def save_fetch_state(state_path: Path, state: dict[str, Any]) -> None:
    """Save incremental fetch state to disk."""
    state_path.write_text(json.dumps(state, indent=2))


def load_existing_row_ids(output_path: Path) -> set[str]:
    """Load existing row IDs from the output JSONL file for deduplication."""
    ids: set[str] = set()
    if not output_path.exists():
        return ids
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                ids.add(row["row_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def append_rows(output_path: Path, rows: list[dict[str, Any]]) -> int:
    """Append rows to the output JSONL file. Returns count of rows written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


# ---------------------------------------------------------------------------
# Pipeline: process one platform
# ---------------------------------------------------------------------------


def process_platform(
    platform: str,
    marketplace_url: str,
    resolved_markets: dict[str, dict[str, Any]],
    timestamp_gt: int,
    existing_ids: set[str],
) -> tuple[list[dict[str, Any]], int]:
    """Process one platform: fetch deliveries, match to resolved markets, build rows.

    Returns (rows, max_delivery_timestamp).
    """
    log.info("%s: fetching deliveries...", platform)
    deliveries = fetch_deliveries(marketplace_url, timestamp_gt)
    log.info("%s: %d deliveries, %d resolved markets", platform, len(deliveries), len(resolved_markets))

    if not deliveries or not resolved_markets:
        return [], 0

    rows: list[dict[str, Any]] = []
    matched = 0
    max_ts = 0

    for delivery in deliveries:
        row_id = _make_row_id(platform, delivery["deliver_id"])
        if row_id in existing_ids:
            continue

        market, confidence = _match_title(delivery["question_title"], resolved_markets)
        if market is None:
            continue

        matched += 1
        row = build_row(delivery, market, confidence, platform)
        rows.append(row)
        max_ts = max(max_ts, delivery["timestamp"])

    log.info(
        "%s: %d deliveries matched to resolved markets, %d rows built",
        platform, matched, len(rows),
    )
    return rows, max_ts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch production prediction data for benchmark scoring.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=int(os.environ.get("BENCHMARK_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS)),
        help="How many days back to fetch (default: 7)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "production_log.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path(__file__).parent / ".fetch_state.json",
        help="Incremental state file path",
    )
    args = parser.parse_args()

    now = int(time.time())
    lookback_ts = now - (args.lookback_days * 86400)

    state = load_fetch_state(args.state_file)
    existing_ids = load_existing_row_ids(args.output)
    log.info("Loaded %d existing row IDs for deduplication", len(existing_ids))

    all_rows: list[dict[str, Any]] = []

    # --- Omen (Gnosis chain) ---
    omen_ts = max(lookback_ts, state.get("omen", {}).get("last_delivery_timestamp", 0))
    log.info("Omen: fetching resolved markets since %s", _ts_to_iso(omen_ts))
    omen_markets = fetch_omen_resolved(omen_ts)

    omen_rows, omen_max_ts = process_platform(
        "omen", MECH_MARKETPLACE_GNOSIS_URL, omen_markets, omen_ts, existing_ids,
    )
    all_rows.extend(omen_rows)

    # --- Polymarket (Polygon chain) ---
    poly_ts = max(lookback_ts, state.get("polymarket", {}).get("last_delivery_timestamp", 0))
    log.info("Polymarket: fetching resolved markets since %s", _ts_to_iso(poly_ts))
    poly_markets = fetch_polymarket_resolved(poly_ts)

    poly_rows, poly_max_ts = process_platform(
        "polymarket", MECH_MARKETPLACE_POLYGON_URL, poly_markets, poly_ts, existing_ids,
    )
    all_rows.extend(poly_rows)

    # Write results
    if all_rows:
        written = append_rows(args.output, all_rows)
        log.info("Appended %d new rows to %s", written, args.output)
    else:
        log.info("No new rows to write")

    # Update incremental state (subtract 1 to catch same-block deliveries on next run)
    if omen_max_ts:
        state["omen"] = {
            "last_delivery_timestamp": omen_max_ts - 1,
            "last_run": _ts_to_iso(now),
        }
    if poly_max_ts:
        state["polymarket"] = {
            "last_delivery_timestamp": poly_max_ts - 1,
            "last_run": _ts_to_iso(now),
        }

    save_fetch_state(args.state_file, state)
    log.info("State saved to %s", args.state_file)

    valid = sum(1 for r in all_rows if r["prediction_parse_status"] == "valid")
    log.info(
        "Summary: %d total rows, %d valid predictions, %d missing/malformed",
        len(all_rows), valid, len(all_rows) - valid,
    )


if __name__ == "__main__":
    main()
