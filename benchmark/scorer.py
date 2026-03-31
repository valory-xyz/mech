"""
Score production prediction data.

Reads production_log.jsonl and computes:
  - Overall Brier score and reliability
  - Per-tool, per-platform, per-category, per-horizon breakdowns
  - Monthly trend

Usage:
    python benchmark/scorer.py
    python benchmark/scorer.py --input path/to/log.jsonl --output path/to/scores.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).parent / "datasets" / "production_log.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "scores.json"

RELIABILITY_GATE = 0.80


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load rows from a JSONL file."""
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Brier score computation
# ---------------------------------------------------------------------------


def brier_score(p_yes: float, outcome: bool) -> float:
    """Compute Brier score for a single prediction."""
    return (p_yes - (1.0 if outcome else 0.0)) ** 2


def compute_group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Brier score and reliability for a group of rows.

    All rows count toward reliability. Only valid rows with
    final_outcome count toward Brier.
    """
    total = len(rows)
    if total == 0:
        return {"brier": None, "reliability": None, "n": 0}

    valid = [
        r for r in rows
        if r["prediction_parse_status"] == "valid"
        and r["final_outcome"] is not None
        and r["p_yes"] is not None
    ]
    reliability = len(valid) / total

    if not valid:
        return {"brier": None, "reliability": round(reliability, 4), "n": total}

    scores = [brier_score(r["p_yes"], r["final_outcome"]) for r in valid]
    avg_brier = sum(scores) / len(scores)

    return {
        "brier": round(avg_brier, 4),
        "reliability": round(reliability, 4),
        "n": total,
    }


# ---------------------------------------------------------------------------
# Horizon classification
# ---------------------------------------------------------------------------


def classify_horizon(lead_time_days: float | None) -> str:
    """Classify prediction lead time into a horizon bucket."""
    if lead_time_days is None:
        return "unknown"
    if lead_time_days < 7:
        return "short_lt_7d"
    if lead_time_days <= 30:
        return "medium_7_30d"
    return "long_gt_30d"


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    """Group rows by a field value."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get(key, "unknown")].append(row)
    return dict(groups)


def group_by_month(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group rows by month and compute stats per month."""
    months: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        predicted_at = row.get("predicted_at")
        if predicted_at:
            month = predicted_at[:7]  # "YYYY-MM"
            months[month].append(row)

    trend = []
    for month in sorted(months):
        stats = compute_group_stats(months[month])
        trend.append({"month": month, "brier": stats["brier"], "n": stats["n"]})
    return trend


def group_by_horizon(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group rows by prediction lead time horizon."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        horizon = classify_horizon(row.get("prediction_lead_time_days"))
        groups[horizon].append(row)

    return {h: compute_group_stats(group) for h, group in groups.items()}


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all scores from production log rows."""
    total = len(rows)
    valid = [
        r for r in rows
        if r["prediction_parse_status"] == "valid"
        and r["final_outcome"] is not None
        and r["p_yes"] is not None
    ]

    overall = compute_group_stats(rows)

    # Per-tool
    by_tool = {
        tool: compute_group_stats(group)
        for tool, group in group_by(rows, "tool_name").items()
    }

    # Per-platform
    by_platform = {
        platform: compute_group_stats(group)
        for platform, group in group_by(rows, "platform").items()
    }

    # Per-category
    by_category = {
        cat: compute_group_stats(group)
        for cat, group in group_by(rows, "category").items()
    }

    # Per-horizon
    by_horizon = group_by_horizon(rows)

    # Monthly trend
    trend = group_by_month(rows)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_rows": total,
        "valid_rows": len(valid),
        "overall": overall,
        "by_tool": by_tool,
        "by_platform": by_platform,
        "by_category": by_category,
        "by_horizon": by_horizon,
        "trend": trend,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score production prediction data.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    rows = load_rows(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")

    result = score(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Scores written to {args.output}")

    # Print summary
    overall = result["overall"]
    print(f"\nOverall: Brier={overall['brier']}, Reliability={overall['reliability']}, n={overall['n']}")

    if overall["reliability"] is not None and overall["reliability"] < RELIABILITY_GATE:
        print(f"WARNING: Reliability {overall['reliability']} is below {RELIABILITY_GATE} gate")

    print("\nBy tool:")
    for tool, stats in sorted(result["by_tool"].items(), key=lambda x: x[1].get("brier") or 999):
        flag = " UNRELIABLE" if stats["reliability"] is not None and stats["reliability"] < RELIABILITY_GATE else ""
        print(f"  {tool}: Brier={stats['brier']}, Reliability={stats['reliability']}, n={stats['n']}{flag}")

    print("\nBy platform:")
    for platform, stats in result["by_platform"].items():
        print(f"  {platform}: Brier={stats['brier']}, n={stats['n']}")

    print("\nTrend:")
    for entry in result["trend"]:
        print(f"  {entry['month']}: Brier={entry['brier']}, n={entry['n']}")


if __name__ == "__main__":
    main()
