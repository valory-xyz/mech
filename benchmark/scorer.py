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
MIN_SAMPLE_SIZE = 30


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
        return {
            "brier": None, "accuracy": None, "sharpness": None,
            "reliability": None, "n": 0, "valid_n": 0,
            "decision_worthy": False,
        }

    valid = [
        r for r in rows
        if r["prediction_parse_status"] == "valid"
        and r["final_outcome"] is not None
        and r["p_yes"] is not None
    ]
    reliability = len(valid) / total
    worthy = len(valid) >= MIN_SAMPLE_SIZE

    if not valid:
        return {
            "brier": None, "accuracy": None, "sharpness": None,
            "reliability": round(reliability, 4),
            "n": total, "valid_n": 0,
            "decision_worthy": False,
        }

    scores = [brier_score(r["p_yes"], r["final_outcome"]) for r in valid]
    avg_brier = sum(scores) / len(scores)

    # p_yes == 0.5 counted as incorrect (no directional signal)
    correct = sum(
        1 for r in valid
        if (r["p_yes"] > 0.5) == r["final_outcome"]
    )
    accuracy = correct / len(valid)

    sharpness = sum(abs(r["p_yes"] - 0.5) for r in valid) / len(valid)

    return {
        "brier": round(avg_brier, 4),
        "accuracy": round(accuracy, 4),
        "sharpness": round(sharpness, 4),
        "reliability": round(reliability, 4),
        "n": total,
        "valid_n": len(valid),
        "decision_worthy": worthy,
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
    """Group rows by month and compute full stats per month."""
    months: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        predicted_at = row.get("predicted_at")
        if predicted_at:
            month = predicted_at[:7]  # "YYYY-MM"
            months[month].append(row)

    trend = []
    for month in sorted(months):
        stats = compute_group_stats(months[month])
        stats["month"] = month
        trend.append(stats)
    return trend


def group_by_horizon(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group rows by prediction lead time horizon."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        horizon = classify_horizon(row.get("prediction_lead_time_days"))
        groups[horizon].append(row)

    return {h: compute_group_stats(group) for h, group in groups.items()}


def _composite_key(row: dict[str, Any], fields: list[str]) -> str:
    """Build a composite grouping key from multiple fields."""
    return " | ".join(str(row.get(f, "unknown")) for f in fields)


def group_by_composite(
    rows: list[dict[str, Any]],
    fields: list[str],
    *,
    horizon: bool = False,
) -> dict[str, Any]:
    """Group rows by a composite key, optionally sub-grouped by horizon.

    When *horizon* is False, returns ``{key: stats}``.
    When *horizon* is True, returns ``{key: {horizon_bucket: stats}}``.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _composite_key(row, fields)
        groups[key].append(row)

    if not horizon:
        return {k: compute_group_stats(g) for k, g in groups.items()}

    result: dict[str, dict[str, Any]] = {}
    for key, group in groups.items():
        result[key] = group_by_horizon(group)
    return result


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

CALIBRATION_BINS = [
    (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
    (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01),
]


def _bin_label(lo: float, hi: float) -> str:
    """Human-readable label for a calibration bin."""
    hi_display = 1.0 if hi > 1.0 else hi
    return f"{lo:.1f}-{hi_display:.1f}"


def compute_calibration(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bucket valid predictions by p_yes range, compare predicted vs realized.

    Returns a list of bin dicts sorted by probability range:
    ``{"bin": "0.7-0.8", "avg_predicted": 0.74, "realized_rate": 0.81, "n": 42, "gap": -0.07}``

    Positive gap = overconfident (predicted > realized).
    Negative gap = underconfident (predicted < realized).
    """
    valid = [
        r for r in rows
        if r.get("prediction_parse_status") == "valid"
        and r.get("final_outcome") is not None
        and r.get("p_yes") is not None
    ]

    bins: dict[str, list[dict[str, Any]]] = {
        _bin_label(lo, hi): [] for lo, hi in CALIBRATION_BINS
    }

    for row in valid:
        p = row["p_yes"]
        for lo, hi in CALIBRATION_BINS:
            if lo <= p < hi:
                bins[_bin_label(lo, hi)].append(row)
                break

    result = []
    for lo, hi in CALIBRATION_BINS:
        label = _bin_label(lo, hi)
        group = bins[label]
        if not group:
            continue
        avg_pred = sum(r["p_yes"] for r in group) / len(group)
        realized = sum(1 for r in group if r["final_outcome"]) / len(group)
        gap = round(avg_pred - realized, 4)
        result.append({
            "bin": label,
            "avg_predicted": round(avg_pred, 4),
            "realized_rate": round(realized, 4),
            "n": len(group),
            "gap": gap,
        })

    return result


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all scores from production log rows."""
    total = len(rows)
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

    # Tool × platform cross breakdown
    by_tool_platform = group_by_composite(rows, ["tool_name", "platform"])

    # Tool × platform × horizon breakdown
    by_tool_platform_horizon = group_by_composite(
        rows, ["tool_name", "platform"], horizon=True,
    )

    # Monthly trend
    trend = group_by_month(rows)

    # Calibration — overall and per-tool
    calibration = compute_calibration(rows)
    calibration_by_tool = {
        tool: compute_calibration(group)
        for tool, group in group_by(rows, "tool_name").items()
    }

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_rows": total,
        "valid_rows": overall["valid_n"],
        "overall": overall,
        "by_tool": by_tool,
        "by_platform": by_platform,
        "by_category": by_category,
        "by_horizon": by_horizon,
        "by_tool_platform": by_tool_platform,
        "by_tool_platform_horizon": by_tool_platform_horizon,
        "trend": trend,
        "calibration": calibration,
        "calibration_by_tool": calibration_by_tool,
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
    print(
        f"\nOverall: Brier={overall['brier']}, Accuracy={overall['accuracy']},"
        f" Sharpness={overall['sharpness']}, Reliability={overall['reliability']},"
        f" n={overall['n']}"
    )

    if overall["reliability"] is not None and overall["reliability"] < RELIABILITY_GATE:
        print(f"WARNING: Reliability {overall['reliability']} is below {RELIABILITY_GATE} gate")

    print("\nBy tool (decision-worthy):")
    ranked = sorted(result["by_tool"].items(), key=lambda x: x[1].get("brier") or 999)
    for tool, stats in ranked:
        flags = []
        if stats["reliability"] is not None and stats["reliability"] < RELIABILITY_GATE:
            flags.append("UNRELIABLE")
        if not stats["decision_worthy"]:
            flags.append(f"LOW-SAMPLE<{MIN_SAMPLE_SIZE}")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  {tool}: Brier={stats['brier']}, Acc={stats['accuracy']}, Sharp={stats['sharpness']}, n={stats['n']}{suffix}")

    print("\nBy platform:")
    for platform, stats in result["by_platform"].items():
        print(f"  {platform}: Brier={stats['brier']}, n={stats['n']}")

    print("\nBy tool × platform:")
    for key, stats in sorted(
        result["by_tool_platform"].items(),
        key=lambda x: x[1].get("brier") or 999,
    ):
        print(f"  {key}: Brier={stats['brier']}, n={stats['n']}")

    print("\nBy tool × platform × horizon:")
    for key, horizons in sorted(result["by_tool_platform_horizon"].items()):
        print(f"  {key}:")
        for h, stats in sorted(horizons.items()):
            print(f"    {h}: Brier={stats['brier']}, n={stats['n']}")

    print("\nTrend:")
    for entry in result["trend"]:
        print(f"  {entry['month']}: Brier={entry['brier']}, Acc={entry['accuracy']}, n={entry['n']}")

    print("\nCalibration (overall):")
    print(f"  {'Bin':<10} {'Predicted':>10} {'Realized':>10} {'Gap':>8} {'n':>6}")
    for b in result["calibration"]:
        direction = "over" if b["gap"] > 0 else "under" if b["gap"] < 0 else ""
        print(
            f"  {b['bin']:<10} {b['avg_predicted']:>10.4f} {b['realized_rate']:>10.4f}"
            f" {b['gap']:>+8.4f} {b['n']:>6}  {direction}"
        )


if __name__ == "__main__":
    main()
