"""
Generate a human-readable benchmark report.

Reads scores.json (aggregates) and production_log.jsonl (individual rows)
to produce a markdown report with rankings, weak spots, and highlights.

Usage:
    python benchmark/analyze.py
    python benchmark/analyze.py --scores path/to/scores.json --log path/to/log.jsonl --output path/to/report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SCORES = Path(__file__).parent / "results" / "scores.json"
DEFAULT_LOG = Path(__file__).parent / "datasets" / "production_log.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "report.md"

BRIER_RANDOM = 0.25
BRIER_WEAK_THRESHOLD = 0.40
RELIABILITY_ISSUE_THRESHOLD = 0.90
SAMPLE_SIZE_WARNING = 20
TREND_WORSENING_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_scores(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Individual prediction scoring (for top/bottom lists)
# ---------------------------------------------------------------------------


def row_brier(row: dict[str, Any]) -> float | None:
    if (
        row.get("prediction_parse_status") != "valid"
        or row.get("p_yes") is None
        or row.get("final_outcome") is None
    ):
        return None
    outcome = 1.0 if row["final_outcome"] else 0.0
    return (row["p_yes"] - outcome) ** 2


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def section_overall(scores: dict[str, Any]) -> str:
    o = scores["overall"]
    if scores["total_rows"] == 0:
        return "## Overall\n\nNo predictions to score."

    rel_str = f"{o['reliability']:.0%}" if o["reliability"] is not None else "N/A"
    brier_str = str(o["brier"]) if o["brier"] is not None else "N/A"
    acc_str = f"{o['accuracy']:.0%}" if o.get("accuracy") is not None else "N/A"
    sharp_str = f"{o['sharpness']:.4f}" if o.get("sharpness") is not None else "N/A"
    lines = [
        "## Overall",
        "",
        f"- Predictions scored: {scores['valid_rows']} / {scores['total_rows']}"
        f" ({rel_str} reliability)",
        f"- Overall Brier: {brier_str}",
        f"  - 0.0 = perfect, 0.25 = random guessing, 1.0 = maximally wrong",
        f"- Accuracy: {acc_str}",
        f"- Sharpness: {sharp_str}",
        f"  - 0.0 = all predictions at 50/50, 0.5 = maximally decisive",
    ]
    return "\n".join(lines)


def _sample_label(stats: dict[str, Any]) -> str:
    """Return a sample-size label for ranking context."""
    if stats.get("decision_worthy") is False:
        return " ⚠ low sample"
    return ""


def section_tool_ranking(scores: dict[str, Any]) -> str:
    tools = scores.get("by_tool", {})
    ranked = sorted(
        tools.items(),
        key=lambda x: x[1].get("brier") if x[1].get("brier") is not None else 999,
    )

    lines = ["## Tool Ranking", ""]
    for i, (tool, stats) in enumerate(ranked, 1):
        flags = ""
        if stats.get("reliability") is not None and stats["reliability"] < RELIABILITY_ISSUE_THRESHOLD:
            flags = f" — {stats['reliability']:.0%} reliability"
        flags += _sample_label(stats)
        brier = stats["brier"] if stats["brier"] is not None else "N/A"
        acc = f"{stats['accuracy']:.0%}" if stats.get("accuracy") is not None else "N/A"
        sharp = f"{stats['sharpness']:.4f}" if stats.get("sharpness") is not None else "N/A"
        lines.append(
            f"{i}. **{tool}** — Brier: {brier}, Acc: {acc},"
            f" Sharp: {sharp} (n={stats['n']}){flags}"
        )

    return "\n".join(lines)


def section_platform(scores: dict[str, Any]) -> str:
    platforms = scores.get("by_platform", {})
    lines = ["## Platform Comparison", ""]
    for platform, stats in sorted(platforms.items(), key=lambda x: x[1].get("brier") if x[1].get("brier") is not None else 999):
        lines.append(f"- **{platform}**: Brier: {stats['brier']} (n={stats['n']})")
    return "\n".join(lines)


def section_weak_spots(scores: dict[str, Any]) -> str:
    lines = ["## Weak Spots", ""]
    found = False

    for section_name, section_key in [
        ("category", "by_category"),
        ("platform", "by_platform"),
        ("tool", "by_tool"),
    ]:
        for name, stats in (scores.get(section_key) or {}).items():
            brier = stats.get("brier")
            if brier is not None and brier > BRIER_WEAK_THRESHOLD:
                found = True
                label = "anti-predictive (worse than coin flip)" if brier > 0.5 else "weak performance"
                lines.append(
                    f"- **{name}** ({section_name}): Brier {brier:.4f} (n={stats['n']})"
                    f" — {label}"
                )

    if not found:
        lines.append(f"No weak spots detected (all Brier scores below {BRIER_WEAK_THRESHOLD}).")

    return "\n".join(lines)


def section_reliability_issues(scores: dict[str, Any]) -> str:
    lines = ["## Reliability Issues", ""]
    found = False

    for tool, stats in (scores.get("by_tool") or {}).items():
        rel = stats.get("reliability")
        if rel is not None and rel < RELIABILITY_ISSUE_THRESHOLD:
            found = True
            error_pct = (1 - rel) * 100
            lines.append(f"- **{tool}**: {error_pct:.1f}% error/malformed rate")

    if not found:
        lines.append("All tools above 90% reliability.")

    return "\n".join(lines)


def _deduplicate_by_question(
    scored: list[tuple[float, dict[str, Any]]],
    keep: str = "worst",
) -> list[tuple[float, dict[str, Any]]]:
    """Keep only the worst (or best) prediction per unique question."""
    seen: dict[str, tuple[float, dict[str, Any]]] = {}
    for b, row in scored:
        q = row.get("question_text", "")
        if q not in seen:
            seen[q] = (b, row)
        elif keep == "worst" and b > seen[q][0]:
            seen[q] = (b, row)
        elif keep == "best" and b < seen[q][0]:
            seen[q] = (b, row)
    return list(seen.values())


def _format_prediction_list(
    rows: list[dict[str, Any]],
    title: str,
    n: int,
    reverse: bool,
    keep: str,
) -> str:
    scored = []
    for row in rows:
        b = row_brier(row)
        if b is not None:
            scored.append((b, row))

    scored.sort(key=lambda x: x[0], reverse=reverse)
    scored = _deduplicate_by_question(scored, keep=keep)
    scored.sort(key=lambda x: x[0], reverse=reverse)

    lines = [f"## {title}", ""]
    for i, (b, row) in enumerate(scored[:n], 1):
        outcome_str = "Yes" if row["final_outcome"] else "No"
        q = row["question_text"]
        if len(q) > 80:
            q = q[:77] + "..."
        lines.append(
            f'{i}. "{q}"'
            f"\n   {row['tool_name']} predicted p_yes={row['p_yes']:.2f},"
            f" outcome: {outcome_str} (Brier: {b:.4f})"
            f"\n   Category: {row.get('category', '?')},"
            f" Platform: {row['platform']}"
        )

    return "\n".join(lines)


def section_worst_predictions(rows: list[dict[str, Any]], n: int = 10) -> str:
    return _format_prediction_list(rows, "Worst Predictions", n, reverse=True, keep="worst")


def section_best_predictions(rows: list[dict[str, Any]], n: int = 10) -> str:
    return _format_prediction_list(rows, "Best Predictions", n, reverse=False, keep="best")


def section_trend(scores: dict[str, Any]) -> str:
    trend = scores.get("trend", [])
    lines = ["## Trend", ""]

    if not trend:
        lines.append("No trend data available.")
        return "\n".join(lines)

    for entry in trend:
        lines.append(f"- {entry['month']}: Brier {entry['brier']} (n={entry['n']})")

    # Check for worsening
    if len(trend) >= 2:
        prev = trend[-2]["brier"]
        curr = trend[-1]["brier"]
        if prev is not None and curr is not None:
            delta = curr - prev
            if delta > TREND_WORSENING_THRESHOLD:
                lines.append(
                    f"\n**Warning:** Brier worsened by {delta:.4f}"
                    f" ({prev:.4f} → {curr:.4f}) in the last month."
                )

    return "\n".join(lines)


def section_sample_size_warnings(scores: dict[str, Any]) -> str:
    lines = ["## Sample Size Warnings", ""]
    found = False

    for cat, stats in (scores.get("by_category") or {}).items():
        if stats["n"] < SAMPLE_SIZE_WARNING:
            found = True
            lines.append(f"- **{cat}**: only {stats['n']} questions — treat with caution")

    if not found:
        lines.append("All categories have sufficient sample size.")

    return "\n".join(lines)


def section_tool_platform(scores: dict[str, Any]) -> str:
    """Tool × platform cross breakdown table."""
    data = scores.get("by_tool_platform", {})
    if not data:
        return "## Tool × Platform\n\nNo cross-breakdown data available."

    lines = [
        "## Tool × Platform",
        "",
        "| Tool | Platform | Brier | Accuracy | Sharpness | n |",
        "|------|----------|-------|----------|-----------|---|",
    ]
    for key, stats in sorted(
        data.items(),
        key=lambda x: x[1].get("brier") if x[1].get("brier") is not None else 999,
    ):
        parts = key.split(" | ")
        tool = parts[0] if parts else key
        platform = parts[1] if len(parts) > 1 else "?"
        brier = f"{stats['brier']:.4f}" if stats.get("brier") is not None else "N/A"
        acc = f"{stats['accuracy']:.0%}" if stats.get("accuracy") is not None else "N/A"
        sharp = f"{stats['sharpness']:.4f}" if stats.get("sharpness") is not None else "N/A"
        label = _sample_label(stats)
        lines.append(f"| {tool} | {platform} | {brier} | {acc} | {sharp} | {stats['n']}{label} |")

    return "\n".join(lines)


def section_tool_platform_horizon(scores: dict[str, Any]) -> str:
    """Tool × platform × horizon breakdown."""
    data = scores.get("by_tool_platform_horizon", {})
    if not data:
        return "## Tool × Platform × Horizon\n\nNo horizon breakdown data available."

    lines = [
        "## Tool × Platform × Horizon",
        "",
        "| Tool | Platform | Horizon | Brier | Accuracy | n |",
        "|------|----------|---------|-------|----------|---|",
    ]
    for key, horizons in sorted(data.items()):
        parts = key.split(" | ")
        tool = parts[0] if parts else key
        platform = parts[1] if len(parts) > 1 else "?"
        for horizon in ["short_lt_7d", "medium_7_30d", "long_gt_30d"]:
            stats = horizons.get(horizon)
            if not stats or stats.get("n", 0) == 0:
                continue
            brier = f"{stats['brier']:.4f}" if stats.get("brier") is not None else "N/A"
            acc = f"{stats['accuracy']:.0%}" if stats.get("accuracy") is not None else "N/A"
            h_label = {"short_lt_7d": "<7d", "medium_7_30d": "7-30d", "long_gt_30d": ">30d"}[horizon]
            lines.append(f"| {tool} | {platform} | {h_label} | {brier} | {acc} | {stats['n']} |")

    return "\n".join(lines)


def section_calibration(scores: dict[str, Any]) -> str:
    """Calibration analysis — are predictions overconfident or underconfident?"""
    cal = scores.get("calibration", [])
    if not cal:
        return "## Calibration\n\nNo calibration data available."

    lines = [
        "## Calibration",
        "",
        "| Predicted Range | Avg Predicted | Realized Yes-Rate | Gap | n |",
        "|-----------------|---------------|-------------------|-----|---|",
    ]
    for bucket in cal:
        if bucket.get("n", 0) == 0:
            continue
        avg_p = f"{bucket['avg_predicted']:.2f}"
        realized = f"{bucket['realized_rate']:.2f}"
        gap = bucket["gap"]
        gap_str = f"{gap:+.2f}"
        lines.append(f"| {bucket['bin']} | {avg_p} | {realized} | {gap_str} | {bucket['n']} |")

    # Summary interpretation
    lines.append("")
    high_conf = [b for b in cal if b.get("avg_predicted", 0) > 0.7 and b.get("n", 0) > 0]
    low_conf = [b for b in cal if b.get("avg_predicted", 0) < 0.3 and b.get("n", 0) > 0]
    if high_conf:
        avg_gap = sum(b["gap"] for b in high_conf) / len(high_conf)
        if avg_gap < -0.1:
            lines.append("**High-confidence predictions are overconfident** — predicted high yes-probability"
                         " but realized rate is much lower.")
        elif avg_gap > 0.1:
            lines.append("**High-confidence predictions are underconfident** — realized rate exceeds predictions.")
    if low_conf:
        avg_gap = sum(b["gap"] for b in low_conf) / len(low_conf)
        if avg_gap > 0.1:
            lines.append("**Low-confidence predictions are underconfident** — predicted low yes-probability"
                         " but events happen more often than predicted.")

    return "\n".join(lines)


def section_parse_breakdown(rows: list[dict[str, Any]]) -> str:
    """Per-tool parse status breakdown."""
    by_tool: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_tool[row.get("tool_name", "unknown")][row.get("prediction_parse_status", "unknown")] += 1

    lines = [
        "## Parse/Error Breakdown by Tool",
        "",
        "| Tool | Valid | Malformed | Missing | Error | Total |",
        "|------|-------|-----------|---------|-------|-------|",
    ]
    for tool in sorted(by_tool):
        c = by_tool[tool]
        total = sum(c.values())
        lines.append(
            f"| {tool} | {c.get('valid', 0)} | {c.get('malformed', 0)}"
            f" | {c.get('missing_fields', 0)} | {c.get('error', 0)} | {total} |"
        )

    return "\n".join(lines)


def section_latency(rows: list[dict[str, Any]]) -> str:
    """Latency breakdown by tool."""
    by_tool: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        lat = row.get("latency_s")
        if lat is not None and lat > 0:
            by_tool[row.get("tool_name", "unknown")].append(lat)

    if not by_tool:
        return "## Latency\n\nNo latency data available."

    lines = [
        "## Latency (seconds)",
        "",
        "| Tool | Median | Mean | p95 | n |",
        "|------|--------|------|-----|---|",
    ]
    for tool in sorted(by_tool, key=lambda t: statistics.median(by_tool[t])):
        vals = sorted(by_tool[tool])
        med = statistics.median(vals)
        mean = statistics.mean(vals)
        p95_idx = min(int(len(vals) * 0.95), len(vals) - 1)
        p95 = vals[p95_idx]
        lines.append(f"| {tool} | {med:.0f}s | {mean:.0f}s | {p95:.0f}s | {len(vals)} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def generate_report(scores: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    date = scores.get("generated_at", "")[:10] or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    sections = [
        f"# Benchmark Report — {date}",
        section_overall(scores),
        section_tool_ranking(scores),
        section_platform(scores),
        section_tool_platform(scores),
        section_tool_platform_horizon(scores),
        section_calibration(scores),
        section_weak_spots(scores),
        section_reliability_issues(scores),
        section_parse_breakdown(rows),
        section_latency(rows),
        section_worst_predictions(rows),
        section_best_predictions(rows),
        section_trend(scores),
        section_sample_size_warnings(scores),
    ]

    return "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate benchmark report from scores and production log.",
    )
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    scores = load_scores(args.scores)
    rows = load_rows(args.log)
    print(f"Loaded scores ({scores['total_rows']} rows) and {len(rows)} raw rows")

    report = generate_report(scores, rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written to {args.output}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
