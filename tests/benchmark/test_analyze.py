# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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
"""Tests for benchmark/analyze.py"""

from typing import Any

import pytest

from benchmark.analyze import (
    generate_report,
    row_brier,
    section_overall,
    section_sample_size_warnings,
    section_trend,
    section_weak_spots,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scores(
    brier: float | None = 0.3,
    reliability: float | None = 0.95,
    total: int = 100,
    valid: int = 95,
    by_tool: dict | None = None,
    by_platform: dict | None = None,
    by_category: dict | None = None,
    trend: list | None = None,
) -> dict[str, Any]:
    """Build a minimal scores dict for testing."""
    return {
        "generated_at": "2026-03-31T06:00:00Z",
        "total_rows": total,
        "valid_rows": valid,
        "overall": {"brier": brier, "reliability": reliability, "n": total},
        "by_tool": by_tool or {},
        "by_platform": by_platform or {},
        "by_category": by_category or {},
        "by_horizon": {},
        "trend": trend or [],
    }


# ---------------------------------------------------------------------------
# section_overall
# ---------------------------------------------------------------------------


class TestSectionOverall:

    def test_normal(self) -> None:
        result = section_overall(_scores(brier=0.31, reliability=0.95))
        assert "0.31" in result
        assert "95%" in result

    def test_empty_dataset(self) -> None:
        result = section_overall(_scores(brier=None, reliability=None, total=0, valid=0))
        assert "No predictions to score" in result

    def test_all_invalid(self) -> None:
        result = section_overall(_scores(brier=None, reliability=0.0, total=5, valid=0))
        assert "N/A" in result  # Brier is N/A


# ---------------------------------------------------------------------------
# section_weak_spots
# ---------------------------------------------------------------------------


class TestSectionWeakSpots:

    def test_anti_predictive_label(self) -> None:
        """Brier > 0.5 should say 'anti-predictive'."""
        s = _scores(by_category={"social": {"brier": 0.81, "n": 100, "reliability": 0.9}})
        result = section_weak_spots(s)
        assert "anti-predictive" in result

    def test_weak_performance_label(self) -> None:
        """Brier between 0.4 and 0.5 should say 'weak performance'."""
        s = _scores(by_category={"tech": {"brier": 0.45, "n": 100, "reliability": 0.9}})
        result = section_weak_spots(s)
        assert "weak performance" in result
        assert "anti-predictive" not in result

    def test_no_weak_spots(self) -> None:
        s = _scores(by_category={"crypto": {"brier": 0.2, "n": 100, "reliability": 0.9}})
        result = section_weak_spots(s)
        assert "No weak spots" in result

    def test_threshold_boundary(self) -> None:
        """Brier exactly at threshold (0.40) should NOT be flagged."""
        s = _scores(by_tool={"test": {"brier": 0.40, "n": 50, "reliability": 1.0}})
        result = section_weak_spots(s)
        assert "No weak spots" in result


# ---------------------------------------------------------------------------
# section_trend
# ---------------------------------------------------------------------------


class TestSectionTrend:

    def test_worsening_alert(self) -> None:
        trend = [
            {"month": "2026-01", "brier": 0.20, "n": 50},
            {"month": "2026-02", "brier": 0.25, "n": 60},  # +0.05 > 0.02
        ]
        result = section_trend(_scores(trend=trend))
        assert "Warning" in result

    def test_no_alert_when_stable(self) -> None:
        trend = [
            {"month": "2026-01", "brier": 0.20, "n": 50},
            {"month": "2026-02", "brier": 0.21, "n": 60},  # +0.01 < 0.02
        ]
        result = section_trend(_scores(trend=trend))
        assert "Warning" not in result

    def test_empty_trend(self) -> None:
        result = section_trend(_scores(trend=[]))
        assert "No trend data" in result


# ---------------------------------------------------------------------------
# section_sample_size_warnings
# ---------------------------------------------------------------------------


class TestSectionSampleSizeWarnings:

    def test_small_category_warned(self) -> None:
        s = _scores(by_category={"weather": {"brier": 0.3, "n": 4, "reliability": 1.0}})
        result = section_sample_size_warnings(s)
        assert "weather" in result
        assert "4 questions" in result

    def test_large_category_not_warned(self) -> None:
        s = _scores(by_category={"crypto": {"brier": 0.3, "n": 200, "reliability": 1.0}})
        result = section_sample_size_warnings(s)
        assert "sufficient sample size" in result


# ---------------------------------------------------------------------------
# row_brier
# ---------------------------------------------------------------------------


class TestRowBrier:

    def test_valid_row(self) -> None:
        row = {"prediction_parse_status": "valid", "p_yes": 0.9, "final_outcome": True}
        assert abs(row_brier(row) - 0.01) < 1e-10

    def test_invalid_row_returns_none(self) -> None:
        row = {"prediction_parse_status": "malformed", "p_yes": None, "final_outcome": True}
        assert row_brier(row) is None

    def test_missing_outcome_returns_none(self) -> None:
        row = {"prediction_parse_status": "valid", "p_yes": 0.5, "final_outcome": None}
        assert row_brier(row) is None


# ---------------------------------------------------------------------------
# generate_report (integration)
# ---------------------------------------------------------------------------


class TestGenerateReport:

    def test_has_all_sections(self) -> None:
        s = _scores(
            by_tool={"tool-a": {"brier": 0.3, "n": 50, "reliability": 1.0}},
            by_platform={"omen": {"brier": 0.4, "n": 50, "reliability": 1.0}},
            by_category={"crypto": {"brier": 0.2, "n": 50, "reliability": 1.0}},
            trend=[{"month": "2026-03", "brier": 0.3, "n": 50}],
        )
        rows = [
            {"prediction_parse_status": "valid", "p_yes": 0.9, "final_outcome": True,
             "question_text": "Will X?", "tool_name": "tool-a", "platform": "omen",
             "category": "crypto"},
        ]
        report = generate_report(s, rows)

        assert "# Benchmark Report" in report
        assert "## Overall" in report
        assert "## Tool Ranking" in report
        assert "## Platform Comparison" in report
        assert "## Weak Spots" in report
        assert "## Reliability Issues" in report
        assert "## Worst Predictions" in report
        assert "## Best Predictions" in report
        assert "## Trend" in report
        assert "## Sample Size Warnings" in report

    def test_empty_data_no_crash(self) -> None:
        s = _scores(brier=None, reliability=None, total=0, valid=0)
        report = generate_report(s, [])
        assert "# Benchmark Report" in report
        assert "No predictions to score" in report
