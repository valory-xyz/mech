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
"""Tests for benchmark/scorer.py"""

from typing import Any

import pytest

from benchmark.scorer import (
    brier_score,
    classify_horizon,
    compute_group_stats,
    group_by,
    group_by_horizon,
    group_by_month,
    score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    p_yes: float = 0.5,
    outcome: bool = True,
    status: str = "valid",
    tool: str = "test-tool",
    platform: str = "omen",
    category: str = "other",
    lead_days: float | None = 2.0,
    predicted_at: str = "2026-03-15T10:00:00Z",
) -> dict[str, Any]:
    """Build a minimal production_log row for testing."""
    return {
        "prediction_parse_status": status,
        "p_yes": p_yes if status == "valid" else None,
        "p_no": (1 - p_yes) if status == "valid" else None,
        "final_outcome": outcome,
        "tool_name": tool,
        "platform": platform,
        "category": category,
        "prediction_lead_time_days": lead_days,
        "predicted_at": predicted_at,
    }


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------


class TestBrierScore:

    def test_perfect_prediction_yes(self) -> None:
        assert brier_score(1.0, True) == 0.0

    def test_perfect_prediction_no(self) -> None:
        assert brier_score(0.0, False) == 0.0

    def test_worst_prediction_yes(self) -> None:
        assert brier_score(0.0, True) == 1.0

    def test_worst_prediction_no(self) -> None:
        assert brier_score(1.0, False) == 1.0

    def test_random_guessing(self) -> None:
        assert brier_score(0.5, True) == 0.25
        assert brier_score(0.5, False) == 0.25

    def test_real_example(self) -> None:
        """p_yes=0.13, outcome=True → (0.13 - 1)² = 0.7569"""
        result = brier_score(0.13, True)
        assert abs(result - 0.7569) < 1e-10


# ---------------------------------------------------------------------------
# compute_group_stats
# ---------------------------------------------------------------------------


class TestComputeGroupStats:

    def test_all_valid(self) -> None:
        rows = [
            _row(p_yes=0.9, outcome=True),   # (0.9-1)²  = 0.01
            _row(p_yes=0.8, outcome=False),   # (0.8-0)²  = 0.64
            _row(p_yes=0.6, outcome=True),    # (0.6-1)²  = 0.16
        ]
        stats = compute_group_stats(rows)
        expected_brier = round((0.01 + 0.64 + 0.16) / 3, 4)
        assert stats["brier"] == expected_brier
        assert stats["reliability"] == 1.0
        assert stats["n"] == 3

    def test_mixed_valid_and_malformed(self) -> None:
        rows = [
            _row(p_yes=0.5, outcome=True),
            _row(status="malformed"),
            _row(status="error"),
        ]
        stats = compute_group_stats(rows)
        assert stats["reliability"] == round(1 / 3, 4)
        assert stats["n"] == 3
        # Brier only from the 1 valid row: (0.5 - 1)² = 0.25
        assert stats["brier"] == 0.25

    def test_all_invalid(self) -> None:
        rows = [_row(status="malformed"), _row(status="error")]
        stats = compute_group_stats(rows)
        assert stats["brier"] is None
        assert stats["reliability"] == 0.0
        assert stats["n"] == 2

    def test_empty(self) -> None:
        stats = compute_group_stats([])
        assert stats["brier"] is None
        assert stats["reliability"] is None
        assert stats["n"] == 0


# ---------------------------------------------------------------------------
# classify_horizon
# ---------------------------------------------------------------------------


class TestClassifyHorizon:

    def test_short(self) -> None:
        assert classify_horizon(0.5) == "short_lt_7d"
        assert classify_horizon(6.9) == "short_lt_7d"

    def test_medium(self) -> None:
        assert classify_horizon(7.0) == "medium_7_30d"
        assert classify_horizon(15.0) == "medium_7_30d"
        assert classify_horizon(30.0) == "medium_7_30d"

    def test_long(self) -> None:
        assert classify_horizon(30.1) == "long_gt_30d"
        assert classify_horizon(90.0) == "long_gt_30d"

    def test_none(self) -> None:
        assert classify_horizon(None) == "unknown"


# ---------------------------------------------------------------------------
# group_by
# ---------------------------------------------------------------------------


class TestGroupBy:

    def test_groups_by_tool(self) -> None:
        rows = [
            _row(tool="tool-a"),
            _row(tool="tool-a"),
            _row(tool="tool-b"),
        ]
        groups = group_by(rows, "tool_name")
        assert len(groups["tool-a"]) == 2
        assert len(groups["tool-b"]) == 1

    def test_missing_key_goes_to_unknown(self) -> None:
        rows = [{"other_field": "x"}]
        groups = group_by(rows, "tool_name")
        assert "unknown" in groups


# ---------------------------------------------------------------------------
# group_by_horizon
# ---------------------------------------------------------------------------


class TestGroupByHorizon:

    def test_buckets(self) -> None:
        rows = [
            _row(lead_days=3.0),   # short
            _row(lead_days=15.0),  # medium
            _row(lead_days=45.0),  # long
            _row(lead_days=None),  # unknown
        ]
        result = group_by_horizon(rows)
        assert "short_lt_7d" in result
        assert "medium_7_30d" in result
        assert "long_gt_30d" in result
        assert "unknown" in result
        assert result["short_lt_7d"]["n"] == 1
        assert result["medium_7_30d"]["n"] == 1
        assert result["long_gt_30d"]["n"] == 1


# ---------------------------------------------------------------------------
# group_by_month
# ---------------------------------------------------------------------------


class TestGroupByMonth:

    def test_monthly_trend(self) -> None:
        rows = [
            _row(predicted_at="2026-01-10T10:00:00Z"),
            _row(predicted_at="2026-01-20T10:00:00Z"),
            _row(predicted_at="2026-02-05T10:00:00Z"),
        ]
        trend = group_by_month(rows)
        months = [t["month"] for t in trend]
        assert months == ["2026-01", "2026-02"]
        assert trend[0]["n"] == 2
        assert trend[1]["n"] == 1

    def test_null_predicted_at_excluded(self) -> None:
        rows = [
            _row(predicted_at="2026-03-01T10:00:00Z"),
            _row(predicted_at=None),
        ]
        trend = group_by_month(rows)
        assert len(trend) == 1
        assert trend[0]["n"] == 1


# ---------------------------------------------------------------------------
# score (full pipeline)
# ---------------------------------------------------------------------------


class TestScore:

    def test_full_scoring(self) -> None:
        rows = [
            _row(p_yes=0.9, outcome=True, tool="tool-a", platform="omen", category="crypto"),
            _row(p_yes=0.8, outcome=False, tool="tool-a", platform="polymarket", category="politics"),
            _row(p_yes=0.5, outcome=True, tool="tool-b", platform="omen", category="crypto"),
            _row(status="malformed", tool="tool-b", platform="omen", category="other"),
        ]
        result = score(rows)

        assert result["total_rows"] == 4
        assert result["valid_rows"] == 3

        # Overall
        assert result["overall"]["n"] == 4
        assert result["overall"]["reliability"] == 0.75  # 3/4

        # By tool
        assert "tool-a" in result["by_tool"]
        assert "tool-b" in result["by_tool"]
        assert result["by_tool"]["tool-a"]["n"] == 2
        assert result["by_tool"]["tool-b"]["n"] == 2

        # By platform
        assert result["by_platform"]["omen"]["n"] == 3
        assert result["by_platform"]["polymarket"]["n"] == 1

        # By category
        assert result["by_category"]["crypto"]["n"] == 2

    def test_empty_input(self) -> None:
        result = score([])
        assert result["total_rows"] == 0
        assert result["valid_rows"] == 0
        assert result["overall"]["brier"] is None

    def test_hand_calculated_brier(self) -> None:
        """Verify overall Brier against manual calculation."""
        rows = [
            _row(p_yes=0.13, outcome=True),   # (0.13-1)² = 0.7569
            _row(p_yes=0.90, outcome=True),   # (0.90-1)² = 0.01
            _row(p_yes=0.80, outcome=False),  # (0.80-0)² = 0.64
            _row(p_yes=0.60, outcome=True),   # (0.60-1)² = 0.16
            _row(p_yes=0.30, outcome=False),  # (0.30-0)² = 0.09
        ]
        result = score(rows)
        expected = round((0.7569 + 0.01 + 0.64 + 0.16 + 0.09) / 5, 4)
        assert result["overall"]["brier"] == expected

    def test_output_has_all_keys(self) -> None:
        result = score([_row()])
        expected_keys = [
            "generated_at", "total_rows", "valid_rows", "overall",
            "by_tool", "by_platform", "by_category", "by_horizon", "trend",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
