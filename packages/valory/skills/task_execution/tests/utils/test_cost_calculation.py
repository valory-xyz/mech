# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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
"""Tests for task_execution.utils.cost_calculation."""

import logging
import math
from typing import Any

import pytest

from packages.valory.skills.task_execution.utils.cost_calculation import (
    DEFAULT_PRICE,
    get_cost_for_done_task,
)


def _task(request_id: str = "req-1", **extra: Any) -> dict:
    """Helper to build a minimal done_task dict."""
    return {"request_id": request_id, **extra}


class TestGetCostForDoneTask:
    """Tests for get_cost_for_done_task."""

    def test_valid_cost_integer(self) -> None:
        """A cost_dict with total_cost=1 → 100 credits."""
        task = _task(cost_dict={"total_cost": 1})
        assert get_cost_for_done_task(task) == 100

    @pytest.mark.parametrize(
        "total_cost, expected",
        [
            (0.01, 1),  # 0.01 * 100 = 1.0 → ceil = 1
            (0.005, 1),  # 0.005 * 100 = 0.5 → ceil = 1
            (0.015, 2),  # 0.015 * 100 = 1.5 → ceil = 2
            (0.5, 50),
            (2.5, 250),
        ],
    )
    def test_valid_cost_fractional_ceil(self, total_cost: float, expected: int) -> None:
        """ceil() is applied correctly for fractional costs."""
        task = _task(cost_dict={"total_cost": total_cost})
        result = get_cost_for_done_task(task)
        assert result == expected
        assert result == math.ceil(total_cost * 100)

    def test_missing_cost_dict_returns_default(self) -> None:
        """No cost_dict key → returns DEFAULT_PRICE."""
        task = _task()
        assert get_cost_for_done_task(task) == DEFAULT_PRICE

    def test_empty_cost_dict_returns_default(self) -> None:
        """Empty cost_dict → returns DEFAULT_PRICE."""
        task = _task(cost_dict={})
        assert get_cost_for_done_task(task) == DEFAULT_PRICE

    def test_empty_cost_dict_logs_warning(self, caplog: Any) -> None:
        """Empty cost_dict logs a warning containing the request_id."""
        task = _task(request_id="rid-42", cost_dict={})
        with caplog.at_level(logging.WARNING):
            get_cost_for_done_task(task)
        assert "rid-42" in caplog.text

    def test_missing_total_cost_returns_default(self) -> None:
        """cost_dict present but no total_cost → returns DEFAULT_PRICE."""
        task = _task(cost_dict={"other_key": 99})
        assert get_cost_for_done_task(task) == DEFAULT_PRICE

    def test_total_cost_none_returns_default(self) -> None:
        """total_cost=None → returns DEFAULT_PRICE."""
        task = _task(cost_dict={"total_cost": None})
        assert get_cost_for_done_task(task) == DEFAULT_PRICE

    def test_total_cost_none_logs_warning(self, caplog: Any) -> None:
        """total_cost=None logs a warning containing the request_id."""
        task = _task(request_id="rid-99", cost_dict={"total_cost": None})
        with caplog.at_level(logging.WARNING):
            get_cost_for_done_task(task)
        assert "rid-99" in caplog.text

    def test_custom_fallback_price(self) -> None:
        """fallback_price parameter is used when cost data is missing."""
        task = _task(cost_dict={})
        assert get_cost_for_done_task(task, fallback_price=42) == 42

    def test_default_price_constant(self) -> None:
        """DEFAULT_PRICE is 1."""
        assert DEFAULT_PRICE == 1
