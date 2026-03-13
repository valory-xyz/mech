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
"""Tests for task_execution.utils.benchmarks."""

import logging

import pytest

from packages.valory.skills.task_execution.utils.benchmarks import (
    PRICE_NUM_TOKENS,
    TokenCounterCallback,
)


def _dummy_counter(prompt: str, model: str) -> int:
    """Dummy token counter: counts words in prompt."""
    return len(prompt.split())


MODEL = "gpt-4"
INPUT_PRICE = TokenCounterCallback.TOKEN_PRICES[MODEL]["input"]
OUTPUT_PRICE = TokenCounterCallback.TOKEN_PRICES[MODEL]["output"]


class TestTokenCounterCallbackInit:
    """Tests for TokenCounterCallback.__init__."""

    def test_initial_cost_dict_zeros(self) -> None:
        """Test initial cost_dict values are all zero."""
        cb = TokenCounterCallback()
        for v in cb.cost_dict.values():
            assert v == 0

    def test_actual_model_is_none(self) -> None:
        """Test actual_model is None after init."""
        cb = TokenCounterCallback()
        assert cb.actual_model is None


class TestTokenToCost:
    """Tests for TokenCounterCallback.token_to_cost."""

    def test_calculation(self) -> None:
        """Test token_to_cost calculation."""
        result = TokenCounterCallback.token_to_cost(1000, MODEL, "input")
        assert result == pytest.approx(1000 / PRICE_NUM_TOKENS * INPUT_PRICE)

    def test_zero_tokens(self) -> None:
        """Test token_to_cost with zero tokens returns 0.0."""
        assert TokenCounterCallback.token_to_cost(0, MODEL, "output") == 0.0


class TestCalculateCost:
    """Tests for TokenCounterCallback.calculate_cost."""

    def test_with_token_key(self) -> None:
        """Test calculate_cost with input_tokens key."""
        cb = TokenCounterCallback()
        cb.calculate_cost("input", MODEL, _dummy_counter, input_tokens=500)
        assert cb.cost_dict["input_tokens"] == 500
        expected_cost = TokenCounterCallback.token_to_cost(500, MODEL, "input")
        assert cb.cost_dict["input_cost"] == pytest.approx(expected_cost)

    def test_with_prompt_key(self) -> None:
        """Test calculate_cost with input_prompt key."""
        cb = TokenCounterCallback()
        prompt = "hello world foo"  # 3 tokens from _dummy_counter
        cb.calculate_cost("input", MODEL, _dummy_counter, input_prompt=prompt)
        assert cb.cost_dict["input_tokens"] == 3

    def test_missing_both_keys_logs_warning_and_returns_early(self, caplog) -> None:  # type: ignore
        """Test calculate_cost logs warning and returns early when both keys missing."""
        cb = TokenCounterCallback()
        with caplog.at_level(logging.WARNING):
            cb.calculate_cost("input", MODEL, _dummy_counter)
        assert "input" in caplog.text
        # cost_dict should remain unchanged — no tokens counted
        assert cb.cost_dict["input_tokens"] == 0
        assert cb.cost_dict["input_cost"] == 0


class TestTokenCounterCallbackCall:
    """Tests for TokenCounterCallback.__call__."""

    def test_call_records_model(self) -> None:
        """Test __call__ records the model."""
        cb = TokenCounterCallback()
        cb(MODEL, _dummy_counter, input_tokens=100, output_tokens=50)
        assert cb.actual_model == MODEL

    def test_call_updates_totals(self) -> None:
        """Test __call__ updates total tokens and cost."""
        cb = TokenCounterCallback()
        cb(MODEL, _dummy_counter, input_tokens=1000, output_tokens=500)
        assert cb.cost_dict["total_tokens"] == 1500
        expected_total = TokenCounterCallback.token_to_cost(
            1000, MODEL, "input"
        ) + TokenCounterCallback.token_to_cost(500, MODEL, "output")
        assert cb.cost_dict["total_cost"] == pytest.approx(expected_total)

    def test_unsupported_model_raises_value_error(self) -> None:
        """Test __call__ raises ValueError for unsupported model."""
        cb = TokenCounterCallback()
        with pytest.raises(ValueError, match="not supported"):
            cb("unknown-model-xyz", _dummy_counter, input_tokens=10, output_tokens=10)

    def test_exception_in_calculate_cost_is_caught(self, caplog) -> None:  # type: ignore
        """If calculate_cost raises internally, the error is logged, not re-raised."""
        cb = TokenCounterCallback()

        # Force an error by providing a token_counter that raises
        def bad_counter(prompt: str, model: str) -> int:
            raise RuntimeError("counter broke")

        with caplog.at_level(logging.ERROR):
            cb(MODEL, bad_counter, input_prompt="test prompt")
        assert "Error in TokenCounterCallback" in caplog.text
