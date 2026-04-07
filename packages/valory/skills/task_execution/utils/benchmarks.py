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
"""Benchmarking for tools."""

import logging
from typing import Any, Callable, Dict, Optional, Union

PRICE_NUM_TOKENS = 1000


class TokenCounterCallback:
    """Callback to count the number of tokens used in a generation."""

    TOKEN_PRICES = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
        "gpt-4o-2024-08-06": {"input": 0.0025, "output": 0.01},
        "gpt-4.1-2025-04-14": {"input": 0.002, "output": 0.008},
        "claude-2": {"input": 0.008, "output": 0.024},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
        "claude-4-sonnet-20250514": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "cohere/command-r-plus": {"input": 0.003, "output": 0.015},
        "databricks/dbrx-instruct:nitro": {"input": 0.0009, "output": 0.0009},
        "nousresearch/nous-hermes-2-mixtral-8x7b-sft": {
            "input": 0.00054,
            "output": 0.00054,
        },
        "openai/gpt-4.1:online": {"input": 0.002, "output": 0.008},
        "x-ai/grok-4.1-fast:online": {"input": 0.0002, "output": 0.0005},
        "google/gemini-2.5-flash:online": {"input": 0.0003, "output": 0.0025},
        "anthropic/claude-haiku-4.5:online": {"input": 0.001, "output": 0.005},
        "anthropic/claude-sonnet-4:online": {"input": 0.003, "output": 0.015},
    }

    def __init__(self) -> None:
        """Initialize the callback."""
        self.actual_model: Optional[str] = None
        self.cost_dict: Dict[str, Union[int, float]] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_cost": 0,
            "output_cost": 0,
            "extra_cost": 0,
            "total_cost": 0,
        }

    @staticmethod
    def token_to_cost(tokens: int, model: str, tokens_type: str) -> float:
        """Converts a number of tokens to a cost in dollars."""
        return (
            tokens
            / PRICE_NUM_TOKENS
            * TokenCounterCallback.TOKEN_PRICES[model][tokens_type]
        )

    def calculate_cost(
        self, tokens_type: str, model: str, token_counter: Callable, **kwargs: Any
    ) -> None:
        """Calculate the cost of a generation."""
        # Check if it its prompt or tokens are passed in
        prompt_key = f"{tokens_type}_prompt"
        token_key = f"{tokens_type}_tokens"
        if prompt_key in kwargs:
            tokens = token_counter(kwargs[prompt_key], model)
        elif token_key in kwargs:
            tokens = kwargs[token_key]
        else:
            logging.warning(f"No {token_key}_tokens or {tokens_type}_prompt found.")
            return
        cost = self.token_to_cost(tokens, model, tokens_type)
        self.cost_dict[token_key] += tokens
        self.cost_dict[f"{tokens_type}_cost"] += cost

    def __call__(self, model: str, token_counter: Callable, **kwargs: Any) -> None:
        """Callback to count the number of tokens used in a generation.

        :param model: model identifier used for cost lookup.
        :param token_counter: function to count tokens from raw text.
        :param kwargs: token counting params plus optional ``call_cost`` (gross provider cost in USD).
        """
        self.actual_model = model
        if model not in list(TokenCounterCallback.TOKEN_PRICES.keys()):
            raise ValueError(f"Model {model} not supported.")
        try:
            # Snapshot token costs before adding this call, so we can
            # compute the delta and derive extra_cost from call_cost.
            input_cost_before = self.cost_dict["input_cost"]
            output_cost_before = self.cost_dict["output_cost"]
            self.calculate_cost("input", model, token_counter, **kwargs)
            self.calculate_cost("output", model, token_counter, **kwargs)

            call_cost = kwargs.get("call_cost")
            if call_cost is not None:
                # Derive extra_cost so that total matches call_cost.
                this_token_cost = (self.cost_dict["input_cost"] - input_cost_before) + (
                    self.cost_dict["output_cost"] - output_cost_before
                )
                surcharge = max(0.0, float(call_cost) - this_token_cost)
                if surcharge > 0:
                    self.cost_dict["extra_cost"] += surcharge
            self.cost_dict["total_tokens"] = (
                self.cost_dict["input_tokens"] + self.cost_dict["output_tokens"]
            )
            self.cost_dict["total_cost"] = (
                self.cost_dict["input_cost"]
                + self.cost_dict["output_cost"]
                + self.cost_dict["extra_cost"]
            )
        except Exception as e:
            logging.error(f"Error in TokenCounterCallback: {e}")
