# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
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
INPUT = "input"
OUTPUT = "output"
COST_PER_TOKEN = "_cost_per_token"
INPUT_COST_PER_TOKEN = f"{INPUT}{COST_PER_TOKEN}"
OUTPUT_COST_PER_TOKEN = f"{OUTPUT}{COST_PER_TOKEN}"


class TokenCounterCallback:
    """Callback to count the number of tokens used in a generation."""

    TOKEN_PRICES = {
        "gpt-3.5-turbo": {INPUT_COST_PER_TOKEN: 0.0005, OUTPUT_COST_PER_TOKEN: 0.0015},
        "gpt-3.5-turbo-0125": {
            INPUT_COST_PER_TOKEN: 0.0005,
            OUTPUT_COST_PER_TOKEN: 0.0015,
        },
        "gpt-3.5-turbo-1106": {
            INPUT_COST_PER_TOKEN: 0.001,
            OUTPUT_COST_PER_TOKEN: 0.002,
        },
        "gpt-4": {INPUT_COST_PER_TOKEN: 0.03, OUTPUT_COST_PER_TOKEN: 0.06},
        "gpt-4-turbo-preview": {
            INPUT_COST_PER_TOKEN: 0.01,
            OUTPUT_COST_PER_TOKEN: 0.03,
        },
        "gpt-4-0125-preview": {INPUT_COST_PER_TOKEN: 0.01, OUTPUT_COST_PER_TOKEN: 0.03},
        "gpt-4-1106-preview": {INPUT_COST_PER_TOKEN: 0.01, OUTPUT_COST_PER_TOKEN: 0.03},
        "gpt-4o-2024-08-06": {
            INPUT_COST_PER_TOKEN: 0.0025,
            OUTPUT_COST_PER_TOKEN: 0.01,
        },
        "gpt-4.1-2025-04-14": {
            INPUT_COST_PER_TOKEN: 0.002,
            OUTPUT_COST_PER_TOKEN: 0.008,
        },
        "claude-2": {INPUT_COST_PER_TOKEN: 0.008, OUTPUT_COST_PER_TOKEN: 0.024},
        "claude-3-haiku-20240307": {
            INPUT_COST_PER_TOKEN: 0.00025,
            OUTPUT_COST_PER_TOKEN: 0.00125,
        },
        "claude-3-5-sonnet-20240620": {
            INPUT_COST_PER_TOKEN: 0.003,
            OUTPUT_COST_PER_TOKEN: 0.015,
        },
        "claude-3-opus-20240229": {
            INPUT_COST_PER_TOKEN: 0.015,
            OUTPUT_COST_PER_TOKEN: 0.075,
        },
        "cohere/command-r-plus": {
            INPUT_COST_PER_TOKEN: 0.003,
            OUTPUT_COST_PER_TOKEN: 0.015,
        },
        "databricks/dbrx-instruct:nitro": {
            INPUT_COST_PER_TOKEN: 0.0009,
            OUTPUT_COST_PER_TOKEN: 0.0009,
        },
        "nousresearch/nous-hermes-2-mixtral-8x7b-sft": {
            INPUT_COST_PER_TOKEN: 0.00054,
            OUTPUT_COST_PER_TOKEN: 0.00054,
        },
    }

    def __init__(self, info: Optional[dict] = None) -> None:
        """Initialize the callback."""
        if info:
            self.TOKEN_PRICES = info

        self.cost_dict: Dict[str, Union[int, float]] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_cost": 0,
            "output_cost": 0,
            "total_cost": 0,
        }

    @staticmethod
    def token_to_cost(tokens: int, model: str, tokens_type: str) -> float:
        """Converts a number of tokens to a cost in dollars."""
        cost_key = f"{tokens_type}{COST_PER_TOKEN}"
        return (
            tokens
            / PRICE_NUM_TOKENS
            * TokenCounterCallback.TOKEN_PRICES[model][cost_key]
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
        cost = self.token_to_cost(tokens, model, tokens_type)
        self.cost_dict[token_key] += tokens
        self.cost_dict[f"{tokens_type}_cost"] += cost

    def __call__(self, model: str, token_counter: Callable, **kwargs: Any) -> None:
        """Callback to count the number of tokens used in a generation."""
        if model not in list(TokenCounterCallback.TOKEN_PRICES.keys()):
            raise ValueError(f"Model {model} not supported.")
        try:
            self.calculate_cost(INPUT, model, token_counter, **kwargs)
            self.calculate_cost(OUTPUT, model, token_counter, **kwargs)
            self.cost_dict["total_tokens"] = (
                self.cost_dict["input_tokens"] + self.cost_dict["output_tokens"]
            )
            self.cost_dict["total_cost"] = (
                self.cost_dict["input_cost"] + self.cost_dict["output_cost"]
            )
        except Exception as e:
            logging.error(f"Error in TokenCounterCallback: {e}")
