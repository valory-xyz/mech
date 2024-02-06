# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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
from typing import Any, Dict, Union

import anthropic
import tiktoken
from tiktoken import Encoding


PRICE_NUM_TOKENS = 1000


def encoding_for_model(model: str) -> Encoding:
    """Get the encoding for a model."""
    return tiktoken.encoding_for_model(model)


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    if "claude" in model:
        return anthropic.Anthropic().count_tokens(text)

    enc = encoding_for_model(model)
    return len(enc.encode(text))


class TokenCounterCallback:
    """Callback to count the number of tokens used in a generation."""

    TOKEN_PRICES = {
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "claude-2": {"input": 0.008, "output": 0.024},
    }

    def __init__(self) -> None:
        """Initialize the callback."""
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
        return (
            tokens
            / PRICE_NUM_TOKENS
            * TokenCounterCallback.TOKEN_PRICES[model][tokens_type]
        )

    def calculate_cost(self, tokens_type: str, model: str, **kwargs: Any) -> None:
        """Calculate the cost of a generation."""
        # Check if it its prompt or tokens are passed in
        prompt_key = f"{tokens_type}_prompt"
        token_key = f"{tokens_type}_tokens"
        if prompt_key in kwargs:
            tokens = count_tokens(kwargs[prompt_key], model)
        elif token_key in kwargs:
            tokens = kwargs[token_key]
        else:
            logging.warning(f"No {token_key}_tokens or {tokens_type}_prompt found.")
        cost = self.token_to_cost(tokens, model, tokens_type)
        self.cost_dict[token_key] += tokens
        self.cost_dict[f"{tokens_type}_cost"] += cost

    def __call__(self, model: str, **kwargs: Any) -> None:
        """Callback to count the number of tokens used in a generation."""
        if model not in list(TokenCounterCallback.TOKEN_PRICES.keys()):
            raise ValueError(f"Model {model} not supported.")
        try:
            self.calculate_cost("input", model, **kwargs)
            self.calculate_cost("output", model, **kwargs)
            self.cost_dict["total_tokens"] = (
                self.cost_dict["input_tokens"] + self.cost_dict["output_tokens"]
            )
            self.cost_dict["total_cost"] = (
                self.cost_dict["input_cost"] + self.cost_dict["output_cost"]
            )
        except Exception as e:
            logging.error(f"Error in TokenCounterCallback: {e}")
