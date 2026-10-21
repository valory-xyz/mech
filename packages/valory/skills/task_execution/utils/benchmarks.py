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

    # USD per 1k tokens, from each vendor's price page. Every entry is sent
    # by a bundled tool, on the current tools or on a release still deployed;
    # a model must be added here before a tool starts sending it, or its
    # tokens are not counted.
    TOKEN_PRICES = {
        # OpenAI list prices, developers.openai.com/api/docs/pricing
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
        "gpt-4o-2024-08-06": {"input": 0.0025, "output": 0.01},
        "gpt-4.1-2025-04-14": {"input": 0.002, "output": 0.008},
        "gpt-4.1-mini-2025-04-14": {"input": 0.0004, "output": 0.0016},
        "gpt-6.1-sol": {"input": 0.002, "output": 0.010},
        "gpt-6-luna": {"input": 0.0001, "output": 0.0005},
        # Anthropic list prices, claude.com/pricing
        "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
        "claude-fable-5": {"input": 0.01, "output": 0.05},
        "claude-sonnet-5-5": {"input": 0.002, "output": 0.010},
        # OpenRouter passes the provider list price through. The ":online" web
        # search is billed per request on top of the tokens and reaches the
        # mech through ``call_cost``, not through this table.
        "openai/gpt-4.1:online": {"input": 0.002, "output": 0.008},
        "openai/gpt-6.1-sol:online": {"input": 0.002, "output": 0.010},
        "x-ai/grok-4.3:online": {"input": 0.00125, "output": 0.0025},
        "google/gemini-2.5-flash:online": {"input": 0.0003, "output": 0.0025},
        "anthropic/claude-sonnet-4:online": {"input": 0.003, "output": 0.015},
        "anthropic/claude-haiku-4.5:online": {"input": 0.001, "output": 0.005},
        "anthropic/claude-sonnet-5.5:online": {"input": 0.002, "output": 0.010},
        # Self-hosted, so this is not a provider list price: it is the hosted
        # market rate for a 14B model, used as the per-token stand-in for a
        # server whose real cost is a flat hourly rate.
        "olas-predict-r1-14b": {"input": 0.0002, "output": 0.0002},
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
            # Calls to a model outside the table that reported no cost of
            # their own. While above zero the tally is incomplete.
            "unpriced_calls": 0,
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

    def _record_unpriced_model(self, model: str, call_cost: Any) -> None:
        """Account for a call to a model the price table does not carry.

        Only the provider-reported ``call_cost`` is counted; the tokens are
        not. A call that reports no cost is counted in ``unpriced_calls``
        instead, which marks the whole tally as incomplete so the request
        is charged the tool floor or the cap rather than the partial sum.

        :param model: the unpriced model identifier.
        :param call_cost: the provider-reported cost in USD, if any.
        """
        if call_cost is not None and float(call_cost) > 0:
            self.cost_dict["extra_cost"] += float(call_cost)
            self.cost_dict["total_cost"] = (
                self.cost_dict["input_cost"]
                + self.cost_dict["output_cost"]
                + self.cost_dict["extra_cost"]
            )
            logging.warning(
                f"Model {model} is not in the price table; counting only its "
                f"reported cost of {float(call_cost)} USD."
            )
            return
        self.cost_dict["unpriced_calls"] += 1
        logging.warning(
            f"Model {model} is not in the price table and reported no cost; "
            "its tokens are not counted and the request's cost is incomplete."
        )

    def __call__(self, model: str, token_counter: Callable, **kwargs: Any) -> None:
        """Callback to count the number of tokens used in a generation.

        :param model: model identifier used for cost lookup.
        :param token_counter: function to count tokens from raw text.
        :param kwargs: token counting params plus optional ``call_cost`` (gross provider cost in USD).
        """
        self.actual_model = model
        try:
            if model not in TokenCounterCallback.TOKEN_PRICES:
                self._record_unpriced_model(model, kwargs.get("call_cost"))
                return
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
