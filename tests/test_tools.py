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
"""This module contains tool tests."""
from typing import List, Any

from packages.valory.customs.openai_request import openai_request
from packages.valory.skills.task_execution.utils.apis import KeyChain
from packages.valory.skills.task_execution.utils.benchmarks import TokenCounterCallback
from tests.constants import (
    OPENAI_SECRET_KEY,
    GOOGLE_API_KEY,
    CLAUDE_API_KEY,
    OPENROUTER_API_KEY,
)


class BaseToolTest:
    """Base tool test class."""

    keys = KeyChain(
        {
            "openai": [OPENAI_SECRET_KEY],
            "google_api_key": [GOOGLE_API_KEY],
            "anthropic": [CLAUDE_API_KEY],
            "openrouter": [OPENROUTER_API_KEY],
        }
    )
    models: List = [None]
    tools: List[str]
    prompts: List[str]
    tool_module: Any = None
    tool_callable: str = "run"

    def _validate_response(self, response: Any) -> None:
        """Validate response."""
        assert type(response) == tuple, "Response of the tool must be a tuple."
        assert len(response) == 5, "Response must have 5 elements."
        assert type(response[0]) == str, "Response[0] must be a string."
        assert type(response[1]) == str, "Response[1] must be a string."
        assert (
            type(response[2]) == dict or response[2] is None
        ), "Response[2] must be a dictionary or None."
        assert (
            type(response[3]) == TokenCounterCallback or response[3] is None
        ), "Response[3] must be a TokenCounterCallback or None."
        assert type(response[4]) == KeyChain, "Response[4] must be a KeyChain object."

    def test_run(self) -> None:
        """Test run method."""
        assert self.tools, "Tools must be provided."
        assert self.prompts, "Prompts must be provided."
        assert self.tool_module, "Callable function must be provided."

        for model in self.models:
            for tool in self.tools:
                for prompt in self.prompts:
                    kwargs = dict(
                        prompt=prompt,
                        tool=tool,
                        api_keys=self.keys,
                        counter_callback=TokenCounterCallback(),
                    )
                    func = getattr(self.tool_module, self.tool_callable)
                    response = func(**kwargs)
                    self._validate_response(response)


class TestPredictionOnline(BaseToolTest):
    """Test Prediction Online."""

    tools = openai_request.ALLOWED_TOOLS
    prompts = [
        'Please take over the role of a Data Scientist to evaluate the given question. With the given question "Will Apple release iPhone 17 by March 2025?" and the `yes` option represented by `Yes` and the `no` option represented by `No`, what are the respective probabilities of `p_yes` and `p_no` occurring?'
    ]
    tool_module = openai_request
