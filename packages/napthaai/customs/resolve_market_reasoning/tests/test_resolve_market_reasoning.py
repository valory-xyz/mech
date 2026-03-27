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

"""Unit tests for resolve_market_reasoning: thread-safe client and offline tiktoken."""

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import packages.napthaai.customs.resolve_market_reasoning.resolve_market_reasoning as module
from packages.napthaai.customs.resolve_market_reasoning.resolve_market_reasoning import (
    OpenAIClientManager,
    fetch_additional_information,
    get_embeddings,
    multi_queries,
)


class TestOpenAIClientManager:
    """Verify OpenAIClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_instance(self) -> None:
        """__enter__ returns a fresh OpenAI client, __exit__ closes it."""
        mgr = OpenAIClientManager(api_key="sk-test")
        with patch(
            "packages.napthaai.customs.resolve_market_reasoning.resolve_market_reasoning.OpenAI"
        ) as MockOpenAI:
            mock_instance = MagicMock()
            MockOpenAI.return_value = mock_instance

            with mgr as client:
                assert client is mock_instance

            mock_instance.close.assert_called_once()

    def test_no_global_client_variable(self) -> None:
        """The module must not define a module-level 'client' variable."""
        source = Path(module.__file__).read_text(encoding="utf-8")
        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.lstrip()
            if stripped.startswith("client:") or stripped.startswith("client ="):
                if not line.startswith(" ") and not line.startswith("\t"):
                    pytest.fail(
                        f"Module-level 'client' variable found at line {i}: {line}"
                    )


class TestFunctionsAcceptClient:
    """Verify refactored functions accept client as an explicit parameter."""

    def test_multi_queries_requires_client_param(self) -> None:
        """multi_queries requires client_ as first param."""
        params = list(inspect.signature(multi_queries).parameters)
        assert params[0] == "client_"

    def test_fetch_additional_information_requires_client_param(self) -> None:
        """fetch_additional_information requires client_ as first param."""
        params = list(inspect.signature(fetch_additional_information).parameters)
        assert params[0] == "client_"

    def test_get_embeddings_requires_client_param(self) -> None:
        """get_embeddings requires client as first param."""
        params = list(inspect.signature(get_embeddings).parameters)
        assert params[0] == "client"
