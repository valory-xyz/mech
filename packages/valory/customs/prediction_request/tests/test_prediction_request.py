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

"""Unit tests for prediction_request: thread-safe client and offline tiktoken."""

import inspect
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from tiktoken import get_encoding

import packages.valory.customs.prediction_request.prediction_request as module
from packages.valory.customs.prediction_request.prediction_request import (
    LLMClientManager,
    count_tokens,
    fetch_additional_information,
    fetch_multi_queries_with_retry,
    generate_prediction_with_retry,
)

TIKTOKEN_CACHE = Path(module.__file__).parent / "tiktoken_cache"
EXPECTED_CACHE_FILES = {
    "9b5ad71b2ce5302211f9c61530b329a4922fc6a4",  # cl100k_base
    "fb374d419588a4632f3f557e76b4b70aebbca790",  # o200k_base
}


class TestLLMClientManager:
    """Verify LLMClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_instance(self) -> None:
        """__enter__ returns a fresh LLMClient, __exit__ closes it."""
        mock_keys: Any = {"openai": "sk-test"}
        mgr = LLMClientManager(api_keys=mock_keys, model="gpt-4o-2024-08-06")
        with patch(
            "packages.valory.customs.prediction_request.prediction_request.LLMClient"
        ) as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance

            with mgr as client:
                assert client is mock_instance
                MockClient.assert_called_once_with(mock_keys, "openai")

            mock_instance.client.close.assert_called_once()

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

    def test_concurrent_contexts_are_independent(self) -> None:
        """Two concurrent LLMClientManager contexts get independent clients."""
        clients_seen: list = []

        def create_and_record(key_suffix: str) -> None:
            mock_keys: Any = {"openai": f"sk-{key_suffix}"}
            mgr = LLMClientManager(api_keys=mock_keys, model="gpt-4o-2024-08-06")
            with patch(
                "packages.valory.customs.prediction_request.prediction_request.LLMClient"
            ) as MockClient:
                mock_instance = MagicMock(name=f"client-{key_suffix}")
                MockClient.return_value = mock_instance
                with mgr as client:
                    clients_seen.append(id(client))

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(create_and_record, s) for s in ("a", "b")]
            for f in as_completed(futures):
                f.result()

        assert (
            len(set(clients_seen)) == 2
        ), "Concurrent contexts must get independent clients"


class TestFunctionsAcceptClient:
    """Verify refactored functions accept client as an explicit parameter."""

    def test_count_tokens_without_client_uses_tiktoken(self) -> None:
        """count_tokens falls back to tiktoken when client is None."""
        token_count = count_tokens("hello world", "gpt-4o-2024-08-06")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_with_client_for_claude(self) -> None:
        """count_tokens uses Anthropic tokenizer when client is provided."""
        mock_client = MagicMock()
        mock_client.llm_provider = "anthropic"
        mock_client.client.messages.count_tokens.return_value = SimpleNamespace(
            input_tokens=42
        )

        result = count_tokens(
            "hello world", "claude-4-sonnet-20250514", client=mock_client
        )
        assert result == 42
        mock_client.client.messages.count_tokens.assert_called_once()

    def test_count_tokens_claude_without_client_uses_fallback(self) -> None:
        """count_tokens for Claude models without client uses cl100k_base fallback."""
        token_count = count_tokens("hello world", "claude-4-sonnet-20250514")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_generate_prediction_requires_client_param(self) -> None:
        """generate_prediction_with_retry requires client as first param."""
        params = list(inspect.signature(generate_prediction_with_retry).parameters)
        assert params[0] == "client"

    def test_fetch_additional_information_requires_client_param(self) -> None:
        """fetch_additional_information requires client as first param."""
        params = list(inspect.signature(fetch_additional_information).parameters)
        assert params[0] == "client"

    def test_fetch_multi_queries_requires_client_param(self) -> None:
        """fetch_multi_queries_with_retry requires client as first param."""
        params = list(inspect.signature(fetch_multi_queries_with_retry).parameters)
        assert params[0] == "client"


class TestTiktokenOfflineCache:
    """Verify tiktoken cache files are bundled and usable."""

    def test_cache_dir_exists(self) -> None:
        """tiktoken_cache directory exists in the package."""
        assert TIKTOKEN_CACHE.is_dir()

    def test_has_all_cache_files(self) -> None:
        """Package bundles both cl100k_base and o200k_base."""
        actual = {f.name for f in TIKTOKEN_CACHE.iterdir()}
        assert EXPECTED_CACHE_FILES.issubset(actual)

    def test_cache_files_are_non_empty(self) -> None:
        """All cache files should contain data."""
        for name in EXPECTED_CACHE_FILES:
            assert (TIKTOKEN_CACHE / name).stat().st_size > 0

    def test_tiktoken_loads_from_bundled_cache(self) -> None:
        """Tiktoken can load encodings from the bundled cache."""
        with patch.dict(os.environ, {"TIKTOKEN_CACHE_DIR": str(TIKTOKEN_CACHE)}):
            enc = get_encoding("cl100k_base")
            assert len(enc.encode("hello world")) > 0
            enc2 = get_encoding("o200k_base")
            assert len(enc2.encode("hello world")) > 0

    def test_module_sets_tiktoken_cache_dir(self) -> None:
        """Module sets TIKTOKEN_CACHE_DIR on load."""
        actual = os.environ.get("TIKTOKEN_CACHE_DIR", "")
        assert actual != ""
        assert Path(actual).is_dir()
