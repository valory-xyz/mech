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

"""Unit tests for prediction_request_reasoning: thread-safe client and offline tiktoken."""

import inspect
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tiktoken import get_encoding

import packages.napthaai.customs.prediction_request_reasoning.prediction_request_reasoning as module
from packages.napthaai.customs.prediction_request_reasoning.prediction_request_reasoning import (
    LLMClientManager,
    count_tokens,
    do_reasoning_with_retry,
    fetch_additional_information,
    multi_questions_response,
)

TIKTOKEN_CACHE = Path(module.__file__).parent / "tiktoken_cache"
EXPECTED_CACHE_FILES = {
    "9b5ad71b2ce5302211f9c61530b329a4922fc6a4",
    "fb374d419588a4632f3f557e76b4b70aebbca790",
}


class TestLLMClientManager:
    """Verify LLMClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_tuple(self) -> None:
        """__enter__ returns a (client, client_embedding) tuple."""
        mock_keys = {"openai": "sk-test"}
        mgr = LLMClientManager(
            api_keys=mock_keys, model="gpt-4o-2024-08-06", embedding_provider="openai"
        )
        with patch(
            "packages.napthaai.customs.prediction_request_reasoning.prediction_request_reasoning.LLMClient"
        ) as MockClient:
            mock_llm = MagicMock(name="llm")
            mock_embed = MagicMock(name="embed")
            MockClient.side_effect = [mock_llm, mock_embed]

            with mgr as (llm_client, embedding_client):
                assert llm_client is mock_llm
                assert embedding_client is mock_embed

    def test_no_global_client_variable(self) -> None:
        """The module must not define module-level client variables."""
        source = Path(module.__file__).read_text(encoding="utf-8")
        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.lstrip()
            if stripped.startswith("client:") or stripped.startswith("client ="):
                if not line.startswith(" ") and not line.startswith("\t"):
                    pytest.fail(
                        f"Module-level 'client' variable found at line {i}: {line}"
                    )
            if stripped.startswith("client_embedding:") or stripped.startswith(
                "client_embedding ="
            ):
                if not line.startswith(" ") and not line.startswith("\t"):
                    pytest.fail(
                        f"Module-level 'client_embedding' variable found at line {i}: {line}"
                    )


class TestFunctionsAcceptClient:
    """Verify refactored functions accept client as an explicit parameter."""

    def test_count_tokens_without_client_uses_tiktoken(self) -> None:
        """count_tokens falls back to tiktoken when client is None."""
        token_count = count_tokens("hello world", "gpt-4o-2024-08-06")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_claude_without_client_uses_fallback(self) -> None:
        """count_tokens for Claude models without client uses cl100k_base fallback."""
        token_count = count_tokens("hello world", "claude-4-sonnet-20250514")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_multi_questions_response_requires_client_param(self) -> None:
        """multi_questions_response requires client as first param."""
        params = list(inspect.signature(multi_questions_response).parameters)
        assert params[0] == "client"

    def test_do_reasoning_requires_client_param(self) -> None:
        """do_reasoning_with_retry requires client as first param."""
        params = list(inspect.signature(do_reasoning_with_retry).parameters)
        assert params[0] == "client"

    def test_fetch_additional_information_requires_client_param(self) -> None:
        """fetch_additional_information requires client as first param."""
        params = list(inspect.signature(fetch_additional_information).parameters)
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
