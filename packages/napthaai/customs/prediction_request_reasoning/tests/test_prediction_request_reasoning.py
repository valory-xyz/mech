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
from packages.napthaai.customs.prediction_request_reasoning import tiktoken_data
from packages.napthaai.customs.prediction_request_reasoning.prediction_request_reasoning import (
    LLMClientManager,
    _ensure_tiktoken_cache,
    count_tokens,
    do_reasoning_with_retry,
    fetch_additional_information,
    multi_questions_response,
)


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
    """Verify tiktoken data is bundled and decodable."""

    def test_ensure_tiktoken_cache_creates_files(self) -> None:
        """_ensure_tiktoken_cache writes decoded BPE files to cache dir."""
        _ensure_tiktoken_cache()
        cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR", "")
        assert cache_dir != "", "TIKTOKEN_CACHE_DIR should be set"
        assert Path(cache_dir).is_dir()
        files = list(Path(cache_dir).iterdir())
        assert len(files) >= 2

    def test_tiktoken_loads_from_decoded_cache(self) -> None:
        """Tiktoken can load encodings after _ensure_tiktoken_cache runs."""
        _ensure_tiktoken_cache()
        enc = get_encoding("cl100k_base")
        assert len(enc.encode("hello world")) > 0
        enc2 = get_encoding("o200k_base")
        assert len(enc2.encode("hello world")) > 0

    def test_tiktoken_data_module_exists(self) -> None:
        """tiktoken_data.py has the expected constants."""
        assert hasattr(tiktoken_data, "CL100K_BASE")
        assert hasattr(tiktoken_data, "O200K_BASE")
        assert hasattr(tiktoken_data, "CL100K_CACHE_NAME")
        assert hasattr(tiktoken_data, "O200K_CACHE_NAME")
        assert len(tiktoken_data.CL100K_BASE) > 0
        assert len(tiktoken_data.O200K_BASE) > 0
