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

"""Unit tests for superforcaster: thread-safe client and offline tiktoken."""

import inspect
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tiktoken import get_encoding

import packages.valory.customs.superforcaster.superforcaster as module
from packages.valory.customs.superforcaster import tiktoken_data
from packages.valory.customs.superforcaster.superforcaster import (
    OpenAIClientManager,
    _ensure_tiktoken_cache,
    generate_prediction_with_retry,
)


class TestOpenAIClientManager:
    """Verify OpenAIClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_instance(self) -> None:
        """__enter__ returns a fresh OpenAIClient, __exit__ closes it."""
        mgr = OpenAIClientManager(api_key="sk-test")
        with patch(
            "packages.valory.customs.superforcaster.superforcaster.OpenAIClient"
        ) as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance

            with mgr as client:
                assert client is mock_instance
                MockClient.assert_called_once_with(api_key="sk-test")

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

    def test_generate_prediction_requires_client_param(self) -> None:
        """generate_prediction_with_retry requires client as first param."""
        params = list(inspect.signature(generate_prediction_with_retry).parameters)
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
