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

"""Tests for thread-safety of LLM client management and tiktoken offline support."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# prediction_request: LLMClientManager no longer uses globals
# ---------------------------------------------------------------------------


class TestPredictionRequestClientManager:
    """Verify LLMClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_instance(self) -> None:
        """__enter__ returns a fresh LLMClient, __exit__ closes it."""
        from packages.valory.customs.prediction_request.prediction_request import (
            LLMClientManager,
        )

        mock_keys = {"openai": "sk-test"}
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
        import packages.valory.customs.prediction_request.prediction_request as mod

        # The module should not have a top-level `client` attribute that is None
        # (it may have the word 'client' in function params, but not as a global)
        source = Path(mod.__file__).read_text(encoding="utf-8")
        # Check there's no line like `client: Optional[LLMClient] = None`
        # or `client = None` at module level (not inside a class/function)
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if stripped.startswith("client:") or stripped.startswith("client ="):
                # Only flag if it's at module level (no indentation)
                if not line.startswith(" ") and not line.startswith("\t"):
                    pytest.fail(
                        f"Module-level 'client' variable found at line {i}: {line}"
                    )

    def test_concurrent_contexts_are_independent(self) -> None:
        """Two concurrent LLMClientManager contexts get independent clients."""
        from packages.valory.customs.prediction_request.prediction_request import (
            LLMClientManager,
        )

        clients_seen = []

        def create_and_record(key_suffix: str) -> None:
            mock_keys = {"openai": f"sk-{key_suffix}"}
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

        # Each context must produce a distinct client object
        assert (
            len(set(clients_seen)) == 2
        ), "Concurrent contexts must get independent clients"


class TestPredictionRequestFunctionsAcceptClient:
    """Verify refactored functions accept client as an explicit parameter."""

    def test_count_tokens_without_client_uses_tiktoken(self) -> None:
        """count_tokens falls back to tiktoken when client is None."""
        from packages.valory.customs.prediction_request.prediction_request import (
            count_tokens,
        )

        token_count = count_tokens("hello world", "gpt-4o-2024-08-06")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_with_client_for_claude(self) -> None:
        """count_tokens uses Anthropic tokenizer when client is provided for Claude models."""
        from packages.valory.customs.prediction_request.prediction_request import (
            count_tokens,
        )

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

    def test_generate_prediction_requires_client_param(self) -> None:
        """generate_prediction_with_retry requires client as first param."""
        import inspect

        from packages.valory.customs.prediction_request.prediction_request import (
            generate_prediction_with_retry,
        )

        sig = inspect.signature(generate_prediction_with_retry)
        params = list(sig.parameters.keys())
        assert (
            params[0] == "client"
        ), f"First param should be 'client', got '{params[0]}'"

    def test_fetch_additional_information_requires_client_param(self) -> None:
        """fetch_additional_information requires client as first param."""
        import inspect

        from packages.valory.customs.prediction_request.prediction_request import (
            fetch_additional_information,
        )

        sig = inspect.signature(fetch_additional_information)
        params = list(sig.parameters.keys())
        assert (
            params[0] == "client"
        ), f"First param should be 'client', got '{params[0]}'"

    def test_fetch_multi_queries_requires_client_param(self) -> None:
        """fetch_multi_queries_with_retry requires client as first param."""
        import inspect

        from packages.valory.customs.prediction_request.prediction_request import (
            fetch_multi_queries_with_retry,
        )

        sig = inspect.signature(fetch_multi_queries_with_retry)
        params = list(sig.parameters.keys())
        assert (
            params[0] == "client"
        ), f"First param should be 'client', got '{params[0]}'"


# ---------------------------------------------------------------------------
# superforcaster: OpenAIClientManager no longer uses globals
# ---------------------------------------------------------------------------


class TestSuperforcasterClientManager:
    """Verify OpenAIClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_instance(self) -> None:
        """__enter__ returns a fresh OpenAIClient, __exit__ closes it."""
        from packages.valory.customs.superforcaster.superforcaster import (
            OpenAIClientManager,
        )

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
        import packages.valory.customs.superforcaster.superforcaster as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if stripped.startswith("client:") or stripped.startswith("client ="):
                if not line.startswith(" ") and not line.startswith("\t"):
                    pytest.fail(
                        f"Module-level 'client' variable found at line {i}: {line}"
                    )

    def test_generate_prediction_requires_client_param(self) -> None:
        """generate_prediction_with_retry requires client as first param."""
        import inspect

        from packages.valory.customs.superforcaster.superforcaster import (
            generate_prediction_with_retry,
        )

        sig = inspect.signature(generate_prediction_with_retry)
        params = list(sig.parameters.keys())
        assert (
            params[0] == "client"
        ), f"First param should be 'client', got '{params[0]}'"


# ---------------------------------------------------------------------------
# tiktoken offline cache
# ---------------------------------------------------------------------------


class TestTiktokenOfflineCache:
    """Verify tiktoken cache files are bundled and usable."""

    PREDICTION_REQUEST_CACHE = Path(
        "packages/valory/customs/prediction_request/tiktoken_cache"
    )
    SUPERFORCASTER_CACHE = Path("packages/valory/customs/superforcaster/tiktoken_cache")
    EXPECTED_FILES = {
        "9b5ad71b2ce5302211f9c61530b329a4922fc6a4",  # cl100k_base
        "fb374d419588a4632f3f557e76b4b70aebbca790",  # o200k_base
    }

    def test_prediction_request_cache_dir_exists(self) -> None:
        """tiktoken_cache directory exists in prediction_request package."""
        assert self.PREDICTION_REQUEST_CACHE.is_dir()

    def test_superforcaster_cache_dir_exists(self) -> None:
        """tiktoken_cache directory exists in superforcaster package."""
        assert self.SUPERFORCASTER_CACHE.is_dir()

    def test_prediction_request_has_all_cache_files(self) -> None:
        """prediction_request bundles both cl100k_base and o200k_base."""
        actual = {f.name for f in self.PREDICTION_REQUEST_CACHE.iterdir()}
        assert self.EXPECTED_FILES.issubset(
            actual
        ), f"Missing cache files: {self.EXPECTED_FILES - actual}"

    def test_superforcaster_has_all_cache_files(self) -> None:
        """superforcaster bundles both cl100k_base and o200k_base."""
        actual = {f.name for f in self.SUPERFORCASTER_CACHE.iterdir()}
        assert self.EXPECTED_FILES.issubset(
            actual
        ), f"Missing cache files: {self.EXPECTED_FILES - actual}"

    def test_cache_files_are_non_empty(self) -> None:
        """All cache files should contain data."""
        for cache_dir in (self.PREDICTION_REQUEST_CACHE, self.SUPERFORCASTER_CACHE):
            for expected_file in self.EXPECTED_FILES:
                path = cache_dir / expected_file
                assert path.stat().st_size > 0, f"Cache file is empty: {path}"

    def test_tiktoken_loads_from_bundled_cache(self) -> None:
        """tiktoken can load encodings from the bundled cache without network."""
        from tiktoken import get_encoding

        cache_dir = str(self.PREDICTION_REQUEST_CACHE.resolve())
        with patch.dict(os.environ, {"TIKTOKEN_CACHE_DIR": cache_dir}):
            enc = get_encoding("cl100k_base")
            tokens = enc.encode("hello world")
            assert len(tokens) > 0

            enc2 = get_encoding("o200k_base")
            tokens2 = enc2.encode("hello world")
            assert len(tokens2) > 0

    def test_prediction_request_sets_tiktoken_cache_dir(self) -> None:
        """prediction_request module sets TIKTOKEN_CACHE_DIR on load."""
        import packages.valory.customs.prediction_request.prediction_request as mod

        mod_dir = Path(mod.__file__).parent
        expected = str(mod_dir / "tiktoken_cache")
        actual = os.environ.get("TIKTOKEN_CACHE_DIR", "")
        # Either it matches our bundled path or was overridden by env
        assert actual != "", "TIKTOKEN_CACHE_DIR should be set after module import"
        assert (
            actual == expected or Path(actual).is_dir()
        ), f"TIKTOKEN_CACHE_DIR should point to a valid directory, got: {actual}"
