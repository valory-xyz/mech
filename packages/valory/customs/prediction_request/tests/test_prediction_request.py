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

"""Unit tests for prediction_request: thread-safe client, offline tiktoken, and source_content."""

import inspect
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import packages.valory.customs.prediction_request.prediction_request as module
from packages.valory.customs.prediction_request.prediction_request import (
    ExtendedDocument,
    LLMClientManager,
    count_tokens,
    extract_texts,
    fetch_additional_information,
    fetch_multi_queries_with_retry,
    generate_prediction_with_retry,
    run,
)


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


MODULE = "packages.valory.customs.prediction_request.prediction_request"


def _make_html_future(url: str, html: str) -> tuple:
    """Create a (future, url) pair with a fake HTML response."""
    response = MagicMock()
    response.status_code = 200
    response.text = html
    response.content = b"<html>"
    future: Future = Future()
    future.set_result(response)
    return (future, url)


def _make_pdf_future(url: str) -> tuple:
    """Create a (future, url) pair with a fake PDF response."""
    response = MagicMock()
    response.status_code = 200
    response.text = ""
    response.content = b"%PDF-1.4 fake content"
    future: Future = Future()
    future.set_result(response)
    return (future, url)


def _make_failed_future(url: str, status: int = 404) -> tuple:
    """Create a (future, url) pair with a non-200 response."""
    response = MagicMock()
    response.status_code = status
    future: Future = Future()
    future.set_result(response)
    return (future, url)


class TestExtractTextsCapture:
    """Verify extract_texts captures source content correctly."""

    @patch(f"{MODULE}.process_in_batches")
    def test_cleaned_mode_stores_extracted_text(self, mock_batches: MagicMock) -> None:
        """In cleaned mode (default), extracted text is stored instead of raw HTML."""
        html = "<html><body>Hello world</body></html>"
        mock_batches.return_value = [[_make_html_future("http://example.com", html)]]

        _, raw_sc = extract_texts(["http://example.com"], num_words=300)

        assert raw_sc["mode"] == "cleaned"
        assert "http://example.com" in raw_sc["pages"]
        assert raw_sc["pages"]["http://example.com"] != html
        assert "Hello world" in raw_sc["pages"]["http://example.com"]
        assert not raw_sc["pdfs"]

    @patch(f"{MODULE}.process_in_batches")
    def test_raw_mode_stores_html(self, mock_batches: MagicMock) -> None:
        """In raw mode, raw HTML is stored."""
        html = "<html><body>Hello world</body></html>"
        mock_batches.return_value = [[_make_html_future("http://example.com", html)]]

        _, raw_sc = extract_texts(
            ["http://example.com"], num_words=300, source_content_mode="raw"
        )

        assert raw_sc["mode"] == "raw"
        assert raw_sc["pages"]["http://example.com"] == html

    @patch(f"{MODULE}.extract_text_from_pdf")
    @patch(f"{MODULE}.process_in_batches")
    def test_pdf_captured(
        self, mock_batches: MagicMock, mock_pdf_extract: MagicMock
    ) -> None:
        """PDF responses are stored in raw_source_content['pdfs']."""
        mock_batches.return_value = [[_make_pdf_future("http://example.com/doc.pdf")]]
        mock_pdf_extract.return_value = ExtendedDocument(
            text="pdf content", url="http://example.com/doc.pdf"
        )

        _, raw_sc = extract_texts(["http://example.com/doc.pdf"], num_words=300)

        assert "http://example.com/doc.pdf" in raw_sc["pdfs"]
        assert raw_sc["pdfs"]["http://example.com/doc.pdf"] == "pdf content"
        assert not raw_sc["pages"]

    @patch(f"{MODULE}.extract_text_from_pdf")
    @patch(f"{MODULE}.process_in_batches")
    def test_failed_pdf_stores_empty_string(
        self, mock_batches: MagicMock, mock_pdf_extract: MagicMock
    ) -> None:
        """When extract_text_from_pdf returns None, empty string is stored."""
        mock_batches.return_value = [[_make_pdf_future("http://example.com/doc.pdf")]]
        mock_pdf_extract.return_value = None

        _, raw_sc = extract_texts(["http://example.com/doc.pdf"], num_words=300)

        assert raw_sc["pdfs"]["http://example.com/doc.pdf"] == ""

    @patch(f"{MODULE}.extract_text_from_pdf")
    @patch(f"{MODULE}.process_in_batches")
    def test_mixed_html_and_pdf(
        self, mock_batches: MagicMock, mock_pdf_extract: MagicMock
    ) -> None:
        """Both HTML and PDF are captured in their respective keys."""
        html = "<html><body>page</body></html>"
        mock_batches.return_value = [
            [
                _make_html_future("http://example.com", html),
                _make_pdf_future("http://example.com/doc.pdf"),
            ]
        ]
        mock_pdf_extract.return_value = ExtendedDocument(
            text="pdf text", url="http://example.com/doc.pdf"
        )

        _, raw_sc = extract_texts(
            ["http://example.com", "http://example.com/doc.pdf"], num_words=300
        )

        assert "http://example.com" in raw_sc["pages"]
        assert "http://example.com/doc.pdf" in raw_sc["pdfs"]

    @patch(f"{MODULE}.process_in_batches")
    def test_non_200_not_captured(self, mock_batches: MagicMock) -> None:
        """Non-200 responses are not stored in raw_source_content."""
        mock_batches.return_value = [[_make_failed_future("http://example.com", 404)]]

        docs, raw_sc = extract_texts(["http://example.com"], num_words=300)

        assert not raw_sc["pages"]
        assert not raw_sc["pdfs"]
        assert docs == []


class TestFetchReplayPath:
    """Verify fetch_additional_information replays from structured source_content."""

    def test_cleaned_mode_uses_text_directly(self) -> None:
        """In cleaned mode, cached text is used directly without re-extraction."""
        source_content = {
            "mode": "cleaned",
            "pages": {
                "http://example.com": "test content here",
            },
            "pdfs": {},
        }
        mock_client = MagicMock()

        result, raw_sc, _ = fetch_additional_information(
            client=mock_client,
            user_prompt="test",
            engine="gpt-4o",
            temperature=0.0,
            max_tokens=100,
            google_api_key=None,
            google_engine=None,
            serper_api_key=None,
            search_provider="google",
            num_urls=3,
            num_words=300,
            source_content=source_content,
        )

        assert raw_sc is source_content
        assert "test content here" in result
        assert "http://example.com" in result

    def test_raw_mode_re_extracts(self) -> None:
        """In raw mode, HTML is re-extracted via extract_text."""
        source_content = {
            "mode": "raw",
            "pages": {
                "http://example.com": "<html><body>test content here</body></html>",
            },
            "pdfs": {},
        }
        mock_client = MagicMock()

        result, raw_sc, _ = fetch_additional_information(
            client=mock_client,
            user_prompt="test",
            engine="gpt-4o",
            temperature=0.0,
            max_tokens=100,
            google_api_key=None,
            google_engine=None,
            serper_api_key=None,
            search_provider="google",
            num_urls=3,
            num_words=300,
            source_content=source_content,
        )

        assert raw_sc is source_content
        assert "http://example.com" in result

    def test_missing_mode_defaults_to_cleaned(self) -> None:
        """Source content without mode key defaults to cleaned (no re-extraction)."""
        source_content = {
            "pages": {
                "http://example.com": "already cleaned text",
            },
            "pdfs": {},
        }
        mock_client = MagicMock()

        result, raw_sc, _ = fetch_additional_information(
            client=mock_client,
            user_prompt="test",
            engine="gpt-4o",
            temperature=0.0,
            max_tokens=100,
            google_api_key=None,
            google_engine=None,
            serper_api_key=None,
            search_provider="google",
            num_urls=3,
            num_words=300,
            source_content=source_content,
        )

        assert "already cleaned text" in result

    def test_pdfs_replayed(self) -> None:
        """Pdfs in source_content are loaded as ExtendedDocuments."""
        source_content = {
            "pages": {},
            "pdfs": {
                "http://example.com/doc.pdf": "pdf extracted text for testing",
            },
        }
        mock_client = MagicMock()

        result, raw_sc, _ = fetch_additional_information(
            client=mock_client,
            user_prompt="test",
            engine="gpt-4o",
            temperature=0.0,
            max_tokens=100,
            google_api_key=None,
            google_engine=None,
            serper_api_key=None,
            search_provider="google",
            num_urls=3,
            num_words=300,
            source_content=source_content,
        )

        assert "pdf extracted text for testing" in result
        assert "http://example.com/doc.pdf" in result

    def test_empty_source_content(self) -> None:
        """Empty source_content produces empty result without error."""
        source_content: dict = {"pages": {}, "pdfs": {}}
        mock_client = MagicMock()

        result, raw_sc, _ = fetch_additional_information(
            client=mock_client,
            user_prompt="test",
            engine="gpt-4o",
            temperature=0.0,
            max_tokens=100,
            google_api_key=None,
            google_engine=None,
            serper_api_key=None,
            search_provider="google",
            num_urls=3,
            num_words=300,
            source_content=source_content,
        )

        assert result == ""


def _make_mock_api_keys(return_source_content: str = "false") -> MagicMock:
    """Create a mock KeyChain-like api_keys object."""
    services = {
        "openai": ["sk-test"],
        "google_api_key": ["gk-test"],
        "google_engine_id": ["ge-test"],
        "serperapi": ["sp-test"],
        "search_provider": ["google"],
        "return_source_content": [return_source_content],
    }
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: services[key][0]
    mock.get = lambda key, default="": services.get(key, [default])[0]
    return mock


class TestRunFlagBehavior:
    """Verify return_source_content flag controls used_params."""

    @patch(f"{MODULE}.generate_prediction_with_retry")
    @patch(f"{MODULE}.fetch_additional_information")
    @patch(f"{MODULE}.LLMClientManager")
    def test_flag_on_includes_source_content(
        self,
        mock_mgr: MagicMock,
        mock_fetch: MagicMock,
        mock_gen: MagicMock,
    ) -> None:
        """When return_source_content is true, source_content is in used_params."""
        mock_mgr.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)
        fake_sc = {"pages": {"http://x.com": "<html>hi</html>"}, "pdfs": {}}
        mock_fetch.return_value = ("info", fake_sc, None)
        mock_gen.return_value = ("prediction", None)

        result = run(
            tool="prediction-online",
            model="gpt-4.1-2025-04-14",
            prompt="test",
            api_keys=_make_mock_api_keys("true"),
        )

        used_params = result[4]
        assert "source_content" in used_params
        assert used_params["source_content"] is fake_sc

    @patch(f"{MODULE}.generate_prediction_with_retry")
    @patch(f"{MODULE}.fetch_additional_information")
    @patch(f"{MODULE}.LLMClientManager")
    def test_flag_off_excludes_source_content(
        self,
        mock_mgr: MagicMock,
        mock_fetch: MagicMock,
        mock_gen: MagicMock,
    ) -> None:
        """When return_source_content is false, source_content is not in used_params."""
        mock_mgr.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)
        mock_fetch.return_value = ("info", {}, None)
        mock_gen.return_value = ("prediction", None)

        result = run(
            tool="prediction-online",
            model="gpt-4.1-2025-04-14",
            prompt="test",
            api_keys=_make_mock_api_keys("false"),
        )

        used_params = result[4]
        assert "source_content" not in used_params

    @patch(f"{MODULE}.generate_prediction_with_retry")
    @patch(f"{MODULE}.fetch_additional_information")
    @patch(f"{MODULE}.LLMClientManager")
    def test_flag_missing_defaults_off(
        self,
        mock_mgr: MagicMock,
        mock_fetch: MagicMock,
        mock_gen: MagicMock,
    ) -> None:
        """When return_source_content is not in api_keys, defaults to off."""
        mock_mgr.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)
        mock_fetch.return_value = ("info", {}, None)
        mock_gen.return_value = ("prediction", None)

        # api_keys without return_source_content key
        services = {"openai": ["sk-test"], "search_provider": ["google"]}
        mock_keys = MagicMock()
        mock_keys.__getitem__ = lambda self, key: services[key][0]
        mock_keys.get = lambda key, default="": services.get(key, [default])[0]

        result = run(
            tool="prediction-online",
            model="gpt-4.1-2025-04-14",
            prompt="test",
            api_keys=mock_keys,
        )

        used_params = result[4]
        assert "source_content" not in used_params

    @patch(f"{MODULE}.LLMClientManager")
    def test_invalid_source_content_mode_returns_error(
        self, mock_mgr: MagicMock
    ) -> None:
        """Invalid source_content_mode returns error string (caught by @with_key_rotation)."""
        mock_mgr.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)

        services = {
            "openai": ["sk-test"],
            "search_provider": ["google"],
            "source_content_mode": ["invalid"],
        }
        mock_keys = MagicMock()
        mock_keys.__getitem__ = lambda self, key: services[key][0]
        mock_keys.get = lambda key, default="": services.get(key, [default])[0]

        result = run(
            tool="prediction-online",
            model="gpt-4.1-2025-04-14",
            prompt="test",
            api_keys=mock_keys,
        )

        assert "Invalid source_content_mode" in result[0]
