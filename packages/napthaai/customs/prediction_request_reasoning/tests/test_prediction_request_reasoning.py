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

"""Unit tests for prediction_request_reasoning: thread-safe client, offline tiktoken, and source_content."""

import inspect
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

import packages.napthaai.customs.prediction_request_reasoning.prediction_request_reasoning as module
from packages.napthaai.customs.prediction_request_reasoning.prediction_request_reasoning import (
    ExtendedDocument,
    LLMClientManager,
    count_tokens,
    do_reasoning_with_retry,
    extract_texts,
    fetch_additional_information,
    multi_questions_response,
    run,
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


REASONING_MODULE = "packages.napthaai.customs.prediction_request_reasoning.prediction_request_reasoning"


def _make_html_future(url: str, html: str) -> tuple:
    """Create a (future, url) pair with a fake HTML response."""
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.text = html
    response.content = b"<html>"
    future: Future = Future()
    future.set_result(response)
    return (future, url)


def _make_pdf_future(url: str) -> tuple:
    """Create a (future, url) pair with a fake PDF response."""
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.text = ""
    response.content = b"%PDF-1.4 fake content"
    future: Future = Future()
    future.set_result(response)
    return (future, url)


class TestExtractTextsCapture:
    """Verify extract_texts captures raw source content correctly."""

    @patch(f"{REASONING_MODULE}.process_in_batches")
    def test_cleaned_mode_stores_extracted_text(self, mock_batches: MagicMock) -> None:
        """In cleaned mode (default), extracted text is stored instead of raw HTML."""
        html = "<html><body>Hello world</body></html>"
        mock_batches.return_value = [[_make_html_future("http://example.com", html)]]

        _, raw_sc = extract_texts(["http://example.com"])

        assert raw_sc["mode"] == "cleaned"
        assert "http://example.com" in raw_sc["pages"]
        assert raw_sc["pages"]["http://example.com"] != html
        assert "Hello world" in raw_sc["pages"]["http://example.com"]
        assert not raw_sc["pdfs"]

    @patch(f"{REASONING_MODULE}.process_in_batches")
    def test_raw_mode_stores_html(self, mock_batches: MagicMock) -> None:
        """In raw mode, raw HTML is stored."""
        html = "<html><body>Hello world</body></html>"
        mock_batches.return_value = [[_make_html_future("http://example.com", html)]]

        _, raw_sc = extract_texts(["http://example.com"], source_content_mode="raw")

        assert raw_sc["mode"] == "raw"
        assert raw_sc["pages"]["http://example.com"] == html

    @patch(f"{REASONING_MODULE}.extract_text_from_pdf")
    @patch(f"{REASONING_MODULE}.process_in_batches")
    def test_pdf_captured(
        self, mock_batches: MagicMock, mock_pdf_extract: MagicMock
    ) -> None:
        """PDF responses are stored in raw_source_content['pdfs']."""
        mock_batches.return_value = [[_make_pdf_future("http://example.com/doc.pdf")]]
        mock_pdf_extract.return_value = ExtendedDocument(
            text="pdf content", url="http://example.com/doc.pdf"
        )

        _, raw_sc = extract_texts(["http://example.com/doc.pdf"])

        assert "http://example.com/doc.pdf" in raw_sc["pdfs"]
        assert raw_sc["pdfs"]["http://example.com/doc.pdf"] == "pdf content"
        assert not raw_sc["pages"]

    @patch(f"{REASONING_MODULE}.extract_text_from_pdf")
    @patch(f"{REASONING_MODULE}.process_in_batches")
    def test_failed_pdf_stores_empty_string(
        self, mock_batches: MagicMock, mock_pdf_extract: MagicMock
    ) -> None:
        """When extract_text_from_pdf returns None, empty string is stored."""
        mock_batches.return_value = [[_make_pdf_future("http://example.com/doc.pdf")]]
        mock_pdf_extract.return_value = None

        _, raw_sc = extract_texts(["http://example.com/doc.pdf"])

        assert raw_sc["pdfs"]["http://example.com/doc.pdf"] == ""

    @patch(f"{REASONING_MODULE}.extract_text_from_pdf")
    @patch(f"{REASONING_MODULE}.process_in_batches")
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

        _, raw_sc = extract_texts(["http://example.com", "http://example.com/doc.pdf"])

        assert "http://example.com" in raw_sc["pages"]
        assert "http://example.com/doc.pdf" in raw_sc["pdfs"]

    @patch(f"{REASONING_MODULE}.process_in_batches")
    def test_non_200_not_captured(self, mock_batches: MagicMock) -> None:
        """Non-200 responses are not stored in raw_source_content."""
        response = MagicMock(spec=requests.Response)
        response.status_code = 404
        future: Future = Future()
        future.set_result(response)
        mock_batches.return_value = [[(future, "http://example.com")]]

        _, raw_sc = extract_texts(["http://example.com"])

        assert not raw_sc["pages"]
        assert not raw_sc["pdfs"]


class TestFetchReplayPath:
    """Verify fetch_additional_information replays from structured source_content."""

    @patch(f"{REASONING_MODULE}.reciprocal_rank_refusion")
    @patch(f"{REASONING_MODULE}.find_similar_chunks")
    @patch(f"{REASONING_MODULE}.get_embeddings")
    @patch(f"{REASONING_MODULE}.multi_questions_response")
    @patch(f"{REASONING_MODULE}.multi_queries")
    def test_cleaned_mode_uses_text_directly(
        self,
        mock_queries: MagicMock,
        mock_questions: MagicMock,
        mock_embeddings: MagicMock,
        mock_similar: MagicMock,
        mock_refusion: MagicMock,
    ) -> None:
        """In cleaned mode, cached text is used directly without re-extraction."""
        source_content = {
            "mode": "cleaned",
            "pages": {
                "http://example.com": "test content here",
            },
            "pdfs": {},
        }
        mock_queries.return_value = (["test query"], None)
        mock_questions.return_value = (["question 1"], None)
        doc = ExtendedDocument(text="test content here", url="http://example.com")
        mock_embeddings.return_value = [doc]
        mock_similar.return_value = [doc]
        mock_refusion.return_value = [doc]

        result, raw_sc, _, _ = fetch_additional_information(
            client=MagicMock(),
            client_embedding=MagicMock(),
            prompt="test",
            model="gpt-4.1-2025-04-14",
            google_api_key=None,
            google_engine_id=None,
            serper_api_key=None,
            search_provider="google",
            source_content=source_content,
        )

        assert raw_sc is source_content
        assert "test content here" in result
        assert "http://example.com" in result

    @patch(f"{REASONING_MODULE}.reciprocal_rank_refusion")
    @patch(f"{REASONING_MODULE}.find_similar_chunks")
    @patch(f"{REASONING_MODULE}.get_embeddings")
    @patch(f"{REASONING_MODULE}.multi_questions_response")
    @patch(f"{REASONING_MODULE}.multi_queries")
    def test_raw_mode_re_extracts(
        self,
        mock_queries: MagicMock,
        mock_questions: MagicMock,
        mock_embeddings: MagicMock,
        mock_similar: MagicMock,
        mock_refusion: MagicMock,
    ) -> None:
        """In raw mode, HTML is re-extracted via extract_text."""
        source_content = {
            "mode": "raw",
            "pages": {
                "http://example.com": "<html><body>test content here</body></html>",
            },
            "pdfs": {},
        }
        mock_queries.return_value = (["test query"], None)
        mock_questions.return_value = (["question 1"], None)
        doc = ExtendedDocument(text="test content", url="http://example.com")
        mock_embeddings.return_value = [doc]
        mock_similar.return_value = [doc]
        mock_refusion.return_value = [doc]

        result, raw_sc, _, _ = fetch_additional_information(
            client=MagicMock(),
            client_embedding=MagicMock(),
            prompt="test",
            model="gpt-4.1-2025-04-14",
            google_api_key=None,
            google_engine_id=None,
            serper_api_key=None,
            search_provider="google",
            source_content=source_content,
        )

        assert raw_sc is source_content
        assert "http://example.com" in result

    @patch(f"{REASONING_MODULE}.multi_queries")
    def test_empty_source_content_raises(self, mock_queries: MagicMock) -> None:
        """Empty source_content raises ValueError (no valid documents)."""
        source_content: dict = {"pages": {}, "pdfs": {}}
        mock_queries.return_value = (["test query"], None)

        with pytest.raises(ValueError, match="No valid documents"):
            fetch_additional_information(
                client=MagicMock(),
                client_embedding=MagicMock(),
                prompt="test",
                model="gpt-4.1-2025-04-14",
                google_api_key=None,
                google_engine_id=None,
                serper_api_key=None,
                search_provider="google",
                source_content=source_content,
            )


def _make_mock_api_keys(return_source_content: str = "false") -> MagicMock:
    """Create a mock api_keys object (KeyChain-like) for run()."""
    services = {
        "openai": "sk-test",
        "google_api_key": None,
        "google_engine_id": None,
        "serperapi": None,
        "search_provider": "google",
        "return_source_content": return_source_content,
    }
    mock_keys = MagicMock()
    mock_keys.__getitem__ = MagicMock(side_effect=lambda k: services[k])
    mock_keys.get = MagicMock(
        side_effect=lambda k, default=None: services.get(k, default)
    )
    mock_keys.max_retries = MagicMock(
        return_value={"openai": 0, "anthropic": 0, "google_api_key": 0, "openrouter": 0}
    )
    return mock_keys


class TestRunFlagBehavior:
    """Verify return_source_content flag controls source_content in used_params."""

    @patch(
        f"{REASONING_MODULE}.parser_prediction_response", return_value='{"p_yes": 0.5}'
    )
    @patch(f"{REASONING_MODULE}.do_reasoning_with_retry")
    @patch(f"{REASONING_MODULE}.fetch_additional_information")
    @patch(f"{REASONING_MODULE}.LLMClientManager")
    def test_flag_on_includes_source_content(
        self,
        mock_mgr: MagicMock,
        mock_fetch: MagicMock,
        mock_reasoning: MagicMock,
        mock_parser: MagicMock,
    ) -> None:
        """When return_source_content is 'true', used_params contains source_content."""
        mock_llm = MagicMock()
        mock_embed = MagicMock()
        mock_mgr.return_value.__enter__ = MagicMock(return_value=(mock_llm, mock_embed))
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)

        mock_fetch.return_value = (
            "additional info",
            {"pages": {"http://x.com": "<html/>"}},
            ["query1"],
            None,
        )
        mock_reasoning.return_value = ("reasoning result", None)

        mock_llm.completions.return_value = MagicMock(
            content="<p_yes>0.5</p_yes>",
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        result = run(
            tool="prediction-request-reasoning",
            model="gpt-4.1-2025-04-14",
            prompt="test",
            api_keys=_make_mock_api_keys("true"),
        )

        used_params = result[4]
        assert "source_content" in used_params

    @patch(
        f"{REASONING_MODULE}.parser_prediction_response", return_value='{"p_yes": 0.5}'
    )
    @patch(f"{REASONING_MODULE}.do_reasoning_with_retry")
    @patch(f"{REASONING_MODULE}.fetch_additional_information")
    @patch(f"{REASONING_MODULE}.LLMClientManager")
    def test_flag_off_excludes_source_content(
        self,
        mock_mgr: MagicMock,
        mock_fetch: MagicMock,
        mock_reasoning: MagicMock,
        mock_parser: MagicMock,
    ) -> None:
        """When return_source_content is 'false', used_params omits source_content."""
        mock_llm = MagicMock()
        mock_embed = MagicMock()
        mock_mgr.return_value.__enter__ = MagicMock(return_value=(mock_llm, mock_embed))
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)

        mock_fetch.return_value = (
            "additional info",
            {"pages": {}},
            ["query1"],
            None,
        )
        mock_reasoning.return_value = ("reasoning result", None)

        mock_llm.completions.return_value = MagicMock(
            content="<p_yes>0.5</p_yes>",
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        result = run(
            tool="prediction-request-reasoning",
            model="gpt-4.1-2025-04-14",
            prompt="test",
            api_keys=_make_mock_api_keys("false"),
        )

        used_params = result[4]
        assert "source_content" not in used_params
