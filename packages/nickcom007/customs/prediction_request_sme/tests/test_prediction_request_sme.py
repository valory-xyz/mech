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

"""Unit tests for prediction_request_sme: thread-safe client, offline tiktoken, and source_content."""

import inspect
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import packages.nickcom007.customs.prediction_request_sme.prediction_request_sme as module
from packages.nickcom007.customs.prediction_request_sme.prediction_request_sme import (
    OpenAIClientManager,
    extract_texts,
    fetch_additional_information,
    get_sme_role,
    run,
)


class TestOpenAIClientManager:
    """Verify OpenAIClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_instance(self) -> None:
        """__enter__ returns a fresh OpenAI client, __exit__ closes it."""
        mgr = OpenAIClientManager(api_key="sk-test")
        with patch(
            "packages.nickcom007.customs.prediction_request_sme.prediction_request_sme.OpenAI"
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

    def test_fetch_additional_information_requires_client_param(self) -> None:
        """fetch_additional_information requires client as first param."""
        params = list(inspect.signature(fetch_additional_information).parameters)
        assert params[0] == "client"

    def test_get_sme_role_requires_client_param(self) -> None:
        """get_sme_role requires client as first param."""
        params = list(inspect.signature(get_sme_role).parameters)
        assert params[0] == "client"


SME_MODULE = "packages.nickcom007.customs.prediction_request_sme.prediction_request_sme"


def _make_html_future(url: str, html: str) -> tuple:
    """Create a (future, url) pair with a fake HTML response."""
    response = MagicMock()
    response.status_code = 200
    response.text = html
    future: Future = Future()
    future.set_result(response)
    return (future, url)


class TestExtractTextsCapture:
    """Verify extract_texts captures source content correctly."""

    @patch(f"{SME_MODULE}.process_in_batches")
    def test_cleaned_mode_stores_extracted_text(self, mock_batches: MagicMock) -> None:
        """In cleaned mode (default), extracted text is stored instead of raw HTML."""
        html = "<html><body>Hello world</body></html>"
        mock_batches.return_value = [[_make_html_future("http://example.com", html)]]

        _, raw_sc = extract_texts(["http://example.com"], num_words=300)

        assert raw_sc["mode"] == "cleaned"
        assert "http://example.com" in raw_sc["pages"]
        assert raw_sc["pages"]["http://example.com"] != html
        assert "Hello world" in raw_sc["pages"]["http://example.com"]

    @patch(f"{SME_MODULE}.process_in_batches")
    def test_raw_mode_stores_html(self, mock_batches: MagicMock) -> None:
        """In raw mode, raw HTML is stored."""
        html = "<html><body>Hello world</body></html>"
        mock_batches.return_value = [[_make_html_future("http://example.com", html)]]

        _, raw_sc = extract_texts(
            ["http://example.com"], num_words=300, source_content_mode="raw"
        )

        assert raw_sc["mode"] == "raw"
        assert raw_sc["pages"]["http://example.com"] == html

    @patch(f"{SME_MODULE}.process_in_batches")
    def test_non_200_not_captured(self, mock_batches: MagicMock) -> None:
        """Non-200 responses are not stored in raw_source_content."""
        response = MagicMock()
        response.status_code = 404
        future: Future = Future()
        future.set_result(response)
        mock_batches.return_value = [[(future, "http://example.com")]]

        _, raw_sc = extract_texts(["http://example.com"], num_words=300)

        assert not raw_sc["pages"]


class TestFetchReplayPath:
    """Verify fetch_additional_information replays from structured source_content."""

    def _make_mock_client(self) -> MagicMock:
        mock_client = MagicMock()
        mock_client.moderations.create.return_value = MagicMock(
            results=[MagicMock(flagged=False)]
        )
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"queries": ["test"]}'))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )
        return mock_client

    def test_cleaned_mode_uses_text_directly(self) -> None:
        """In cleaned mode, cached text is used directly without re-extraction."""
        source_content = {
            "mode": "cleaned",
            "pages": {
                "http://example.com": "test content here",
            },
        }

        result, raw_sc, _ = fetch_additional_information(
            client=self._make_mock_client(),
            prompt="test",
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

    def test_raw_mode_re_extracts(self) -> None:
        """In raw mode, HTML is re-extracted via extract_text."""
        source_content = {
            "mode": "raw",
            "pages": {
                "http://example.com": "<html><body>test content here</body></html>",
            },
        }

        result, raw_sc, _ = fetch_additional_information(
            client=self._make_mock_client(),
            prompt="test",
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
        """Source content without mode key defaults to cleaned."""
        source_content = {
            "pages": {
                "http://example.com": "already cleaned text",
            },
        }

        result, _, _ = fetch_additional_information(
            client=self._make_mock_client(),
            prompt="test",
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

    def test_empty_source_content(self) -> None:
        """Empty source_content produces empty result without error."""
        source_content: dict = {"pages": {}}

        result, _, _ = fetch_additional_information(
            client=self._make_mock_client(),
            prompt="test",
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

    @patch(f"{SME_MODULE}.fetch_additional_information")
    @patch(f"{SME_MODULE}.get_sme_role")
    @patch(f"{SME_MODULE}.OpenAIClientManager")
    def test_flag_on_includes_source_content(
        self,
        mock_mgr: MagicMock,
        mock_sme_role: MagicMock,
        mock_fetch: MagicMock,
    ) -> None:
        """When return_source_content is 'true', used_params contains source_content."""
        mock_client = MagicMock()
        mock_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)

        mock_sme_role.return_value = (None, "You are a helpful assistant.", None)
        mock_fetch.return_value = (
            "additional info",
            {"pages": {"http://x.com": "<html/>"}},
            None,
        )

        mock_client.moderations.create.return_value = MagicMock(
            results=[MagicMock(flagged=False)]
        )
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"p_yes": 0.5}'))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        result = run(
            tool="prediction-online-sme",
            model="gpt-4.1-2025-04-14",
            prompt="test",
            api_keys=_make_mock_api_keys("true"),
        )

        used_params = result[4]
        assert "source_content" in used_params

    @patch(f"{SME_MODULE}.fetch_additional_information")
    @patch(f"{SME_MODULE}.get_sme_role")
    @patch(f"{SME_MODULE}.OpenAIClientManager")
    def test_flag_off_excludes_source_content(
        self,
        mock_mgr: MagicMock,
        mock_sme_role: MagicMock,
        mock_fetch: MagicMock,
    ) -> None:
        """When return_source_content is 'false', used_params omits source_content."""
        mock_client = MagicMock()
        mock_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)

        mock_sme_role.return_value = (None, "You are a helpful assistant.", None)
        mock_fetch.return_value = ("additional info", {"pages": {}}, None)

        mock_client.moderations.create.return_value = MagicMock(
            results=[MagicMock(flagged=False)]
        )
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"p_yes": 0.5}'))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        result = run(
            tool="prediction-online-sme",
            model="gpt-4.1-2025-04-14",
            prompt="test",
            api_keys=_make_mock_api_keys("false"),
        )

        used_params = result[4]
        assert "source_content" not in used_params
