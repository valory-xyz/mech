# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
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

"""Unit tests for resolve_market_jury."""

import json
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
    VoterResult,
    _all_agree,
    _build_consensus_result,
    _compute_agreement,
    _decided_votes,
    _extract_json,
    _parse_vote,
    run,
)

MODULE = "packages.valory.customs.resolve_market_jury.resolve_market_jury"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vote(
    voter: str = "v1",
    has_occurred: bool = True,
    is_determinable: bool = True,
    is_valid: bool = True,
    confidence: float = 0.9,
    reasoning: str = "test",
    error: Optional[str] = None,
) -> VoterResult:
    """Build a VoterResult for testing."""
    return VoterResult(
        voter=voter,
        model="test-model",
        is_valid=is_valid,
        is_determinable=is_determinable,
        has_occurred=has_occurred,
        confidence=confidence,
        reasoning=reasoning,
        sources=["http://example.com"],
        error=error,
    )


def _mock_api_keys() -> MagicMock:
    """Build a mock KeyChain."""
    keys = MagicMock()
    keys.__getitem__ = MagicMock(return_value="fake-key")
    keys.max_retries.return_value = {"openai": 1, "openrouter": 1}
    keys.rotate = MagicMock()
    return keys


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    """Tests for JSON extraction from LLM responses."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ('{"a": 1}', {"a": 1}),
            ('  {"a": 1}  ', {"a": 1}),
            ('```json\n{"a": 1}\n```', {"a": 1}),
            ('```\n{"a": 1}\n```', {"a": 1}),
            ('Here is the result: {"a": 1} and more text.', {"a": 1}),
            ("no json here", None),
            ("{not: valid json}", None),
            ("```json\n{bad}\n```", None),
        ],
        ids=[
            "plain_json",
            "whitespace",
            "markdown_fenced",
            "markdown_no_lang",
            "embedded_in_text",
            "no_json",
            "invalid_braces",
            "invalid_fences",
        ],
    )
    def test_extract(self, text: str, expected: Optional[dict]) -> None:
        """Extract JSON from various formats."""
        assert _extract_json(text) == expected


# ---------------------------------------------------------------------------
# _parse_vote
# ---------------------------------------------------------------------------


class TestParseVote:
    """Tests for parsing LLM text into VoterResult."""

    def test_valid_response(self) -> None:
        """Parse a well-formed voter response."""
        raw = json.dumps(
            {
                "is_valid": True,
                "is_determinable": True,
                "has_occurred": False,
                "confidence": 0.85,
                "reasoning": "Found evidence",
                "sources": ["http://a.com"],
            }
        )
        result = _parse_vote(raw, "test", "model")
        assert result.has_occurred is False
        assert result.confidence == 0.85
        assert result.error is None

    def test_unparseable(self) -> None:
        """Return error VoterResult for garbage input."""
        result = _parse_vote("garbage", "test", "model")
        assert result.error is not None
        assert "Unparseable" in result.error

    def test_missing_fields_use_defaults(self) -> None:
        """Missing fields should get defaults."""
        result = _parse_vote('{"is_valid": true}', "test", "model")
        assert result.is_valid is True
        assert result.has_occurred is None
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Consensus helpers
# ---------------------------------------------------------------------------


class TestDecidedVotes:
    """Tests for _decided_votes filtering."""

    @pytest.mark.parametrize(
        "bad_vote, expected_count",
        [
            (_vote(is_determinable=False), 1),
            (_vote(is_valid=False), 1),
            (_vote(error="failed"), 1),
        ],
        ids=["indeterminate", "invalid", "error"],
    )
    def test_filters_bad_votes(
        self, bad_vote: VoterResult, expected_count: int
    ) -> None:
        """Bad votes are excluded, good ones kept."""
        votes = [bad_vote, _vote()]
        assert len(_decided_votes(votes)) == expected_count

    def test_all_valid(self) -> None:
        """All valid votes pass through."""
        votes = [_vote(), _vote(), _vote()]
        assert len(_decided_votes(votes)) == 3


class TestAllAgree:
    """Tests for _all_agree consensus check."""

    @pytest.mark.parametrize(
        "votes, expected",
        [
            ([_vote(has_occurred=True), _vote(has_occurred=True)], True),
            ([_vote(has_occurred=False), _vote(has_occurred=False)], True),
            ([_vote(has_occurred=True), _vote(has_occurred=False)], False),
            ([_vote()], False),
            (
                [
                    _vote(has_occurred=True),
                    _vote(has_occurred=True),
                    _vote(is_determinable=False),
                ],
                True,
            ),
        ],
        ids=[
            "unanimous_yes",
            "unanimous_no",
            "disagreement",
            "single",
            "ignores_indet",
        ],
    )
    def test_all_agree(self, votes: list, expected: bool) -> None:
        """Check consensus detection."""
        assert _all_agree(votes) is expected


class TestBuildConsensusResult:
    """Tests for _build_consensus_result."""

    def test_builds_result(self) -> None:
        """Consensus result has correct fields."""
        votes = [_vote(has_occurred=True), _vote(has_occurred=True)]
        result = _build_consensus_result(votes)
        assert result["has_occurred"] is True
        assert result["is_valid"] is True
        assert result["agreement_ratio"] == 1.0
        assert "judge skipped" in result["judge_reasoning"]


class TestComputeAgreement:
    """Tests for _compute_agreement."""

    def test_full_agreement(self) -> None:
        """All votes same gives 1.0."""
        votes = [_vote(has_occurred=True)] * 3
        assert _compute_agreement(votes) == 1.0

    def test_majority(self) -> None:
        """2-1 split gives 2/3."""
        votes = [
            _vote(has_occurred=True),
            _vote(has_occurred=True),
            _vote(has_occurred=False),
        ]
        assert abs(_compute_agreement(votes) - 2 / 3) < 0.01

    def test_empty(self) -> None:
        """No decided votes gives 0.0."""
        votes = [_vote(is_determinable=False)]
        assert _compute_agreement(votes) == 0.0


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class TestAdapterOpenai:
    """Tests for OpenAI adapter."""

    def test_responses_api(self) -> None:
        """Uses Responses API when available."""
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.text = '{"is_valid": true, "has_occurred": true, "confidence": 0.9}'
        mock_item.content = None
        mock_client.responses.create.return_value.output = [mock_item]

        with patch(f"{MODULE}.openai.OpenAI", return_value=mock_client):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _adapter_openai,
            )

            result = _adapter_openai("openai", "gpt-4.1", "prompt", "key")
            assert result.has_occurred is True

    def test_fallback_chat_completions(self) -> None:
        """Falls back to chat completions when responses API missing."""
        mock_client = MagicMock(spec=[])  # no 'responses' attr
        mock_client.chat = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"is_valid": true, "has_occurred": false, "confidence": 0.8}'
                )
            )
        ]

        with patch(f"{MODULE}.openai.OpenAI", return_value=mock_client):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _adapter_openai,
            )

            result = _adapter_openai("openai", "gpt-4.1", "prompt", "key")
            assert result.has_occurred is False

    def test_responses_api_nested_content(self) -> None:
        """Handles response items with nested content blocks."""
        mock_client = MagicMock()
        inner_block = MagicMock()
        inner_block.text = '{"is_valid": true, "has_occurred": true, "confidence": 0.9}'
        outer_item = MagicMock(spec=[])  # no 'text' attr
        outer_item.content = [inner_block]
        mock_client.responses.create.return_value.output = [outer_item]

        with patch(f"{MODULE}.openai.OpenAI", return_value=mock_client):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _adapter_openai,
            )

            result = _adapter_openai("openai", "gpt-4.1", "prompt", "key")
            assert result.has_occurred is True

    def test_responses_api_mixed_items(self) -> None:
        """Handles mix of items: some with text, some with content, some with neither."""
        mock_client = MagicMock()
        # Item with neither text nor content
        skip_item = MagicMock(spec=[])  # no text, no content
        # Item with content but inner block has no text
        no_text_block = MagicMock(spec=[])  # no text attr
        content_item = MagicMock(spec=[])
        content_item.content = [no_text_block]
        # Item with text (the actual response)
        text_item = MagicMock()
        text_item.text = '{"is_valid": true, "has_occurred": false, "confidence": 0.8}'
        mock_client.responses.create.return_value.output = [
            skip_item,
            content_item,
            text_item,
        ]

        with patch(f"{MODULE}.openai.OpenAI", return_value=mock_client):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _adapter_openai,
            )

            result = _adapter_openai("openai", "gpt-4.1", "prompt", "key")
            assert result.has_occurred is False


class TestAdapterOpenrouter:
    """Tests for OpenRouter adapter."""

    def test_appends_online(self) -> None:
        """Appends :online to model slug."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"is_valid": true, "has_occurred": true, "confidence": 0.9}'
                )
            )
        ]

        with patch(f"{MODULE}.openai.OpenAI", return_value=mock_client):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _adapter_openrouter,
            )

            result = _adapter_openrouter("grok", "x-ai/grok", "prompt", "key")
            call_args = mock_client.chat.completions.create.call_args
            assert ":online" in call_args.kwargs["model"]
            assert result.has_occurred is True

    def test_already_online(self) -> None:
        """Does not double-append :online."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content='{"is_valid": true}'))
        ]

        with patch(f"{MODULE}.openai.OpenAI", return_value=mock_client):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _adapter_openrouter,
            )

            _adapter_openrouter("grok", "model:online", "prompt", "key")
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "model:online"


# ---------------------------------------------------------------------------
# _run_judge
# ---------------------------------------------------------------------------


class TestRunJudge:
    """Tests for the judge function."""

    def test_judge_returns_verdict(self) -> None:
        """Judge parses valid JSON response."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"is_valid": true, "is_determinable": true, '
                    '"has_occurred": false, "judge_reasoning": "majority"}'
                )
            )
        ]

        with patch(f"{MODULE}.openai.OpenAI", return_value=mock_client):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _run_judge,
            )

            votes = [_vote(has_occurred=True), _vote(has_occurred=False)]
            result = _run_judge("question?", votes, "key")
            assert result["has_occurred"] is False

    def test_judge_unparseable(self) -> None:
        """Judge returns fallback on garbage response."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="not json at all"))
        ]

        with patch(f"{MODULE}.openai.OpenAI", return_value=mock_client):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _run_judge,
            )

            result = _run_judge("q?", [_vote()], "key")
            assert result["is_valid"] is False
            assert "Unparseable" in result["judge_reasoning"]

    def test_judge_retries_on_529(self) -> None:
        """Judge retries on 529 overloaded error."""
        mock_client = MagicMock()
        err = MagicMock()
        err.status_code = 529
        mock_client.chat.completions.create.side_effect = [
            __import__("openai").APIStatusError(
                message="overloaded", response=err, body=None
            ),
            MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content='{"is_valid": true, "has_occurred": true}'
                        )
                    )
                ]
            ),
        ]

        with (
            patch(f"{MODULE}.openai.OpenAI", return_value=mock_client),
            patch(f"{MODULE}.time.sleep"),
        ):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _run_judge,
            )

            result = _run_judge("q?", [_vote()], "key")
            assert result["has_occurred"] is True

    def test_judge_retries_exhausted_raises(self) -> None:
        """Judge raises when all retries are exhausted."""
        mock_client = MagicMock()
        err = MagicMock()
        err.status_code = 529
        api_err = __import__("openai").APIStatusError(
            message="overloaded", response=err, body=None
        )
        mock_client.chat.completions.create.side_effect = api_err

        with (
            patch(f"{MODULE}.openai.OpenAI", return_value=mock_client),
            patch(f"{MODULE}.time.sleep"),
            pytest.raises(__import__("openai").APIStatusError),
        ):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _run_judge,
            )

            _run_judge("q?", [_vote()], "key")


# ---------------------------------------------------------------------------
# collect_votes / cast_vote
# ---------------------------------------------------------------------------


class TestCollectVotes:
    """Tests for vote collection."""

    def test_collects_from_all_voters(self) -> None:
        """Collects votes from all registered voters."""
        mock_result = _vote()

        with patch(f"{MODULE}.cast_vote", return_value=mock_result):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                collect_votes,
            )

            results = collect_votes("question?", ["openai", "grok"], _mock_api_keys())
            assert len(results) == 2

    def test_handles_voter_failure(self) -> None:
        """Failed voters are skipped, others still collected."""
        call_count = 0

        def side_effect(*args: str, **kwargs: str) -> VoterResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API down")
            return _vote()

        with patch(f"{MODULE}.cast_vote", side_effect=side_effect):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                collect_votes,
            )

            results = collect_votes("q?", ["a", "b"], _mock_api_keys())
            assert len(results) == 1


class TestCastVote:
    """Tests for the cast_vote facade."""

    def test_delegates_to_adapter(self) -> None:
        """cast_vote looks up registry and calls adapter."""
        mock_result = _vote()
        mock_adapter = MagicMock(return_value=mock_result)

        with patch(f"{MODULE}._ADAPTERS", {"_adapter_openai": mock_adapter}):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                cast_vote,
            )

            keys = _mock_api_keys()
            result = cast_vote("openai", "question?", keys)
            assert result is mock_result
            mock_adapter.assert_called_once()


# ---------------------------------------------------------------------------
# with_key_rotation
# ---------------------------------------------------------------------------


class TestWithKeyRotation:
    """Tests for the key rotation decorator."""

    def test_success_appends_api_keys(self) -> None:
        """Successful call appends api_keys to result tuple."""
        keys = _mock_api_keys()

        @__import__(
            "packages.valory.customs.resolve_market_jury.resolve_market_jury",
            fromlist=["with_key_rotation"],
        ).with_key_rotation
        def fake_run(**kwargs: str) -> tuple:
            return ("result", None, None, None)

        result = fake_run(api_keys=keys, tool="resolve-market-jury-v1")
        assert len(result) == 5
        assert result[0] == "result"
        assert result[4] is keys

    def test_broad_exception_returns_error(self) -> None:
        """Unexpected exceptions return error string."""
        keys = _mock_api_keys()

        @__import__(
            "packages.valory.customs.resolve_market_jury.resolve_market_jury",
            fromlist=["with_key_rotation"],
        ).with_key_rotation
        def fake_run(**kwargs: str) -> tuple:
            raise TypeError("bad")

        result = fake_run(api_keys=keys)
        assert "bad" in result[0]

    def test_rate_limit_rotates_keys(self) -> None:
        """Verify RateLimitError triggers key rotation and retry."""
        keys = _mock_api_keys()
        keys.max_retries.return_value = {"openai": 2, "openrouter": 2}
        call_count = 0

        @__import__(
            "packages.valory.customs.resolve_market_jury.resolve_market_jury",
            fromlist=["with_key_rotation"],
        ).with_key_rotation
        def fake_run(**kwargs: str) -> tuple:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise __import__("openai").RateLimitError(
                    message="rate limited",
                    response=MagicMock(status_code=429),
                    body=None,
                )
            return ("ok", None, None, None)

        result = fake_run(api_keys=keys)
        assert result[0] == "ok"
        assert keys.rotate.called

    def test_rate_limit_exhausted_raises(self) -> None:
        """Verify RateLimitError raises when retries exhausted."""
        keys = _mock_api_keys()
        keys.max_retries.return_value = {"openai": 0, "openrouter": 0}

        @__import__(
            "packages.valory.customs.resolve_market_jury.resolve_market_jury",
            fromlist=["with_key_rotation"],
        ).with_key_rotation
        def fake_run(**kwargs: str) -> tuple:
            raise __import__("openai").RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body=None,
            )

        with pytest.raises(__import__("openai").RateLimitError):
            fake_run(api_keys=keys)


# ---------------------------------------------------------------------------
# run() -- the main entry point
# ---------------------------------------------------------------------------


class TestRun:
    """Tests for the run() entry point."""

    def test_invalid_tool_returns_error(self) -> None:
        """Invalid tool name returns error string (caught by decorator)."""
        keys = _mock_api_keys()
        result = run(prompt="q?", tool="bad-tool", api_keys=keys)
        assert "not supported" in result[0]

    def test_cost_mode(self) -> None:
        """delivery_rate=0 returns cost via callback."""
        keys = _mock_api_keys()
        cb = MagicMock(return_value=0.05)
        # The decorator tries result + (api_keys,) but cost mode returns
        # a float. TypeError is caught by the decorator's broad except.
        # We verify the callback was invoked and the error message mentions
        # the type issue (confirming cost mode was reached).
        result = run(
            prompt="q?",
            tool="resolve-market-jury-v1",
            api_keys=keys,
            delivery_rate=0,
            counter_callback=cb,
        )
        cb.assert_called_once()
        # Decorator catches TypeError from float + tuple, returns error string
        assert "unsupported operand" in result[0]

    def test_cost_mode_no_callback_returns_error(self) -> None:
        """delivery_rate=0 without callback returns error (caught by decorator)."""
        keys = _mock_api_keys()
        result = run(
            prompt="q?",
            tool="resolve-market-jury-v1",
            api_keys=keys,
            delivery_rate=0,
        )
        assert "counter callback" in result[0]

    def test_unanimous_skips_judge(self) -> None:
        """Unanimous votes skip the judge."""
        keys = _mock_api_keys()
        unanimous_votes = [_vote(has_occurred=True)] * 3

        with (
            patch(f"{MODULE}.collect_votes", return_value=unanimous_votes),
            patch(f"{MODULE}._run_judge") as mock_judge,
        ):
            result = run(
                prompt="q?",
                tool="resolve-market-jury-v1",
                api_keys=keys,
            )
            mock_judge.assert_not_called()
            parsed = json.loads(result[0])
            assert parsed["has_occurred"] is True

    def test_disagreement_calls_judge(self) -> None:
        """Disagreeing votes invoke the judge."""
        keys = _mock_api_keys()
        mixed_votes = [_vote(has_occurred=True), _vote(has_occurred=False)]
        judge_verdict = {
            "is_valid": True,
            "is_determinable": True,
            "has_occurred": False,
            "judge_reasoning": "majority wins",
        }

        with (
            patch(f"{MODULE}.collect_votes", return_value=mixed_votes),
            patch(f"{MODULE}._run_judge", return_value=judge_verdict),
        ):
            result = run(
                prompt="q?",
                tool="resolve-market-jury-v1",
                api_keys=keys,
            )
            parsed = json.loads(result[0])
            assert parsed["has_occurred"] is False
            assert parsed["judge_reasoning"] == "majority wins"

    def test_no_votes_returns_failure(self) -> None:
        """No successful votes returns failure result."""
        keys = _mock_api_keys()

        with patch(f"{MODULE}.collect_votes", return_value=[]):
            result = run(
                prompt="q?",
                tool="resolve-market-jury-v1",
                api_keys=keys,
            )
            parsed = json.loads(result[0])
            assert parsed["is_valid"] is False
            assert parsed["n_successful"] == 0
