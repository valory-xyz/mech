# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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
"""Tests for task_submission_abci.behaviours — non-generator and key-branch coverage."""

import contextlib
import json
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, cast
from unittest.mock import MagicMock, call, patch

import pytest
from aea_ledger_ethereum import EthereumApi

from packages.valory.contracts.complementary_service_metadata.contract import (
    ComplementaryServiceMetadata,
)
from packages.valory.contracts.hash_checkpoint.contract import HashCheckpointContract
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.task_execution.behaviours import (
    PREDICT_API_EVENTS,
    SETTLING_NONCES_BY_SENDER,
)
from packages.valory.skills.task_submission_abci import behaviours as beh_mod
from packages.valory.skills.task_submission_abci.behaviours import (
    DONE_TASKS,
    DeliverBehaviour,
    ENQUEUED_AT_LOCAL,
    FundsSplittingBehaviour,
    IS_OFFCHAIN,
    LAST_TX,
    MAX_SETTLEMENT_ATTEMPTS,
    MECH_ADDRESS,
    MarketplaceData,
    MarketplaceKeys,
    NONCE,
    OffchainDataKey,
    OffchainDataValue,
    OffchainKeys,
    PAYMENT_MODEL,
    SENDER,
    SETTLED_COUNTED_TX_KEY,
    SETTLEMENT_ATTEMPTS_KEY,
    SETTLEMENT_ATTEMPT_PERIOD_KEY,
    SETTLEMENT_OUTCOME_CONTRACT_ERROR,
    SETTLEMENT_OUTCOME_DROPPED,
    SETTLEMENT_OUTCOME_SETTLED,
    SETTLEMENT_OUTCOME_SIM_FAILED,
    SOURCE_OFFCHAIN,
    SOURCE_ONCHAIN,
    TaskExecutionBaseBehaviour,
    TaskPoolingBehaviour,
    TaskSubmissionRoundBehaviour,
    TransactionPreparationBehaviour,
    ZERO_ADDRESS,
    ZERO_IPFS_HASH,
)
from packages.valory.skills.task_submission_abci.rounds import decode_tx_payload
from packages.valory.skills.task_submission_abci.tests.conftest import (
    _error_contract_msg,
    _error_ledger_msg,
    _gen_returning,
    _make_benchmark_ctx,
    _make_ctx,
    _make_fs_ctx,
    _make_full_ctx,
    _make_lock,
    _mock_to_multihash,
    _noop_gen,
    _noop_gen_with_args,
    _raw_tx_contract_msg,
    _run_gen,
    _state_contract_msg,
    _state_ledger_msg,
)

# ---------------------------------------------------------------------------
# Helpers: concrete subclasses + minimal skill context
# ---------------------------------------------------------------------------

_CAST_ROUND = type("DummyRound", (), {})


class _SyncedDataMixin:
    """Mixin that overrides synchronized_data to read from _synchronized_data."""

    _synchronized_data: Any = None

    @property
    def synchronized_data(self) -> Any:  # type: ignore
        """Return the test-provided synchronized data."""
        return self._synchronized_data


class _DummyBase(_SyncedDataMixin, TaskExecutionBaseBehaviour):
    """Minimal concrete subclass for testing TaskExecutionBaseBehaviour."""

    matching_round: Type[AbstractRound] = _CAST_ROUND  # type: ignore

    def async_act(self) -> Generator[None, None, None]:
        yield from ()


class _DummyPooling(_SyncedDataMixin, TaskPoolingBehaviour):
    """Minimal concrete subclass for testing TaskPoolingBehaviour."""

    matching_round: Type[AbstractRound] = _CAST_ROUND  # type: ignore

    def async_act(self) -> Generator[None, None, None]:
        yield from ()


class _DummyDeliver(_SyncedDataMixin, DeliverBehaviour):
    """Minimal concrete subclass for testing DeliverBehaviour."""

    matching_round: Type[AbstractRound] = _CAST_ROUND  # type: ignore

    def async_act(self) -> Generator[None, None, None]:
        yield from ()


class _DummyFunds(_SyncedDataMixin, FundsSplittingBehaviour):
    """Minimal concrete subclass for testing FundsSplittingBehaviour."""

    matching_round: Type[AbstractRound] = _CAST_ROUND  # type: ignore

    def async_act(self) -> Generator[None, None, None]:
        yield from ()


class _DummyTransPrep(_SyncedDataMixin, TransactionPreparationBehaviour):
    """Minimal concrete subclass for testing TransactionPreparationBehaviour."""

    def async_act(self) -> Generator[None, None, None]:
        yield from ()


# ---------------------------------------------------------------------------
# TaskExecutionBaseBehaviour — properties & non-generator methods
# ---------------------------------------------------------------------------


class TestDoneTasksProperty:
    """Tests for TestDoneTasksProperty."""

    def test_returns_copy_of_tasks(self) -> None:
        """Test returns copy of tasks."""
        tasks = [{"request_id": "r1"}]
        ctx = _make_ctx(done_tasks=tasks)
        b = _DummyBase(name="b", skill_context=ctx)
        result = b.done_tasks
        assert result == tasks
        # deepcopy — mutation doesn't affect shared_state
        result.append({"request_id": "r2"})
        assert len(ctx.shared_state[DONE_TASKS]) == 1


class TestPaymentModelProperty:
    """Tests for TestPaymentModelProperty."""

    def test_returns_none_by_default_and_value_when_set(self) -> None:
        """Test returns None when not set, and the value when set."""
        ctx = _make_ctx()
        b = _DummyBase(name="b", skill_context=ctx)
        assert b.payment_model is None
        ctx.shared_state[PAYMENT_MODEL] = "native"
        assert b.payment_model == "native"


class TestDoneTasksLock:
    """Test Done Tasks Lock."""

    def test_returns_lock_from_shared_state(self) -> None:
        """Test returns lock from shared state."""
        lock = _make_lock()
        ctx = _make_ctx(lock=lock)
        b = _DummyBase(name="b", skill_context=ctx)
        assert b.done_tasks_lock() is lock


class TestMechAddresses:
    """Test Mech Addresses."""

    def test_returns_from_params(self) -> None:
        """Test returns from params."""
        ctx = _make_ctx(agent_mech_addresses=["0xA", "0xB"])
        b = _DummyBase(name="b", skill_context=ctx)
        assert b.mech_addresses == ["0xA", "0xB"]


class TestRemoveTasks:
    """Test Remove Tasks."""

    def test_remove_submitted_task(self) -> None:
        """Test remove submitted task."""
        tasks = [{"request_id": "r1"}, {"request_id": "r2"}]
        ctx = _make_ctx(done_tasks=tasks)
        b = _DummyBase(name="b", skill_context=ctx)
        b.remove_tasks([{"request_id": "r1"}])
        # r1 was submitted → removed; r2 stays
        remaining = ctx.shared_state[DONE_TASKS]
        assert len(remaining) == 1
        assert remaining[0]["request_id"] == "r2"

    def test_no_change_when_empty_submitted(self) -> None:
        """Test no change when empty submitted."""
        tasks = [{"request_id": "r1"}]
        ctx = _make_ctx(done_tasks=tasks)
        b = _DummyBase(name="b", skill_context=ctx)
        b.remove_tasks([])
        assert len(ctx.shared_state[DONE_TASKS]) == 1

    def test_all_submitted_leaves_empty(self) -> None:
        """Test all submitted leaves empty."""
        tasks = [{"request_id": "r1"}, {"request_id": "r2"}]
        ctx = _make_ctx(done_tasks=tasks)
        b = _DummyBase(name="b", skill_context=ctx)
        b.remove_tasks(tasks)
        assert ctx.shared_state[DONE_TASKS] == []

    def test_remove_submitted_task_mixed_types(self) -> None:
        """``remove_tasks`` matches across the ``str`` / ``int`` split.

        Mixed ``request_id`` types (``str`` vs ``int``) can survive
        an in-place restart. Both sides are ``str``-normalised on the
        equality check so a delivered task doesn't silently escape
        removal and get re-emitted.
        """
        # ``str`` in shared_state; ``int`` in submitted.
        done_tasks_str: List[Dict[str, Any]] = [{"request_id": "5"}]
        ctx = _make_ctx(done_tasks=done_tasks_str)
        b = _DummyBase(name="b", skill_context=ctx)
        b.remove_tasks([{"request_id": 5}])
        assert ctx.shared_state[DONE_TASKS] == []

        # Same, other direction.
        done_tasks_int: List[Dict[str, Any]] = [{"request_id": 5}]
        ctx = _make_ctx(done_tasks=done_tasks_int)
        b = _DummyBase(name="b", skill_context=ctx)
        b.remove_tasks([{"request_id": "5"}])
        assert ctx.shared_state[DONE_TASKS] == []


class TestRemoveTasksById:
    """Tests for :meth:`TaskExecutionBaseBehaviour.remove_tasks_by_id`."""

    def test_removes_matching_task(self) -> None:
        """Passing an id list prunes matching tasks and leaves the rest."""
        tasks = [{"request_id": "r1"}, {"request_id": "r2"}]
        ctx = _make_ctx(done_tasks=tasks)
        b = _DummyBase(name="b", skill_context=ctx)
        b.remove_tasks_by_id(["r1"])
        remaining = ctx.shared_state[DONE_TASKS]
        assert len(remaining) == 1
        assert remaining[0]["request_id"] == "r2"

    def test_empty_id_list_is_noop(self) -> None:
        """An empty id list leaves shared_state untouched."""
        tasks = [{"request_id": "r1"}]
        ctx = _make_ctx(done_tasks=tasks)
        b = _DummyBase(name="b", skill_context=ctx)
        b.remove_tasks_by_id([])
        assert ctx.shared_state[DONE_TASKS] == tasks

    def test_all_ids_leaves_empty(self) -> None:
        """Passing every id present clears shared_state."""
        tasks = [{"request_id": "r1"}, {"request_id": "r2"}]
        ctx = _make_ctx(done_tasks=tasks)
        b = _DummyBase(name="b", skill_context=ctx)
        b.remove_tasks_by_id(["r1", "r2"])
        assert ctx.shared_state[DONE_TASKS] == []

    def test_int_shared_state_matches_str_id(self) -> None:
        """A legacy ``int`` in shared_state is matched by a ``str`` id.

        The id list side is typed ``List[str]``; callers must go
        through :func:`extract_request_ids` which normalises to
        ``str`` at the write site. The shared_state side, in
        contrast, may still carry legacy ``int`` request_ids from a
        pre-fix boot, so this direction of the equality is normalised
        via ``str(done_task.get("request_id"))`` on lookup.
        """
        ctx = _make_ctx(done_tasks=[{"request_id": 5}])
        b = _DummyBase(name="b", skill_context=ctx)
        b.remove_tasks_by_id(["5"])
        assert ctx.shared_state[DONE_TASKS] == []

    def test_id_not_in_shared_state_is_noop(self) -> None:
        """An id absent from shared_state doesn't affect other entries.

        Guards the case where the executor lost state (restart or
        prune-race) between the on-chain settle and the next prune.
        The id is still valid for the marketplace side; we just skip
        the pop safely.
        """
        tasks = [{"request_id": "r1"}]
        ctx = _make_ctx(done_tasks=tasks)
        b = _DummyBase(name="b", skill_context=ctx)
        b.remove_tasks_by_id(["r-missing"])
        assert ctx.shared_state[DONE_TASKS] == tasks

    def test_pre_settlement_remove_tasks_preserves_predict_api_cache(self) -> None:
        """``remove_tasks_by_id`` does not prune ``PREDICT_API_EVENTS``.

        The event cache prune belongs to ``handle_submitted_tasks``
        (post-settlement). The pre-settlement caller
        ``TransactionPreparationBehaviour.get_payload_content`` also
        calls ``remove_tasks_by_id`` on the simulation-skip path;
        popping the cache there strands the event because the task
        row is still present in the consensus ``done_tasks``.
        """
        now = time.time()
        ctx = _make_ctx(done_tasks=[{"request_id": "r1"}, {"request_id": "r2"}])
        ctx.shared_state[PREDICT_API_EVENTS] = {
            "r1": {"event": {"src": "off"}, "written_at": now},
            "r2": {"event": {"src": "off"}, "written_at": now},
        }
        b = _DummyBase(name="b", skill_context=ctx)
        b.remove_tasks_by_id(["r1"])
        surviving = ctx.shared_state[PREDICT_API_EVENTS]
        assert "r1" in surviving
        assert "r2" in surviving


class TestSetGauge:
    """Test Set Gauge."""

    def test_set_gauge_with_labels(self) -> None:
        """Test set gauge with labels."""
        ctx = _make_ctx()
        b = _DummyBase(name="b", skill_context=ctx)
        metric = MagicMock()
        b.set_gauge(metric, 42, chain="gnosis")
        metric.labels.assert_called_with(chain="gnosis")
        metric.labels().set.assert_called_with(42)
        metric.labels().set_to_current_time.assert_called()

    def test_set_gauge_without_labels(self) -> None:
        """Test set gauge without labels."""
        ctx = _make_ctx()
        b = _DummyBase(name="b", skill_context=ctx)
        metric = MagicMock()
        b.set_gauge(metric, 100)
        metric.set.assert_called_with(100)
        metric.set_to_current_time.assert_called()


class TestObserveHistogram:
    """Test Observe Histogram."""

    def test_observe_with_labels(self) -> None:
        """Test observe with labels."""
        ctx = _make_ctx()
        b = _DummyBase(name="b", skill_context=ctx)
        metric = MagicMock()
        b.observe_histogram(metric, 3.14, tool="my_tool")
        metric.labels.assert_called_with(tool="my_tool")
        metric.labels().observe.assert_called_with(3.14)

    def test_observe_without_labels(self) -> None:
        """Test observe without labels."""
        ctx = _make_ctx()
        b = _DummyBase(name="b", skill_context=ctx)
        metric = MagicMock()
        b.observe_histogram(metric, 1.5)
        metric.observe.assert_called_with(1.5)


# ---------------------------------------------------------------------------
# TaskPoolingBehaviour — non-generator methods
# ---------------------------------------------------------------------------


class TestSetTx:
    """Test Set Tx."""

    def test_stores_tx_hash_and_timestamp(self) -> None:
        """Test stores tx hash and timestamp."""
        ctx = _make_ctx()
        b = _DummyPooling(name="b", skill_context=ctx)
        before = time.time()
        b.set_tx("0xhash")
        after = time.time()
        tx, ts = ctx.shared_state[LAST_TX]
        assert tx == "0xhash"
        assert before <= ts <= after


class TestCheckLastTxStatus:
    """Test Check Last Tx Status."""

    def test_returns_true_with_hash_when_final_tx_hash_exists(self) -> None:
        """Test returns true with hash when final tx hash exists."""
        ctx = _make_ctx()
        b = _DummyPooling(name="b", skill_context=ctx)
        # Patch synchronized_data on the behaviour
        mock_sd = MagicMock()
        mock_sd.final_tx_hash = "0xfinal"
        b._synchronized_data = mock_sd
        result = b.check_last_tx_status()
        assert result == (True, "0xfinal")
        # Also verify set_tx was called
        assert ctx.shared_state[LAST_TX][0] == "0xfinal"

    def test_returns_false_empty_when_exception_raised(self) -> None:
        """Test returns false empty when exception raised."""
        ctx = _make_ctx()
        b = _DummyPooling(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        type(mock_sd).final_tx_hash = property(
            lambda self: (_ for _ in ()).throw(ValueError("not set"))
        )
        b._synchronized_data = mock_sd
        result = b.check_last_tx_status()
        assert result == (False, "")

    def test_returns_false_when_final_tx_hash_is_none(self) -> None:
        """Test returns false when final tx hash is none."""
        ctx = _make_ctx()
        b = _DummyPooling(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.final_tx_hash = None
        b._synchronized_data = mock_sd
        result = b.check_last_tx_status()
        assert result == (False, "")


class TestGetDoneTasks:
    """Test Get Done Tasks."""

    def test_returns_tasks_immediately_when_available(self) -> None:
        """Test returns tasks immediately when available."""
        tasks = [{"request_id": "r1"}]
        ctx = _make_ctx(done_tasks=tasks)
        b = _DummyPooling(name="b", skill_context=ctx)
        result = _run_gen(b.get_done_tasks(timeout=5.0))
        assert result == tasks

    def test_returns_empty_list_after_timeout(self) -> None:
        """Test returns empty list after timeout."""
        ctx = _make_ctx(done_tasks=[])
        b = _DummyPooling(name="b", skill_context=ctx)

        sleep_called = []

        def fake_sleep(seconds: float) -> Generator[None, None, None]:  # type: ignore[misc]
            sleep_called.append(seconds)
            if False:
                yield

        with patch.object(b, "sleep", side_effect=fake_sleep):
            result = _run_gen(b.get_done_tasks(timeout=0.01))
        assert result == []


# ---------------------------------------------------------------------------
# DeliverBehaviour — pure method _update_current_delivery_report
# ---------------------------------------------------------------------------


class TestUpdateCurrentDeliveryReport:
    """Test Update Current Delivery Report."""

    def _make_b(self) -> _DummyDeliver:
        ctx = _make_ctx()
        return _DummyDeliver(name="b", skill_context=ctx)

    def test_new_agent_and_tool(self) -> None:
        """Test new agent and tool."""
        b = self._make_b()
        task = {"task_executor_address": "agent-0", "tool": "tool-a"}
        result = b._update_current_delivery_report({}, [task])
        assert result == {"agent-0": {"tool-a": 1}}

    def test_increments_existing_tool(self) -> None:
        """Test increments existing tool."""
        b = self._make_b()
        current = {"agent-0": {"tool-a": 5}}
        task = {"task_executor_address": "agent-0", "tool": "tool-a"}
        result = b._update_current_delivery_report(current, [task])
        assert result["agent-0"]["tool-a"] == 6

    def test_new_tool_for_existing_agent(self) -> None:
        """Test new tool for existing agent."""
        b = self._make_b()
        current = {"agent-0": {"tool-a": 2}}
        task = {"task_executor_address": "agent-0", "tool": "tool-b"}
        result = b._update_current_delivery_report(current, [task])
        assert result["agent-0"]["tool-a"] == 2
        assert result["agent-0"]["tool-b"] == 1

    def test_multiple_tasks_multiple_agents(self) -> None:
        """Test multiple tasks multiple agents."""
        b = self._make_b()
        tasks = [
            {"task_executor_address": "agent-0", "tool": "tool-a"},
            {"task_executor_address": "agent-1", "tool": "tool-b"},
            {"task_executor_address": "agent-0", "tool": "tool-a"},
        ]
        result = b._update_current_delivery_report({}, tasks)
        assert result["agent-0"]["tool-a"] == 2
        assert result["agent-1"]["tool-b"] == 1

    def test_empty_tasks_returns_unchanged(self) -> None:
        """Test empty tasks returns unchanged."""
        b = self._make_b()
        current = {"agent-0": {"tool-a": 3}}
        result = b._update_current_delivery_report(current, [])
        assert result == {"agent-0": {"tool-a": 3}}


# ---------------------------------------------------------------------------
# DeliverBehaviour — get_delivery_report (generator with mocked _get_current_delivery_report)
# ---------------------------------------------------------------------------


class TestGetDeliveryReport:
    """Test Get Delivery Report."""

    def _make_b(self, done_tasks: Any = None) -> _DummyDeliver:
        ctx = _make_ctx()
        b = _DummyDeliver(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.done_tasks = done_tasks or []
        b._synchronized_data = mock_sd
        return b

    def test_returns_none_when_current_report_is_none(self) -> None:
        """Test returns none when current report is none."""
        b = self._make_b()
        with patch.object(
            b, "_get_current_delivery_report", side_effect=_gen_returning(None)
        ):
            result = _run_gen(b.get_delivery_report())
        assert result is None

    def test_returns_updated_report(self) -> None:
        """Test returns updated report."""
        task = {"task_executor_address": "agent-0", "tool": "tool-x"}
        b = self._make_b(done_tasks=[task])
        mock_sd = MagicMock()
        mock_sd.done_tasks = [task]
        b._synchronized_data = mock_sd
        with patch.object(
            b, "_get_current_delivery_report", side_effect=_gen_returning({})
        ):
            result = _run_gen(b.get_delivery_report())
        assert result == {"agent-0": {"tool-x": 1}}


# ---------------------------------------------------------------------------
# Enum constants test
# ---------------------------------------------------------------------------


class TestEnumConstants:
    """Verify the module-level Enum classes are accessible."""

    def test_enum_values(self) -> None:
        """All contract enum constants have expected values."""
        assert OffchainKeys.DELIVER_WITH_SIGNATURES.value == "deliverWithSignatures"
        assert OffchainDataKey.REQUEST_DATA_KEY.value == "requestData"
        assert OffchainDataValue.IPFS_HASH.value == "ipfs_hash"
        assert MarketplaceKeys.REQUEST_IDS.value == "requestIds"
        assert MarketplaceData.REQUEST_ID.value == "requestId"


# ---------------------------------------------------------------------------
# TaskPoolingBehaviour._fetch_tx_block_number — generator branches
# ---------------------------------------------------------------------------


class TestFetchTxBlockNumber:
    """Test Fetch Tx Block Number."""

    def _make_b(self) -> _DummyPooling:
        ctx = _make_ctx()
        return _DummyPooling(name="b", skill_context=ctx)

    def test_returns_none_when_no_response(self) -> None:
        """Test returns none when no response."""
        b = self._make_b()
        with patch.object(
            b, "get_transaction_receipt", side_effect=_gen_returning(None)
        ):
            result = _run_gen(b._fetch_tx_block_number("0xhash"))
        assert result is None

    def test_returns_none_when_block_number_missing(self) -> None:
        """Test returns none when block number missing."""
        b = self._make_b()
        response = {"status": "1"}  # No blockNumber key
        with patch.object(
            b, "get_transaction_receipt", side_effect=_gen_returning(response)
        ):
            result = _run_gen(b._fetch_tx_block_number("0xhash"))
        assert result is None

    def test_returns_none_when_block_number_invalid(self) -> None:
        """Test returns none when block number invalid."""
        b = self._make_b()
        response = {"blockNumber": "not-an-int"}
        with patch.object(
            b, "get_transaction_receipt", side_effect=_gen_returning(response)
        ):
            result = _run_gen(b._fetch_tx_block_number("0xhash"))
        assert result is None

    def test_returns_int_when_valid_block_number(self) -> None:
        """Test returns int when valid block number."""
        b = self._make_b()
        response = {"blockNumber": 12345}
        with patch.object(
            b, "get_transaction_receipt", side_effect=_gen_returning(response)
        ):
            result = _run_gen(b._fetch_tx_block_number("0xhash"))
        assert result == 12345

    def test_returns_int_when_block_number_is_string_int(self) -> None:
        """Test returns int when block number is string int."""
        b = self._make_b()
        response = {"blockNumber": "9999"}
        with patch.object(
            b, "get_transaction_receipt", side_effect=_gen_returning(response)
        ):
            result = _run_gen(b._fetch_tx_block_number("0xhash"))
        assert result == 9999


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_num_requests_delivered and _get_num_reqs_by_agent
# ---------------------------------------------------------------------------


class TestGetNumReqsByAgent:
    """Test Get Num Reqs By Agent."""

    def _make_b(self) -> _DummyFunds:
        ctx = _make_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_none_when_delivery_report_is_none(self) -> None:
        """Test returns none when delivery report is none."""
        b = self._make_b()
        with patch.object(b, "get_delivery_report", side_effect=_gen_returning(None)):
            result = _run_gen(b._get_num_reqs_by_agent())
        assert result is None

    def test_aggregates_tool_counts_per_agent(self) -> None:
        """Test aggregates tool counts per agent."""
        b = self._make_b()
        report = {
            "agent-0": {"tool-a": 3, "tool-b": 2},
            "agent-1": {"tool-a": 1},
        }
        with patch.object(b, "get_delivery_report", side_effect=_gen_returning(report)):
            result = _run_gen(b._get_num_reqs_by_agent())
        assert result == {"agent-0": 5, "agent-1": 1}

    def test_empty_delivery_report(self) -> None:
        """Test empty delivery report."""
        b = self._make_b()
        with patch.object(b, "get_delivery_report", side_effect=_gen_returning({})):
            result = _run_gen(b._get_num_reqs_by_agent())
        assert result == {}


class TestGetNumRequestsDelivered:
    """Test Get Num Requests Delivered."""

    def _make_b(self) -> _DummyFunds:
        ctx = _make_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_none_when_reqs_by_agent_is_none(self) -> None:
        """Test returns none when reqs by agent is none."""
        b = self._make_b()
        with patch.object(
            b, "_get_num_reqs_by_agent", side_effect=_gen_returning(None)
        ):
            result = _run_gen(b._get_num_requests_delivered())
        assert result is None

    def test_returns_sum_of_all_agents(self) -> None:
        """Test returns sum of all agents."""
        b = self._make_b()
        reqs_by_agent = {"agent-0": 5, "agent-1": 3}
        with patch.object(
            b, "_get_num_reqs_by_agent", side_effect=_gen_returning(reqs_by_agent)
        ):
            result = _run_gen(b._get_num_requests_delivered())
        assert result == 8

    def test_returns_zero_for_empty_agents(self) -> None:
        """Test returns zero for empty agents."""
        b = self._make_b()
        with patch.object(b, "_get_num_reqs_by_agent", side_effect=_gen_returning({})):
            result = _run_gen(b._get_num_requests_delivered())
        assert result == 0


# ---------------------------------------------------------------------------
# TaskPoolingBehaviour.get_payload_content
# ---------------------------------------------------------------------------


class TestGetPayloadContent:
    """Test Get Payload Content."""

    def test_returns_json_of_done_tasks(self) -> None:
        """Test returns json of done tasks."""
        ctx = _make_ctx()
        b = _DummyPooling(name="b", skill_context=ctx)
        tasks = [{"request_id": "r1", "tool": "t1"}]
        with patch.object(b, "get_done_tasks", side_effect=_gen_returning(tasks)):
            result = _run_gen(b.get_payload_content())
        import json as _json

        assert _json.loads(result) == tasks


# ---------------------------------------------------------------------------
# TaskPoolingBehaviour.handle_submitted_tasks
# ---------------------------------------------------------------------------


class TestHandleSubmittedTasks:
    """Test Handle Submitted Tasks."""

    def _make_b(self) -> "_DummyPooling":
        ctx = _make_full_ctx()
        ctx.shared_state["mech_delivery_last_block_number"] = MagicMock()
        b = _DummyPooling(name="b", skill_context=ctx)
        return b

    def _patch_sd(self, b: Any, submitted_ids: List[str]) -> Any:
        """Wire the mocked synchronized_data with a fixed id list.

        :param b: the behaviour under test to attach the mock onto.
        :param submitted_ids: value the property returns.
        :return: a no-op context manager so the caller's ``with`` block stays uniform.
        """
        mock_sd = MagicMock()
        mock_sd.submitted_request_ids = submitted_ids
        b._synchronized_data = mock_sd
        return contextlib.nullcontext()

    def test_status_false_removes_nothing(self) -> None:
        """A failed prior-tx status leaves ``shared_state[DONE_TASKS]`` untouched."""
        b = self._make_b()
        seeded = [{"request_id": "r1", "tool": "t1"}]
        b.context.shared_state[DONE_TASKS] = list(seeded)
        with patch.object(b, "check_last_tx_status", return_value=(False, "")):
            result = _run_gen(b.handle_submitted_tasks())
        assert result is None
        assert b.context.shared_state[DONE_TASKS] == seeded

    def test_status_true_empty_tasks(self) -> None:
        """Test status true empty tasks."""
        b = self._make_b()
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            self._patch_sd(b, submitted_ids=[]),
        ):
            result = _run_gen(b.handle_submitted_tasks())
        assert result is None

    def test_status_true_with_tasks_no_block_number(self) -> None:
        """The task is pruned from shared_state and its histogram fires."""
        task = {"request_id": "r1", "tool": "t1", "start_time": time.perf_counter()}
        b = self._make_b()
        b.context.shared_state[DONE_TASKS] = [task]
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            self._patch_sd(b, submitted_ids=["r1"]),
            patch.object(b, "_fetch_tx_block_number", side_effect=_gen_returning(None)),
            patch.object(b, "observe_histogram") as mock_hist,
        ):
            result = _run_gen(b.handle_submitted_tasks())
        assert result is None
        # End-to-end contract: the submitted id is gone from shared_state.
        assert b.context.shared_state[DONE_TASKS] == []
        # Histogram MUST fire for the task in shared_state; a regression
        # that flips the ``if task is None`` guard would silently skip
        # the emit without this assertion.
        mock_hist.assert_called_once()
        _, kwargs = mock_hist.call_args
        assert kwargs["tool"] == "t1"

    def test_status_true_with_tasks_and_block_number(self) -> None:
        """Same as above but with a block number → gauge fires too."""
        task = {"request_id": "r1", "tool": "t1", "start_time": time.perf_counter()}
        b = self._make_b()
        b.context.shared_state[DONE_TASKS] = [task]
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            self._patch_sd(b, submitted_ids=["r1"]),
            patch.object(
                b, "_fetch_tx_block_number", side_effect=_gen_returning(12345)
            ),
            patch.object(b, "observe_histogram") as mock_hist,
            patch.object(b, "set_gauge") as mock_gauge,
        ):
            result = _run_gen(b.handle_submitted_tasks())
        assert result is None
        assert b.context.shared_state[DONE_TASKS] == []
        mock_hist.assert_called_once()
        _, kwargs = mock_hist.call_args
        assert kwargs["tool"] == "t1"
        mock_gauge.assert_called_once()

    def test_reads_submitted_request_ids_when_present(self) -> None:
        """Primary path: the id list drives the shared_state prune."""
        task = {"request_id": "r1", "tool": "t1", "start_time": time.perf_counter()}
        other = {"request_id": "r2", "tool": "t2", "start_time": time.perf_counter()}
        b = self._make_b()
        b.context.shared_state[DONE_TASKS] = [task, other]
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            self._patch_sd(b, submitted_ids=["r1"]),
            patch.object(b, "_fetch_tx_block_number", side_effect=_gen_returning(None)),
            patch.object(b, "observe_histogram"),
        ):
            _run_gen(b.handle_submitted_tasks())
        # r1 pruned, r2 kept.
        assert b.context.shared_state[DONE_TASKS] == [other]

    def test_empty_submitted_ids_is_noop(self) -> None:
        """An empty id list → early return, shared_state untouched."""
        b = self._make_b()
        seeded = [{"request_id": "r1", "tool": "t1"}]
        b.context.shared_state[DONE_TASKS] = list(seeded)
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            self._patch_sd(b, submitted_ids=[]),
        ):
            _run_gen(b.handle_submitted_tasks())
        assert b.context.shared_state[DONE_TASKS] == seeded

    def test_submitted_ids_pruned_from_predict_api_event_cache(self) -> None:
        """Post-settlement path drops submitted ids from ``PREDICT_API_EVENTS``.

        Anchors the prune to ``handle_submitted_tasks`` (the only
        caller that has proof ``PostTxSettlementBehaviour`` already
        POSTed). Guards against a regression that removes the pop
        loop and silently leaks one 30-60 KB event per delivered
        task until the 24h TTL fires.
        """
        now = time.time()
        task = {"request_id": "r1", "tool": "t1", "start_time": time.perf_counter()}
        b = self._make_b()
        b.context.shared_state[DONE_TASKS] = [task]
        b.context.shared_state[PREDICT_API_EVENTS] = {
            "r1": {"event": {"src": "off"}, "written_at": now},
            "r2": {"event": {"src": "off"}, "written_at": now},
        }
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            self._patch_sd(b, submitted_ids=["r1"]),
            patch.object(b, "_fetch_tx_block_number", side_effect=_gen_returning(None)),
            patch.object(b, "observe_histogram"),
        ):
            _run_gen(b.handle_submitted_tasks())
        surviving = b.context.shared_state[PREDICT_API_EVENTS]
        assert "r1" not in surviving
        assert "r2" in surviving

    def test_histogram_skipped_when_task_absent_from_shared_state(self) -> None:
        """Skip the histogram when the task is absent from shared state.

        The prune loop still runs (the delivery landed on-chain);
        only the timing metric is dropped for that entry. Guards the
        ``if task is None`` continue branch so a regression that
        flips the guard would surface here instead of silently
        emitting garbage timings.
        """
        b = self._make_b()
        b.context.shared_state[DONE_TASKS] = []  # id present in ids, absent here
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            self._patch_sd(b, submitted_ids=["r1"]),
            patch.object(b, "_fetch_tx_block_number", side_effect=_gen_returning(None)),
            patch.object(b, "observe_histogram") as mock_hist,
        ):
            _run_gen(b.handle_submitted_tasks())
        mock_hist.assert_not_called()
        # Prune still ran (no-op here since shared_state was empty).
        assert b.context.shared_state[DONE_TASKS] == []

    @pytest.mark.parametrize(
        "task",
        [
            pytest.param(
                {"request_id": "r1", "start_time": 0.0},
                id="tool-missing",
            ),
            pytest.param(
                {"request_id": "r1", "tool": "t1"},
                id="start_time-missing",
            ),
        ],
    )
    def test_histogram_skipped_when_task_missing_tool_or_start_time(
        self, task: Dict[str, Any]
    ) -> None:
        """Skip the histogram when ``tool`` or ``start_time`` is absent.

        Task-execution failures produce entries without ``tool``, and
        early-abort paths can produce entries without ``start_time``.
        Emitting a histogram sample against either as ``None`` would
        push a nonsense label / negative duration into the metric.
        Both halves of the guard are exercised so a mutation that
        weakens either half surfaces here.

        :param task: parametrised done-task shape (missing ``tool``
            in one case, missing ``start_time`` in the other).
        """
        b = self._make_b()
        b.context.shared_state[DONE_TASKS] = [task]
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            self._patch_sd(b, submitted_ids=["r1"]),
            patch.object(b, "_fetch_tx_block_number", side_effect=_gen_returning(None)),
            patch.object(b, "observe_histogram") as mock_hist,
        ):
            _run_gen(b.handle_submitted_tasks())
        mock_hist.assert_not_called()
        # Prune still ran end-to-end despite the metric skip.
        assert b.context.shared_state[DONE_TASKS] == []

    def test_prune_acquires_done_tasks_lock(self) -> None:
        """The live prune path MUST hold ``done_tasks_lock``.

        Without this, dropping the ``with`` around the prune would
        pass every other test in the suite silently — the background
        executor's concurrent append could then torn-read the metrics
        loop's snapshot. Directly assert the lock is entered.
        """
        b = self._make_b()
        b.context.shared_state[DONE_TASKS] = [
            {"request_id": "r1", "tool": "t1", "start_time": time.perf_counter()},
        ]
        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(return_value=None)
        mock_lock.__exit__ = MagicMock(return_value=None)
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            self._patch_sd(b, submitted_ids=["r1"]),
            patch.object(b, "_fetch_tx_block_number", side_effect=_gen_returning(None)),
            patch.object(b, "observe_histogram"),
            patch.object(b, "done_tasks_lock", return_value=mock_lock),
        ):
            _run_gen(b.handle_submitted_tasks())
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()

    def test_str_normalisation_matches_legacy_int_on_live_path(self) -> None:
        """Live prune path str-normalises the shared_state side of the match.

        The reviewer flagged that ``0518c00e`` moved the prune inline
        and the only test guarding the ``str()`` normalisation lived
        on ``remove_tasks_by_id`` (which the live path used to
        bypass). After the consolidation the live path calls
        ``remove_tasks_by_id`` again, but pin it end-to-end here so a
        future re-inlining that drops the normalisation surfaces.
        """
        b = self._make_b()
        # Shared state carries the legacy ``int`` shape; the incoming
        # id list is ``List[str]`` per the type contract.
        b.context.shared_state[DONE_TASKS] = [
            {"request_id": 5, "tool": "t1", "start_time": time.perf_counter()},
        ]
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            self._patch_sd(b, submitted_ids=["5"]),
            patch.object(b, "_fetch_tx_block_number", side_effect=_gen_returning(None)),
            patch.object(b, "observe_histogram"),
        ):
            _run_gen(b.handle_submitted_tasks())
        assert b.context.shared_state[DONE_TASKS] == []

    def test_prune_runs_before_block_number_fetch(self) -> None:
        """The prune completes before the ``_fetch_tx_block_number`` yield.

        A cancellation at the yield must not leave the delivered
        batch un-pruned. Sending ``GeneratorExit`` at the first yield
        would mid-execute the metrics block and prune only if either
        happens after the yield. Pin the invariant by asserting that
        with the yield replaced by a raise, ``shared_state`` is
        already pruned.
        """
        b = self._make_b()
        b.context.shared_state[DONE_TASKS] = [
            {"request_id": "r1", "tool": "t1", "start_time": time.perf_counter()},
        ]

        def _raising_yield(_tx_hash: str) -> Generator[None, None, None]:
            if False:
                yield  # pragma: no cover - satisfy typing
            raise RuntimeError("simulated yield failure")

        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            self._patch_sd(b, submitted_ids=["r1"]),
            patch.object(b, "_fetch_tx_block_number", side_effect=_raising_yield),
            patch.object(b, "observe_histogram"),
        ):
            with pytest.raises(RuntimeError, match="simulated yield failure"):
                _run_gen(b.handle_submitted_tasks())
        # If the prune had been moved below the yield, this would
        # still be ``[{'request_id': 'r1', ...}]``.
        assert b.context.shared_state[DONE_TASKS] == []


# ---------------------------------------------------------------------------
# End-to-end hand-off contract (round-side write ↔ behaviour-side read)
# ---------------------------------------------------------------------------


class TestSubmittedRequestIdsRoundTrip:
    """Pins the write→read hand-off using real ``SynchronizedData``.

    The other behaviour-side tests replace ``synchronized_data`` with
    a ``MagicMock``; that leaves the real property (including its
    ``list[str]`` validation) unexercised on the behaviour path, so a
    rename or deletion of the property would pass silently. This
    class exercises the actual round writer, feeds the resulting
    synced data to the behaviour, and asserts the prune reaches
    ``shared_state[DONE_TASKS]``.
    """

    def _make_sync_data_with_ids(self, ids: List[str]) -> Any:
        """Build a real ``SynchronizedData`` carrying ``submitted_request_ids``."""
        from packages.valory.skills.abstract_round_abci.base import (
            AbciAppDB,
            get_name,
        )
        from packages.valory.skills.task_submission_abci.rounds import (
            SynchronizedData,
        )

        data: dict = {
            "participants": [["agent-0", "agent-1", "agent-2"]],
            "consensus_threshold": [3],
            "all_participants": [["agent-0", "agent-1", "agent-2"]],
            get_name(SynchronizedData.final_tx_hash): ["0xhash"],
            get_name(SynchronizedData.submitted_request_ids): [ids],
        }
        return SynchronizedData(AbciAppDB(data))

    def test_end_to_end_prune_using_real_synchronized_data(self) -> None:
        """A real ``SynchronizedData`` drives the prune to shared_state.

        Round writer → behaviour reader → shared_state prune. If the
        property name or ``list[str]`` validation changes, this test
        catches the drift.
        """
        ctx = _make_full_ctx()
        ctx.shared_state["mech_delivery_last_block_number"] = MagicMock()
        b = _DummyPooling(name="b", skill_context=ctx)
        b._synchronized_data = self._make_sync_data_with_ids(["req-a", "req-b"])
        b.context.shared_state[DONE_TASKS] = [
            {"request_id": "req-a", "tool": "t1", "start_time": time.perf_counter()},
            {"request_id": "req-b", "tool": "t2", "start_time": time.perf_counter()},
            {"request_id": "req-keep", "tool": "t3"},
        ]
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            patch.object(b, "_fetch_tx_block_number", side_effect=_gen_returning(None)),
            patch.object(b, "observe_histogram"),
        ):
            _run_gen(b.handle_submitted_tasks())
        remaining_ids = [
            task["request_id"] for task in b.context.shared_state[DONE_TASKS]
        ]
        assert remaining_ids == ["req-keep"]

    def test_skipped_task_survives_settlement_end_to_end(self) -> None:
        """Round writer → behaviour reader: a sim-skipped task is NOT pruned.

        ``PostTxSettlementRound`` hands off only ``tx_included_request_ids``.
        A task that was pooled but left out of the multisend must still be
        in ``shared_state[DONE_TASKS]`` afterwards so it is re-pooled.
        """
        from packages.valory.skills.abstract_round_abci.base import (
            AbciAppDB,
            get_name,
        )
        from packages.valory.skills.task_submission_abci.payloads import (
            PostTxSettlementPayload,
        )
        from packages.valory.skills.task_submission_abci.rounds import (
            PostTxSettlementRound,
            SynchronizedData,
        )

        participants = ["agent-0", "agent-1", "agent-2"]
        done = [{"request_id": "req-a"}, {"request_id": "req-skipped"}]
        db = AbciAppDB(
            {
                "participants": [participants],
                "consensus_threshold": [3],
                "all_participants": [participants],
                get_name(SynchronizedData.final_tx_hash): ["0xhash"],
                get_name(SynchronizedData.done_tasks): [done],
                get_name(SynchronizedData.tx_included_request_ids): [["req-a"]],
            }
        )
        round_ = PostTxSettlementRound(
            synchronized_data=SynchronizedData(db), context=MagicMock()
        )
        round_.collection = {
            p: PostTxSettlementPayload(sender=p, content="done") for p in participants
        }
        result = round_.end_block()
        assert result is not None
        new_sd, _ = result

        ctx = _make_full_ctx()
        ctx.shared_state["mech_delivery_last_block_number"] = MagicMock()
        b = _DummyPooling(name="b", skill_context=ctx)
        b._synchronized_data = new_sd
        b.context.shared_state[DONE_TASKS] = [
            {"request_id": "req-a", "tool": "t1", "start_time": time.perf_counter()},
            {
                "request_id": "req-skipped",
                "tool": "t2",
                "start_time": time.perf_counter(),
            },
        ]
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            patch.object(b, "_fetch_tx_block_number", side_effect=_gen_returning(None)),
            patch.object(b, "observe_histogram"),
        ):
            _run_gen(b.handle_submitted_tasks())
        remaining_ids = [
            task["request_id"] for task in b.context.shared_state[DONE_TASKS]
        ]
        assert remaining_ids == ["req-skipped"]


class TestHandleSubmittedTasksMetrics:
    """``handle_submitted_tasks`` labels settlement metrics by path."""

    def _make_b(self, tasks: List[Dict[str, Any]]) -> "_DummyPooling":
        ctx = _make_full_ctx()
        ctx.shared_state["mech_delivery_last_block_number"] = MagicMock()
        b = _DummyPooling(name="b", skill_context=ctx)
        b.context.shared_state[DONE_TASKS] = tasks
        mock_sd = MagicMock()
        mock_sd.submitted_request_ids = [str(t["request_id"]) for t in tasks]
        b._synchronized_data = mock_sd
        return b

    @pytest.mark.parametrize(
        "is_offchain, expected_source",
        [(True, SOURCE_OFFCHAIN), (False, SOURCE_ONCHAIN), (None, SOURCE_ONCHAIN)],
        ids=["offchain", "onchain", "flag-missing"],
    )
    def test_delivery_time_carries_source(
        self, is_offchain: Optional[bool], expected_source: str
    ) -> None:
        """The delivery-time histogram splits by ``source``.

        :param is_offchain: the task's ``is_offchain`` flag (``None`` = absent).
        :param expected_source: the label value that must be emitted.
        """
        task: Dict[str, Any] = {
            "request_id": "r1",
            "tool": "t1",
            "start_time": time.perf_counter(),
            MECH_ADDRESS: "0xMECH",
        }
        if is_offchain is not None:
            task[IS_OFFCHAIN] = is_offchain
        b = self._make_b([task])
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            patch.object(b, "_fetch_tx_block_number", side_effect=_gen_returning(None)),
            patch.object(b, "observe_histogram") as mock_hist,
        ):
            _run_gen(b.handle_submitted_tasks())
        _, kwargs = mock_hist.call_args
        assert kwargs["source"] == expected_source


class TestCountSettledFromSyncedData:
    """``settled`` is counted from synced data so it survives an executor restart."""

    _ME = "0xSELF"

    def _make_self(
        self,
        done_tasks: List[Dict[str, Any]],
        included: List[str],
        tx_hash: str = "0xTX",
    ) -> Any:
        self_ = SimpleNamespace(
            synchronized_data=SimpleNamespace(
                done_tasks=done_tasks,
                tx_included_request_ids=included,
                final_tx_hash=tx_hash,
            ),
            context=SimpleNamespace(agent_address=self._ME, shared_state={}),
            count_settlement=MagicMock(),
            metrics_mech_label=lambda: "0xLABEL",
        )
        return self_

    @staticmethod
    def _task(rid: str, executor: str, **extra: Any) -> Dict[str, Any]:
        return {"request_id": rid, "task_executor_address": executor, **extra}

    def test_counts_own_included_tasks_with_source_and_mech(self) -> None:
        """Only this agent's tasks that were in the tx are counted, once each."""
        done = [
            self._task("r-off", self._ME, is_offchain=True, mech_address="0xM1"),
            self._task("r-on", self._ME),
            self._task("r-other", "0xOTHER", is_offchain=True),
            self._task("r-skipped", self._ME, is_offchain=True),
        ]
        self_ = self._make_self(done, included=["r-off", "r-on", "r-other"])
        beh_mod.PostTxSettlementBehaviour.count_settled_from_synced_data(self_)
        assert self_.count_settlement.call_args_list == [
            call(SETTLEMENT_OUTCOME_SETTLED, SOURCE_OFFCHAIN, "0xM1"),
            call(SETTLEMENT_OUTCOME_SETTLED, SOURCE_ONCHAIN, "0xLABEL"),
        ]

    def test_nothing_included_counts_nothing(self) -> None:
        """Delivery-rate settlement and all-skipped periods count no settled tasks."""
        self_ = self._make_self([self._task("r1", self._ME)], included=[])
        beh_mod.PostTxSettlementBehaviour.count_settled_from_synced_data(self_)
        self_.count_settlement.assert_not_called()

    def test_int_request_id_matches_str_included_id(self) -> None:
        """On-chain ids are ``int`` on the task and ``str`` in the envelope."""
        self_ = self._make_self([self._task(cast(str, 42), self._ME)], included=["42"])
        beh_mod.PostTxSettlementBehaviour.count_settled_from_synced_data(self_)
        self_.count_settlement.assert_called_once()

    def test_missing_executor_address_is_not_counted(self) -> None:
        """A task without an executor stamp is nobody's to count."""
        self_ = self._make_self([{"request_id": "r1"}], included=["r1"])
        beh_mod.PostTxSettlementBehaviour.count_settled_from_synced_data(self_)
        self_.count_settlement.assert_not_called()

    def test_round_re_entry_counts_the_same_tx_once(self) -> None:
        """The post-settlement round self-loops; the same tx hash is counted once."""
        self_ = self._make_self([self._task("r1", self._ME)], included=["r1"])
        beh_mod.PostTxSettlementBehaviour.count_settled_from_synced_data(self_)
        beh_mod.PostTxSettlementBehaviour.count_settled_from_synced_data(self_)
        self_.count_settlement.assert_called_once()
        assert self_.context.shared_state[SETTLED_COUNTED_TX_KEY] == "0xTX"

    def test_a_new_tx_hash_is_counted_again(self) -> None:
        """The guard is per confirmed tx, not a one-shot latch."""
        self_ = self._make_self([self._task("r1", self._ME)], included=["r1"])
        beh_mod.PostTxSettlementBehaviour.count_settled_from_synced_data(self_)
        self_.synchronized_data.final_tx_hash = "0xTX2"
        beh_mod.PostTxSettlementBehaviour.count_settled_from_synced_data(self_)
        assert self_.count_settlement.call_count == 2


class TestCountSettlement:
    """``count_settlement`` forwards bounded labels to the module counter."""

    def test_increments_with_chain_and_mech_labels(self) -> None:
        """Labels come from params + the caller; ``amount`` is the increment."""
        b = _DummyPooling(name="b", skill_context=_make_full_ctx())
        with patch.object(beh_mod.mech_settlement_total, "labels") as mock_labels:
            b.count_settlement(SETTLEMENT_OUTCOME_SETTLED, SOURCE_OFFCHAIN, "0xM", 3)
        mock_labels.assert_called_once_with(
            outcome=SETTLEMENT_OUTCOME_SETTLED,
            source=SOURCE_OFFCHAIN,
            chain="100",
            mech_address="0xm",
        )
        mock_labels.return_value.inc.assert_called_once_with(3)

    @pytest.mark.parametrize("amount", [0, -1])
    def test_non_positive_amount_is_noop(self, amount: int) -> None:
        """A zero / negative count never touches the counter.

        :param amount: the increment to request.
        """
        b = _DummyPooling(name="b", skill_context=_make_full_ctx())
        with patch.object(beh_mod.mech_settlement_total, "labels") as mock_labels:
            b.count_settlement(
                SETTLEMENT_OUTCOME_SIM_FAILED, SOURCE_ONCHAIN, "0xM", amount
            )
        mock_labels.assert_not_called()


class TestLocalSettlementCounting:
    """Settlement outcomes are attributed only to the agent that executed the task."""

    def _make_b(self, local_ids: List[str]) -> "_DummyTransPrep":
        ctx = _make_full_ctx(done_tasks=[{"request_id": rid} for rid in local_ids])
        return _DummyTransPrep(name="b", skill_context=ctx)

    def test_local_request_ids_filters_to_locally_held_tasks(self) -> None:
        """Only ids present in this agent's DONE_TASKS come back, as ``str``."""
        b = self._make_b(["r1", "r3"])
        assert b.local_request_ids(["r1", "r2", "r3", "r4"]) == {"r1", "r3"}
        assert b.local_request_ids([]) == set()

    def test_count_local_settlement_uses_local_subset_size(self) -> None:
        """The increment is the number of locally held ids, not the batch size."""
        b = self._make_b(["r1"])
        with patch.object(b, "count_settlement") as mock_count:
            b.count_local_settlement(
                SETTLEMENT_OUTCOME_CONTRACT_ERROR, SOURCE_OFFCHAIN, "0xM", ["r1", "r2"]
            )
        mock_count.assert_called_once_with(
            SETTLEMENT_OUTCOME_CONTRACT_ERROR, SOURCE_OFFCHAIN, "0xM", 1
        )

    def test_metrics_mech_label_prefers_marketplace_mech(self) -> None:
        """The label matches the task_execution side: marketplace mech first."""
        ctx = _make_full_ctx(
            mech_to_config={
                "0xlegacy": SimpleNamespace(is_marketplace_mech=False),
                "0xmarket": SimpleNamespace(is_marketplace_mech=True),
            }
        )
        b = _DummyTransPrep(name="b", skill_context=ctx)
        assert b.metrics_mech_label() == "0xmarket"


class TestNoteSettlementSkipped:
    """Per-period retry charge and one-at-a-time drop for tasks whose deliver simulation failed."""

    PERIOD = 5

    @staticmethod
    def _task(
        request_id: str, attempts: Optional[int] = None, nonce: int = 7
    ) -> Dict[str, Any]:
        task: Dict[str, Any] = {
            "request_id": request_id,
            IS_OFFCHAIN: True,
            SENDER: "0xSENDER",
            NONCE: nonce,
            "tool": "t1",
        }
        if attempts is not None:
            task[SETTLEMENT_ATTEMPTS_KEY] = attempts
        return task

    def _make_b(self, local_tasks: List[Dict[str, Any]]) -> "_DummyTransPrep":
        ctx = _make_full_ctx(done_tasks=local_tasks)
        ctx.shared_state[SETTLING_NONCES_BY_SENDER] = {"0xSENDER": {7, 8}}
        return _DummyTransPrep(name="b", skill_context=ctx)

    def _skip(self, b: Any, ids: List[str], period: int = PERIOD) -> MagicMock:
        with patch.object(b, "count_settlement") as mock_count:
            b.note_settlement_skipped(ids, SOURCE_OFFCHAIN, "0xMECH", period)
        return mock_count

    @staticmethod
    def _amounts(mock_count: MagicMock) -> Dict[str, int]:
        return {c.args[0]: c.args[3] for c in mock_count.call_args_list}

    def test_first_skip_stamps_attempt_and_keeps_task(self) -> None:
        """One failure: attempt count becomes 1, task stays, counted as sim_failed."""
        b = self._make_b([self._task("r1")])
        mock_count = self._skip(b, ["r1"])
        local = b.context.shared_state[DONE_TASKS]
        assert [t["request_id"] for t in local] == ["r1"]
        assert local[0][SETTLEMENT_ATTEMPTS_KEY] == 1
        assert local[0][SETTLEMENT_ATTEMPT_PERIOD_KEY] == self.PERIOD
        assert b.context.shared_state[SETTLING_NONCES_BY_SENDER] == {"0xSENDER": {7, 8}}
        assert self._amounts(mock_count) == {
            SETTLEMENT_OUTCOME_SIM_FAILED: 1,
            SETTLEMENT_OUTCOME_DROPPED: 0,
        }

    def test_re_entry_in_the_same_period_charges_nothing(self) -> None:
        """A tx-prep round re-run at the same height must not burn a second attempt.

        The round self-loops on its 60s timeout and the framework rebuilds the
        behaviour, so the whole builder runs again against the same local task.
        """
        b = self._make_b([self._task("r1")])
        self._skip(b, ["r1"])
        second = self._skip(b, ["r1"])
        local = b.context.shared_state[DONE_TASKS]
        assert local[0][SETTLEMENT_ATTEMPTS_KEY] == 1
        assert self._amounts(second) == {
            SETTLEMENT_OUTCOME_SIM_FAILED: 0,
            SETTLEMENT_OUTCOME_DROPPED: 0,
        }

    def test_next_period_charges_a_second_attempt(self) -> None:
        """A new period is a new attempt."""
        b = self._make_b([self._task("r1")])
        self._skip(b, ["r1"], period=1)
        self._skip(b, ["r1"], period=2)
        assert b.context.shared_state[DONE_TASKS][0][SETTLEMENT_ATTEMPTS_KEY] == 2

    def test_cap_burns_only_across_periods_never_within_one(self) -> None:
        """MAX re-entries in one period leave the task queued at attempt 1."""
        b = self._make_b([self._task("r1")])
        for _ in range(MAX_SETTLEMENT_ATTEMPTS + 1):
            self._skip(b, ["r1"])
        local = b.context.shared_state[DONE_TASKS]
        assert [t["request_id"] for t in local] == ["r1"]
        assert local[0][SETTLEMENT_ATTEMPTS_KEY] == 1

    def test_reaching_the_cap_drops_task_and_releases_settling_nonce(self) -> None:
        """The Nth period removes the task locally and frees its nonce slot."""
        b = self._make_b(
            [
                self._task("r1", attempts=MAX_SETTLEMENT_ATTEMPTS - 1),
                self._task("r-other", nonce=8),
            ]
        )
        mock_count = self._skip(b, ["r1"])
        local = b.context.shared_state[DONE_TASKS]
        assert [t["request_id"] for t in local] == ["r-other"]
        # Only the dropped task's nonce (7) is released; 8 belongs to another task.
        assert b.context.shared_state[SETTLING_NONCES_BY_SENDER] == {"0xSENDER": {8}}
        assert self._amounts(mock_count) == {
            SETTLEMENT_OUTCOME_DROPPED: 1,
            SETTLEMENT_OUTCOME_SIM_FAILED: 0,
        }

    def test_only_the_lowest_nonce_at_cap_is_dropped_per_period(self) -> None:
        """A whole group at the cap loses one task per period, lowest nonce first."""
        b = self._make_b(
            [
                self._task("r-late", attempts=MAX_SETTLEMENT_ATTEMPTS - 1, nonce=9),
                self._task("r-early", attempts=MAX_SETTLEMENT_ATTEMPTS - 1, nonce=7),
                self._task("r-young", attempts=0, nonce=8),
            ]
        )
        mock_count = self._skip(b, ["r-late", "r-early", "r-young"])
        local = b.context.shared_state[DONE_TASKS]
        assert [t["request_id"] for t in local] == ["r-late", "r-young"]
        # The survivor at the cap keeps its count and is retried, not reset.
        assert local[0][SETTLEMENT_ATTEMPTS_KEY] == MAX_SETTLEMENT_ATTEMPTS
        assert self._amounts(mock_count) == {
            SETTLEMENT_OUTCOME_DROPPED: 1,
            SETTLEMENT_OUTCOME_SIM_FAILED: 2,
        }

    def test_onchain_task_without_nonce_is_dropped_by_position(self) -> None:
        """On-chain groups have no wire nonce; the first candidate goes."""
        first = {"request_id": 1, SETTLEMENT_ATTEMPTS_KEY: MAX_SETTLEMENT_ATTEMPTS - 1}
        second = {"request_id": 2, SETTLEMENT_ATTEMPTS_KEY: MAX_SETTLEMENT_ATTEMPTS - 1}
        b = self._make_b([first, second])
        with patch.object(b, "count_settlement") as mock_count:
            b.note_settlement_skipped(["1", "2"], SOURCE_ONCHAIN, "0xMECH", self.PERIOD)
        assert [t["request_id"] for t in b.context.shared_state[DONE_TASKS]] == [2]
        assert self._amounts(mock_count) == {
            SETTLEMENT_OUTCOME_DROPPED: 1,
            SETTLEMENT_OUTCOME_SIM_FAILED: 1,
        }
        assert mock_count.call_args_list[0].args[1] == SOURCE_ONCHAIN

    def test_below_cap_does_not_drop(self) -> None:
        """Attempts at cap-2 become cap-1 after this skip: still retried next period."""
        b = self._make_b([self._task("r1", attempts=MAX_SETTLEMENT_ATTEMPTS - 2)])
        self._skip(b, ["r1"])
        local = b.context.shared_state[DONE_TASKS]
        assert len(local) == 1
        assert local[0][SETTLEMENT_ATTEMPTS_KEY] == MAX_SETTLEMENT_ATTEMPTS - 1

    def test_non_owner_agent_changes_nothing_and_counts_nothing(self) -> None:
        """An agent without the task locally leaves shared state alone and counts 0."""
        b = self._make_b([self._task("r-unrelated")])
        mock_count = self._skip(b, ["r1", "r2"])
        local = b.context.shared_state[DONE_TASKS]
        assert [t["request_id"] for t in local] == ["r-unrelated"]
        assert SETTLEMENT_ATTEMPTS_KEY not in local[0]
        assert self._amounts(mock_count) == {
            SETTLEMENT_OUTCOME_DROPPED: 0,
            SETTLEMENT_OUTCOME_SIM_FAILED: 0,
        }

    def test_empty_id_list_is_noop(self) -> None:
        """Nothing skipped → nothing stamped, nothing counted."""
        b = self._make_b([self._task("r1")])
        mock_count = self._skip(b, [])
        assert SETTLEMENT_ATTEMPTS_KEY not in b.context.shared_state[DONE_TASKS][0]
        mock_count.assert_not_called()


class TestOffchainUnsettledGauges:
    """The delivered-but-unsettled backlog gauges published at pooling time."""

    def _make_b(self) -> "_DummyPooling":
        return _DummyPooling(name="b", skill_context=_make_full_ctx())

    def test_counts_only_offchain_and_reports_oldest_age(self) -> None:
        """On-chain tasks are ignored; age is measured from the oldest stamp."""
        b = self._make_b()
        now = time.time()
        done = [
            {"request_id": "on", IS_OFFCHAIN: False, ENQUEUED_AT_LOCAL: now - 999},
            {"request_id": "a", IS_OFFCHAIN: True, ENQUEUED_AT_LOCAL: now - 120},
            {"request_id": "b", IS_OFFCHAIN: True, ENQUEUED_AT_LOCAL: now - 30},
        ]
        with patch.object(b, "set_gauge") as mock_gauge:
            b.update_offchain_unsettled_gauges(done)
        calls = {c.args[0]: (c.args[1], c.kwargs) for c in mock_gauge.call_args_list}
        count_value, count_labels = calls[beh_mod.mech_offchain_unsettled_delivered]
        age_value, _ = calls[beh_mod.mech_offchain_unsettled_oldest_age_seconds]
        assert count_value == 2
        assert count_labels == {"chain": "100", "mech_address": "0xmech"}
        assert 119 <= age_value <= 121

    def test_empty_backlog_publishes_zeros(self) -> None:
        """No off-chain tasks → both gauges 0 (not stale, not skipped)."""
        b = self._make_b()
        with patch.object(b, "set_gauge") as mock_gauge:
            b.update_offchain_unsettled_gauges([{"request_id": "on"}])
        values = {c.args[0]: c.args[1] for c in mock_gauge.call_args_list}
        assert values[beh_mod.mech_offchain_unsettled_delivered] == 0
        assert values[beh_mod.mech_offchain_unsettled_oldest_age_seconds] == 0

    def test_missing_stamp_does_not_break_age(self) -> None:
        """A task without a receive stamp still counts but is ignored for age."""
        b = self._make_b()
        done = [
            {"request_id": "a", IS_OFFCHAIN: True},
            {"request_id": "b", IS_OFFCHAIN: True, ENQUEUED_AT_LOCAL: "bad"},
        ]
        with patch.object(b, "set_gauge") as mock_gauge:
            b.update_offchain_unsettled_gauges(done)
        values = {c.args[0]: c.args[1] for c in mock_gauge.call_args_list}
        assert values[beh_mod.mech_offchain_unsettled_delivered] == 2
        assert values[beh_mod.mech_offchain_unsettled_oldest_age_seconds] == 0

    def test_get_payload_content_publishes_gauges(self) -> None:
        """The pooling payload path is what refreshes the gauges each period."""
        b = self._make_b()
        done = [{"request_id": "a", IS_OFFCHAIN: True}]
        with (
            patch.object(b, "get_done_tasks", side_effect=_gen_returning(done)),
            patch.object(b, "update_offchain_unsettled_gauges") as mock_update,
        ):
            payload = _run_gen(b.get_payload_content())
        assert json.loads(payload) == done
        mock_update.assert_called_once_with(done)


# ---------------------------------------------------------------------------
# DeliverBehaviour._get_current_delivery_report
# ---------------------------------------------------------------------------


class TestGetCurrentDeliveryReport:
    """Test Get Current Delivery Report."""

    def _make_b(self) -> "_DummyDeliver":
        ctx = _make_full_ctx()
        b = _DummyDeliver(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        b._synchronized_data = mock_sd
        return b

    def _patch_sd(self, b: Any) -> Any:
        return contextlib.nullcontext()

    def test_returns_none_on_contract_error(self) -> None:
        """Test returns none on contract error."""
        b = self._make_b()
        with (
            self._patch_sd(b),
            patch.object(
                b,
                "get_contract_api_response",
                side_effect=_gen_returning(_error_contract_msg()),
            ),
        ):
            result = _run_gen(b._get_current_delivery_report())
        assert result is None

    def test_returns_empty_dict_for_zero_ipfs_hash(self) -> None:
        """Test returns empty dict for zero ipfs hash."""
        b = self._make_b()
        msg = _state_contract_msg({"data": ZERO_IPFS_HASH})
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(msg)
            ),
        ):
            result = _run_gen(b._get_current_delivery_report())
        assert result == {}

    def test_returns_none_when_ipfs_fetch_fails(self) -> None:
        """Test returns none when ipfs fetch fails."""
        b = self._make_b()
        # Use a real CIDv1 hash string that CID.from_string can parse
        valid_cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
        msg = _state_contract_msg({"data": valid_cid})
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(msg)
            ),
            patch.object(b, "get_from_ipfs", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b._get_current_delivery_report())
        assert result is None

    def test_returns_usage_data_on_success(self) -> None:
        """Test returns usage data on success."""
        b = self._make_b()
        valid_cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
        msg = _state_contract_msg({"data": valid_cid})
        usage_data = {"agent-0": {"tool-a": 3}}
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(msg)
            ),
            patch.object(b, "get_from_ipfs", side_effect=_gen_returning(usage_data)),
        ):
            result = _run_gen(b._get_current_delivery_report())
        assert result == usage_data


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_balance
# ---------------------------------------------------------------------------


class TestGetBalance:
    """Test Get Balance."""

    def _make_b(self) -> "_DummyFunds":
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_none_on_ledger_error(self) -> None:
        """Test returns none on ledger error."""
        b = self._make_b()
        with patch.object(
            b,
            "get_ledger_api_response",
            side_effect=_gen_returning(_error_ledger_msg()),
        ):
            result = _run_gen(b._get_balance("0xAGENT"))
        assert result is None

    def test_returns_balance_on_success(self) -> None:
        """Test returns balance on success."""
        b = self._make_b()
        msg = _state_ledger_msg({"get_balance_result": 500})
        with patch.object(
            b, "get_ledger_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(b._get_balance("0xAGENT"))
        assert result == 500


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_mech_payment_type
# ---------------------------------------------------------------------------


class TestGetMechPaymentType:
    """Test Get Mech Payment Type."""

    def test_shortcut_when_payment_model_set(self) -> None:
        """Test shortcut when payment model set."""
        # When payment_model is set and address matches agent_mech_contract_address
        payment_type = b"\x00" * 32
        ctx = _make_full_ctx(payment_model=payment_type)
        ctx.params.agent_mech_contract_address = "0xPRIMARY"
        ctx.params.agent_mech_contract_addresses = ["0xPRIMARY"]
        b = _DummyFunds(name="b", skill_context=ctx)
        # _get_mech_payment_type is a generator function; when shortcut path taken, no yields
        result = _run_gen(b._get_mech_payment_type("0xPRIMARY"))
        assert result == payment_type

    def test_fetches_via_contract_api_on_non_primary(self) -> None:
        """Test fetches via contract api on non primary."""
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        mech_type_bytes = b"\xab" * 32
        msg = _state_contract_msg({"mech_type": mech_type_bytes})
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(b._get_mech_payment_type("0xOTHER"))
        assert result == mech_type_bytes


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_balance_tracker_address
# ---------------------------------------------------------------------------


class TestGetBalanceTrackerAddress:
    """Test Get Balance Tracker Address."""

    def test_returns_address_on_success(self) -> None:
        """Test returns address on success."""
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        msg = _state_contract_msg({"data": "0xTRACKER"})
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(b._get_balance_tracker_address(b"\x00" * 32))
        assert result == "0xTRACKER"


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._adjust_mech_balance
# ---------------------------------------------------------------------------


class TestAdjustMechBalance:
    """Test Adjust Mech Balance."""

    def test_adjusts_balance_by_token_credit_ratio(self) -> None:
        """Test adjusts balance by token credit ratio."""
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        ratio = 2 * (10**18)  # ratio * mech_balance // 1e18 = 2 * mech_balance
        msg = _state_contract_msg({"token_credit_ratio": ratio})
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(b._adjust_mech_balance("0xTRACKER", 100))
        assert result == 200


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_mech_info
# ---------------------------------------------------------------------------


class TestGetMechInfo:
    """Test Get Mech Info."""

    def _make_b(self) -> "_DummyFunds":
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_none_when_mech_type_none(self) -> None:
        """Test returns none when mech type none."""
        b = self._make_b()
        with patch.object(
            b, "_get_mech_payment_type", side_effect=_gen_returning(None)
        ):
            result = _run_gen(b._get_mech_info("0xMECH"))
        assert result is None

    def test_returns_none_when_tracker_address_none(self) -> None:
        """Test returns none when tracker address none."""
        b = self._make_b()
        with (
            patch.object(
                b, "_get_mech_payment_type", side_effect=_gen_returning(b"\x00" * 32)
            ),
            patch.object(
                b, "_get_balance_tracker_address", side_effect=_gen_returning(None)
            ),
        ):
            result = _run_gen(b._get_mech_info("0xMECH"))
        assert result is None

    def test_returns_tuple_on_success(self) -> None:
        """Test returns tuple on success."""
        b = self._make_b()
        mech_type = b"\xaa" * 32
        msg = _state_contract_msg({"mech_balance": 999})
        with (
            patch.object(
                b, "_get_mech_payment_type", side_effect=_gen_returning(mech_type)
            ),
            patch.object(
                b, "_get_balance_tracker_address", side_effect=_gen_returning("0xTRACK")
            ),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(msg)
            ),
        ):
            result = _run_gen(b._get_mech_info("0xMECH"))
        assert result == (mech_type, "0xTRACK", 999)


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_process_payment_tx
# ---------------------------------------------------------------------------


class TestGetProcessPaymentTx:
    """Test Get Process Payment Tx."""

    def test_returns_dict_on_success(self) -> None:
        """Test returns dict on success."""
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        b._synchronized_data = mock_sd
        tx_data = b"\xde\xad"
        msg = _state_contract_msg({"data": tx_data, "simulation_ok": True})
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(b._get_process_payment_tx("0xMECH", "0xTRACK"))
        assert result["simulation_ok"] is True
        assert result["data"] == tx_data


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_fee and _get_max_fee_factor
# ---------------------------------------------------------------------------


class TestGetFeeAndMaxFeeFactor:
    """Test Get Fee And Max Fee Factor."""

    def test_get_fee_returns_int(self) -> None:
        """Test get fee returns int."""
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        msg = _state_contract_msg({"data": 250})
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(b._get_fee())
        assert result == 250

    def test_get_max_fee_factor_returns_int(self) -> None:
        """Test get max fee factor returns int."""
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        msg = _state_contract_msg({"max_fee_factor": 10000})
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(b._get_max_fee_factor("0xTRACK"))
        assert result == 10000


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._calculate_mech_profits
# ---------------------------------------------------------------------------


class TestCalculateMechProfits:
    """Test Calculate Mech Profits."""

    def _make_b(self) -> "_DummyFunds":
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_none_when_fee_is_none(self) -> None:
        """Test returns none when fee is none."""
        b = self._make_b()
        with patch.object(b, "_get_fee", side_effect=_gen_returning(None)):
            result = _run_gen(b._calculate_mech_profits("0xTRACK", 1000))
        assert result is None

    def test_returns_none_when_max_fee_factor_is_none(self) -> None:
        """Test returns none when max fee factor is none."""
        b = self._make_b()
        with (
            patch.object(b, "_get_fee", side_effect=_gen_returning(100)),
            patch.object(b, "_get_max_fee_factor", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b._calculate_mech_profits("0xTRACK", 1000))
        assert result is None

    def test_calculates_profits_on_success(self) -> None:
        """Test calculates profits on success."""
        b = self._make_b()
        # fee=100, MAX_FEE_FACTOR=10000, mech_balance=1000
        # marketplace_fee = (1000 * 100 + (10000-1)) // 10000 = (100000 + 9999) // 10000 = 109999 // 10000 = 10
        # profits = 1000 - 10 = 990
        with (
            patch.object(b, "_get_fee", side_effect=_gen_returning(100)),
            patch.object(b, "_get_max_fee_factor", side_effect=_gen_returning(10000)),
        ):
            result = _run_gen(b._calculate_mech_profits("0xTRACK", 1000))
        assert result == 990


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._split_funds branches
# ---------------------------------------------------------------------------


class TestSplitFunds:
    """Test Split Funds."""

    def _make_b(self) -> "_DummyFunds":
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        b.context.params.service_owner_share = 1000  # 10%
        return b

    def test_returns_none_when_service_owner_none(self) -> None:
        """Test returns none when service owner none."""
        b = self._make_b()
        with patch.object(b, "_get_service_owner", side_effect=_gen_returning(None)):
            result = _run_gen(b._split_funds(1000))
        assert result is None

    def test_returns_none_when_agent_funding_amounts_none(self) -> None:
        """Test returns none when agent funding amounts none."""
        b = self._make_b()
        with (
            patch.object(
                b, "_get_service_owner", side_effect=_gen_returning("0xOWNER")
            ),
            patch.object(
                b, "_get_agent_funding_amounts", side_effect=_gen_returning(None)
            ),
        ):
            result = _run_gen(b._split_funds(1000))
        assert result is None

    def test_returns_none_when_funds_by_operator_none(self) -> None:
        """Test returns none when funds by operator none."""
        b = self._make_b()
        with (
            patch.object(
                b, "_get_service_owner", side_effect=_gen_returning("0xOWNER")
            ),
            patch.object(
                b,
                "_get_agent_funding_amounts",
                side_effect=_gen_returning({"agent-0": 50}),
            ),
            patch.object(b, "_get_funds_by_operator", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b._split_funds(1000))
        assert result is None

    def test_proportional_split_when_agent_amounts_exceed_profits(self) -> None:
        """Test proportional split when agent amounts exceed profits."""
        b = self._make_b()
        # agent_funding_amounts = {"a0": 80, "a1": 60} → total=140 > profits=100
        # a0 share = (80 * 100) // 140 = 57, a1 share = (60 * 100) // 140 = 42
        with (
            patch.object(
                b, "_get_service_owner", side_effect=_gen_returning("0xOWNER")
            ),
            patch.object(
                b,
                "_get_agent_funding_amounts",
                side_effect=_gen_returning({"a0": 80, "a1": 60}),
            ),
        ):
            result = _run_gen(b._split_funds(100))
        assert result is not None
        # a0 share = (80 * 100) // 140 = 57, a1 share = (60 * 100) // 140 = 42
        assert result["a0"] == 57
        assert result["a1"] == 42

    def test_full_success_split(self) -> None:
        """Test full success split."""
        b = self._make_b()
        # Funding: agent-0 gets 50, total 50 <= profits 200
        # After agents: profits 150, service_owner_share 10pct = 15
        # Operator gets 135
        with (
            patch.object(
                b, "_get_service_owner", side_effect=_gen_returning("0xOWNER")
            ),
            patch.object(
                b,
                "_get_agent_funding_amounts",
                side_effect=_gen_returning({"agent-0": 50}),
            ),
            patch.object(
                b, "_get_funds_by_operator", side_effect=_gen_returning({"op-0": 135})
            ),
        ):
            result = _run_gen(b._split_funds(200))
        assert result is not None
        assert result["agent-0"] == 50
        assert result["0xOWNER"] == 15  # 10% of 150
        assert result["op-0"] == 135


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_transfer_tx
# ---------------------------------------------------------------------------


class TestGetTransferTx:
    """Test Get Transfer Tx."""

    def test_returns_dict_on_success(self) -> None:
        """Test returns dict on success."""
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        tx_data = b"\xbe\xef"
        msg = _state_contract_msg({"data": tx_data})
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(b._get_transfer_tx("0xMECH", "0xRECEIVER", 100))
        assert result["to"] == "0xMECH"
        assert result["data"] == tx_data


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_token_address / _get_token_transfer_tx_data / _get_token_transfer_tx
# ---------------------------------------------------------------------------


class TestTokenTransfer:
    """Test Token Transfer."""

    def _make_b(self) -> "_DummyFunds":
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_get_token_address_success(self) -> None:
        """Test get token address success."""
        b = self._make_b()
        msg = _state_contract_msg({"token_address": "0xTOKEN"})  # nosec B105
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(b._get_token_address("0xTRACK"))
        assert result == "0xTOKEN"

    def test_get_token_transfer_tx_data_success(self) -> None:
        """Test get token transfer tx data success."""
        b = self._make_b()
        tx_data = b"\xca\xfe"
        msg = _state_contract_msg({"data": tx_data})
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(
                b._get_token_transfer_tx_data("0xTOKEN", "0xRECEIVER", 100)
            )
        assert result == tx_data

    def test_get_token_transfer_tx_returns_none_when_data_none(self) -> None:
        """Test get token transfer tx returns none when data none."""
        b = self._make_b()
        with patch.object(
            b, "_get_token_transfer_tx_data", side_effect=_gen_returning(None)
        ):
            result = _run_gen(
                b._get_token_transfer_tx("0xMECH", "0xTOKEN", "0xRECEIVER", 100)
            )
        assert result is None

    def test_get_token_transfer_tx_success(self) -> None:
        """Test get token transfer tx success."""
        b = self._make_b()
        tx_data = b"\xba\xbe"
        msg = _state_contract_msg({"data": tx_data})
        with (
            patch.object(
                b,
                "_get_token_transfer_tx_data",
                side_effect=_gen_returning(b"\xca\xfe"),
            ),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(msg)
            ),
        ):
            result = _run_gen(
                b._get_token_transfer_tx("0xMECH", "0xTOKEN", "0xRECEIVER", 100)
            )
        assert result["to"] == "0xMECH"


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_service_owner
# ---------------------------------------------------------------------------


class TestGetServiceOwner:
    """Test Get Service Owner."""

    def _make_b(self) -> "_DummyFunds":
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_none_on_contract_error(self) -> None:
        """Test returns none on contract error."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._get_service_owner(1))
        assert result is None

    def test_returns_owner_on_success(self) -> None:
        """Test returns owner on success."""
        b = self._make_b()
        msg = _state_contract_msg({"service_owner": "0xOWNER"})
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(b._get_service_owner(1))
        assert result == "0xOWNER"


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_funds_by_operator
# ---------------------------------------------------------------------------


class TestGetFundsByOperator:
    """Test Get Funds By Operator."""

    def _make_b(self) -> "_DummyFunds":
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_empty_dict_when_operator_share_is_zero(self) -> None:
        """Test returns empty dict when operator share is zero."""
        b = self._make_b()
        result = _run_gen(b._get_funds_by_operator(0))
        assert result == {}

    def test_returns_none_when_reqs_by_agent_none(self) -> None:
        """Test returns none when reqs by agent none."""
        b = self._make_b()
        with patch.object(
            b, "_get_num_reqs_by_agent", side_effect=_gen_returning(None)
        ):
            result = _run_gen(b._get_funds_by_operator(100))
        assert result is None

    def test_returns_zero_per_agent_when_total_reqs_zero(self) -> None:
        """Test returns zero per agent when total reqs zero."""
        b = self._make_b()
        reqs_by_agent = {"a0": 0, "a1": 0}
        with patch.object(
            b, "_get_num_reqs_by_agent", side_effect=_gen_returning(reqs_by_agent)
        ):
            result = _run_gen(b._get_funds_by_operator(100))
        assert result == {"a0": 0, "a1": 0}

    def test_returns_none_when_accumulate_fails(self) -> None:
        """Test returns none when accumulate fails."""
        b = self._make_b()
        with (
            patch.object(
                b, "_get_num_reqs_by_agent", side_effect=_gen_returning({"a0": 5})
            ),
            patch.object(
                b, "_accumulate_reqs_by_operator", side_effect=_gen_returning(None)
            ),
        ):
            result = _run_gen(b._get_funds_by_operator(100))
        assert result is None

    def test_splits_by_operator_proportionally(self) -> None:
        """Test splits by operator proportionally."""
        b = self._make_b()
        # a0 → op0 (3 reqs), a1 → op0 (2 reqs), a2 → op1 (5 reqs)
        # after accumulate: op0→5, op1→5 (total 10)
        # share for each = (100 * 5) // 10 = 50
        with (
            patch.object(
                b,
                "_get_num_reqs_by_agent",
                side_effect=_gen_returning({"a0": 3, "a1": 2, "a2": 5}),
            ),
            patch.object(
                b,
                "_accumulate_reqs_by_operator",
                side_effect=_gen_returning({"op0": 5, "op1": 5}),
            ),
        ):
            result = _run_gen(b._get_funds_by_operator(100))
        assert result == {"op0": 50, "op1": 50}

    def test_filters_zero_address_operator(self) -> None:
        """Test filters zero address operator."""
        b = self._make_b()
        # zero address gets filtered, its reqs removed from total
        # op1→5 reqs (total valid = 5), share = (100 * 5) // 5 = 100
        with (
            patch.object(
                b,
                "_get_num_reqs_by_agent",
                side_effect=_gen_returning({"a0": 3, "a1": 5}),
            ),
            patch.object(
                b,
                "_accumulate_reqs_by_operator",
                side_effect=_gen_returning({ZERO_ADDRESS: 3, "op1": 5}),
            ),
        ):
            result = _run_gen(b._get_funds_by_operator(100))
        assert ZERO_ADDRESS not in result
        assert result["op1"] == 100


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._accumulate_reqs_by_operator
# ---------------------------------------------------------------------------


class TestAccumulateReqsByOperator:
    """Test Accumulate Reqs By Operator."""

    def test_accumulates_reqs_by_operator(self) -> None:
        """Test accumulates reqs by operator."""
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        # agent_to_operator mapping: a0→op0, a1→op0, a2→op1
        msg = _state_contract_msg({"a0": "op0", "a1": "op0", "a2": "op1"})
        reqs_by_agent = {"a0": 3, "a1": 2, "a2": 5}
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(b._accumulate_reqs_by_operator(reqs_by_agent))
        assert result == {"op0": 5, "op1": 5}


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_agent_balances / _get_agent_funding_amounts
# ---------------------------------------------------------------------------


class TestGetAgentBalances:
    """Test Get Agent Balances."""

    def _make_b(self, participants: Any = None) -> "_DummyFunds":
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.all_participants = participants or ["a0", "a1"]
        mock_sd.mech_agent_balance = MagicMock()
        b._synchronized_data = mock_sd
        return b

    def _patch_sd(self, b: Any) -> Any:
        return contextlib.nullcontext()

    def test_returns_none_when_any_balance_is_none(self) -> None:
        """Test returns none when any balance is none."""
        b = self._make_b(["a0"])
        with (
            self._patch_sd(b),
            patch.object(b, "_get_balance", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b._get_agent_balances())
        assert result is None

    def test_returns_balances_dict_on_success(self) -> None:
        """Test returns balances dict on success."""
        b = self._make_b(["a0", "a1"])
        balances = {"a0": 200, "a1": 50}

        def _balance_gen(addr: Any) -> Generator[None, None, Any]:
            """Generator that returns balance for given address."""
            if False:
                yield
            return balances[addr]

        with (
            self._patch_sd(b),
            patch.object(
                b, "_get_balance", side_effect=lambda addr: _balance_gen(addr)
            ),
            patch.object(b, "set_gauge"),
        ):
            result = _run_gen(b._get_agent_balances())
        assert result["a0"] == 200
        assert result["a1"] == 50


class TestGetAgentFundingAmounts:
    """Test Get Agent Funding Amounts."""

    def _make_b(self, participants: Any = None) -> "_DummyFunds":
        ctx = _make_full_ctx()
        ctx.params.minimum_agent_balance = 100
        ctx.params.agent_funding_amount = 200
        b = _DummyFunds(name="b", skill_context=ctx)
        return b

    def test_returns_none_when_agent_balances_none(self) -> None:
        """Test returns none when agent balances none."""
        b = self._make_b()
        with patch.object(b, "_get_agent_balances", side_effect=_gen_returning(None)):
            result = _run_gen(b._get_agent_funding_amounts())
        assert result is None

    def test_returns_funding_amounts_for_underfunded_agents(self) -> None:
        """Test returns funding amounts for underfunded agents."""
        b = self._make_b()
        # a0 has 200 (above 100 min), a1 has 50 (below 100 min)
        with patch.object(
            b, "_get_agent_balances", side_effect=_gen_returning({"a0": 200, "a1": 50})
        ):
            result = _run_gen(b._get_agent_funding_amounts())
        assert "a0" not in result
        assert result["a1"] == 200  # agent_funding_amount

    def test_returns_empty_when_all_agents_funded(self) -> None:
        """Test returns empty when all agents funded."""
        b = self._make_b()
        with patch.object(
            b, "_get_agent_balances", side_effect=_gen_returning({"a0": 200, "a1": 150})
        ):
            result = _run_gen(b._get_agent_funding_amounts())
        assert result == {}


# ---------------------------------------------------------------------------
# TrackingBehaviour methods
# ---------------------------------------------------------------------------


class TestSaveUsageToIpfs:
    """Test Save Usage To Ipfs."""

    def _make_b(self) -> "_DummyTransPrep":
        ctx = _make_full_ctx()
        return _DummyTransPrep(name="b", skill_context=ctx)

    def test_returns_none_when_ipfs_fails(self) -> None:
        """Test returns none when ipfs fails."""
        b = self._make_b()
        with patch.object(b, "send_to_ipfs", side_effect=_gen_returning(None)):
            result = _run_gen(b._save_usage_to_ipfs({"data": 1}))
        assert result is None

    def test_returns_hash_on_success(self) -> None:
        """Test returns hash on success."""
        b = self._make_b()
        with patch.object(b, "send_to_ipfs", side_effect=_gen_returning("bafyhash")):
            result = _run_gen(b._save_usage_to_ipfs({"data": 1}))
        assert result == "bafyhash"


class TestGetCheckpointTx:
    """Test Get Checkpoint Tx."""

    def test_returns_dict_on_success(self) -> None:
        """Test returns dict on success."""
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        tx_data = b"\xaa\xbb"
        msg = _state_contract_msg({"data": tx_data})
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(b._get_checkpoint_tx("0xHASH", "ab" * 16))
        assert result["to"] == "0xHASH"
        assert result["data"] == tx_data


class TestGetUpdateUsageTx:
    """Test Get Update Usage Tx."""

    def _make_b(self) -> "_DummyTransPrep":
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.done_tasks = []
        b._synchronized_data = mock_sd
        return b

    def _patch_sd(self, b: Any) -> Any:
        return contextlib.nullcontext()

    def test_returns_none_when_delivery_report_none(self) -> None:
        """Test returns none when delivery report none."""
        b = self._make_b()
        with (
            self._patch_sd(b),
            patch.object(b, "get_delivery_report", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_update_usage_tx())
        assert result is None

    def test_returns_none_when_ipfs_save_fails(self) -> None:
        """Test returns none when ipfs save fails."""
        b = self._make_b()
        with (
            self._patch_sd(b),
            patch.object(
                b,
                "get_delivery_report",
                side_effect=_gen_returning({"agent-0": {"t": 1}}),
            ),
            patch.object(b, "_save_usage_to_ipfs", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_update_usage_tx())
        assert result is None

    def test_returns_tx_on_success(self) -> None:
        """Test returns tx on success."""
        b = self._make_b()
        tx = {"to": "0xHASH", "value": 0, "data": b"\x00"}
        with (
            _mock_to_multihash() as mtm,
            self._patch_sd(b),
            patch.object(
                b,
                "get_delivery_report",
                side_effect=_gen_returning({"agent-0": {"t": 1}}),
            ),
            patch.object(
                b, "_save_usage_to_ipfs", side_effect=_gen_returning("bafyhash")
            ),
            patch.object(b, "_get_checkpoint_tx", side_effect=_gen_returning(tx)),
            patch(
                "packages.valory.skills.task_submission_abci.behaviours.to_v1",
                return_value="bafyhashv1",
            ),
        ):
            mtm.return_value = "ab" * 16
            result = _run_gen(b.get_update_usage_tx())
        assert result == tx


# ---------------------------------------------------------------------------
# HashUpdateBehaviour._get_latest_hash / _should_update_hash / get_mech_update_hash_tx
# ---------------------------------------------------------------------------


class TestHashUpdateBehaviour:
    """Test Hash Update Behaviour."""

    def _make_b(self) -> "_DummyTransPrep":
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        return b

    def test_get_latest_hash_success(self) -> None:
        """Test get lahash success."""
        b = self._make_b()
        hash_data = b"\xcc" * 32
        msg = _state_contract_msg({"data": hash_data})
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(b._get_latest_hash())
        assert result == hash_data

    def test_should_update_hash_returns_false_when_latest_is_none(self) -> None:
        """Test should update hash returns false when lais none."""
        b = self._make_b()
        b.context.params.task_mutable_params.latest_metadata_hash = None
        with patch.object(b, "_get_latest_hash", side_effect=_gen_returning(None)):
            result = _run_gen(b._should_update_hash())
        assert result is False

    def test_should_update_hash_returns_false_when_configured_hash_empty(self) -> None:
        """Test should update hash returns false when configured hash empty."""
        b = self._make_b()
        b.context.params.task_mutable_params.latest_metadata_hash = None
        with (
            _mock_to_multihash() as mtm,
            patch.object(
                b, "_get_latest_hash", side_effect=_gen_returning(b"\x00" * 32)
            ),
        ):
            mtm.return_value = ""
            result = _run_gen(b._should_update_hash())
        assert result is False

    def test_should_update_hash_returns_false_when_hashes_same(self) -> None:
        """Test should update hash returns false when hashes same."""
        b = self._make_b()
        # latest_metadata_hash and configured_hash must both be strings for equality
        b.context.params.task_mutable_params.latest_metadata_hash = "same_hash"
        with _mock_to_multihash() as mtm:
            mtm.return_value = "same_hash"
            result = _run_gen(b._should_update_hash())
        assert result is False

    def test_should_update_hash_returns_true_when_hashes_differ(self) -> None:
        """Test should update hash returns true when hashes differ."""
        b = self._make_b()
        b.context.params.task_mutable_params.latest_metadata_hash = b"old_hash"
        with _mock_to_multihash() as mtm:
            mtm.return_value = "new_hash"
            result = _run_gen(b._should_update_hash())
        assert result is True

    def test_get_mech_update_hash_tx_returns_none_when_no_update_needed(self) -> None:
        """Test get mech update hash tx returns none when no update needed."""
        b = self._make_b()
        with patch.object(b, "_should_update_hash", side_effect=_gen_returning(False)):
            result = _run_gen(b.get_mech_update_hash_tx())
        assert result is None

    def test_get_mech_update_hash_tx_returns_tx_on_success(self) -> None:
        """Test get mech update hash tx returns tx on success."""
        b = self._make_b()
        b.context.params.task_mutable_params.latest_metadata_hash = b"old"
        tx_data = b"\xde\xca"
        msg = _state_contract_msg({"data": tx_data})
        with (
            _mock_to_multihash() as mtm,
            patch.object(b, "_should_update_hash", side_effect=_gen_returning(True)),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(msg)
            ),
        ):
            mtm.return_value = "ab" * 16
            result = _run_gen(b.get_mech_update_hash_tx())
        assert result["to"] == "0xMETA"
        assert result["data"] == tx_data


# ---------------------------------------------------------------------------
# TransactionPreparationBehaviour._get_deliver_tx
# ---------------------------------------------------------------------------


class TestGetDeliverTx:
    """Test Get Deliver Tx."""

    def _make_b(self) -> "_DummyTransPrep":
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        return b

    def test_routes_to_marketplace_when_is_marketplace_mech(self) -> None:
        """Test routes to marketplace when is marketplace mech."""
        b = self._make_b()
        task = {
            "request_id": "r1",
            "is_marketplace_mech": True,
            "mech_address": "0xMECH",
        }
        tx = {"to": "0xMECH", "value": 0, "data": b"\x00", "simulation_ok": True}
        with patch.object(
            b, "_get_deliver_marketplace_tx", side_effect=_gen_returning(tx)
        ):
            result = _run_gen(b._get_deliver_tx(task))
        assert result == tx

    def test_routes_to_agent_mech_when_not_marketplace(self) -> None:
        """Test routes to agent mech when not marketplace."""
        b = self._make_b()
        task = {"request_id": "r1", "is_marketplace_mech": False}
        tx = {"to": "0xMECH", "value": 0, "data": b"\x00", "simulation_ok": True}
        with patch.object(
            b, "_get_agent_mech_deliver_tx", side_effect=_gen_returning(tx)
        ):
            result = _run_gen(b._get_deliver_tx(task))
        assert result == tx


# ---------------------------------------------------------------------------
# TransactionPreparationBehaviour._get_agent_mech_deliver_tx / _get_deliver_marketplace_tx
# ---------------------------------------------------------------------------


class TestDeliverTxMethods:
    """Test Deliver Tx Methods."""

    def _make_b(self) -> "_DummyTransPrep":
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        b._synchronized_data = mock_sd
        return b

    def _patch_sd(self, b: Any) -> Any:
        return contextlib.nullcontext()

    def test_agent_mech_deliver_tx_success(self) -> None:
        """Test agent mech deliver tx success."""
        b = self._make_b()
        task_data = {
            "mech_address": "0xMECH",
            "request_id": "r1",
            "task_result": b"\xaa",
            "request_id_nonce": 1,
        }
        msg = _state_contract_msg({"data": b"\xdd", "simulation_ok": True})
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(msg)
            ),
        ):
            result = _run_gen(b._get_agent_mech_deliver_tx(task_data))
        assert result["to"] == "0xMECH"
        assert result["simulation_ok"] is True

    def test_deliver_marketplace_tx_success(self) -> None:
        """Test deliver marketplace tx success."""
        b = self._make_b()
        task_data = {
            "mech_address": "0xMECH",
            "request_id": "r1",
            "task_result": b"\xaa",
        }
        msg = _state_contract_msg({"data": b"\xdd", "simulation_ok": True})
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(msg)
            ),
        ):
            result = _run_gen(b._get_deliver_marketplace_tx(task_data))
        assert result["to"] == "0xMECH"
        assert result["simulation_ok"] is True


# ---------------------------------------------------------------------------
# TransactionPreparationBehaviour._get_safe_tx_hash / _to_multisend
# ---------------------------------------------------------------------------


class TestSafeTxHashAndMultisend:
    """Test Safe Tx Hash And Multisend."""

    def _make_b(self) -> "_DummyTransPrep":
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        b._synchronized_data = mock_sd
        return b

    def _patch_sd(self, b: Any) -> Any:
        return contextlib.nullcontext()

    def test_get_safe_tx_hash_returns_none_on_error(self) -> None:
        """Test get safe tx hash returns none on error."""
        b = self._make_b()
        with (
            self._patch_sd(b),
            patch.object(
                b,
                "get_contract_api_response",
                side_effect=_gen_returning(_error_contract_msg()),
            ),
        ):
            result = _run_gen(b._get_safe_tx_hash(b"\x00" * 32))
        assert result is None

    def test_get_safe_tx_hash_returns_hash_on_success(self) -> None:
        """Test get safe tx hash returns hash on success."""
        b = self._make_b()
        msg = _state_contract_msg({"tx_hash": "0xabcdef"})
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(msg)
            ),
        ):
            result = _run_gen(b._get_safe_tx_hash(b"\x00" * 32))
        assert result == "abcdef"  # strips "0x"

    def test_to_multisend_returns_none_on_non_raw_tx_response(self) -> None:
        """Test to multisend returns none on non raw tx response."""
        b = self._make_b()
        with (
            self._patch_sd(b),
            patch.object(
                b,
                "get_contract_api_response",
                side_effect=_gen_returning(_error_contract_msg()),
            ),
        ):
            result = _run_gen(
                b._to_multisend([{"to": "0xA", "value": 0, "data": b"\x00"}])
            )
        assert result is None

    def test_to_multisend_returns_none_when_safe_tx_hash_fails(self) -> None:
        """Test to multisend returns none when safe tx hash fails."""
        b = self._make_b()
        raw_msg = _raw_tx_contract_msg({"data": "0x" + "ab" * 8})
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(raw_msg)
            ),
            patch.object(b, "_get_safe_tx_hash", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(
                b._to_multisend([{"to": "0xA", "value": 0, "data": b"\x00"}])
            )
        assert result is None


# ---------------------------------------------------------------------------
# TransactionPreparationBehaviour._get_offchain_tasks_deliver_data
# ---------------------------------------------------------------------------


class TestGetOffchainTasksDeliverData:
    """Test Get Offchain Tasks Deliver Data."""

    def _make_b(self) -> "_DummyTransPrep":
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        return b

    def _patch_sd(self, b: Any, done_tasks: Any) -> Any:
        mock_sd = MagicMock()
        mock_sd.done_tasks = done_tasks
        mock_sd.safe_contract_address = "0xSAFE"
        b._synchronized_data = mock_sd
        return contextlib.nullcontext()

    @staticmethod
    def _offchain_task(
        request_id: str, sender: str = "0xSENDER", nonce: int = 1
    ) -> Dict:
        return {
            "request_id": request_id,
            "is_offchain": True,
            "nonce": nonce,
            "sender": sender,
            "ipfs_hash": "0x" + "ab" * 16,
            "signature": "0x" + "cd" * 32,
            "task_result": "de" * 16,
            "delivery_rate": 100,
        }

    def test_returns_empty_list_when_no_offchain_tasks(self) -> None:
        """Test returns empty list when no offchain tasks."""
        b = self._make_b()
        non_offchain = [{"request_id": "r1", "is_offchain": False}]
        with self._patch_sd(b, non_offchain):
            result = _run_gen(b._get_offchain_tasks_deliver_data())
        assert result == ([], [])

    def test_skips_tasks_with_failed_simulation(self) -> None:
        """A sim-failed sender group is left out of both the txs and the included ids."""
        b = self._make_b()
        task = self._offchain_task("r1")
        msg = _state_contract_msg({"data": b"\xdd", "simulation_ok": False})
        with (
            self._patch_sd(b, [task]),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(msg)
            ),
            patch.object(b, "note_settlement_skipped") as mock_skip,
        ):
            result = _run_gen(b._get_offchain_tasks_deliver_data())
        assert result == ([], [])
        # The skip bookkeeping (retry stamp / drop cap / metric) fires for
        # exactly the ids in the failed group.
        mock_skip.assert_called_once_with(
            ["r1"], SOURCE_OFFCHAIN, "0xMECH", b.synchronized_data.period_count
        )

    def test_sim_failure_only_excludes_the_failing_sender_group(self) -> None:
        """Two senders, one fails simulation: the other's ids are still included."""
        b = self._make_b()
        ok_task = self._offchain_task("r-ok", sender="0xOK")
        bad_task = self._offchain_task("r-bad", sender="0xBAD")
        by_requester = {
            "0xOK": _state_contract_msg({"data": b"\x01", "simulation_ok": True}),
            "0xBAD": _state_contract_msg({"data": b"\x02", "simulation_ok": False}),
        }

        def _respond(*_args: Any, **kwargs: Any) -> Any:
            return _gen_returning(by_requester[kwargs["requester"]])()

        with (
            self._patch_sd(b, [ok_task, bad_task]),
            patch.object(b, "get_contract_api_response", side_effect=_respond),
            patch.object(b, "note_settlement_skipped") as mock_skip,
        ):
            tx_list, included = _run_gen(b._get_offchain_tasks_deliver_data())
        assert [tx["data"] for tx in tx_list] == [b"\x01"]
        assert included == ["r-ok"]
        mock_skip.assert_called_once_with(
            ["r-bad"], SOURCE_OFFCHAIN, "0xMECH", b.synchronized_data.period_count
        )

    def test_appends_tx_when_simulation_ok(self) -> None:
        """Test appends tx when simulation ok."""
        b = self._make_b()
        task = self._offchain_task("r1")
        msg = _state_contract_msg({"data": b"\xdd", "simulation_ok": True})
        with (
            self._patch_sd(b, [task]),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(msg)
            ),
        ):
            tx_list, included = _run_gen(b._get_offchain_tasks_deliver_data())
        assert len(tx_list) == 1
        assert tx_list[0]["to"] == "0xMECH"
        assert included == ["r1"]


# ---------------------------------------------------------------------------
# TransactionPreparationBehaviour._get_is_nvm_mech / _get_encoded_deliver_data
# ---------------------------------------------------------------------------


class TestNvmMechHelpers:
    """Test Nvm Mech Helpers."""

    def _make_b(self) -> "_DummyTransPrep":
        ctx = _make_full_ctx()
        return _DummyTransPrep(name="b", skill_context=ctx)

    def test_get_is_nvm_mech_returns_bool(self) -> None:
        """Test get is nvm mech returns bool."""
        b = self._make_b()
        msg = _state_contract_msg({"data": True})
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            result = _run_gen(b._get_is_nvm_mech("0xMECH"))
        assert result is True

    def test_get_encoded_deliver_data_with_requests(self) -> None:
        """Test get encoded deliver data with requests."""
        b = self._make_b()
        encoded_data = b"\xee\xff"
        msg = _state_contract_msg({"data": encoded_data})
        request_ids = [b"\x00" * 32]
        datas = [b"\x01" * 32]
        delivery_rates = [100]
        with patch.object(
            b, "get_contract_api_response", side_effect=_gen_returning(msg)
        ):
            final_ids, final_datas = _run_gen(
                b._get_encoded_deliver_data(request_ids, datas, delivery_rates)
            )
        assert len(final_ids) == 1
        assert final_datas[0] == encoded_data

    def test_get_encoded_deliver_data_empty(self) -> None:
        """Test get encoded deliver data empty."""
        b = self._make_b()
        final_ids, final_datas = _run_gen(b._get_encoded_deliver_data([], [], []))
        assert final_ids == []
        assert final_datas == []


# ---------------------------------------------------------------------------
# TransactionPreparationBehaviour._get_marketplace_tasks_deliver_data
# ---------------------------------------------------------------------------


class TestGetMarketplaceTasksDeliverData:
    """Test Get Marketplace Tasks Deliver Data."""

    def _make_b(self) -> "_DummyTransPrep":
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        return b

    def test_returns_empty_list_for_no_marketplace_tasks(self) -> None:
        """Test returns empty list for no marketplace tasks."""
        b = self._make_b()
        result = _run_gen(b._get_marketplace_tasks_deliver_data([]))
        assert result == ([], [])

    def test_appends_tx_for_non_nvm_mech_with_simulation_ok(self) -> None:
        """Test appends tx for non nvm mech with simulation ok."""
        b = self._make_b()
        task = {
            "mech_address": "0xMECH",
            "request_id": 12345,
            "requestId": 12345,
            "task_result": "de" * 16,
        }
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        b._synchronized_data = mock_sd
        msg_deliver = _state_contract_msg({"data": b"\xdd", "simulation_ok": True})
        with (
            patch.object(b, "_get_is_nvm_mech", side_effect=_gen_returning(False)),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(msg_deliver)
            ),
        ):
            tx_list, included = _run_gen(b._get_marketplace_tasks_deliver_data([task]))
        assert len(tx_list) == 1
        # Reported as the ``str`` form the prune site keys on, not bytes32.
        assert included == ["12345"]

    def test_skips_tasks_with_failed_simulation(self) -> None:
        """A sim-failed mech group is left out of txs and ids and charged one attempt."""
        b = self._make_b()
        task = {
            "mech_address": "0xMECH",
            "request_id": 12345,
            "requestId": 12345,
            "task_result": "de" * 16,
        }
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        mock_sd.period_count = 4
        b._synchronized_data = mock_sd
        msg_deliver = _state_contract_msg({"data": b"\xdd", "simulation_ok": False})
        with (
            patch.object(b, "_get_is_nvm_mech", side_effect=_gen_returning(False)),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(msg_deliver)
            ),
            patch.object(b, "note_settlement_skipped") as mock_skip,
        ):
            result = _run_gen(b._get_marketplace_tasks_deliver_data([task]))
        assert result == ([], [])
        mock_skip.assert_called_once_with(["12345"], SOURCE_ONCHAIN, "0xMECH", 4)

    def test_contract_error_after_sim_failure_does_not_recount_those_ids(self) -> None:
        """Ids already charged sim_failed are excluded from the contract_error count."""
        b = self._make_b()
        bad = {
            "mech_address": "0xBAD",
            "request_id": 1,
            "requestId": 1,
            "task_result": "aa",
        }
        good = {
            "mech_address": "0xGOOD",
            "request_id": 2,
            "requestId": 2,
            "task_result": "bb",
        }
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        b._synchronized_data = mock_sd
        b.context.shared_state[DONE_TASKS] = [dict(bad), dict(good)]
        responses = {
            "0xBAD": _state_contract_msg({"data": b"\x01", "simulation_ok": False}),
            "0xGOOD": _error_contract_msg(),
        }

        def _respond(*_args: Any, **kwargs: Any) -> Any:
            return _gen_returning(responses[kwargs["contract_address"]])()

        with (
            patch.object(b, "_get_is_nvm_mech", side_effect=_gen_returning(False)),
            patch.object(b, "get_contract_api_response", side_effect=_respond),
            patch.object(b, "count_settlement") as mock_count,
        ):
            result = _run_gen(b._get_marketplace_tasks_deliver_data([bad, good]))
        assert result == (None, [])
        by_outcome = {
            (c.args[0], c.args[2]): c.args[3] for c in mock_count.call_args_list
        }
        assert by_outcome[(SETTLEMENT_OUTCOME_SIM_FAILED, "0xBAD")] == 1
        assert by_outcome[(SETTLEMENT_OUTCOME_CONTRACT_ERROR, "0xBAD")] == 0
        assert by_outcome[(SETTLEMENT_OUTCOME_CONTRACT_ERROR, "0xGOOD")] == 1


# ---------------------------------------------------------------------------
# TaskSubmissionRoundBehaviour class attributes
# ---------------------------------------------------------------------------


class TestTaskSubmissionRoundBehaviour:
    """Test Task Submission Round Behaviour."""

    def test_round_behaviour_composition(self) -> None:
        """Initial behaviour, app class, and behaviour set are wired correctly."""
        from packages.valory.skills.task_submission_abci.behaviours import (
            TransactionPreparationBehaviour,
        )
        from packages.valory.skills.task_submission_abci.rounds import (
            TaskSubmissionAbciApp,
        )

        assert (
            TaskSubmissionRoundBehaviour.initial_behaviour_cls is TaskPoolingBehaviour
        )
        assert TaskSubmissionRoundBehaviour.abci_app_cls is TaskSubmissionAbciApp
        assert (
            TransactionPreparationBehaviour in TaskSubmissionRoundBehaviour.behaviours
        )
        assert TaskPoolingBehaviour in TaskSubmissionRoundBehaviour.behaviours


# ---------------------------------------------------------------------------
# Tests for get_split_profit_txs in FundsSplittingBehaviour
# ---------------------------------------------------------------------------


class TestGetSplitProfitTxs:
    """Test Get Split Profit Txs."""

    def _make_b(self) -> "_DummyFunds":
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_empty_list_when_no_split(self) -> None:
        """Test returns empty list when no split."""
        b = self._make_b()
        with patch.object(
            b, "_should_split_profits", side_effect=_gen_returning(False)
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result == []

    def test_returns_none_when_mech_info_unavailable(self) -> None:
        """Test returns none when mech info unavailable."""
        b = self._make_b()
        b.context.params.agent_mech_contract_addresses = ["0xMECH"]
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None

    def test_returns_none_when_profits_is_none(self) -> None:
        """Test returns none when profits is none."""
        b = self._make_b()
        # non-NVM mech type (plain bytes)
        mech_info = (b"\x00" * 32, "0xTRACK", 1000)
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(
                b, "_calculate_mech_profits", side_effect=_gen_returning(None)
            ),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None

    def test_returns_none_when_process_payment_tx_is_none(self) -> None:
        """Test returns none when process payment tx is none."""
        b = self._make_b()
        mech_info = (b"\x00" * 32, "0xTRACK", 1000)
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(
                b, "_get_process_payment_tx", side_effect=_gen_returning(None)
            ),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None

    def test_returns_none_when_simulation_fails(self) -> None:
        """Test returns none when simulation fails."""
        b = self._make_b()
        mech_info = (b"\x00" * 32, "0xTRACK", 1000)
        process_tx = {
            "to": "0xTRACK",
            "value": 0,
            "data": b"\x00",
            "simulation_ok": False,
        }
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(
                b, "_get_process_payment_tx", side_effect=_gen_returning(process_tx)
            ),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None

    def test_returns_none_when_split_funds_is_none(self) -> None:
        """Test returns none when split funds is none."""
        b = self._make_b()
        mech_info = (b"\x00" * 32, "0xTRACK", 1000)
        process_tx = {
            "to": "0xTRACK",
            "value": 0,
            "data": b"\x00",
            "simulation_ok": True,
        }
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(
                b,
                "_get_process_payment_tx",
                side_effect=_gen_returning(dict(process_tx)),
            ),
            patch.object(b, "_split_funds", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None

    def test_returns_none_when_transfer_tx_is_none(self) -> None:
        """Test returns none when transfer tx is none."""
        b = self._make_b()
        mech_info = (b"\x00" * 32, "0xTRACK", 1000)
        process_tx = {
            "to": "0xTRACK",
            "value": 0,
            "data": b"\x00",
            "simulation_ok": True,
        }
        split_funds = {"op0": 100}  # amount > 0 → need transfer tx
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(
                b,
                "_get_process_payment_tx",
                side_effect=_gen_returning(dict(process_tx)),
            ),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
            patch.object(b, "_get_transfer_tx", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None

    def test_skips_zero_amount_receivers(self) -> None:
        """Test skips zero amount receivers."""
        b = self._make_b()
        mech_info = (b"\x00" * 32, "0xTRACK", 1000)
        process_tx = {
            "to": "0xTRACK",
            "value": 0,
            "data": b"\x00",
            "simulation_ok": True,
        }
        split_funds = {"op0": 0}  # amount=0 → skip, no transfer tx needed
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(
                b,
                "_get_process_payment_tx",
                side_effect=_gen_returning(dict(process_tx)),
            ),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        # Only process_payment_tx (no transfer txs for zero amounts), but txs is empty if only zeros
        # Actually process_payment_tx IS appended before split_funds, so txs=[process_tx]
        assert result is not None

    def test_returns_txs_with_native_transfer(self) -> None:
        """Test returns txs with native transfer."""
        b = self._make_b()
        # Non-NVM, non-TOKEN mech type → uses _get_transfer_tx
        mech_info = (b"\x00" * 32, "0xTRACK", 1000)
        process_tx = {
            "to": "0xTRACK",
            "value": 0,
            "data": b"\x00",
            "simulation_ok": True,
        }
        split_funds = {"op0": 900}
        transfer_tx = {"to": "0xMECH", "value": 0, "data": b"\x01"}
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(
                b,
                "_get_process_payment_tx",
                side_effect=_gen_returning(dict(process_tx)),
            ),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
            patch.object(
                b, "_get_transfer_tx", side_effect=_gen_returning(transfer_tx)
            ),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is not None
        assert len(result) == 2  # process_payment_tx + transfer_tx

    def test_nvm_mech_adjusts_balance(self) -> None:
        """Test nvm mech adjusts balance."""
        b = self._make_b()
        # NVM type - use PAYMENT_TYPE_NATIVE_NVM hex
        from packages.valory.skills.task_submission_abci.behaviours import (
            PAYMENT_TYPE_NATIVE_NVM,
        )

        nvm_type = bytes.fromhex(PAYMENT_TYPE_NATIVE_NVM)
        mech_info = (nvm_type, "0xTRACK", 1000)
        process_tx = {
            "to": "0xTRACK",
            "value": 0,
            "data": b"\x00",
            "simulation_ok": True,
        }
        split_funds = {"op0": 0}
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_adjust_mech_balance", side_effect=_gen_returning(500)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(400)),
            patch.object(
                b,
                "_get_process_payment_tx",
                side_effect=_gen_returning(dict(process_tx)),
            ),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is not None  # process_tx was added

    def test_nvm_mech_returns_none_when_adjust_fails(self) -> None:
        """Test nvm mech returns none when adjust fails."""
        b = self._make_b()
        from packages.valory.skills.task_submission_abci.behaviours import (
            PAYMENT_TYPE_NATIVE_NVM,
        )

        nvm_type = bytes.fromhex(PAYMENT_TYPE_NATIVE_NVM)
        mech_info = (nvm_type, "0xTRACK", 1000)
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_adjust_mech_balance", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None


# ---------------------------------------------------------------------------
# _to_multisend full success path
# ---------------------------------------------------------------------------


class TestToMultisendSuccess:
    """Test To Multisend Success."""

    def _make_b(self) -> "_DummyTransPrep":
        ctx = _make_full_ctx()
        ctx.params.manual_gas_limit = 0
        b = _DummyTransPrep(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        b._synchronized_data = mock_sd
        return b

    def _patch_sd(self, b: Any) -> Any:
        return contextlib.nullcontext()

    def test_to_multisend_full_success(self) -> None:
        """Test to multisend full success."""
        b = self._make_b()
        raw_msg = _raw_tx_contract_msg({"data": "0x" + "ab" * 8})
        tx_hash = "cd" * 32
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(raw_msg)
            ),
            patch.object(b, "_get_safe_tx_hash", side_effect=_gen_returning(tx_hash)),
            patch(
                "packages.valory.skills.task_submission_abci.behaviours.hash_payload_to_hex",
                return_value="encoded_payload",
            ),
        ):
            result = _run_gen(
                b._to_multisend([{"to": "0xA", "value": 0, "data": b"\x00"}])
            )
        assert result == "encoded_payload"


# ---------------------------------------------------------------------------
# TransactionPreparationBehaviour.get_payload_content
# ---------------------------------------------------------------------------


class TestTransactionPreparationGetPayloadContent:
    """Test Transaction Preparation Get Payload Content."""

    def _make_b(self) -> "_DummyTransPrep":
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.done_tasks = []
        mock_sd.safe_contract_address = "0xSAFE"
        b._synchronized_data = mock_sd
        return b

    def _patch_sd(self, b: Any) -> Any:
        return contextlib.nullcontext()

    def test_returns_error_payload_when_update_usage_tx_none(self) -> None:
        """Test returns error payload when update usage tx none."""
        b = self._make_b()
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_mech_update_hash_tx", side_effect=_gen_returning(None)
            ),
            patch.object(b, "get_split_profit_txs", side_effect=_gen_returning([])),
            patch.object(
                b,
                "_get_offchain_tasks_deliver_data",
                side_effect=_gen_returning(([], [])),
            ),
            patch.object(
                b,
                "_get_marketplace_tasks_deliver_data",
                side_effect=_gen_returning(([], [])),
            ),
            patch.object(b, "get_update_usage_tx", side_effect=_gen_returning(None)),
        ):
            from packages.valory.skills.task_submission_abci.rounds import (
                TransactionPreparationRound,
            )

            result = _run_gen(b.get_payload_content())
        assert result == TransactionPreparationRound.ERROR_PAYLOAD

    def test_returns_error_payload_when_multisend_fails(self) -> None:
        """Test returns error payload when multisend fails."""
        b = self._make_b()
        usage_tx = {"to": "0xHASH", "value": 0, "data": b"\x00"}
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_mech_update_hash_tx", side_effect=_gen_returning(None)
            ),
            patch.object(b, "get_split_profit_txs", side_effect=_gen_returning([])),
            patch.object(
                b,
                "_get_offchain_tasks_deliver_data",
                side_effect=_gen_returning(([], [])),
            ),
            patch.object(
                b,
                "_get_marketplace_tasks_deliver_data",
                side_effect=_gen_returning(([], [])),
            ),
            patch.object(
                b, "get_update_usage_tx", side_effect=_gen_returning(usage_tx)
            ),
            patch.object(b, "_to_multisend", side_effect=_gen_returning(None)),
        ):
            from packages.valory.skills.task_submission_abci.rounds import (
                TransactionPreparationRound,
            )

            result = _run_gen(b.get_payload_content())
        assert result == TransactionPreparationRound.ERROR_PAYLOAD

    def test_returns_multisend_str_on_full_success(self) -> None:
        """Test returns multisend str on full success."""
        b = self._make_b()
        usage_tx = {"to": "0xHASH", "value": 0, "data": b"\x00"}
        hash_tx = {"to": "0xMETA", "value": 0, "data": b"\x01"}
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_mech_update_hash_tx", side_effect=_gen_returning(hash_tx)
            ),
            patch.object(b, "get_split_profit_txs", side_effect=_gen_returning([])),
            patch.object(
                b,
                "_get_offchain_tasks_deliver_data",
                side_effect=_gen_returning(([], [])),
            ),
            patch.object(
                b,
                "_get_marketplace_tasks_deliver_data",
                side_effect=_gen_returning(([], [])),
            ),
            patch.object(
                b, "get_update_usage_tx", side_effect=_gen_returning(usage_tx)
            ),
            patch.object(
                b, "_to_multisend", side_effect=_gen_returning("encoded_multisend")
            ),
        ):
            result = _run_gen(b.get_payload_content())
        # The vote is the envelope, never the bare multisend string.
        assert result != "encoded_multisend"
        assert decode_tx_payload(result) == {
            "tx_hash": "encoded_multisend",
            "included_request_ids": [],
        }

    def test_included_ids_aggregate_across_all_three_deliver_paths(self) -> None:
        """Off-chain, marketplace and legacy ids all land in the envelope, in that order."""
        b = self._make_b()
        legacy = {"is_marketplace_mech": False, "request_id": "r-legacy"}
        b._synchronized_data.done_tasks = [legacy]
        deliver_tx = {
            "to": "0xMECH",
            "value": 0,
            "data": b"\x00",
            "simulation_ok": True,
        }
        usage_tx = {"to": "0xHASH", "value": 0, "data": b"\x00"}
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_mech_update_hash_tx", side_effect=_gen_returning(None)
            ),
            patch.object(b, "get_split_profit_txs", side_effect=_gen_returning([])),
            patch.object(
                b,
                "_get_offchain_tasks_deliver_data",
                side_effect=_gen_returning(([{"to": "0xA"}], ["r-off"])),
            ),
            patch.object(
                b,
                "_get_marketplace_tasks_deliver_data",
                side_effect=_gen_returning(([{"to": "0xB"}], ["r-mkt"])),
            ),
            patch.object(
                b, "_get_deliver_tx", side_effect=_gen_returning(dict(deliver_tx))
            ),
            patch.object(
                b, "get_update_usage_tx", side_effect=_gen_returning(usage_tx)
            ),
            patch.object(b, "_to_multisend", side_effect=_gen_returning("encoded")),
        ):
            result = _run_gen(b.get_payload_content())
        assert decode_tx_payload(result) == {
            "tx_hash": "encoded",
            "included_request_ids": ["r-off", "r-mkt", "r-legacy"],
        }

    def test_abandoned_offchain_batch_contributes_no_ids(self) -> None:
        """A ``(None, [])`` off-chain result adds nothing to the envelope."""
        b = self._make_b()
        usage_tx = {"to": "0xHASH", "value": 0, "data": b"\x00"}
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_mech_update_hash_tx", side_effect=_gen_returning(None)
            ),
            patch.object(b, "get_split_profit_txs", side_effect=_gen_returning([])),
            patch.object(
                b,
                "_get_offchain_tasks_deliver_data",
                side_effect=_gen_returning((None, [])),
            ),
            patch.object(
                b,
                "_get_marketplace_tasks_deliver_data",
                side_effect=_gen_returning(([], [])),
            ),
            patch.object(
                b, "get_update_usage_tx", side_effect=_gen_returning(usage_tx)
            ),
            patch.object(b, "_to_multisend", side_effect=_gen_returning("encoded")),
        ):
            result = _run_gen(b.get_payload_content())
        assert decode_tx_payload(result) == {
            "tx_hash": "encoded",
            "included_request_ids": [],
        }

    def test_returns_error_when_deliver_tx_is_none(self) -> None:
        """Test returns error when deliver tx is none."""
        b = self._make_b()
        b._synchronized_data.done_tasks = [
            {"is_marketplace_mech": False, "request_id": "r1"}
        ]
        usage_tx = {"to": "0xHASH", "value": 0, "data": b"\x00"}
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_mech_update_hash_tx", side_effect=_gen_returning(None)
            ),
            patch.object(b, "get_split_profit_txs", side_effect=_gen_returning([])),
            patch.object(
                b,
                "_get_offchain_tasks_deliver_data",
                side_effect=_gen_returning(([], [])),
            ),
            patch.object(
                b,
                "_get_marketplace_tasks_deliver_data",
                side_effect=_gen_returning(([], [])),
            ),
            patch.object(b, "_get_deliver_tx", side_effect=_gen_returning(None)),
            patch.object(
                b, "get_update_usage_tx", side_effect=_gen_returning(usage_tx)
            ),
        ):
            from packages.valory.skills.task_submission_abci.rounds import (
                TransactionPreparationRound,
            )

            result = _run_gen(b.get_payload_content())
        assert result == TransactionPreparationRound.ERROR_PAYLOAD

    def test_skips_deliver_with_failed_simulation(self) -> None:
        """Test skips deliver with failed simulation."""
        b = self._make_b()
        task = {"is_marketplace_mech": False, "request_id": "r1"}
        b._synchronized_data.done_tasks = [task]
        b.context.shared_state[DONE_TASKS] = [dict(task)]
        deliver_tx = {
            "to": "0xMECH",
            "value": 0,
            "data": b"\x00",
            "simulation_ok": False,
        }
        usage_tx = {"to": "0xHASH", "value": 0, "data": b"\x00"}
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_mech_update_hash_tx", side_effect=_gen_returning(None)
            ),
            patch.object(b, "get_split_profit_txs", side_effect=_gen_returning([])),
            patch.object(
                b,
                "_get_offchain_tasks_deliver_data",
                side_effect=_gen_returning(([], [])),
            ),
            patch.object(
                b,
                "_get_marketplace_tasks_deliver_data",
                side_effect=_gen_returning(([], [])),
            ),
            patch.object(
                b, "_get_deliver_tx", side_effect=_gen_returning(dict(deliver_tx))
            ),
            patch.object(
                b, "get_update_usage_tx", side_effect=_gen_returning(usage_tx)
            ),
            patch.object(b, "_to_multisend", side_effect=_gen_returning("encoded")),
            patch.object(b, "note_settlement_skipped") as mock_skip,
        ):
            result = _run_gen(b.get_payload_content())
        # deliver skipped but the vote still completes, with the skipped
        # id absent from the envelope so it is not pruned as settled. The
        # task is retried through the same cap as the other two paths.
        assert decode_tx_payload(result) == {
            "tx_hash": "encoded",
            "included_request_ids": [],
        }
        mock_skip.assert_called_once_with(
            ["r1"], SOURCE_ONCHAIN, "0xmech", b.synchronized_data.period_count
        )

    def test_appends_response_tx_when_present(self) -> None:
        """Test appends response tx when present."""
        b = self._make_b()
        response_tx = {"to": "0xRESP", "value": 0, "data": b"\x02"}
        task = {
            "is_marketplace_mech": False,
            "request_id": "r1",
            "transaction": response_tx,
        }
        b._synchronized_data.done_tasks = [task]
        deliver_tx = {
            "to": "0xMECH",
            "value": 0,
            "data": b"\x00",
            "simulation_ok": True,
        }
        usage_tx = {"to": "0xHASH", "value": 0, "data": b"\x00"}
        with (
            self._patch_sd(b),
            patch.object(
                b, "get_mech_update_hash_tx", side_effect=_gen_returning(None)
            ),
            patch.object(b, "get_split_profit_txs", side_effect=_gen_returning([])),
            patch.object(
                b,
                "_get_offchain_tasks_deliver_data",
                side_effect=_gen_returning(([], [])),
            ),
            patch.object(
                b,
                "_get_marketplace_tasks_deliver_data",
                side_effect=_gen_returning(([], [])),
            ),
            patch.object(
                b, "_get_deliver_tx", side_effect=_gen_returning(dict(deliver_tx))
            ),
            patch.object(
                b, "get_update_usage_tx", side_effect=_gen_returning(usage_tx)
            ),
            patch.object(b, "_to_multisend", side_effect=_gen_returning("encoded")),
        ):
            result = _run_gen(b.get_payload_content())
        assert decode_tx_payload(result) == {
            "tx_hash": "encoded",
            "included_request_ids": ["r1"],
        }


# ---------------------------------------------------------------------------
# _get_marketplace_tasks_deliver_data with NVM mech (lines 1801-1812)
# ---------------------------------------------------------------------------


class _DummyBaseNoMixin(TaskExecutionBaseBehaviour):
    """Dummy without mixin — for testing the real synchronized_data property."""

    matching_round: Type[AbstractRound] = _CAST_ROUND  # type: ignore

    def async_act(self) -> Generator[None, None, None]:
        yield from ()


class TestSynchronizedDataProperty:
    """Cover line 159: TaskExecutionBaseBehaviour.synchronized_data property."""

    def test_returns_synchronized_data_from_shared_state(self) -> None:
        """Test returns synchronized data from shared state."""
        ctx = _make_full_ctx()
        b = _DummyBaseNoMixin(name="b", skill_context=ctx)
        # ctx.state.synchronized_data is set in _make_full_ctx
        result = b.synchronized_data
        assert result is ctx.state.synchronized_data


class TestGetSplitProfitTxsTokenMech:
    """Cover lines 597-613, 630-631: token type mech path in get_split_profit_txs."""

    def _make_b(self) -> "_DummyFunds":
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_none_when_no_mech_addresses(self) -> None:
        """Test returns none when no mech addresses."""
        b = self._make_b()
        b.context.params.agent_mech_contract_addresses = []
        with patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None  # txs=[] → return None

    def test_token_mech_returns_none_when_token_address_none(self) -> None:
        """Test token mech returns none when token address none."""
        b = self._make_b()
        from packages.valory.skills.task_submission_abci.behaviours import (
            PAYMENT_TYPE_TOKEN,
        )

        token_type = bytes.fromhex(PAYMENT_TYPE_TOKEN)
        mech_info = (token_type, "0xTRACK", 1000)
        process_tx = {
            "to": "0xTRACK",
            "value": 0,
            "data": b"\x00",
            "simulation_ok": True,
        }
        split_funds = {"op0": 500}  # non-zero amount triggers token path
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(
                b,
                "_get_process_payment_tx",
                side_effect=_gen_returning(dict(process_tx)),
            ),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
            patch.object(b, "_get_token_address", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None

    def test_token_mech_success_with_token_transfer(self) -> None:
        """Test token mech success with token transfer."""
        b = self._make_b()
        from packages.valory.skills.task_submission_abci.behaviours import (
            PAYMENT_TYPE_TOKEN,
        )

        token_type = bytes.fromhex(PAYMENT_TYPE_TOKEN)
        mech_info = (token_type, "0xTRACK", 1000)
        process_tx = {
            "to": "0xTRACK",
            "value": 0,
            "data": b"\x00",
            "simulation_ok": True,
        }
        split_funds = {"op0": 500}
        token_transfer_tx = {"to": "0xMECH", "value": 0, "data": b"\x02"}
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(
                b,
                "_get_process_payment_tx",
                side_effect=_gen_returning(dict(process_tx)),
            ),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
            patch.object(
                b, "_get_token_address", side_effect=_gen_returning("0xTOKEN")
            ),
            patch.object(
                b,
                "_get_token_transfer_tx",
                side_effect=_gen_returning(token_transfer_tx),
            ),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is not None
        assert len(result) == 2  # process_tx + token_transfer_tx

    def test_fixed_price_token_usdc_routes_to_erc20_branch(self) -> None:
        """USDC mech (FixedPriceTokenUSDC paymentType) must take the ERC20 branch.

        Guards against the regression where an unrecognised token-payment
        paymentType falls through to _get_transfer_tx and encodes a native
        transfer, which reverts on-chain because the mech contract holds the
        token but no native balance.
        """
        b = self._make_b()
        from packages.valory.skills.task_submission_abci.behaviours import (
            PAYMENT_TYPE_TOKEN_USDC,
        )

        usdc_type = bytes.fromhex(PAYMENT_TYPE_TOKEN_USDC)
        mech_info = (usdc_type, "0xTRACK", 1000)
        process_tx = {
            "to": "0xTRACK",
            "value": 0,
            "data": b"\x00",
            "simulation_ok": True,
        }
        split_funds = {"0xSERVICE_OWNER": 900}
        token_transfer_tx = {"to": "0xMECH", "value": 0, "data": b"\x02"}
        native_transfer_sentinel = MagicMock()
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(
                b,
                "_get_process_payment_tx",
                side_effect=_gen_returning(dict(process_tx)),
            ),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
            patch.object(
                b, "_get_token_address", side_effect=_gen_returning("0xTOKEN")
            ),
            patch.object(
                b,
                "_get_token_transfer_tx",
                side_effect=_gen_returning(token_transfer_tx),
            ),
            patch.object(b, "_get_transfer_tx", native_transfer_sentinel),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is not None
        assert len(result) == 2  # process_tx + token_transfer_tx (not native)
        native_transfer_sentinel.assert_not_called()

    def test_unrecognised_payment_type_warns_before_native_fallback(self) -> None:
        """Unknown paymentType falls back to native transfer AND logs a warning.

        Guards against the regression where a new token rail is added to
        the ``PAYMENT_TYPE_*`` constants but forgotten in
        ``TOKEN_PAYMENT_TYPES`` — the withdraw path silently reverts
        on-chain with GS013. The warning surfaces the misconfiguration
        in agent logs before the on-chain revert lands.
        """
        b = self._make_b()
        unknown_type = bytes.fromhex(
            "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        )
        mech_info = (unknown_type, "0xTRACK", 1000)
        process_tx = {
            "to": "0xTRACK",
            "value": 0,
            "data": b"\x00",
            "simulation_ok": True,
        }
        split_funds = {"0xSERVICE_OWNER": 900}
        native_transfer_tx = {"to": "0xSERVICE_OWNER", "value": 900, "data": b""}
        warn_spy = MagicMock()
        b.context.logger.warning = warn_spy  # type: ignore[assignment]
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(
                b,
                "_get_process_payment_tx",
                side_effect=_gen_returning(dict(process_tx)),
            ),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
            patch.object(
                b, "_get_transfer_tx", side_effect=_gen_returning(native_transfer_tx)
            ),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is not None
        warn_spy.assert_called_once()
        warn_msg = warn_spy.call_args[0][0]
        assert "Unrecognised mech_type" in warn_msg
        assert unknown_type.hex() in warn_msg
        assert "TOKEN_PAYMENT_TYPES" in warn_msg

    def test_known_native_payment_type_does_not_warn(self) -> None:
        """Legitimate ``PAYMENT_TYPE_NATIVE`` mechs must not trip the warning.

        The unrecognised-mech-type warning is scoped to hashes outside
        both ``TOKEN_PAYMENT_TYPES`` and ``KNOWN_NATIVE_PAYMENT_TYPES``;
        a plain native rail is the intended path.
        """
        b = self._make_b()
        from packages.valory.skills.task_submission_abci.behaviours import (
            PAYMENT_TYPE_NATIVE,
        )

        native_type = bytes.fromhex(PAYMENT_TYPE_NATIVE)
        mech_info = (native_type, "0xTRACK", 1000)
        process_tx = {
            "to": "0xTRACK",
            "value": 0,
            "data": b"\x00",
            "simulation_ok": True,
        }
        split_funds = {"0xSERVICE_OWNER": 900}
        native_transfer_tx = {"to": "0xSERVICE_OWNER", "value": 900, "data": b""}
        warn_spy = MagicMock()
        b.context.logger.warning = warn_spy  # type: ignore[assignment]
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(
                b,
                "_get_process_payment_tx",
                side_effect=_gen_returning(dict(process_tx)),
            ),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
            patch.object(
                b, "_get_transfer_tx", side_effect=_gen_returning(native_transfer_tx)
            ),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is not None
        warn_spy.assert_not_called()


class TestMarketplaceNvmMechPath:
    """Test Marketplace Nvm Mech Path."""

    def _make_b(self) -> "_DummyTransPrep":
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        return b

    def test_nvm_mech_encodes_data(self) -> None:
        """Test nvm mech encodes data."""
        b = self._make_b()
        task = {
            "mech_address": "0xMECH",
            "request_id": 12345,
            "requestId": 12345,
            "task_result": "de" * 16,
        }
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        b._synchronized_data = mock_sd
        encoded_ids = [b"\x00" * 32]
        encoded_datas = [b"\xab" * 16]
        msg_deliver = _state_contract_msg({"data": b"\xdd", "simulation_ok": True})
        with (
            patch.object(b, "_get_is_nvm_mech", side_effect=_gen_returning(True)),
            patch.object(
                b,
                "_get_encoded_deliver_data",
                side_effect=_gen_returning((encoded_ids, encoded_datas)),
            ),
            patch.object(
                b, "get_contract_api_response", side_effect=_gen_returning(msg_deliver)
            ),
        ):
            tx_list, included = _run_gen(b._get_marketplace_tasks_deliver_data([task]))
        assert len(tx_list) == 1
        # NVM encoding rewrites the ABI ids; the reported ids stay the raw ones.
        assert included == ["12345"]


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour — _should_split_profits / _split_funds
# (previously in test_funds_split.py; conftest fixtures are inlined here)
# ---------------------------------------------------------------------------


class TestFundsSplittingBehaviourSplit:
    """Tests for FundsSplittingBehaviour._should_split_profits and _split_funds."""

    @pytest.fixture
    def fs_ctx(self) -> SimpleNamespace:
        """Test fs ctx."""
        return _make_fs_ctx()

    @pytest.fixture
    def fs_behaviour(self, fs_ctx: SimpleNamespace) -> _DummyFunds:
        """Test fs behaviour."""
        return _DummyFunds(name="fs", skill_context=fs_ctx)

    @pytest.fixture
    def patch_mech_info(
        self, monkeypatch: pytest.MonkeyPatch, fs_behaviour: _DummyFunds
    ) -> Callable[[Dict[str, int]], None]:
        """Test patch mech info."""

        def _apply(balances_by_addr: Dict[str, int]) -> None:
            def _fake(
                self: _DummyFunds, mech_address: str
            ) -> Generator[None, None, Optional[Tuple[bytes, str, int]]]:
                if False:
                    yield
                bal = balances_by_addr.get(mech_address)
                if bal is None:
                    return None
                return b"\x00", "0xBalanceTracker", bal

            monkeypatch.setattr(_DummyFunds, "_get_mech_info", _fake)

        return _apply

    @pytest.fixture(autouse=True)
    def _no_agent_deficits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default: no agent needs funding; individual tests override via monkeypatch."""

        def _stub(self: _DummyFunds) -> Generator[None, None, Optional[Dict[str, int]]]:
            if False:
                yield
            return {}

        monkeypatch.setattr(_DummyFunds, "_get_agent_funding_amounts", _stub)

    def test_split_true_at_exact_threshold(
        self, fs_behaviour: _DummyFunds, fs_ctx: SimpleNamespace, patch_mech_info: Any
    ) -> None:
        """Return True when the mech balance equals the threshold."""
        fs_ctx.params.profit_split_balance = 10
        fs_ctx.params.agent_mech_contract_addresses = ["0xA"]
        patch_mech_info({"0xA": 10})
        assert _run_gen(fs_behaviour._should_split_profits()) is True

    def test_split_false_below_threshold(
        self, fs_behaviour: _DummyFunds, fs_ctx: SimpleNamespace, patch_mech_info: Any
    ) -> None:
        """Return False when the mech balance is below the threshold."""
        fs_ctx.params.profit_split_balance = 10
        fs_ctx.params.agent_mech_contract_addresses = ["0xA"]
        patch_mech_info({"0xA": 9})
        assert _run_gen(fs_behaviour._should_split_profits()) is False

    def test_balance_logic_avoids_old_modulo_flakiness(
        self, fs_behaviour: _DummyFunds, fs_ctx: SimpleNamespace, patch_mech_info: Any
    ) -> None:
        """Threshold logic triggers once balance surpasses target (avoids 9→11 miss)."""
        fs_ctx.params.profit_split_balance = 10
        fs_ctx.params.agent_mech_contract_addresses = ["0xA"]
        patch_mech_info({"0xA": 11})
        assert _run_gen(fs_behaviour._should_split_profits()) is True

    def test_error_on_missing_mech_info_returns_false(
        self, fs_behaviour: _DummyFunds, fs_ctx: SimpleNamespace, patch_mech_info: Any
    ) -> None:
        """If a mech returns None from _get_mech_info, method returns False."""
        fs_ctx.params.profit_split_balance = 10
        fs_ctx.params.agent_mech_contract_addresses = ["0xMissing"]
        patch_mech_info({})
        assert _run_gen(fs_behaviour._should_split_profits()) is False

    def test_agent_dips_below_minimum_triggers_split_via_deficit(
        self,
        fs_behaviour: _DummyFunds,
        fs_ctx: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When an agent needs funding, _should_split_profits() returns True immediately."""
        ONE_XDAI = 10**18
        fs_ctx.params.minimum_agent_balance = 10**17
        fs_ctx.params.agent_funding_amount = 2 * 10**17
        fs_ctx.params.profit_split_balance = ONE_XDAI
        fs_ctx.params.agent_mech_contract_addresses = ["0xMECH"]

        calls: Dict[str, int] = {"funds": 0}

        def _get_agent_funding_amounts_deficit(
            self: Any,
        ) -> Generator[None, None, Optional[Dict[str, int]]]:
            calls["funds"] += 1
            if False:
                yield
            return {"0xAGENT": fs_ctx.params.agent_funding_amount}

        monkeypatch.setattr(
            type(fs_behaviour),
            "_get_agent_funding_amounts",
            _get_agent_funding_amounts_deficit,
        )

        assert _run_gen(fs_behaviour._should_split_profits()) is True
        assert calls["funds"] == 1

    def test_owner_share_rounding_wei_error(
        self,
        fs_behaviour: _DummyFunds,
        fs_ctx: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """int(0.1 * profits) != profits // 10 for large integers — code uses //."""
        profits: int = 1_000_000_000_000_003_139
        fs_ctx.params.service_owner_share = 1000
        fs_ctx.params.on_chain_service_id = 1

        def _no_deficits(self: Any) -> Generator[None, None, Optional[Dict[str, int]]]:
            if False:
                yield
            return {}

        def _owner(self: Any, _sid: int) -> Generator[None, None, Optional[str]]:
            if False:
                yield
            return "0xOWNER"

        def _ops(
            self: Any, operator_share: int
        ) -> Generator[None, None, Optional[Dict[str, int]]]:
            if False:
                yield
            return {"0xOP": operator_share}

        monkeypatch.setattr(
            type(fs_behaviour), "_get_agent_funding_amounts", _no_deficits
        )
        monkeypatch.setattr(type(fs_behaviour), "_get_service_owner", _owner)
        monkeypatch.setattr(type(fs_behaviour), "_get_funds_by_operator", _ops)

        split = _run_gen(fs_behaviour._split_funds(profits))
        assert split is not None
        assert split["0xOWNER"] == profits // 10

    def test_split_exact_allocation_no_deficits_bps(
        self,
        fs_behaviour: _DummyFunds,
        fs_ctx: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No agent deficits: owner gets bps share, operator gets remainder. Sum == profits."""
        profits: int = 1_000_000_000_000_003_139
        fs_ctx.params.service_owner_share = 1_000
        fs_ctx.params.on_chain_service_id = 1

        def _no_deficits(self: Any) -> Generator[None, None, Optional[Dict[str, int]]]:
            if False:
                yield
            return {}

        def _owner(self: Any, _sid: int) -> Generator[None, None, Optional[str]]:
            if False:
                yield
            return "0xOWNER"

        def _ops(
            self: Any, operator_share: int
        ) -> Generator[None, None, Optional[Dict[str, int]]]:
            if False:
                yield
            return {"0xOP": operator_share}

        monkeypatch.setattr(
            type(fs_behaviour), "_get_agent_funding_amounts", _no_deficits
        )
        monkeypatch.setattr(type(fs_behaviour), "_get_service_owner", _owner)
        monkeypatch.setattr(type(fs_behaviour), "_get_funds_by_operator", _ops)

        split = _run_gen(fs_behaviour._split_funds(profits))
        assert split is not None
        owner_expected = profits * fs_ctx.params.service_owner_share // 10_000
        operator_expected = profits - owner_expected
        assert split == {"0xOWNER": owner_expected, "0xOP": operator_expected}
        assert sum(split.values()) == profits

    def test_split_with_agent_deficits_exact_totals_bps(
        self,
        fs_behaviour: _DummyFunds,
        fs_ctx: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With agent deficits, agents funded first; remainder split owner/operator."""
        profits: int = 2_000_000_000_000_000_000
        fs_ctx.params.service_owner_share = 1_000
        fs_ctx.params.on_chain_service_id = 1

        agent_funding_amount = 200_000_000_000_000_000
        deficits_map: Dict[str, int] = {
            "0xA1": agent_funding_amount,
            "0xA2": agent_funding_amount,
        }
        total_deficits = sum(deficits_map.values())

        def _deficits(self: Any) -> Generator[None, None, Optional[Dict[str, int]]]:
            if False:
                yield
            return dict(deficits_map)

        def _owner(self: Any, _sid: int) -> Generator[None, None, Optional[str]]:
            if False:
                yield
            return "0xOWNER"

        def _ops(
            self: Any, operator_share: int
        ) -> Generator[None, None, Optional[Dict[str, int]]]:
            if False:
                yield
            return {"0xOP": operator_share}

        monkeypatch.setattr(type(fs_behaviour), "_get_agent_funding_amounts", _deficits)
        monkeypatch.setattr(type(fs_behaviour), "_get_service_owner", _owner)
        monkeypatch.setattr(type(fs_behaviour), "_get_funds_by_operator", _ops)

        split = _run_gen(fs_behaviour._split_funds(profits))
        assert split is not None
        remainder = profits - total_deficits
        owner_expected = remainder * fs_ctx.params.service_owner_share // 10_000
        operator_expected = remainder - owner_expected
        expected = dict(deficits_map)
        expected["0xOWNER"] = owner_expected
        expected["0xOP"] = operator_expected
        assert split == expected
        assert sum(split.values()) == profits

    def test_split_when_deficits_exceed_profits_proportional_only_to_agents(
        self,
        fs_behaviour: _DummyFunds,
        fs_ctx: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If total agent deficits > profits, split proportionally among agents only."""
        profits: int = 900
        deficits_map: Dict[str, int] = {"0xA": 600, "0xB": 600}

        def _deficits(self: Any) -> Generator[None, None, Optional[Dict[str, int]]]:
            if False:
                yield
            return dict(deficits_map)

        def _owner(self: Any, _sid: int) -> Generator[None, None, Optional[str]]:
            if False:
                yield
            return "0xOWNER"

        monkeypatch.setattr(type(fs_behaviour), "_get_agent_funding_amounts", _deficits)
        monkeypatch.setattr(type(fs_behaviour), "_get_service_owner", _owner)

        split = _run_gen(fs_behaviour._split_funds(profits))
        assert split is not None
        total_need = sum(deficits_map.values())
        expected_a = (deficits_map["0xA"] * profits) // total_need
        expected_b = (deficits_map["0xB"] * profits) // total_need
        assert "0xOWNER" not in split
        assert "0xOP" not in split
        assert split == {"0xA": expected_a, "0xB": expected_b}
        assert sum(split.values()) <= profits


# ---------------------------------------------------------------------------
# Tests for async_act: TaskPoolingBehaviour and TransactionPreparationBehaviour
# ---------------------------------------------------------------------------


class TestTaskPoolingBehaviourAsyncAct:
    """Drive TaskPoolingBehaviour.async_act through the real generator body."""

    def test_async_act_runs_to_done(self) -> None:
        """async_act builds a payload and marks itself done."""
        ctx = _make_benchmark_ctx()
        b = TaskPoolingBehaviour(name="b", skill_context=ctx)

        def _payload_content() -> Generator[None, None, str]:
            yield from ()
            return json.dumps([])

        with (
            patch.object(b, "handle_submitted_tasks", side_effect=_noop_gen),
            patch.object(b, "get_payload_content", side_effect=_payload_content),
            patch.object(b, "send_a2a_transaction", side_effect=_noop_gen_with_args),
            patch.object(b, "wait_until_round_end", side_effect=_noop_gen),
        ):
            _run_gen(b.async_act())

        assert b.is_done()


class TestTransactionPreparationBehaviourAsyncAct:
    """Drive TransactionPreparationBehaviour.async_act through the real generator body."""

    def test_async_act_runs_to_done(self) -> None:
        """async_act builds a transaction payload and marks itself done."""
        ctx = _make_benchmark_ctx()
        b = TransactionPreparationBehaviour(name="b", skill_context=ctx)

        def _tx_hash() -> Generator[None, None, str]:
            yield from ()
            return "some_tx_hash"

        with (
            patch.object(b, "get_payload_content", side_effect=_tx_hash),
            patch.object(b, "send_a2a_transaction", side_effect=_noop_gen_with_args),
            patch.object(b, "wait_until_round_end", side_effect=_noop_gen),
        ):
            _run_gen(b.async_act())

        assert b.is_done()


# ---------------------------------------------------------------------------
# Contract API error-path tests
# ---------------------------------------------------------------------------


class TestFundsSplittingContractErrors:
    """Test contract API error branches in FundsSplittingBehaviour methods."""

    def _make_b(self) -> "_DummyFunds":
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        b._synchronized_data = SimpleNamespace(
            safe_contract_address="0xSAFE",
            all_participants=frozenset(["0xA"]),
            done_tasks=[],
        )
        return b

    def test_get_mech_payment_type_contract_error(self) -> None:
        """Error from get_mech_type returns None."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._get_mech_payment_type("0xOTHER"))
        assert result is None

    def test_get_balance_tracker_address_contract_error(self) -> None:
        """Error from get_balance_tracker_for_mech_type returns None."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._get_balance_tracker_address(b"\x00" * 32))
        assert result is None

    def test_get_mech_info_contract_error_on_balance(self) -> None:
        """Error from get_mech_balance (3rd call in _get_mech_info) returns None."""
        b = self._make_b()
        with (
            patch.object(
                b, "_get_mech_payment_type", side_effect=_gen_returning(b"\x00" * 32)
            ),
            patch.object(
                b, "_get_balance_tracker_address", side_effect=_gen_returning("0xTRACK")
            ),
            patch.object(
                b,
                "get_contract_api_response",
                side_effect=_gen_returning(_error_contract_msg()),
            ),
        ):
            result = _run_gen(b._get_mech_info("0xMECH"))
        assert result is None

    def test_adjust_mech_balance_contract_error(self) -> None:
        """Error from get_token_credit_ratio returns None."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._adjust_mech_balance("0xTRACKER", 100))
        assert result is None

    def test_get_process_payment_tx_contract_error(self) -> None:
        """Error from get_process_payment_tx returns None."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._get_process_payment_tx("0xMECH", "0xTRACK"))
        assert result is None

    def test_get_fee_contract_error(self) -> None:
        """Error from get_fee returns None."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._get_fee())
        assert result is None

    def test_get_max_fee_factor_contract_error(self) -> None:
        """Error from get_max_fee_factor returns None."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._get_max_fee_factor("0xTRACK"))
        assert result is None

    def test_get_transfer_tx_contract_error(self) -> None:
        """Error from get_exec_tx_data returns None."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._get_transfer_tx("0xMECH", "0xRECEIVER", 100))
        assert result is None

    def test_get_token_address_contract_error(self) -> None:
        """Error from get_token_address returns None."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._get_token_address("0xTRACK"))
        assert result is None

    def test_get_token_transfer_tx_data_contract_error(self) -> None:
        """Error from get_transfer_tx_data returns None."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(
                b._get_token_transfer_tx_data("0xTOKEN", "0xRECEIVER", 100)
            )
        assert result is None

    def test_get_token_transfer_tx_contract_error_on_exec(self) -> None:
        """Error from get_exec_tx_data (2nd call in _get_token_transfer_tx) returns None."""
        b = self._make_b()
        with (
            patch.object(
                b, "_get_token_transfer_tx_data", side_effect=_gen_returning(b"\xaa")
            ),
            patch.object(
                b,
                "get_contract_api_response",
                side_effect=_gen_returning(_error_contract_msg()),
            ),
        ):
            result = _run_gen(
                b._get_token_transfer_tx("0xMECH", "0xTOKEN", "0xRECEIVER", 100)
            )
        assert result is None

    def test_accumulate_reqs_by_operator_contract_error(self) -> None:
        """Error from get_operators_mapping returns None."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._accumulate_reqs_by_operator({"0xA": 5}))
        assert result is None


class TestTransPrepContractErrors:
    """Test contract API error branches in TransactionPreparationBehaviour methods."""

    def _make_b(self) -> "_DummyTransPrep":
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        b._synchronized_data = SimpleNamespace(
            safe_contract_address="0xSAFE",
            all_participants=frozenset(["0xA"]),
            done_tasks=[],
        )
        return b

    def test_get_checkpoint_tx_contract_error(self) -> None:
        """Error from get_checkpoint_data returns None."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._get_checkpoint_tx("0xHASH", "ab" * 16))
        assert result is None

    def test_get_latest_hash_contract_error(self) -> None:
        """Error from get_token_hash returns None."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._get_latest_hash())
        assert result is None

    def test_get_mech_update_hash_tx_contract_error(self) -> None:
        """Error from get_update_hash_tx_data returns None."""
        b = self._make_b()
        # _should_update_hash needs to return True, so make latest_metadata_hash differ
        b.context.params.task_mutable_params.latest_metadata_hash = b"\x00" * 32
        with (
            _mock_to_multihash() as mtm,
            patch.object(
                b,
                "get_contract_api_response",
                side_effect=_gen_returning(_error_contract_msg()),
            ),
        ):
            mtm.return_value = "ab" * 16
            result = _run_gen(b.get_mech_update_hash_tx())
        assert result is None

    def test_get_agent_mech_deliver_tx_contract_error(self) -> None:
        """Error from get_deliver_data returns None."""
        b = self._make_b()
        task_data = {
            "mech_address": "0xMECH",
            "request_id": "r1",
            "task_result": "deadbeef",
            "request_id_nonce": 0,
        }
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._get_agent_mech_deliver_tx(task_data))
        assert result is None

    def test_get_deliver_marketplace_tx_contract_error(self) -> None:
        """Error from get_deliver_to_market_tx returns None."""
        b = self._make_b()
        task_data = {
            "mech_address": "0xMECH",
            "request_id": "r1",
            "task_result": "deadbeef",
        }
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._get_deliver_marketplace_tx(task_data))
        assert result is None

    def test_get_offchain_tasks_deliver_data_contract_error(self) -> None:
        """Error from get_offchain_deliver_data returns None."""
        b = self._make_b()
        offchain_task = {
            "request_id": "r1",
            IS_OFFCHAIN: True,
            NONCE: 1,
            SENDER: "0xSENDER",
            OffchainDataValue.IPFS_HASH.value: "0x" + "aa" * 32,
            OffchainDataValue.SIGNATURE.value: "0x" + "bb" * 65,
            OffchainDataValue.TASK_RESULT.value: "cc" * 16,
            OffchainDataValue.DELIVERY_RATE.value: "10",
        }
        b._synchronized_data = SimpleNamespace(
            safe_contract_address="0xSAFE",
            done_tasks=[offchain_task],
        )
        b.context.shared_state[DONE_TASKS] = [dict(offchain_task)]
        with (
            patch.object(
                b,
                "get_contract_api_response",
                side_effect=_gen_returning(_error_contract_msg()),
            ),
            patch.object(b, "count_settlement") as mock_count,
        ):
            result = _run_gen(b._get_offchain_tasks_deliver_data())
        assert result == (None, [])
        # The whole off-chain batch is abandoned this period: count every task.
        mock_count.assert_called_once_with(
            SETTLEMENT_OUTCOME_CONTRACT_ERROR, SOURCE_OFFCHAIN, "0xMECH", 1
        )

    def test_get_is_nvm_mech_contract_error(self) -> None:
        """Error from get_is_nvm_mech returns None."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._get_is_nvm_mech("0xMECH"))
        assert result is None

    def test_get_encoded_deliver_data_contract_error(self) -> None:
        """Error from get_encoded_data_for_request returns (None, None)."""
        b = self._make_b()
        with patch.object(
            b,
            "get_contract_api_response",
            side_effect=_gen_returning(_error_contract_msg()),
        ):
            result = _run_gen(b._get_encoded_deliver_data([b"\x01"], [b"\x02"], [10]))
        assert result == (None, None)

    def test_get_marketplace_tasks_deliver_data_contract_error(self) -> None:
        """Error from get_marketplace_deliver_data returns None."""
        b = self._make_b()
        marketplace_tasks = [
            {
                MECH_ADDRESS: "0xMECH",
                "request_id": 1,
                MarketplaceData.REQUEST_ID.value: 1,
                MarketplaceData.TASK_RESULT.value: "aa" * 16,
            }
        ]
        # _get_is_nvm_mech must succeed (returning False so we skip encoding)
        # then get_contract_api_response for get_marketplace_deliver_data must fail
        call_count = 0

        def _side_effect(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: _get_is_nvm_mech → STATE with is_nvm=False
                return _gen_returning(_state_contract_msg({"data": False}))(
                    *args, **kwargs
                )
            # Second call: get_marketplace_deliver_data → ERROR
            return _gen_returning(_error_contract_msg())(*args, **kwargs)

        b.context.shared_state[DONE_TASKS] = [dict(marketplace_tasks[0])]
        with (
            patch.object(b, "get_contract_api_response", side_effect=_side_effect),
            patch.object(b, "count_settlement") as mock_count,
        ):
            result = _run_gen(b._get_marketplace_tasks_deliver_data(marketplace_tasks))
        assert result == (None, [])
        mock_count.assert_called_once_with(
            SETTLEMENT_OUTCOME_CONTRACT_ERROR, SOURCE_ONCHAIN, "0xMECH", 1
        )


# ---------------------------------------------------------------------------
# Contract-helper bytes-return invariant
# ---------------------------------------------------------------------------


class TestContractHelpersReturnBytes:
    """Pin the bytes-return invariant on tx-building contract helpers.

    Regression guard for the open-autonomy 0.21.18 refactor that removed the
    HexBytes(data) coercion inside multisend.encode_data. Both helpers below
    used to return the raw encode_abi str; the skill-level tests mock
    get_contract_api_response at the dispatcher boundary and do not exercise
    these helpers' real return shapes, so without this class a future revert
    of bytes.fromhex(data[2:]) would not be caught by CI.

    Lives here rather than under packages/valory/contracts/<name>/tests/
    because first-party contracts do not have their own tests/ infrastructure
    today; introducing it for only two contracts would establish a new
    convention mid-repo.
    """

    _SAMPLE_ENCODE_ABI = "0x5b34eba0" + "ab" * 32

    @staticmethod
    def _mock_contract_instance() -> MagicMock:
        instance = MagicMock()
        instance.encode_abi.return_value = (
            TestContractHelpersReturnBytes._SAMPLE_ENCODE_ABI
        )
        return instance

    def test_hash_checkpoint_get_checkpoint_data_returns_bytes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HashCheckpointContract.get_checkpoint_data must return data as bytes."""
        monkeypatch.setattr(
            HashCheckpointContract,
            "get_instance",
            classmethod(lambda cls, *_, **__: self._mock_contract_instance()),
        )
        result = HashCheckpointContract.get_checkpoint_data(
            ledger_api=MagicMock(spec=EthereumApi),
            contract_address="0x0000000000000000000000000000000000000001",
            data=b"\xab" * 32,
        )
        assert isinstance(result["data"], bytes)
        assert result["data"].hex() == self._SAMPLE_ENCODE_ABI[2:]

    def test_complementary_service_metadata_get_update_hash_tx_data_returns_bytes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ComplementaryServiceMetadata.get_update_hash_tx_data must return data as bytes."""
        monkeypatch.setattr(
            ComplementaryServiceMetadata,
            "get_instance",
            classmethod(lambda cls, *_, **__: self._mock_contract_instance()),
        )
        result = ComplementaryServiceMetadata.get_update_hash_tx_data(
            ledger_api=MagicMock(spec=EthereumApi),
            contract_address="0x0000000000000000000000000000000000000002",
            service_id=42,
            metadata_hash=b"\xcd" * 32,
        )
        assert isinstance(result["data"], bytes)
        assert result["data"].hex() == self._SAMPLE_ENCODE_ABI[2:]


# ---------------------------------------------------------------------------
# Durable preimage stamps: the settlement skill marks each follow-up step on
# the off-chain record so a restart never replays work that already landed.
# ---------------------------------------------------------------------------

from packages.valory.skills.task_execution.utils import preimage  # noqa: E402

STAMP_REQUEST_ID = "r1"
STAMP_TX_HASH = "0x" + "cd" * 32
STAMP_SENDER = "0xSENDER"
STAMP_NONCE = 7


def _delivered_preimage_state(shared_state: Dict[str, Any]) -> None:
    """Switch retention on and seed one delivered record for ``STAMP_REQUEST_ID``."""
    preimage.init_shared_state(shared_state, retention_enabled=True)
    preimage.record_settlement(
        shared_state, STAMP_REQUEST_ID, "{}", "cid", preimage.STATUS_DELIVERED, 1.0
    )
    shared_state[preimage.PREIMAGE_WRITE_QUEUE].clear()


class TestRetryCapStampsAbandoned:
    """The tx-prep retry cap marks the dropped deliver on its preimage record."""

    PERIOD = 3

    def _make_b(self, attempts: int, offchain: bool = True) -> "_DummyTransPrep":
        task = {
            "request_id": STAMP_REQUEST_ID,
            IS_OFFCHAIN: offchain,
            SENDER: STAMP_SENDER,
            NONCE: STAMP_NONCE,
            SETTLEMENT_ATTEMPTS_KEY: attempts,
        }
        ctx = _make_full_ctx(done_tasks=[task])
        ctx.shared_state[SETTLING_NONCES_BY_SENDER] = {STAMP_SENDER: {STAMP_NONCE}}
        _delivered_preimage_state(ctx.shared_state)
        return _DummyTransPrep(name="b", skill_context=ctx)

    def _skip(self, b: Any) -> None:
        with patch.object(b, "count_settlement"):
            b.note_settlement_skipped(
                [STAMP_REQUEST_ID], SOURCE_OFFCHAIN, "0xMECH", self.PERIOD
            )

    def test_dropped_task_is_stamped_abandoned(self) -> None:
        """Reaching the cap stamps ``abandoned_at`` so the drainer never re-queues it."""
        b = self._make_b(attempts=MAX_SETTLEMENT_ATTEMPTS - 1)
        self._skip(b)
        record = b.context.shared_state[preimage.PREIMAGE_RECORDS][STAMP_REQUEST_ID]
        assert b.context.shared_state[DONE_TASKS] == []
        assert isinstance(record[preimage.FIELD_ABANDONED_AT], int)
        assert preimage.is_complete(record, require_posted=True) is True

    def test_onchain_drop_parks_nothing(self) -> None:
        """An on-chain task has no preimage row, so its drop never parks a stamp."""
        b = self._make_b(attempts=MAX_SETTLEMENT_ATTEMPTS - 1, offchain=False)
        b.context.shared_state[preimage.PREIMAGE_RECORDS].clear()
        self._skip(b)
        assert b.context.shared_state[DONE_TASKS] == []
        assert b.context.shared_state[preimage.PREIMAGE_PENDING_STAMPS] == {}

    def test_retried_task_is_not_stamped(self) -> None:
        """Below the cap the task stays queued and the record stays replayable."""
        b = self._make_b(attempts=0)
        self._skip(b)
        record = b.context.shared_state[preimage.PREIMAGE_RECORDS][STAMP_REQUEST_ID]
        assert len(b.context.shared_state[DONE_TASKS]) == 1
        assert record[preimage.FIELD_ABANDONED_AT] is None


class TestPostPredictApiBatchStampsPosted:
    """A 2xx on the delivered / replay batch stamps ``posted_at`` per event."""

    CHAIN_ID = 100

    @staticmethod
    def _event(request_id: str) -> Dict[str, Any]:
        return {
            "request": {"request_id": request_id, "chain_id": 100},
            "response": {"request_id": request_id, "delivery_tx_hash": None},
        }

    def _make_self(self, status_code: int) -> SimpleNamespace:
        shared_state: Dict[str, Any] = {}
        _delivered_preimage_state(shared_state)

        def _get_signature(*_a: Any, **_k: Any) -> Generator[None, None, str]:
            if False:  # pragma: no cover - generator shape only
                yield None
            return "0x" + "ab" * 65

        def _do_request(*_a: Any, **_k: Any) -> Generator[None, None, Any]:
            if False:  # pragma: no cover - generator shape only
                yield None
            return SimpleNamespace(status_code=status_code, body=b"{}")

        self_ = SimpleNamespace(
            context=SimpleNamespace(
                shared_state=shared_state,
                logger=SimpleNamespace(
                    info=lambda *a, **k: None,
                    warning=lambda *a, **k: None,
                    error=lambda *a, **k: None,
                    debug=lambda *a, **k: None,
                ),
                agent_address="0xAGENT",
            ),
            params=SimpleNamespace(predict_api_events_timeout_seconds=5.0),
            get_signature=_get_signature,
            _do_request=_do_request,
            _build_http_request_message=lambda **_k: (object(), object()),
            _drop_swept_from_pending=lambda _ids: None,
        )
        # Bind the real isolation helper so the unbound-method call resolves.
        self_._isolate_rejected_replay = lambda events, status: (
            beh_mod.PostTxSettlementBehaviour._isolate_rejected_replay(
                cast(beh_mod.PostTxSettlementBehaviour, self_), events, status
            )
        )
        return self_

    def _post(
        self,
        self_: SimpleNamespace,
        batch_label: str,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        gen = beh_mod.PostTxSettlementBehaviour._post_predict_api_batch(
            cast(beh_mod.PostTxSettlementBehaviour, self_),
            events=events if events is not None else [self._event(STAMP_REQUEST_ID)],
            mech_address="0x" + "11" * 20,  # EIP-712 encoder needs real addresses
            verifying_contract="0x" + "22" * 20,
            predict_api_url="https://example.invalid/events",
            batch_label=batch_label,
            swept_request_ids=None,
        )
        _run_gen(gen)

    @pytest.mark.parametrize(
        "batch_label", [beh_mod.BATCH_LABEL_DELIVERED, beh_mod.BATCH_LABEL_REPLAY]
    )
    def test_2xx_stamps_posted_at(self, batch_label: str) -> None:
        """Delivered and replay batches stamp the record on success."""
        self_ = self._make_self(status_code=200)
        self._post(self_, batch_label)
        record = self_.context.shared_state[preimage.PREIMAGE_RECORDS][STAMP_REQUEST_ID]
        assert isinstance(record[preimage.FIELD_POSTED_AT], int)
        assert self_.context.shared_state[preimage.PREIMAGE_WRITE_QUEUE] == [
            STAMP_REQUEST_ID
        ]

    def test_sweep_batch_never_stamps(self) -> None:
        """Request-only sweep events are not deliveries; no posted stamp."""
        self_ = self._make_self(status_code=200)
        self._post(self_, beh_mod.BATCH_LABEL_SWEEP)
        record = self_.context.shared_state[preimage.PREIMAGE_RECORDS][STAMP_REQUEST_ID]
        assert record[preimage.FIELD_POSTED_AT] is None

    def test_4xx_on_a_multi_event_replay_shrinks_the_next_batch(self) -> None:
        """The server does not say which event it refused, so isolate by shrinking."""
        self_ = self._make_self(status_code=422)
        self._post(
            self_, beh_mod.BATCH_LABEL_REPLAY, [self._event("a"), self._event("b")]
        )
        ss = self_.context.shared_state
        assert ss[beh_mod.PREDICT_API_REPLAY_SHRINK] is True
        assert (
            ss[preimage.PREIMAGE_RECORDS][STAMP_REQUEST_ID][preimage.FIELD_ABANDONED_AT]
            is None
        )

    def test_4xx_on_a_single_event_replay_marks_that_row_abandoned(self) -> None:
        """A refused one-event batch identifies the culprit; it stops blocking newer rows."""
        self_ = self._make_self(status_code=422)
        self._post(self_, beh_mod.BATCH_LABEL_REPLAY)
        record = self_.context.shared_state[preimage.PREIMAGE_RECORDS][STAMP_REQUEST_ID]
        assert isinstance(record[preimage.FIELD_ABANDONED_AT], int)
        assert preimage.is_complete(record, require_posted=True) is True

    def test_2xx_on_replay_clears_the_shrink_flag(self) -> None:
        """Once a replay batch lands, the batch size goes back to the configured value."""
        self_ = self._make_self(status_code=200)
        self_.context.shared_state[beh_mod.PREDICT_API_REPLAY_SHRINK] = True
        self._post(self_, beh_mod.BATCH_LABEL_REPLAY)
        assert beh_mod.PREDICT_API_REPLAY_SHRINK not in self_.context.shared_state

    @pytest.mark.parametrize(
        "batch_label, status_code",
        [
            (beh_mod.BATCH_LABEL_REPLAY, 503),  # retryable: no isolation
            (beh_mod.BATCH_LABEL_DELIVERED, 422),  # delivered batch is not replayed
        ],
    )
    def test_other_failures_do_not_isolate(
        self, batch_label: str, status_code: int
    ) -> None:
        """Only a 4xx on the replay batch triggers isolation."""
        self_ = self._make_self(status_code=status_code)
        self._post(self_, batch_label, [self._event("a"), self._event("b")])
        assert beh_mod.PREDICT_API_REPLAY_SHRINK not in self_.context.shared_state

    @pytest.mark.parametrize("status_code", [500, 422, 302])
    def test_non_2xx_never_stamps(self, status_code: int) -> None:
        """A rejected or failed POST leaves the record replayable."""
        self_ = self._make_self(status_code=status_code)
        self._post(self_, beh_mod.BATCH_LABEL_DELIVERED)
        record = self_.context.shared_state[preimage.PREIMAGE_RECORDS][STAMP_REQUEST_ID]
        assert record[preimage.FIELD_POSTED_AT] is None
        assert self_.context.shared_state[preimage.PREIMAGE_WRITE_QUEUE] == []


@pytest.mark.parametrize(
    "event, expected",
    [
        ({"request_id": 5, "response": {"request_id": "9"}}, "5"),
        ({"response": {"request_id": 9}}, "9"),
        ({"request": {"request_id": "3"}, "response": "not-a-dict"}, "3"),
        ({"request": "not-a-dict", "response": {}}, None),
        ({}, None),
    ],
)
def test_event_request_id_resolution_order(
    event: Dict[str, Any], expected: Any
) -> None:
    """Top-level id wins, then ``response``, then ``request``; non-dicts are skipped."""
    assert beh_mod._event_request_id(event) == expected


class TestReplayEventsFromPreimage:
    """``_replay_events_from_preimage`` re-sends settled-but-unposted rows."""

    def _make_self(
        self, shared_state: Dict[str, Any], batch_size: int = 50
    ) -> SimpleNamespace:
        infos: List[str] = []
        return SimpleNamespace(
            context=SimpleNamespace(
                shared_state=shared_state,
                logger=SimpleNamespace(
                    info=lambda msg, *a, **k: infos.append(msg % a if a else msg),
                    warning=lambda *a, **k: None,
                    debug=lambda *a, **k: None,
                    error=lambda *a, **k: None,
                ),
            ),
            params=SimpleNamespace(predict_api_replay_batch_size=batch_size),
            _infos=infos,
        )

    def test_events_are_copied_and_stamped_with_their_own_tx_hash(self) -> None:
        """Each replayed event carries the record's settlement hash; the record is untouched."""
        shared_state: Dict[str, Any] = {}
        _delivered_preimage_state(shared_state)
        record = shared_state[preimage.PREIMAGE_RECORDS][STAMP_REQUEST_ID]
        record[preimage.FIELD_SETTLED_TX_HASH] = STAMP_TX_HASH
        record[preimage.FIELD_PREDICT_API_EVENT] = {
            "response": {"request_id": STAMP_REQUEST_ID, "delivery_tx_hash": None}
        }
        self_ = self._make_self(shared_state)
        events = beh_mod.PostTxSettlementBehaviour._replay_events_from_preimage(
            cast(beh_mod.PostTxSettlementBehaviour, self_)
        )
        assert events == [
            {
                "response": {
                    "request_id": STAMP_REQUEST_ID,
                    "delivery_tx_hash": STAMP_TX_HASH,
                }
            }
        ]
        assert (
            record[preimage.FIELD_PREDICT_API_EVENT]["response"]["delivery_tx_hash"]
            is None
        )
        assert self_._infos == [
            "predict-api replay: 1 settled-but-unposted event(s) recovered from "
            "the preimage store."
        ]

    def test_no_rows_means_no_events_and_no_log(self) -> None:
        """An empty store is a quiet no-op."""
        shared_state: Dict[str, Any] = {}
        preimage.init_shared_state(shared_state, retention_enabled=True)
        self_ = self._make_self(shared_state)
        assert (
            beh_mod.PostTxSettlementBehaviour._replay_events_from_preimage(
                cast(beh_mod.PostTxSettlementBehaviour, self_)
            )
            == []
        )
        assert self_._infos == []


class TestCountSettledStampsSettlement:
    """``count_settled_from_synced_data`` stamps ``settled_tx_hash`` on own off-chain rows."""

    _ME = "0xSELF"

    def _make_self(self, done_tasks: List[Dict[str, Any]], included: List[str]) -> Any:
        shared_state: Dict[str, Any] = {}
        _delivered_preimage_state(shared_state)
        return SimpleNamespace(
            synchronized_data=SimpleNamespace(
                done_tasks=done_tasks,
                tx_included_request_ids=included,
                final_tx_hash=STAMP_TX_HASH,
            ),
            context=SimpleNamespace(agent_address=self._ME, shared_state=shared_state),
            count_settlement=MagicMock(),
            metrics_mech_label=lambda: "0xLABEL",
        )

    def _record(self, self_: Any) -> Dict[str, Any]:
        return self_.context.shared_state[preimage.PREIMAGE_RECORDS][STAMP_REQUEST_ID]

    def test_own_offchain_task_in_the_tx_is_stamped_in_the_confirming_round(
        self,
    ) -> None:
        """The stamp lands as soon as the tx confirms, not a period later."""
        task = {
            "request_id": STAMP_REQUEST_ID,
            "task_executor_address": self._ME,
            IS_OFFCHAIN: True,
        }
        self_ = self._make_self([task], included=[STAMP_REQUEST_ID])
        beh_mod.PostTxSettlementBehaviour.count_settled_from_synced_data(self_)
        assert self._record(self_)[preimage.FIELD_SETTLED_TX_HASH] == STAMP_TX_HASH
        assert self_.context.shared_state[preimage.PREIMAGE_WRITE_QUEUE] == [
            STAMP_REQUEST_ID
        ]

    @pytest.mark.parametrize(
        "task, included",
        [
            (
                {"request_id": STAMP_REQUEST_ID, "task_executor_address": "0xSELF"},
                [STAMP_REQUEST_ID],
            ),
            (
                {
                    "request_id": STAMP_REQUEST_ID,
                    "task_executor_address": "0xOTHER",
                    IS_OFFCHAIN: True,
                },
                [STAMP_REQUEST_ID],
            ),
            (
                {
                    "request_id": STAMP_REQUEST_ID,
                    "task_executor_address": "0xSELF",
                    IS_OFFCHAIN: True,
                },
                [],
            ),
        ],
        ids=["onchain", "other-agent", "not-in-tx"],
    )
    def test_other_tasks_are_not_stamped(
        self, task: Dict[str, Any], included: List[str]
    ) -> None:
        """On-chain, other agents' and not-yet-settled tasks leave the record alone."""
        self_ = self._make_self([task], included=included)
        beh_mod.PostTxSettlementBehaviour.count_settled_from_synced_data(self_)
        assert self._record(self_)[preimage.FIELD_SETTLED_TX_HASH] is None
        assert self_.context.shared_state[preimage.PREIMAGE_PENDING_STAMPS] == {}

    def test_absent_row_parks_the_stamp(self) -> None:
        """After a restart the row is not in memory yet; the stamp waits for hydrate."""
        task = {
            "request_id": "r-late",
            "task_executor_address": self._ME,
            IS_OFFCHAIN: True,
        }
        self_ = self._make_self([task], included=["r-late"])
        beh_mod.PostTxSettlementBehaviour.count_settled_from_synced_data(self_)
        assert self_.context.shared_state[preimage.PREIMAGE_PENDING_STAMPS] == {
            "r-late": {preimage.FIELD_SETTLED_TX_HASH: STAMP_TX_HASH}
        }


def test_replay_batch_is_capped_by_the_configured_size() -> None:
    """Only ``predict_api_replay_batch_size`` events go out per round, oldest first."""
    shared_state: Dict[str, Any] = {}
    preimage.init_shared_state(shared_state, retention_enabled=True)
    for rid, settled_at in (("c", 3.0), ("a", 1.0), ("b", 2.0)):
        preimage.record_settlement(
            shared_state, rid, "{}", "cid", preimage.STATUS_DELIVERED, settled_at
        )
        record = shared_state[preimage.PREIMAGE_RECORDS][rid]
        record[preimage.FIELD_SETTLED_TX_HASH] = "0x" + rid
        record[preimage.FIELD_PREDICT_API_EVENT] = {"response": {"request_id": rid}}
    self_ = TestReplayEventsFromPreimage()._make_self(shared_state, batch_size=2)
    events = beh_mod.PostTxSettlementBehaviour._replay_events_from_preimage(
        cast(beh_mod.PostTxSettlementBehaviour, self_)
    )
    assert [e["response"]["request_id"] for e in events] == ["a", "b"]
    # After a rejected replay batch the next one is a single event.
    shared_state[beh_mod.PREDICT_API_REPLAY_SHRINK] = True
    events = beh_mod.PostTxSettlementBehaviour._replay_events_from_preimage(
        cast(beh_mod.PostTxSettlementBehaviour, self_)
    )
    assert [e["response"]["request_id"] for e in events] == ["a"]
