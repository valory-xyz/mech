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

import threading
import time
from types import SimpleNamespace
from typing import Any, Generator, List, Optional, Type
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.task_submission_abci.behaviours import (
    DONE_TASKS,
    DONE_TASKS_LOCK,
    LAST_TX,
    PAYMENT_MODEL,
    ZERO_ADDRESS,
    ZERO_IPFS_HASH,
    DeliverBehaviour,
    FundsSplittingBehaviour,
    TaskExecutionBaseBehaviour,
    TaskPoolingBehaviour,
    TaskSubmissionRoundBehaviour,
    TransactionPreparationBehaviour,
)


# ---------------------------------------------------------------------------
# Helpers: concrete subclasses + minimal skill context
# ---------------------------------------------------------------------------

_CAST_ROUND = type("DummyRound", (), {})


class _DummyBase(TaskExecutionBaseBehaviour):
    """Minimal concrete subclass for testing TaskExecutionBaseBehaviour."""

    matching_round: Type[AbstractRound] = _CAST_ROUND  # type: ignore

    def async_act(self) -> Generator[None, None, None]:
        if False:  # pragma: no cover
            yield
        return None


class _DummyPooling(TaskPoolingBehaviour):
    """Minimal concrete subclass for testing TaskPoolingBehaviour."""

    matching_round: Type[AbstractRound] = _CAST_ROUND  # type: ignore

    def async_act(self) -> Generator[None, None, None]:
        if False:  # pragma: no cover
            yield
        return None


class _DummyDeliver(DeliverBehaviour):
    """Minimal concrete subclass for testing DeliverBehaviour."""

    matching_round: Type[AbstractRound] = _CAST_ROUND  # type: ignore

    def async_act(self) -> Generator[None, None, None]:
        if False:  # pragma: no cover
            yield
        return None


class _DummyFunds(FundsSplittingBehaviour):
    """Minimal concrete subclass for testing FundsSplittingBehaviour."""

    matching_round: Type[AbstractRound] = _CAST_ROUND  # type: ignore

    def async_act(self) -> Generator[None, None, None]:
        if False:  # pragma: no cover
            yield
        return None


def _make_lock() -> threading.Lock:
    return threading.Lock()


def _make_ctx(
    done_tasks=None,
    payment_model=None,
    lock=None,
    agent_mech_addresses=None,
) -> SimpleNamespace:
    if lock is None:
        lock = _make_lock()
    shared_state: dict = {
        DONE_TASKS_LOCK: lock,
        DONE_TASKS: done_tasks if done_tasks is not None else [],
    }
    if payment_model is not None:
        shared_state[PAYMENT_MODEL] = payment_model
    return SimpleNamespace(
        logger=SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        ),
        params=SimpleNamespace(
            profit_split_balance=100,
            on_chain_service_id=1,
            agent_mech_contract_addresses=agent_mech_addresses or ["0xMECH"],
            task_wait_timeout=0.1,
            default_chain_id="100",
        ),
        shared_state=shared_state,
    )


def _run_gen(gen: Generator) -> Any:
    """Run a generator to completion, returning its final value."""
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


# ---------------------------------------------------------------------------
# TaskExecutionBaseBehaviour — properties & non-generator methods
# ---------------------------------------------------------------------------


class TestDoneTasksProperty:
    def test_returns_empty_list_by_default(self):
        ctx = _make_ctx(done_tasks=[])
        b = _DummyBase(name="b", skill_context=ctx)
        assert b.done_tasks == []

    def test_returns_copy_of_tasks(self):
        tasks = [{"request_id": "r1"}]
        ctx = _make_ctx(done_tasks=tasks)
        b = _DummyBase(name="b", skill_context=ctx)
        result = b.done_tasks
        assert result == tasks
        # deepcopy — mutation doesn't affect shared_state
        result.append({"request_id": "r2"})
        assert len(ctx.shared_state[DONE_TASKS]) == 1


class TestPaymentModelProperty:
    def test_returns_none_when_not_set(self):
        ctx = _make_ctx()
        b = _DummyBase(name="b", skill_context=ctx)
        assert b.payment_model is None

    def test_returns_model_when_set(self):
        ctx = _make_ctx()
        ctx.shared_state[PAYMENT_MODEL] = "native"
        b = _DummyBase(name="b", skill_context=ctx)
        assert b.payment_model == "native"


class TestDoneTasksLock:
    def test_returns_lock_from_shared_state(self):
        lock = _make_lock()
        ctx = _make_ctx(lock=lock)
        b = _DummyBase(name="b", skill_context=ctx)
        assert b.done_tasks_lock() is lock


class TestMechAddresses:
    def test_returns_from_params(self):
        ctx = _make_ctx(agent_mech_addresses=["0xA", "0xB"])
        b = _DummyBase(name="b", skill_context=ctx)
        assert b.mech_addresses == ["0xA", "0xB"]


class TestRemoveTasks:
    def test_remove_submitted_task(self):
        tasks = [{"request_id": "r1"}, {"request_id": "r2"}]
        ctx = _make_ctx(done_tasks=tasks)
        b = _DummyBase(name="b", skill_context=ctx)
        b.remove_tasks([{"request_id": "r1"}])
        # r1 was submitted → removed; r2 stays
        remaining = ctx.shared_state[DONE_TASKS]
        assert len(remaining) == 1
        assert remaining[0]["request_id"] == "r2"

    def test_no_change_when_empty_submitted(self):
        tasks = [{"request_id": "r1"}]
        ctx = _make_ctx(done_tasks=tasks)
        b = _DummyBase(name="b", skill_context=ctx)
        b.remove_tasks([])
        assert len(ctx.shared_state[DONE_TASKS]) == 1

    def test_all_submitted_leaves_empty(self):
        tasks = [{"request_id": "r1"}, {"request_id": "r2"}]
        ctx = _make_ctx(done_tasks=tasks)
        b = _DummyBase(name="b", skill_context=ctx)
        b.remove_tasks(tasks)
        assert ctx.shared_state[DONE_TASKS] == []


class TestToMultihash:
    def test_empty_bytes_returns_empty_string(self):
        with patch(
            "packages.valory.skills.task_submission_abci.behaviours.multibase"
        ) as mock_mb:
            mock_mb.decode.return_value = b""
            result = TaskExecutionBaseBehaviour.to_multihash("empty-cid")
        assert result == ""

    def test_valid_bytes_strips_six_hex_chars(self):
        multihash_bytes = bytes.fromhex("1220" + "ab" * 32)
        with (
            patch("packages.valory.skills.task_submission_abci.behaviours.multibase") as mock_mb,
            patch("packages.valory.skills.task_submission_abci.behaviours.multicodec") as mock_mc,
        ):
            mock_mb.decode.return_value = b"\x01\x70" + multihash_bytes
            mock_mc.remove_prefix.return_value = multihash_bytes
            result = TaskExecutionBaseBehaviour.to_multihash("bafytest")
        assert result == multihash_bytes.hex()[6:]


class TestSetGauge:
    def test_set_gauge_with_labels(self):
        ctx = _make_ctx()
        b = _DummyBase(name="b", skill_context=ctx)
        metric = MagicMock()
        b.set_gauge(metric, 42, chain="gnosis")
        metric.labels.assert_called_with(chain="gnosis")
        metric.labels().set.assert_called_with(42)
        metric.labels().set_to_current_time.assert_called()

    def test_set_gauge_without_labels(self):
        ctx = _make_ctx()
        b = _DummyBase(name="b", skill_context=ctx)
        metric = MagicMock()
        b.set_gauge(metric, 100)
        metric.set.assert_called_with(100)
        metric.set_to_current_time.assert_called()


class TestObserveHistogram:
    def test_observe_with_labels(self):
        ctx = _make_ctx()
        b = _DummyBase(name="b", skill_context=ctx)
        metric = MagicMock()
        b.observe_histogram(metric, 3.14, tool="my_tool")
        metric.labels.assert_called_with(tool="my_tool")
        metric.labels().observe.assert_called_with(3.14)

    def test_observe_without_labels(self):
        ctx = _make_ctx()
        b = _DummyBase(name="b", skill_context=ctx)
        metric = MagicMock()
        b.observe_histogram(metric, 1.5)
        metric.observe.assert_called_with(1.5)


# ---------------------------------------------------------------------------
# TaskPoolingBehaviour — non-generator methods
# ---------------------------------------------------------------------------


class TestSetTx:
    def test_stores_tx_hash_and_timestamp(self):
        ctx = _make_ctx()
        b = _DummyPooling(name="b", skill_context=ctx)
        before = time.time()
        b.set_tx("0xhash")
        after = time.time()
        tx, ts = ctx.shared_state[LAST_TX]
        assert tx == "0xhash"
        assert before <= ts <= after


class TestCheckLastTxStatus:
    def _make_sync_data_mock(self, final_tx_hash_value=None, raise_=False):
        """Create a mock synchronized_data."""
        sd = MagicMock()
        if raise_:
            type(sd).final_tx_hash = property(lambda self: (_ for _ in ()).throw(ValueError("not set")))
        else:
            type(sd).final_tx_hash = property(lambda self: final_tx_hash_value)
        return sd

    def test_returns_true_with_hash_when_final_tx_hash_exists(self):
        ctx = _make_ctx()
        b = _DummyPooling(name="b", skill_context=ctx)
        # Patch synchronized_data on the behaviour
        mock_sd = MagicMock()
        mock_sd.final_tx_hash = "0xfinal"
        with patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: mock_sd)):
            result = b.check_last_tx_status()
        assert result == (True, "0xfinal")
        # Also verify set_tx was called
        assert ctx.shared_state[LAST_TX][0] == "0xfinal"

    def test_returns_false_empty_when_exception_raised(self):
        ctx = _make_ctx()
        b = _DummyPooling(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        type(mock_sd).final_tx_hash = property(lambda self: (_ for _ in ()).throw(ValueError("not set")))
        with patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: mock_sd)):
            result = b.check_last_tx_status()
        assert result == (False, "")

    def test_returns_false_when_final_tx_hash_is_none(self):
        ctx = _make_ctx()
        b = _DummyPooling(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.final_tx_hash = None
        with patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: mock_sd)):
            result = b.check_last_tx_status()
        assert result == (False, "")


class TestGetDoneTasks:
    def test_returns_tasks_immediately_when_available(self):
        tasks = [{"request_id": "r1"}]
        ctx = _make_ctx(done_tasks=tasks)
        b = _DummyPooling(name="b", skill_context=ctx)
        result = _run_gen(b.get_done_tasks(timeout=5.0))
        assert result == tasks

    def test_returns_empty_list_after_timeout(self):
        ctx = _make_ctx(done_tasks=[])
        b = _DummyPooling(name="b", skill_context=ctx)

        sleep_called = []

        def fake_sleep(seconds):
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
    def _make_b(self) -> _DummyDeliver:
        ctx = _make_ctx()
        return _DummyDeliver(name="b", skill_context=ctx)

    def test_new_agent_and_tool(self):
        b = self._make_b()
        task = {"task_executor_address": "agent-0", "tool": "tool-a"}
        result = b._update_current_delivery_report({}, [task])
        assert result == {"agent-0": {"tool-a": 1}}

    def test_increments_existing_tool(self):
        b = self._make_b()
        current = {"agent-0": {"tool-a": 5}}
        task = {"task_executor_address": "agent-0", "tool": "tool-a"}
        result = b._update_current_delivery_report(current, [task])
        assert result["agent-0"]["tool-a"] == 6

    def test_new_tool_for_existing_agent(self):
        b = self._make_b()
        current = {"agent-0": {"tool-a": 2}}
        task = {"task_executor_address": "agent-0", "tool": "tool-b"}
        result = b._update_current_delivery_report(current, [task])
        assert result["agent-0"]["tool-a"] == 2
        assert result["agent-0"]["tool-b"] == 1

    def test_multiple_tasks_multiple_agents(self):
        b = self._make_b()
        tasks = [
            {"task_executor_address": "agent-0", "tool": "tool-a"},
            {"task_executor_address": "agent-1", "tool": "tool-b"},
            {"task_executor_address": "agent-0", "tool": "tool-a"},
        ]
        result = b._update_current_delivery_report({}, tasks)
        assert result["agent-0"]["tool-a"] == 2
        assert result["agent-1"]["tool-b"] == 1

    def test_empty_tasks_returns_unchanged(self):
        b = self._make_b()
        current = {"agent-0": {"tool-a": 3}}
        result = b._update_current_delivery_report(current, [])
        assert result == {"agent-0": {"tool-a": 3}}


# ---------------------------------------------------------------------------
# DeliverBehaviour — get_delivery_report (generator with mocked _get_current_delivery_report)
# ---------------------------------------------------------------------------


class TestGetDeliveryReport:
    def _make_b(self, done_tasks=None) -> _DummyDeliver:
        ctx = _make_ctx()
        b = _DummyDeliver(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.done_tasks = done_tasks or []
        object.__setattr__(b, "_synchronized_data", mock_sd)
        return b

    def test_returns_none_when_current_report_is_none(self):
        b = self._make_b()
        with (
            patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: b._synchronized_data)),
            patch.object(b, "_get_current_delivery_report", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_delivery_report())
        assert result is None

    def test_returns_updated_report(self):
        task = {"task_executor_address": "agent-0", "tool": "tool-x"}
        b = self._make_b(done_tasks=[task])
        mock_sd = MagicMock()
        mock_sd.done_tasks = [task]
        with (
            patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: mock_sd)),
            patch.object(b, "_get_current_delivery_report", side_effect=_gen_returning({})),
        ):
            result = _run_gen(b.get_delivery_report())
        assert result == {"agent-0": {"tool-x": 1}}


# ---------------------------------------------------------------------------
# Enum constants test
# ---------------------------------------------------------------------------


class TestEnumConstants:
    """Verify the module-level Enum classes are accessible."""

    from packages.valory.skills.task_submission_abci.behaviours import (
        OffchainKeys,
        OffchainDataKey,
        OffchainDataValue,
        MarketplaceKeys,
        MarketplaceData,
    )

    def test_offchain_keys_values(self):
        from packages.valory.skills.task_submission_abci.behaviours import OffchainKeys

        assert OffchainKeys.DELIVER_WITH_SIGNATURES.value == "deliverWithSignatures"

    def test_offchain_data_key_values(self):
        from packages.valory.skills.task_submission_abci.behaviours import OffchainDataKey

        assert OffchainDataKey.REQUEST_DATA_KEY.value == "requestData"

    def test_offchain_data_value_values(self):
        from packages.valory.skills.task_submission_abci.behaviours import OffchainDataValue

        assert OffchainDataValue.IPFS_HASH.value == "ipfs_hash"

    def test_marketplace_keys_values(self):
        from packages.valory.skills.task_submission_abci.behaviours import MarketplaceKeys

        assert MarketplaceKeys.REQUEST_IDS.value == "requestIds"

    def test_marketplace_data_values(self):
        from packages.valory.skills.task_submission_abci.behaviours import MarketplaceData

        assert MarketplaceData.REQUEST_ID.value == "requestId"


# ---------------------------------------------------------------------------
# TaskPoolingBehaviour._fetch_tx_block_number — generator branches
# ---------------------------------------------------------------------------


def _gen_returning(value):
    """Create a generator that returns `value` immediately."""

    def _gen(*args, **kwargs):
        if False:
            yield
        return value

    return _gen


class TestFetchTxBlockNumber:
    def _make_b(self) -> _DummyPooling:
        ctx = _make_ctx()
        return _DummyPooling(name="b", skill_context=ctx)

    def test_returns_none_when_no_response(self):
        b = self._make_b()
        with patch.object(b, "get_transaction_receipt", side_effect=_gen_returning(None)):
            result = _run_gen(b._fetch_tx_block_number("0xhash"))
        assert result is None

    def test_returns_none_when_block_number_missing(self):
        b = self._make_b()
        response = {"status": "1"}  # No blockNumber key
        with patch.object(b, "get_transaction_receipt", side_effect=_gen_returning(response)):
            result = _run_gen(b._fetch_tx_block_number("0xhash"))
        assert result is None

    def test_returns_none_when_block_number_invalid(self):
        b = self._make_b()
        response = {"blockNumber": "not-an-int"}
        with patch.object(b, "get_transaction_receipt", side_effect=_gen_returning(response)):
            result = _run_gen(b._fetch_tx_block_number("0xhash"))
        assert result is None

    def test_returns_int_when_valid_block_number(self):
        b = self._make_b()
        response = {"blockNumber": 12345}
        with patch.object(b, "get_transaction_receipt", side_effect=_gen_returning(response)):
            result = _run_gen(b._fetch_tx_block_number("0xhash"))
        assert result == 12345

    def test_returns_int_when_block_number_is_string_int(self):
        b = self._make_b()
        response = {"blockNumber": "9999"}
        with patch.object(b, "get_transaction_receipt", side_effect=_gen_returning(response)):
            result = _run_gen(b._fetch_tx_block_number("0xhash"))
        assert result == 9999


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_num_requests_delivered and _get_num_reqs_by_agent
# ---------------------------------------------------------------------------


class TestGetNumReqsByAgent:
    def _make_b(self) -> _DummyFunds:
        ctx = _make_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_none_when_delivery_report_is_none(self):
        b = self._make_b()
        with patch.object(b, "get_delivery_report", side_effect=_gen_returning(None)):
            result = _run_gen(b._get_num_reqs_by_agent())
        assert result is None

    def test_aggregates_tool_counts_per_agent(self):
        b = self._make_b()
        report = {
            "agent-0": {"tool-a": 3, "tool-b": 2},
            "agent-1": {"tool-a": 1},
        }
        with patch.object(b, "get_delivery_report", side_effect=_gen_returning(report)):
            result = _run_gen(b._get_num_reqs_by_agent())
        assert result == {"agent-0": 5, "agent-1": 1}

    def test_empty_delivery_report(self):
        b = self._make_b()
        with patch.object(b, "get_delivery_report", side_effect=_gen_returning({})):
            result = _run_gen(b._get_num_reqs_by_agent())
        assert result == {}


class TestGetNumRequestsDelivered:
    def _make_b(self) -> _DummyFunds:
        ctx = _make_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_none_when_reqs_by_agent_is_none(self):
        b = self._make_b()
        with patch.object(b, "_get_num_reqs_by_agent", side_effect=_gen_returning(None)):
            result = _run_gen(b._get_num_requests_delivered())
        assert result is None

    def test_returns_sum_of_all_agents(self):
        b = self._make_b()
        reqs_by_agent = {"agent-0": 5, "agent-1": 3}
        with patch.object(b, "_get_num_reqs_by_agent", side_effect=_gen_returning(reqs_by_agent)):
            result = _run_gen(b._get_num_requests_delivered())
        assert result == 8

    def test_returns_zero_for_empty_agents(self):
        b = self._make_b()
        with patch.object(b, "_get_num_reqs_by_agent", side_effect=_gen_returning({})):
            result = _run_gen(b._get_num_requests_delivered())
        assert result == 0


# ---------------------------------------------------------------------------
# Helpers for contract/ledger API message mocks
# ---------------------------------------------------------------------------


def _state_contract_msg(body=None):
    """Return a MagicMock with STATE performative and given body."""
    msg = MagicMock()
    msg.performative = ContractApiMessage.Performative.STATE
    if body is not None:
        msg.state.body = body
    return msg


def _error_contract_msg():
    """Return a MagicMock with ERROR performative (non-STATE)."""
    msg = MagicMock()
    msg.performative = ContractApiMessage.Performative.ERROR
    return msg


def _raw_tx_contract_msg(body=None):
    """Return a MagicMock with RAW_TRANSACTION performative."""
    msg = MagicMock()
    msg.performative = ContractApiMessage.Performative.RAW_TRANSACTION
    if body is not None:
        msg.raw_transaction.body = body
    return msg


def _state_ledger_msg(body=None):
    """Return a MagicMock with STATE performative and given body (ledger API)."""
    msg = MagicMock()
    msg.performative = LedgerApiMessage.Performative.STATE
    if body is not None:
        msg.state.body = body
    return msg


def _error_ledger_msg():
    """Return a MagicMock with non-STATE performative (ledger API)."""
    msg = MagicMock()
    msg.performative = LedgerApiMessage.Performative.ERROR
    return msg


def _make_full_ctx(done_tasks=None, payment_model=None, lock=None, agent_mech_addresses=None):
    """Extended context with all params needed for complex behaviours."""
    ctx = _make_ctx(
        done_tasks=done_tasks,
        payment_model=payment_model,
        lock=lock,
        agent_mech_addresses=agent_mech_addresses,
    )
    ctx.params.agent_mech_contract_address = "0xMECH"
    ctx.params.hash_checkpoint_address = "0xHASH"
    ctx.params.mech_marketplace_address = "0xMARKET"
    ctx.params.complementary_service_metadata_address = "0xMETA"
    ctx.params.metadata_hash = "bafytest"
    ctx.params.task_mutable_params = SimpleNamespace(latest_metadata_hash=None)
    ctx.params.service_registry_address = "0xSERVICE"
    ctx.params.minimum_agent_balance = 100
    ctx.params.agent_funding_amount = 200
    ctx.params.service_owner_share = 1000
    ctx.params.multisend_address = "0xMULTI"
    ctx.params.mech_staking_instance_address = "0xSTAKE"
    ctx.params.mech_max_delivery_rate = 10
    # ctx.state is accessed by BaseBehaviour.shared_state property
    ctx.state = SimpleNamespace(
        mech_delivery_last_block_number=MagicMock(),
        mech_agent_balance=MagicMock(),
        tool_delivery_time=MagicMock(),
        synchronized_data=SimpleNamespace(safe_contract_address="0xSAFE"),
    )
    return ctx


# ---------------------------------------------------------------------------
# TaskPoolingBehaviour.get_payload_content
# ---------------------------------------------------------------------------


class TestGetPayloadContent:
    def test_returns_json_of_done_tasks(self):
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
    def _make_b(self):
        ctx = _make_full_ctx()
        ctx.shared_state["mech_delivery_last_block_number"] = MagicMock()
        b = _DummyPooling(name="b", skill_context=ctx)
        return b

    def _patch_sd(self, b, done_tasks):
        mock_sd = MagicMock()
        mock_sd.done_tasks = done_tasks
        return patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: mock_sd))

    def test_status_false_removes_nothing(self):
        b = self._make_b()
        with patch.object(b, "check_last_tx_status", return_value=(False, "")):
            result = _run_gen(b.handle_submitted_tasks())
        assert result is None

    def test_status_true_empty_tasks(self):
        b = self._make_b()
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            self._patch_sd(b, []),
        ):
            result = _run_gen(b.handle_submitted_tasks())
        assert result is None

    def test_status_true_with_tasks_no_block_number(self):
        task = {"request_id": "r1", "tool": "t1", "start_time": time.perf_counter()}
        b = self._make_b()
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            self._patch_sd(b, [task]),
            patch.object(b, "_fetch_tx_block_number", side_effect=_gen_returning(None)),
            patch.object(b, "observe_histogram"),
        ):
            result = _run_gen(b.handle_submitted_tasks())
        assert result is None

    def test_status_true_with_tasks_and_block_number(self):
        task = {"request_id": "r1", "tool": "t1", "start_time": time.perf_counter()}
        b = self._make_b()
        with (
            patch.object(b, "check_last_tx_status", return_value=(True, "0xhash")),
            self._patch_sd(b, [task]),
            patch.object(b, "_fetch_tx_block_number", side_effect=_gen_returning(12345)),
            patch.object(b, "observe_histogram"),
            patch.object(b, "set_gauge"),
        ):
            result = _run_gen(b.handle_submitted_tasks())
        assert result is None


# ---------------------------------------------------------------------------
# DeliverBehaviour._get_current_delivery_report
# ---------------------------------------------------------------------------


class TestGetCurrentDeliveryReport:
    def _make_b(self):
        ctx = _make_full_ctx()
        b = _DummyDeliver(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        object.__setattr__(b, "_synchronized_data", mock_sd)
        return b

    def _patch_sd(self, b):
        return patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: b._synchronized_data))

    def test_returns_none_on_contract_error(self):
        b = self._make_b()
        with (
            self._patch_sd(b),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(_error_contract_msg())),
        ):
            result = _run_gen(b._get_current_delivery_report())
        assert result is None

    def test_returns_empty_dict_for_zero_ipfs_hash(self):
        b = self._make_b()
        msg = _state_contract_msg({"data": ZERO_IPFS_HASH})
        with (
            self._patch_sd(b),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)),
        ):
            result = _run_gen(b._get_current_delivery_report())
        assert result == {}

    def test_returns_none_when_ipfs_fetch_fails(self):
        b = self._make_b()
        # Use a real CIDv1 hash string that CID.from_string can parse
        valid_cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
        msg = _state_contract_msg({"data": valid_cid})
        with (
            self._patch_sd(b),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)),
            patch.object(b, "get_from_ipfs", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b._get_current_delivery_report())
        assert result is None

    def test_returns_usage_data_on_success(self):
        b = self._make_b()
        valid_cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
        msg = _state_contract_msg({"data": valid_cid})
        usage_data = {"agent-0": {"tool-a": 3}}
        with (
            self._patch_sd(b),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)),
            patch.object(b, "get_from_ipfs", side_effect=_gen_returning(usage_data)),
        ):
            result = _run_gen(b._get_current_delivery_report())
        assert result == usage_data


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_balance
# ---------------------------------------------------------------------------


class TestGetBalance:
    def _make_b(self):
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_none_on_ledger_error(self):
        b = self._make_b()
        with patch.object(b, "get_ledger_api_response", side_effect=_gen_returning(_error_ledger_msg())):
            result = _run_gen(b._get_balance("0xAGENT"))
        assert result is None

    def test_returns_balance_on_success(self):
        b = self._make_b()
        msg = _state_ledger_msg({"get_balance_result": 500})
        with patch.object(b, "get_ledger_api_response", side_effect=_gen_returning(msg)):
            result = _run_gen(b._get_balance("0xAGENT"))
        assert result == 500


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_mech_payment_type
# ---------------------------------------------------------------------------


class TestGetMechPaymentType:
    def test_shortcut_when_payment_model_set(self):
        # When payment_model is set and address matches agent_mech_contract_address
        payment_type = b"\x00" * 32
        ctx = _make_full_ctx(payment_model=payment_type)
        ctx.params.agent_mech_contract_address = "0xPRIMARY"
        ctx.params.agent_mech_contract_addresses = ["0xPRIMARY"]
        b = _DummyFunds(name="b", skill_context=ctx)
        # _get_mech_payment_type is a generator function; when shortcut path taken, no yields
        result = _run_gen(b._get_mech_payment_type("0xPRIMARY"))
        assert result == payment_type

    def test_fetches_via_contract_api_on_non_primary(self):
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        mech_type_bytes = b"\xab" * 32
        msg = _state_contract_msg({"mech_type": mech_type_bytes})
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)):
            result = _run_gen(b._get_mech_payment_type("0xOTHER"))
        assert result == mech_type_bytes


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_balance_tracker_address
# ---------------------------------------------------------------------------


class TestGetBalanceTrackerAddress:
    def test_returns_address_on_success(self):
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        msg = _state_contract_msg({"data": "0xTRACKER"})
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)):
            result = _run_gen(b._get_balance_tracker_address(b"\x00" * 32))
        assert result == "0xTRACKER"


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._adjust_mech_balance
# ---------------------------------------------------------------------------


class TestAdjustMechBalance:
    def test_adjusts_balance_by_token_credit_ratio(self):
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        ratio = 2 * (10 ** 18)  # ratio * mech_balance // 1e18 = 2 * mech_balance
        msg = _state_contract_msg({"token_credit_ratio": ratio})
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)):
            result = _run_gen(b._adjust_mech_balance("0xTRACKER", 100))
        assert result == 200


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_mech_info
# ---------------------------------------------------------------------------


class TestGetMechInfo:
    def _make_b(self):
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_none_when_mech_type_none(self):
        b = self._make_b()
        with patch.object(b, "_get_mech_payment_type", side_effect=_gen_returning(None)):
            result = _run_gen(b._get_mech_info("0xMECH"))
        assert result is None

    def test_returns_none_when_tracker_address_none(self):
        b = self._make_b()
        with (
            patch.object(b, "_get_mech_payment_type", side_effect=_gen_returning(b"\x00" * 32)),
            patch.object(b, "_get_balance_tracker_address", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b._get_mech_info("0xMECH"))
        assert result is None

    def test_returns_tuple_on_success(self):
        b = self._make_b()
        mech_type = b"\xaa" * 32
        msg = _state_contract_msg({"mech_balance": 999})
        with (
            patch.object(b, "_get_mech_payment_type", side_effect=_gen_returning(mech_type)),
            patch.object(b, "_get_balance_tracker_address", side_effect=_gen_returning("0xTRACK")),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)),
        ):
            result = _run_gen(b._get_mech_info("0xMECH"))
        assert result == (mech_type, "0xTRACK", 999)


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_process_payment_tx
# ---------------------------------------------------------------------------


class TestGetProcessPaymentTx:
    def test_returns_dict_on_success(self):
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        tx_data = b"\xde\xad"
        msg = _state_contract_msg({"data": tx_data, "simulation_ok": True})
        with (
            patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: mock_sd)),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)),
        ):
            result = _run_gen(b._get_process_payment_tx("0xMECH", "0xTRACK"))
        assert result["simulation_ok"] is True
        assert result["data"] == tx_data


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_fee and _get_max_fee_factor
# ---------------------------------------------------------------------------


class TestGetFeeAndMaxFeeFactor:
    def test_get_fee_returns_int(self):
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        msg = _state_contract_msg({"data": 250})
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)):
            result = _run_gen(b._get_fee())
        assert result == 250

    def test_get_max_fee_factor_returns_int(self):
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        msg = _state_contract_msg({"max_fee_factor": 10000})
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)):
            result = _run_gen(b._get_max_fee_factor("0xTRACK"))
        assert result == 10000


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._calculate_mech_profits
# ---------------------------------------------------------------------------


class TestCalculateMechProfits:
    def _make_b(self):
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_none_when_fee_is_none(self):
        b = self._make_b()
        with patch.object(b, "_get_fee", side_effect=_gen_returning(None)):
            result = _run_gen(b._calculate_mech_profits("0xTRACK", 1000))
        assert result is None

    def test_returns_none_when_max_fee_factor_is_none(self):
        b = self._make_b()
        with (
            patch.object(b, "_get_fee", side_effect=_gen_returning(100)),
            patch.object(b, "_get_max_fee_factor", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b._calculate_mech_profits("0xTRACK", 1000))
        assert result is None

    def test_calculates_profits_on_success(self):
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
    def _make_b(self):
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        b.context.params.service_owner_share = 1000  # 10%
        return b

    def test_returns_none_when_service_owner_none(self):
        b = self._make_b()
        with patch.object(b, "_get_service_owner", side_effect=_gen_returning(None)):
            result = _run_gen(b._split_funds(1000))
        assert result is None

    def test_returns_none_when_agent_funding_amounts_none(self):
        b = self._make_b()
        with (
            patch.object(b, "_get_service_owner", side_effect=_gen_returning("0xOWNER")),
            patch.object(b, "_get_agent_funding_amounts", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b._split_funds(1000))
        assert result is None

    def test_returns_none_when_funds_by_operator_none(self):
        b = self._make_b()
        with (
            patch.object(b, "_get_service_owner", side_effect=_gen_returning("0xOWNER")),
            patch.object(b, "_get_agent_funding_amounts", side_effect=_gen_returning({"agent-0": 50})),
            patch.object(b, "_get_funds_by_operator", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b._split_funds(1000))
        assert result is None

    def test_proportional_split_when_agent_amounts_exceed_profits(self):
        b = self._make_b()
        # agent_funding_amounts = {"a0": 80, "a1": 60} → total=140 > profits=100
        # a0 share = (80 * 100) // 140 = 57, a1 share = (60 * 100) // 140 = 42
        with (
            patch.object(b, "_get_service_owner", side_effect=_gen_returning("0xOWNER")),
            patch.object(b, "_get_agent_funding_amounts", side_effect=_gen_returning({"a0": 80, "a1": 60})),
        ):
            result = _run_gen(b._split_funds(100))
        assert result is not None
        assert "a0" in result and "a1" in result
        # Total should not exceed profits
        assert result["a0"] + result["a1"] <= 100

    def test_full_success_split(self):
        b = self._make_b()
        # agent_funding_amounts = {"agent-0": 50}, total=50 <= profits=200
        # profits after agents = 150, service_owner_share = 10% = 15
        # operator_share = 135
        with (
            patch.object(b, "_get_service_owner", side_effect=_gen_returning("0xOWNER")),
            patch.object(b, "_get_agent_funding_amounts", side_effect=_gen_returning({"agent-0": 50})),
            patch.object(b, "_get_funds_by_operator", side_effect=_gen_returning({"op-0": 135})),
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
    def test_returns_dict_on_success(self):
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        tx_data = b"\xbe\xef"
        msg = _state_contract_msg({"data": tx_data})
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)):
            result = _run_gen(b._get_transfer_tx("0xMECH", "0xRECEIVER", 100))
        assert result["to"] == "0xMECH"
        assert result["data"] == tx_data


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_token_address / _get_token_transfer_tx_data / _get_token_transfer_tx
# ---------------------------------------------------------------------------


class TestTokenTransfer:
    def _make_b(self):
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_get_token_address_success(self):
        b = self._make_b()
        msg = _state_contract_msg({"token_address": "0xTOKEN"})
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)):
            result = _run_gen(b._get_token_address("0xTRACK"))
        assert result == "0xTOKEN"

    def test_get_token_transfer_tx_data_success(self):
        b = self._make_b()
        tx_data = b"\xca\xfe"
        msg = _state_contract_msg({"data": tx_data})
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)):
            result = _run_gen(b._get_token_transfer_tx_data("0xTOKEN", "0xRECEIVER", 100))
        assert result == tx_data

    def test_get_token_transfer_tx_returns_none_when_data_none(self):
        b = self._make_b()
        with patch.object(b, "_get_token_transfer_tx_data", side_effect=_gen_returning(None)):
            result = _run_gen(b._get_token_transfer_tx("0xMECH", "0xTOKEN", "0xRECEIVER", 100))
        assert result is None

    def test_get_token_transfer_tx_success(self):
        b = self._make_b()
        tx_data = b"\xba\xbe"
        msg = _state_contract_msg({"data": tx_data})
        with (
            patch.object(b, "_get_token_transfer_tx_data", side_effect=_gen_returning(b"\xca\xfe")),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)),
        ):
            result = _run_gen(b._get_token_transfer_tx("0xMECH", "0xTOKEN", "0xRECEIVER", 100))
        assert result["to"] == "0xMECH"


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_service_owner
# ---------------------------------------------------------------------------


class TestGetServiceOwner:
    def _make_b(self):
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_none_on_contract_error(self):
        b = self._make_b()
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(_error_contract_msg())):
            result = _run_gen(b._get_service_owner(1))
        assert result is None

    def test_returns_owner_on_success(self):
        b = self._make_b()
        msg = _state_contract_msg({"service_owner": "0xOWNER"})
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)):
            result = _run_gen(b._get_service_owner(1))
        assert result == "0xOWNER"


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_funds_by_operator
# ---------------------------------------------------------------------------


class TestGetFundsByOperator:
    def _make_b(self):
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_empty_dict_when_operator_share_is_zero(self):
        b = self._make_b()
        result = _run_gen(b._get_funds_by_operator(0))
        assert result == {}

    def test_returns_none_when_reqs_by_agent_none(self):
        b = self._make_b()
        with patch.object(b, "_get_num_reqs_by_agent", side_effect=_gen_returning(None)):
            result = _run_gen(b._get_funds_by_operator(100))
        assert result is None

    def test_returns_zero_per_agent_when_total_reqs_zero(self):
        b = self._make_b()
        reqs_by_agent = {"a0": 0, "a1": 0}
        with patch.object(b, "_get_num_reqs_by_agent", side_effect=_gen_returning(reqs_by_agent)):
            result = _run_gen(b._get_funds_by_operator(100))
        assert result == {"a0": 0, "a1": 0}

    def test_returns_none_when_accumulate_fails(self):
        b = self._make_b()
        with (
            patch.object(b, "_get_num_reqs_by_agent", side_effect=_gen_returning({"a0": 5})),
            patch.object(b, "_accumulate_reqs_by_operator", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b._get_funds_by_operator(100))
        assert result is None

    def test_splits_by_operator_proportionally(self):
        b = self._make_b()
        # a0 → op0 (3 reqs), a1 → op0 (2 reqs), a2 → op1 (5 reqs)
        # after accumulate: op0→5, op1→5 (total 10)
        # share for each = (100 * 5) // 10 = 50
        with (
            patch.object(b, "_get_num_reqs_by_agent", side_effect=_gen_returning({"a0": 3, "a1": 2, "a2": 5})),
            patch.object(b, "_accumulate_reqs_by_operator", side_effect=_gen_returning({"op0": 5, "op1": 5})),
        ):
            result = _run_gen(b._get_funds_by_operator(100))
        assert result == {"op0": 50, "op1": 50}

    def test_filters_zero_address_operator(self):
        b = self._make_b()
        # zero address gets filtered, its reqs removed from total
        # op1→5 reqs (total valid = 5), share = (100 * 5) // 5 = 100
        with (
            patch.object(b, "_get_num_reqs_by_agent", side_effect=_gen_returning({"a0": 3, "a1": 5})),
            patch.object(b, "_accumulate_reqs_by_operator", side_effect=_gen_returning({ZERO_ADDRESS: 3, "op1": 5})),
        ):
            result = _run_gen(b._get_funds_by_operator(100))
        assert ZERO_ADDRESS not in result
        assert result["op1"] == 100


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._accumulate_reqs_by_operator
# ---------------------------------------------------------------------------


class TestAccumulateReqsByOperator:
    def test_accumulates_reqs_by_operator(self):
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        # agent_to_operator mapping: a0→op0, a1→op0, a2→op1
        msg = _state_contract_msg({"a0": "op0", "a1": "op0", "a2": "op1"})
        reqs_by_agent = {"a0": 3, "a1": 2, "a2": 5}
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)):
            result = _run_gen(b._accumulate_reqs_by_operator(reqs_by_agent))
        assert result == {"op0": 5, "op1": 5}


# ---------------------------------------------------------------------------
# FundsSplittingBehaviour._get_agent_balances / _get_agent_funding_amounts
# ---------------------------------------------------------------------------


class TestGetAgentBalances:
    def _make_b(self, participants=None):
        ctx = _make_full_ctx()
        b = _DummyFunds(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.all_participants = participants or ["a0", "a1"]
        mock_sd.mech_agent_balance = MagicMock()
        object.__setattr__(b, "_synchronized_data", mock_sd)
        return b

    def _patch_sd(self, b):
        return patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: b._synchronized_data))

    def test_returns_none_when_any_balance_is_none(self):
        b = self._make_b(["a0"])
        with (
            self._patch_sd(b),
            patch.object(b, "_get_balance", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b._get_agent_balances())
        assert result is None

    def test_returns_balances_dict_on_success(self):
        b = self._make_b(["a0", "a1"])
        balances = {"a0": 200, "a1": 50}

        def _balance_gen(addr):
            """Generator that returns balance for given address."""
            if False:
                yield
            return balances[addr]

        with (
            self._patch_sd(b),
            patch.object(b, "_get_balance", side_effect=lambda addr: _balance_gen(addr)),
            patch.object(b, "set_gauge"),
        ):
            result = _run_gen(b._get_agent_balances())
        assert result["a0"] == 200
        assert result["a1"] == 50


class TestGetAgentFundingAmounts:
    def _make_b(self, participants=None):
        ctx = _make_full_ctx()
        ctx.params.minimum_agent_balance = 100
        ctx.params.agent_funding_amount = 200
        b = _DummyFunds(name="b", skill_context=ctx)
        return b

    def test_returns_none_when_agent_balances_none(self):
        b = self._make_b()
        with patch.object(b, "_get_agent_balances", side_effect=_gen_returning(None)):
            result = _run_gen(b._get_agent_funding_amounts())
        assert result is None

    def test_returns_funding_amounts_for_underfunded_agents(self):
        b = self._make_b()
        # a0 has 200 (above 100 min), a1 has 50 (below 100 min)
        with patch.object(b, "_get_agent_balances", side_effect=_gen_returning({"a0": 200, "a1": 50})):
            result = _run_gen(b._get_agent_funding_amounts())
        assert "a0" not in result
        assert result["a1"] == 200  # agent_funding_amount

    def test_returns_empty_when_all_agents_funded(self):
        b = self._make_b()
        with patch.object(b, "_get_agent_balances", side_effect=_gen_returning({"a0": 200, "a1": 150})):
            result = _run_gen(b._get_agent_funding_amounts())
        assert result == {}


# ---------------------------------------------------------------------------
# TrackingBehaviour methods
# ---------------------------------------------------------------------------


class TestSaveUsageToIpfs:
    def _make_b(self):
        ctx = _make_full_ctx()
        return _DummyTransPrep(name="b", skill_context=ctx)

    def test_returns_none_when_ipfs_fails(self):
        b = self._make_b()
        with patch.object(b, "send_to_ipfs", side_effect=_gen_returning(None)):
            result = _run_gen(b._save_usage_to_ipfs({"data": 1}))
        assert result is None

    def test_returns_hash_on_success(self):
        b = self._make_b()
        with patch.object(b, "send_to_ipfs", side_effect=_gen_returning("bafyhash")):
            result = _run_gen(b._save_usage_to_ipfs({"data": 1}))
        assert result == "bafyhash"


class TestGetCheckpointTx:
    def test_returns_dict_on_success(self):
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        tx_data = b"\xaa\xbb"
        msg = _state_contract_msg({"data": tx_data})
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)):
            result = _run_gen(b._get_checkpoint_tx("0xHASH", "ab" * 16))
        assert result["to"] == "0xHASH"
        assert result["data"] == tx_data


class TestGetUpdateUsageTx:
    def _make_b(self):
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.done_tasks = []
        object.__setattr__(b, "_synchronized_data", mock_sd)
        return b

    def _patch_sd(self, b):
        return patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: b._synchronized_data))

    def test_returns_none_when_delivery_report_none(self):
        b = self._make_b()
        with (
            self._patch_sd(b),
            patch.object(b, "get_delivery_report", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_update_usage_tx())
        assert result is None

    def test_returns_none_when_ipfs_save_fails(self):
        b = self._make_b()
        with (
            self._patch_sd(b),
            patch.object(b, "get_delivery_report", side_effect=_gen_returning({"agent-0": {"t": 1}})),
            patch.object(b, "_save_usage_to_ipfs", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_update_usage_tx())
        assert result is None

    def test_returns_tx_on_success(self):
        b = self._make_b()
        tx = {"to": "0xHASH", "value": 0, "data": b"\x00"}
        with (
            self._patch_sd(b),
            patch.object(b, "get_delivery_report", side_effect=_gen_returning({"agent-0": {"t": 1}})),
            patch.object(b, "_save_usage_to_ipfs", side_effect=_gen_returning("bafyhash")),
            patch.object(b, "_get_checkpoint_tx", side_effect=_gen_returning(tx)),
            patch.object(TaskExecutionBaseBehaviour, "to_multihash", return_value="ab" * 16),
        ):
            with patch("packages.valory.skills.task_submission_abci.behaviours.to_v1", return_value="bafyhashv1"):
                result = _run_gen(b.get_update_usage_tx())
        assert result == tx


# ---------------------------------------------------------------------------
# HashUpdateBehaviour._get_latest_hash / _should_update_hash / get_mech_update_hash_tx
# ---------------------------------------------------------------------------


class TestHashUpdateBehaviour:
    def _make_b(self):
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        return b

    def test_get_latest_hash_success(self):
        b = self._make_b()
        hash_data = b"\xcc" * 32
        msg = _state_contract_msg({"data": hash_data})
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)):
            result = _run_gen(b._get_latest_hash())
        assert result == hash_data

    def test_should_update_hash_returns_false_when_latest_is_none(self):
        b = self._make_b()
        b.context.params.task_mutable_params.latest_metadata_hash = None
        with patch.object(b, "_get_latest_hash", side_effect=_gen_returning(None)):
            result = _run_gen(b._should_update_hash())
        assert result is False

    def test_should_update_hash_returns_false_when_configured_hash_empty(self):
        b = self._make_b()
        b.context.params.task_mutable_params.latest_metadata_hash = None
        with (
            patch.object(b, "_get_latest_hash", side_effect=_gen_returning(b"\x00" * 32)),
            patch.object(type(b), "to_multihash", return_value=""),
        ):
            result = _run_gen(b._should_update_hash())
        assert result is False

    def test_should_update_hash_returns_false_when_hashes_same(self):
        b = self._make_b()
        # latest_metadata_hash and configured_hash must both be strings for equality
        b.context.params.task_mutable_params.latest_metadata_hash = "same_hash"
        with patch.object(type(b), "to_multihash", return_value="same_hash"):
            result = _run_gen(b._should_update_hash())
        assert result is False

    def test_should_update_hash_returns_true_when_hashes_differ(self):
        b = self._make_b()
        b.context.params.task_mutable_params.latest_metadata_hash = b"old_hash"
        with patch.object(type(b), "to_multihash", return_value="new_hash"):
            result = _run_gen(b._should_update_hash())
        assert result is True

    def test_get_mech_update_hash_tx_returns_none_when_no_update_needed(self):
        b = self._make_b()
        with patch.object(b, "_should_update_hash", side_effect=_gen_returning(False)):
            result = _run_gen(b.get_mech_update_hash_tx())
        assert result is None

    def test_get_mech_update_hash_tx_returns_tx_on_success(self):
        b = self._make_b()
        b.context.params.task_mutable_params.latest_metadata_hash = b"old"
        tx_data = b"\xde\xca"
        msg = _state_contract_msg({"data": tx_data})
        with (
            patch.object(b, "_should_update_hash", side_effect=_gen_returning(True)),
            patch.object(type(b), "to_multihash", return_value="ab" * 16),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)),
        ):
            result = _run_gen(b.get_mech_update_hash_tx())
        assert result["to"] == "0xMETA"
        assert result["data"] == tx_data


# ---------------------------------------------------------------------------
# TransactionPreparationBehaviour._get_deliver_tx
# ---------------------------------------------------------------------------


class _DummyTransPrep(TransactionPreparationBehaviour):
    """Minimal concrete subclass for testing TransactionPreparationBehaviour."""

    def async_act(self) -> Generator[None, None, None]:
        if False:  # pragma: no cover
            yield
        return None


class TestGetDeliverTx:
    def _make_b(self):
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        return b

    def test_routes_to_marketplace_when_is_marketplace_mech(self):
        b = self._make_b()
        task = {"request_id": "r1", "is_marketplace_mech": True, "mech_address": "0xMECH"}
        tx = {"to": "0xMECH", "value": 0, "data": b"\x00", "simulation_ok": True}
        with patch.object(b, "_get_deliver_marketplace_tx", side_effect=_gen_returning(tx)):
            result = _run_gen(b._get_deliver_tx(task))
        assert result == tx

    def test_routes_to_agent_mech_when_not_marketplace(self):
        b = self._make_b()
        task = {"request_id": "r1", "is_marketplace_mech": False}
        tx = {"to": "0xMECH", "value": 0, "data": b"\x00", "simulation_ok": True}
        with patch.object(b, "_get_agent_mech_deliver_tx", side_effect=_gen_returning(tx)):
            result = _run_gen(b._get_deliver_tx(task))
        assert result == tx


# ---------------------------------------------------------------------------
# TransactionPreparationBehaviour._get_agent_mech_deliver_tx / _get_deliver_marketplace_tx
# ---------------------------------------------------------------------------


class TestDeliverTxMethods:
    def _make_b(self):
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        object.__setattr__(b, "_synchronized_data", mock_sd)
        return b

    def _patch_sd(self, b):
        return patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: b._synchronized_data))

    def test_agent_mech_deliver_tx_success(self):
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
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)),
        ):
            result = _run_gen(b._get_agent_mech_deliver_tx(task_data))
        assert result["to"] == "0xMECH"
        assert result["simulation_ok"] is True

    def test_deliver_marketplace_tx_success(self):
        b = self._make_b()
        task_data = {
            "mech_address": "0xMECH",
            "request_id": "r1",
            "task_result": b"\xaa",
        }
        msg = _state_contract_msg({"data": b"\xdd", "simulation_ok": True})
        with (
            self._patch_sd(b),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)),
        ):
            result = _run_gen(b._get_deliver_marketplace_tx(task_data))
        assert result["to"] == "0xMECH"
        assert result["simulation_ok"] is True


# ---------------------------------------------------------------------------
# TransactionPreparationBehaviour._get_safe_tx_hash / _to_multisend
# ---------------------------------------------------------------------------


class TestSafeTxHashAndMultisend:
    def _make_b(self):
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        object.__setattr__(b, "_synchronized_data", mock_sd)
        return b

    def _patch_sd(self, b):
        return patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: b._synchronized_data))

    def test_get_safe_tx_hash_returns_none_on_error(self):
        b = self._make_b()
        with (
            self._patch_sd(b),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(_error_contract_msg())),
        ):
            result = _run_gen(b._get_safe_tx_hash(b"\x00" * 32))
        assert result is None

    def test_get_safe_tx_hash_returns_hash_on_success(self):
        b = self._make_b()
        msg = _state_contract_msg({"tx_hash": "0xabcdef"})
        with (
            self._patch_sd(b),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)),
        ):
            result = _run_gen(b._get_safe_tx_hash(b"\x00" * 32))
        assert result == "abcdef"  # strips "0x"

    def test_to_multisend_returns_none_on_non_raw_tx_response(self):
        b = self._make_b()
        with (
            self._patch_sd(b),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(_error_contract_msg())),
        ):
            result = _run_gen(b._to_multisend([{"to": "0xA", "value": 0, "data": b"\x00"}]))
        assert result is None

    def test_to_multisend_returns_none_when_safe_tx_hash_fails(self):
        b = self._make_b()
        raw_msg = _raw_tx_contract_msg({"data": "0x" + "ab" * 8})
        with (
            self._patch_sd(b),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(raw_msg)),
            patch.object(b, "_get_safe_tx_hash", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b._to_multisend([{"to": "0xA", "value": 0, "data": b"\x00"}]))
        assert result is None


# ---------------------------------------------------------------------------
# TransactionPreparationBehaviour._get_offchain_tasks_deliver_data
# ---------------------------------------------------------------------------


class TestGetOffchainTasksDeliverData:
    def _make_b(self):
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        return b

    def _patch_sd(self, b, done_tasks):
        mock_sd = MagicMock()
        mock_sd.done_tasks = done_tasks
        mock_sd.safe_contract_address = "0xSAFE"
        object.__setattr__(b, "_synchronized_data", mock_sd)
        return patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: b._synchronized_data))

    def test_returns_empty_list_when_no_offchain_tasks(self):
        b = self._make_b()
        non_offchain = [{"request_id": "r1", "is_offchain": False}]
        with self._patch_sd(b, non_offchain):
            result = _run_gen(b._get_offchain_tasks_deliver_data())
        assert result == []

    def test_skips_tasks_with_failed_simulation(self):
        b = self._make_b()
        task = {
            "request_id": "r1",
            "is_offchain": True,
            "nonce": 1,
            "sender": "0xSENDER",
            "ipfs_hash": "0x" + "ab" * 16,
            "signature": "0x" + "cd" * 32,
            "task_result": "de" * 16,
            "delivery_rate": 100,
        }
        msg = _state_contract_msg({"data": b"\xdd", "simulation_ok": False})
        with (
            self._patch_sd(b, [task]),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)),
        ):
            result = _run_gen(b._get_offchain_tasks_deliver_data())
        assert result == []

    def test_appends_tx_when_simulation_ok(self):
        b = self._make_b()
        task = {
            "request_id": "r1",
            "is_offchain": True,
            "nonce": 1,
            "sender": "0xSENDER",
            "ipfs_hash": "0x" + "ab" * 16,
            "signature": "0x" + "cd" * 32,
            "task_result": "de" * 16,
            "delivery_rate": 100,
        }
        msg = _state_contract_msg({"data": b"\xdd", "simulation_ok": True})
        with (
            self._patch_sd(b, [task]),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)),
        ):
            result = _run_gen(b._get_offchain_tasks_deliver_data())
        assert len(result) == 1
        assert result[0]["to"] == "0xMECH"


# ---------------------------------------------------------------------------
# TransactionPreparationBehaviour._get_is_nvm_mech / _get_encoded_deliver_data
# ---------------------------------------------------------------------------


class TestNvmMechHelpers:
    def _make_b(self):
        ctx = _make_full_ctx()
        return _DummyTransPrep(name="b", skill_context=ctx)

    def test_get_is_nvm_mech_returns_bool(self):
        b = self._make_b()
        msg = _state_contract_msg({"data": True})
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)):
            result = _run_gen(b._get_is_nvm_mech("0xMECH"))
        assert result is True

    def test_get_encoded_deliver_data_with_requests(self):
        b = self._make_b()
        encoded_data = b"\xee\xff"
        msg = _state_contract_msg({"data": encoded_data})
        request_ids = [b"\x00" * 32]
        datas = [b"\x01" * 32]
        delivery_rates = [100]
        with patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg)):
            final_ids, final_datas = _run_gen(b._get_encoded_deliver_data(request_ids, datas, delivery_rates))
        assert len(final_ids) == 1
        assert final_datas[0] == encoded_data

    def test_get_encoded_deliver_data_empty(self):
        b = self._make_b()
        final_ids, final_datas = _run_gen(b._get_encoded_deliver_data([], [], []))
        assert final_ids == []
        assert final_datas == []


# ---------------------------------------------------------------------------
# TransactionPreparationBehaviour._get_marketplace_tasks_deliver_data
# ---------------------------------------------------------------------------


class TestGetMarketplaceTasksDeliverData:
    def _make_b(self):
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        return b

    def _patch_sd(self, b, done_tasks):
        mock_sd = MagicMock()
        mock_sd.done_tasks = done_tasks
        mock_sd.safe_contract_address = "0xSAFE"
        object.__setattr__(b, "_synchronized_data", mock_sd)
        return patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: b._synchronized_data))

    def test_returns_empty_list_for_no_marketplace_tasks(self):
        b = self._make_b()
        result = _run_gen(b._get_marketplace_tasks_deliver_data([]))
        assert result == []

    def test_appends_tx_for_non_nvm_mech_with_simulation_ok(self):
        b = self._make_b()
        task = {
            "mech_address": "0xMECH",
            "requestId": 12345,
            "task_result": "de" * 16,
        }
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        msg_deliver = _state_contract_msg({"data": b"\xdd", "simulation_ok": True})
        with (
            patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: mock_sd)),
            patch.object(b, "_get_is_nvm_mech", side_effect=_gen_returning(False)),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg_deliver)),
        ):
            result = _run_gen(b._get_marketplace_tasks_deliver_data([task]))
        assert len(result) == 1

    def test_skips_tasks_with_failed_simulation(self):
        b = self._make_b()
        task = {
            "mech_address": "0xMECH",
            "requestId": 12345,
            "task_result": "de" * 16,
        }
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        msg_deliver = _state_contract_msg({"data": b"\xdd", "simulation_ok": False})
        with (
            patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: mock_sd)),
            patch.object(b, "_get_is_nvm_mech", side_effect=_gen_returning(False)),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg_deliver)),
        ):
            result = _run_gen(b._get_marketplace_tasks_deliver_data([task]))
        assert result == []


# ---------------------------------------------------------------------------
# TaskSubmissionRoundBehaviour class attributes
# ---------------------------------------------------------------------------


class TestTaskSubmissionRoundBehaviour:
    def test_initial_behaviour_is_task_pooling(self):
        from packages.valory.skills.task_submission_abci.behaviours import TaskPoolingBehaviour
        assert TaskSubmissionRoundBehaviour.initial_behaviour_cls is TaskPoolingBehaviour

    def test_abci_app_cls_is_task_submission_app(self):
        from packages.valory.skills.task_submission_abci.rounds import TaskSubmissionAbciApp
        assert TaskSubmissionRoundBehaviour.abci_app_cls is TaskSubmissionAbciApp

    def test_behaviours_set_contains_expected_classes(self):
        from packages.valory.skills.task_submission_abci.behaviours import TransactionPreparationBehaviour
        assert TransactionPreparationBehaviour in TaskSubmissionRoundBehaviour.behaviours
        assert TaskPoolingBehaviour in TaskSubmissionRoundBehaviour.behaviours


# ---------------------------------------------------------------------------
# get_split_profit_txs (FundsSplittingBehaviour)
# ---------------------------------------------------------------------------


class TestGetSplitProfitTxs:
    def _make_b(self):
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_empty_list_when_no_split(self):
        b = self._make_b()
        with patch.object(b, "_should_split_profits", side_effect=_gen_returning(False)):
            result = _run_gen(b.get_split_profit_txs())
        assert result == []

    def test_returns_none_when_mech_info_unavailable(self):
        b = self._make_b()
        b.context.params.agent_mech_contract_addresses = ["0xMECH"]
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None

    def test_returns_none_when_profits_is_none(self):
        b = self._make_b()
        # non-NVM mech type (plain bytes)
        mech_info = (b"\x00" * 32, "0xTRACK", 1000)
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None

    def test_returns_none_when_process_payment_tx_is_none(self):
        b = self._make_b()
        mech_info = (b"\x00" * 32, "0xTRACK", 1000)
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(b, "_get_process_payment_tx", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None

    def test_returns_none_when_simulation_fails(self):
        b = self._make_b()
        mech_info = (b"\x00" * 32, "0xTRACK", 1000)
        process_tx = {"to": "0xTRACK", "value": 0, "data": b"\x00", "simulation_ok": False}
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(b, "_get_process_payment_tx", side_effect=_gen_returning(process_tx)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None

    def test_returns_none_when_split_funds_is_none(self):
        b = self._make_b()
        mech_info = (b"\x00" * 32, "0xTRACK", 1000)
        process_tx = {"to": "0xTRACK", "value": 0, "data": b"\x00", "simulation_ok": True}
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(b, "_get_process_payment_tx", side_effect=_gen_returning(dict(process_tx))),
            patch.object(b, "_split_funds", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None

    def test_returns_none_when_transfer_tx_is_none(self):
        b = self._make_b()
        mech_info = (b"\x00" * 32, "0xTRACK", 1000)
        process_tx = {"to": "0xTRACK", "value": 0, "data": b"\x00", "simulation_ok": True}
        split_funds = {"op0": 100}  # amount > 0 → need transfer tx
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(b, "_get_process_payment_tx", side_effect=_gen_returning(dict(process_tx))),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
            patch.object(b, "_get_transfer_tx", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None

    def test_skips_zero_amount_receivers(self):
        b = self._make_b()
        mech_info = (b"\x00" * 32, "0xTRACK", 1000)
        process_tx = {"to": "0xTRACK", "value": 0, "data": b"\x00", "simulation_ok": True}
        split_funds = {"op0": 0}  # amount=0 → skip, no transfer tx needed
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(b, "_get_process_payment_tx", side_effect=_gen_returning(dict(process_tx))),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        # Only process_payment_tx (no transfer txs for zero amounts), but txs is empty if only zeros
        # Actually process_payment_tx IS appended before split_funds, so txs=[process_tx]
        assert result is not None

    def test_returns_txs_with_native_transfer(self):
        b = self._make_b()
        # Non-NVM, non-TOKEN mech type → uses _get_transfer_tx
        mech_info = (b"\x00" * 32, "0xTRACK", 1000)
        process_tx = {"to": "0xTRACK", "value": 0, "data": b"\x00", "simulation_ok": True}
        split_funds = {"op0": 900}
        transfer_tx = {"to": "0xMECH", "value": 0, "data": b"\x01"}
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(b, "_get_process_payment_tx", side_effect=_gen_returning(dict(process_tx))),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
            patch.object(b, "_get_transfer_tx", side_effect=_gen_returning(transfer_tx)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is not None
        assert len(result) == 2  # process_payment_tx + transfer_tx

    def test_nvm_mech_adjusts_balance(self):
        b = self._make_b()
        # NVM type - use PAYMENT_TYPE_NATIVE_NVM hex
        from packages.valory.skills.task_submission_abci.behaviours import PAYMENT_TYPE_NATIVE_NVM
        nvm_type = bytes.fromhex(PAYMENT_TYPE_NATIVE_NVM)
        mech_info = (nvm_type, "0xTRACK", 1000)
        process_tx = {"to": "0xTRACK", "value": 0, "data": b"\x00", "simulation_ok": True}
        split_funds = {"op0": 0}
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_adjust_mech_balance", side_effect=_gen_returning(500)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(400)),
            patch.object(b, "_get_process_payment_tx", side_effect=_gen_returning(dict(process_tx))),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is not None  # process_tx was added

    def test_nvm_mech_returns_none_when_adjust_fails(self):
        b = self._make_b()
        from packages.valory.skills.task_submission_abci.behaviours import PAYMENT_TYPE_NATIVE_NVM
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
    def _make_b(self):
        ctx = _make_full_ctx()
        ctx.params.manual_gas_limit = 0
        b = _DummyTransPrep(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        object.__setattr__(b, "_synchronized_data", mock_sd)
        return b

    def _patch_sd(self, b):
        return patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: b._synchronized_data))

    def test_to_multisend_full_success(self):
        b = self._make_b()
        raw_msg = _raw_tx_contract_msg({"data": "0x" + "ab" * 8})
        tx_hash = "cd" * 32
        with (
            self._patch_sd(b),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(raw_msg)),
            patch.object(b, "_get_safe_tx_hash", side_effect=_gen_returning(tx_hash)),
            patch("packages.valory.skills.task_submission_abci.behaviours.hash_payload_to_hex", return_value="encoded_payload"),
        ):
            result = _run_gen(b._to_multisend([{"to": "0xA", "value": 0, "data": b"\x00"}]))
        assert result == "encoded_payload"


# ---------------------------------------------------------------------------
# TransactionPreparationBehaviour.get_payload_content
# ---------------------------------------------------------------------------


class TestTransactionPreparationGetPayloadContent:
    def _make_b(self):
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        mock_sd = MagicMock()
        mock_sd.done_tasks = []
        mock_sd.safe_contract_address = "0xSAFE"
        object.__setattr__(b, "_synchronized_data", mock_sd)
        return b

    def _patch_sd(self, b):
        return patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: b._synchronized_data))

    def test_returns_error_payload_when_update_usage_tx_none(self):
        b = self._make_b()
        with (
            self._patch_sd(b),
            patch.object(b, "get_mech_update_hash_tx", side_effect=_gen_returning(None)),
            patch.object(b, "get_split_profit_txs", side_effect=_gen_returning([])),
            patch.object(b, "_get_offchain_tasks_deliver_data", side_effect=_gen_returning([])),
            patch.object(b, "_get_marketplace_tasks_deliver_data", side_effect=_gen_returning([])),
            patch.object(b, "get_update_usage_tx", side_effect=_gen_returning(None)),
        ):
            from packages.valory.skills.task_submission_abci.rounds import TransactionPreparationRound
            result = _run_gen(b.get_payload_content())
        assert result == TransactionPreparationRound.ERROR_PAYLOAD

    def test_returns_error_payload_when_multisend_fails(self):
        b = self._make_b()
        usage_tx = {"to": "0xHASH", "value": 0, "data": b"\x00"}
        with (
            self._patch_sd(b),
            patch.object(b, "get_mech_update_hash_tx", side_effect=_gen_returning(None)),
            patch.object(b, "get_split_profit_txs", side_effect=_gen_returning([])),
            patch.object(b, "_get_offchain_tasks_deliver_data", side_effect=_gen_returning([])),
            patch.object(b, "_get_marketplace_tasks_deliver_data", side_effect=_gen_returning([])),
            patch.object(b, "get_update_usage_tx", side_effect=_gen_returning(usage_tx)),
            patch.object(b, "_to_multisend", side_effect=_gen_returning(None)),
        ):
            from packages.valory.skills.task_submission_abci.rounds import TransactionPreparationRound
            result = _run_gen(b.get_payload_content())
        assert result == TransactionPreparationRound.ERROR_PAYLOAD

    def test_returns_multisend_str_on_full_success(self):
        b = self._make_b()
        usage_tx = {"to": "0xHASH", "value": 0, "data": b"\x00"}
        hash_tx = {"to": "0xMETA", "value": 0, "data": b"\x01"}
        with (
            self._patch_sd(b),
            patch.object(b, "get_mech_update_hash_tx", side_effect=_gen_returning(hash_tx)),
            patch.object(b, "get_split_profit_txs", side_effect=_gen_returning([])),
            patch.object(b, "_get_offchain_tasks_deliver_data", side_effect=_gen_returning([])),
            patch.object(b, "_get_marketplace_tasks_deliver_data", side_effect=_gen_returning([])),
            patch.object(b, "get_update_usage_tx", side_effect=_gen_returning(usage_tx)),
            patch.object(b, "_to_multisend", side_effect=_gen_returning("encoded_multisend")),
        ):
            result = _run_gen(b.get_payload_content())
        assert result == "encoded_multisend"

    def test_returns_error_when_deliver_tx_is_none(self):
        b = self._make_b()
        b._synchronized_data.done_tasks = [{"is_marketplace_mech": False, "request_id": "r1"}]
        usage_tx = {"to": "0xHASH", "value": 0, "data": b"\x00"}
        with (
            self._patch_sd(b),
            patch.object(b, "get_mech_update_hash_tx", side_effect=_gen_returning(None)),
            patch.object(b, "get_split_profit_txs", side_effect=_gen_returning([])),
            patch.object(b, "_get_offchain_tasks_deliver_data", side_effect=_gen_returning([])),
            patch.object(b, "_get_marketplace_tasks_deliver_data", side_effect=_gen_returning([])),
            patch.object(b, "_get_deliver_tx", side_effect=_gen_returning(None)),
            patch.object(b, "get_update_usage_tx", side_effect=_gen_returning(usage_tx)),
        ):
            from packages.valory.skills.task_submission_abci.rounds import TransactionPreparationRound
            result = _run_gen(b.get_payload_content())
        assert result == TransactionPreparationRound.ERROR_PAYLOAD

    def test_skips_deliver_with_failed_simulation(self):
        b = self._make_b()
        task = {"is_marketplace_mech": False, "request_id": "r1"}
        b._synchronized_data.done_tasks = [task]
        deliver_tx = {"to": "0xMECH", "value": 0, "data": b"\x00", "simulation_ok": False}
        usage_tx = {"to": "0xHASH", "value": 0, "data": b"\x00"}
        with (
            self._patch_sd(b),
            patch.object(b, "get_mech_update_hash_tx", side_effect=_gen_returning(None)),
            patch.object(b, "get_split_profit_txs", side_effect=_gen_returning([])),
            patch.object(b, "_get_offchain_tasks_deliver_data", side_effect=_gen_returning([])),
            patch.object(b, "_get_marketplace_tasks_deliver_data", side_effect=_gen_returning([])),
            patch.object(b, "_get_deliver_tx", side_effect=_gen_returning(dict(deliver_tx))),
            patch.object(b, "get_update_usage_tx", side_effect=_gen_returning(usage_tx)),
            patch.object(b, "_to_multisend", side_effect=_gen_returning("encoded")),
        ):
            result = _run_gen(b.get_payload_content())
        assert result == "encoded"  # deliver skipped but completed

    def test_appends_response_tx_when_present(self):
        b = self._make_b()
        response_tx = {"to": "0xRESP", "value": 0, "data": b"\x02"}
        task = {"is_marketplace_mech": False, "request_id": "r1", "transaction": response_tx}
        b._synchronized_data.done_tasks = [task]
        deliver_tx = {"to": "0xMECH", "value": 0, "data": b"\x00", "simulation_ok": True}
        usage_tx = {"to": "0xHASH", "value": 0, "data": b"\x00"}
        with (
            self._patch_sd(b),
            patch.object(b, "get_mech_update_hash_tx", side_effect=_gen_returning(None)),
            patch.object(b, "get_split_profit_txs", side_effect=_gen_returning([])),
            patch.object(b, "_get_offchain_tasks_deliver_data", side_effect=_gen_returning([])),
            patch.object(b, "_get_marketplace_tasks_deliver_data", side_effect=_gen_returning([])),
            patch.object(b, "_get_deliver_tx", side_effect=_gen_returning(dict(deliver_tx))),
            patch.object(b, "get_update_usage_tx", side_effect=_gen_returning(usage_tx)),
            patch.object(b, "_to_multisend", side_effect=_gen_returning("encoded")),
        ):
            result = _run_gen(b.get_payload_content())
        assert result == "encoded"


# ---------------------------------------------------------------------------
# _get_marketplace_tasks_deliver_data with NVM mech (lines 1801-1812)
# ---------------------------------------------------------------------------


class TestSynchronizedDataProperty:
    """Cover line 159: TaskExecutionBaseBehaviour.synchronized_data property."""

    def test_returns_synchronized_data_from_shared_state(self):
        ctx = _make_full_ctx()
        b = _DummyBase(name="b", skill_context=ctx)
        # ctx.state.synchronized_data is set in _make_full_ctx
        result = b.synchronized_data
        assert result is ctx.state.synchronized_data


class TestGetSplitProfitTxsTokenMech:
    """Cover lines 597-613, 630-631: token type mech path in get_split_profit_txs."""

    def _make_b(self):
        ctx = _make_full_ctx()
        return _DummyFunds(name="b", skill_context=ctx)

    def test_returns_none_when_no_mech_addresses(self):
        b = self._make_b()
        b.context.params.agent_mech_contract_addresses = []
        with patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None  # txs=[] → return None

    def test_token_mech_returns_none_when_token_address_none(self):
        b = self._make_b()
        from packages.valory.skills.task_submission_abci.behaviours import PAYMENT_TYPE_TOKEN
        token_type = bytes.fromhex(PAYMENT_TYPE_TOKEN)
        mech_info = (token_type, "0xTRACK", 1000)
        process_tx = {"to": "0xTRACK", "value": 0, "data": b"\x00", "simulation_ok": True}
        split_funds = {"op0": 500}  # non-zero amount triggers token path
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(b, "_get_process_payment_tx", side_effect=_gen_returning(dict(process_tx))),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
            patch.object(b, "_get_token_address", side_effect=_gen_returning(None)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is None

    def test_token_mech_success_with_token_transfer(self):
        b = self._make_b()
        from packages.valory.skills.task_submission_abci.behaviours import PAYMENT_TYPE_TOKEN
        token_type = bytes.fromhex(PAYMENT_TYPE_TOKEN)
        mech_info = (token_type, "0xTRACK", 1000)
        process_tx = {"to": "0xTRACK", "value": 0, "data": b"\x00", "simulation_ok": True}
        split_funds = {"op0": 500}
        token_transfer_tx = {"to": "0xMECH", "value": 0, "data": b"\x02"}
        with (
            patch.object(b, "_should_split_profits", side_effect=_gen_returning(True)),
            patch.object(b, "_get_mech_info", side_effect=_gen_returning(mech_info)),
            patch.object(b, "_calculate_mech_profits", side_effect=_gen_returning(900)),
            patch.object(b, "_get_process_payment_tx", side_effect=_gen_returning(dict(process_tx))),
            patch.object(b, "_split_funds", side_effect=_gen_returning(split_funds)),
            patch.object(b, "_get_token_address", side_effect=_gen_returning("0xTOKEN")),
            patch.object(b, "_get_token_transfer_tx", side_effect=_gen_returning(token_transfer_tx)),
        ):
            result = _run_gen(b.get_split_profit_txs())
        assert result is not None
        assert len(result) == 2  # process_tx + token_transfer_tx


class TestMarketplaceNvmMechPath:
    def _make_b(self):
        ctx = _make_full_ctx()
        b = _DummyTransPrep(name="b", skill_context=ctx)
        return b

    def test_nvm_mech_encodes_data(self):
        b = self._make_b()
        task = {
            "mech_address": "0xMECH",
            "requestId": 12345,
            "task_result": "de" * 16,
        }
        mock_sd = MagicMock()
        mock_sd.safe_contract_address = "0xSAFE"
        encoded_ids = [b"\x00" * 32]
        encoded_datas = [b"\xab" * 16]
        msg_deliver = _state_contract_msg({"data": b"\xdd", "simulation_ok": True})
        with (
            patch.object(type(b), "synchronized_data", new_callable=lambda: property(lambda self: mock_sd)),
            patch.object(b, "_get_is_nvm_mech", side_effect=_gen_returning(True)),
            patch.object(b, "_get_encoded_deliver_data", side_effect=_gen_returning((encoded_ids, encoded_datas))),
            patch.object(b, "get_contract_api_response", side_effect=_gen_returning(msg_deliver)),
        ):
            result = _run_gen(b._get_marketplace_tasks_deliver_data([task]))
        assert len(result) == 1
