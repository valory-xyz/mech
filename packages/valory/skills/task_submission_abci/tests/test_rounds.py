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
"""Tests for task_submission_abci.rounds."""

import json
from unittest.mock import MagicMock

import pytest

from packages.valory.skills.abstract_round_abci.base import (
    AbciAppDB,
    BaseSynchronizedData,
)
from packages.valory.skills.task_submission_abci.payloads import (
    TaskPoolingPayload,
    TransactionPayload,
)
from packages.valory.skills.task_submission_abci.rounds import (
    Event,
    FinishedTaskExecutionWithErrorRound,
    FinishedTaskPoolingRound,
    FinishedWithoutTasksRound,
    SynchronizedData,
    TaskPoolingRound,
    TaskSubmissionAbciApp,
    TransactionPreparationRound,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PARTICIPANTS = ["agent-0", "agent-1", "agent-2"]
_THRESHOLD = 3  # must be ceil(2/3 * 3) = 3 for 3 participants


def _make_db(**extra) -> AbciAppDB:
    """Create an AbciAppDB with the required fields."""
    data = {
        "participants": [_PARTICIPANTS],
        "consensus_threshold": [_THRESHOLD],
        "all_participants": [_PARTICIPANTS],
        **{k: [v] for k, v in extra.items()},
    }
    return AbciAppDB(data)


def _make_sync_data(**extra) -> SynchronizedData:
    return SynchronizedData(_make_db(**extra))


def _make_task(request_id: str) -> dict:
    return {"request_id": request_id, "result": "ok"}


def _payload_for(address: str, tasks: list) -> TaskPoolingPayload:
    return TaskPoolingPayload(sender=address, content=json.dumps(tasks))


def _make_pooling_round(payloads: dict, **db_extra) -> TaskPoolingRound:
    """Create a TaskPoolingRound with given payloads dict {address: TaskPoolingPayload}."""
    sync_data = _make_sync_data(**db_extra)
    ctx = MagicMock()
    round_ = TaskPoolingRound(synchronized_data=sync_data, context=ctx)
    round_.collection = payloads
    return round_


def _make_tx_round(payloads: dict, **db_extra) -> TransactionPreparationRound:
    sync_data = _make_sync_data(**db_extra)
    ctx = MagicMock()
    round_ = TransactionPreparationRound(synchronized_data=sync_data, context=ctx)
    round_.collection = payloads
    return round_


# ---------------------------------------------------------------------------
# SynchronizedData tests
# ---------------------------------------------------------------------------


class TestSynchronizedData:
    """Tests for SynchronizedData properties."""

    def test_most_voted_tx_hash(self):
        sd = _make_sync_data(most_voted_tx_hash="0xdeadbeef")
        assert sd.most_voted_tx_hash == "0xdeadbeef"

    def test_most_voted_tx_hash_missing_raises(self):
        sd = _make_sync_data()
        with pytest.raises(ValueError):
            _ = sd.most_voted_tx_hash

    def test_done_tasks_defaults_to_empty_list(self):
        sd = _make_sync_data()
        assert sd.done_tasks == []

    def test_done_tasks_returns_stored_value(self):
        tasks = [{"request_id": "1"}]
        sd = _make_sync_data(done_tasks=tasks)
        assert sd.done_tasks == tasks

    def test_final_tx_hash(self):
        sd = _make_sync_data(final_tx_hash="0xfinal")
        assert sd.final_tx_hash == "0xfinal"

    def test_final_tx_hash_missing_raises(self):
        sd = _make_sync_data()
        with pytest.raises(ValueError):
            _ = sd.final_tx_hash


# ---------------------------------------------------------------------------
# TaskPoolingRound tests
# ---------------------------------------------------------------------------


class TestTaskPoolingRound:
    """Tests for TaskPoolingRound."""

    def test_threshold_not_reached_returns_none(self):
        """Below threshold → end_block returns None."""
        # Only 1 payload, threshold is 3
        payloads = {"agent-0": _payload_for("agent-0", [_make_task("req-1")])}
        round_ = _make_pooling_round(payloads)
        assert round_.end_block() is None

    def test_threshold_reached_with_tasks_returns_done(self):
        """At threshold with tasks → returns (data, Event.DONE)."""
        task = _make_task("req-1")
        payloads = {
            "agent-0": _payload_for("agent-0", [task]),
            "agent-1": _payload_for("agent-1", [task]),
            "agent-2": _payload_for("agent-2", [task]),
        }
        round_ = _make_pooling_round(payloads)
        result = round_.end_block()
        assert result is not None
        data, event = result
        assert event == Event.DONE
        assert isinstance(data, SynchronizedData)
        assert len(data.done_tasks) == 1  # deduplication

    def test_threshold_reached_no_tasks_returns_no_tasks(self):
        """At threshold but all tasks are duplicates after dedup → NO_TASKS."""
        # All agents submit empty list → 0 tasks
        payloads = {
            "agent-0": _payload_for("agent-0", []),
            "agent-1": _payload_for("agent-1", []),
            "agent-2": _payload_for("agent-2", []),
        }
        round_ = _make_pooling_round(payloads)
        result = round_.end_block()
        assert result is not None
        data, event = result
        assert event == Event.NO_TASKS

    def test_deduplication_by_request_id(self):
        """Same request_id from multiple agents → deduplicated to one."""
        task = _make_task("req-dup")
        payloads = {
            "agent-0": _payload_for("agent-0", [task]),
            "agent-1": _payload_for("agent-1", [task]),
            "agent-2": _payload_for("agent-2", [task]),
        }
        round_ = _make_pooling_round(payloads)
        result = round_.end_block()
        assert result is not None
        data, event = result
        assert event == Event.DONE
        assert len(data.done_tasks) == 1
        assert data.done_tasks[0]["request_id"] == "req-dup"

    def test_tasks_sorted_by_request_id(self):
        """Tasks are sorted by request_id in ascending order."""
        tasks_agent0 = [_make_task("req-c"), _make_task("req-a")]
        tasks_agent1 = [_make_task("req-b")]
        tasks_agent2 = []
        payloads = {
            "agent-0": _payload_for("agent-0", tasks_agent0),
            "agent-1": _payload_for("agent-1", tasks_agent1),
            "agent-2": _payload_for("agent-2", tasks_agent2),
        }
        round_ = _make_pooling_round(payloads)
        result = round_.end_block()
        assert result is not None
        data, event = result
        assert event == Event.DONE
        request_ids = [t["request_id"] for t in data.done_tasks]
        assert request_ids == sorted(request_ids)

    def test_collection_threshold_reached_property_true(self):
        payloads = {
            "agent-0": _payload_for("agent-0", []),
            "agent-1": _payload_for("agent-1", []),
            "agent-2": _payload_for("agent-2", []),
        }
        round_ = _make_pooling_round(payloads)
        assert round_.collection_threshold_reached is True

    def test_collection_threshold_reached_property_false(self):
        payloads = {"agent-0": _payload_for("agent-0", [])}
        round_ = _make_pooling_round(payloads)
        assert round_.collection_threshold_reached is False

    def test_unique_tasks_from_different_agents(self):
        """Different request_ids from different agents → all kept."""
        payloads = {
            "agent-0": _payload_for("agent-0", [_make_task("req-1")]),
            "agent-1": _payload_for("agent-1", [_make_task("req-2")]),
            "agent-2": _payload_for("agent-2", [_make_task("req-3")]),
        }
        round_ = _make_pooling_round(payloads)
        result = round_.end_block()
        assert result is not None
        data, event = result
        assert event == Event.DONE
        assert len(data.done_tasks) == 3


# ---------------------------------------------------------------------------
# TransactionPreparationRound tests
# ---------------------------------------------------------------------------


class TestTransactionPreparationRound:
    """Tests for TransactionPreparationRound."""

    def _tx_payload(self, sender: str, content: str) -> TransactionPayload:
        return TransactionPayload(sender=sender, content=content)

    def test_below_threshold_returns_none(self):
        payloads = {"agent-0": self._tx_payload("agent-0", "0xhash")}
        round_ = _make_tx_round(payloads)
        assert round_.end_block() is None

    def test_threshold_with_valid_hash_returns_done(self):
        tx_hash = "0xabcdef"
        payloads = {
            "agent-0": self._tx_payload("agent-0", tx_hash),
            "agent-1": self._tx_payload("agent-1", tx_hash),
            "agent-2": self._tx_payload("agent-2", tx_hash),
        }
        round_ = _make_tx_round(payloads)
        result = round_.end_block()
        assert result is not None
        data, event = result
        assert event == Event.DONE
        assert data.most_voted_tx_hash == tx_hash

    def test_threshold_with_error_payload_returns_error(self):
        payloads = {
            "agent-0": self._tx_payload("agent-0", "error"),
            "agent-1": self._tx_payload("agent-1", "error"),
            "agent-2": self._tx_payload("agent-2", "error"),
        }
        round_ = _make_tx_round(payloads, done_tasks=[_make_task("req-1")])
        result = round_.end_block()
        assert result is not None
        data, event = result
        assert event == Event.ERROR
        # done_tasks are cleared on error
        assert data.done_tasks == []

    def test_no_majority_possible_returns_no_majority(self):
        """All agents vote differently → majority impossible → NO_MAJORITY."""
        payloads = {
            "agent-0": self._tx_payload("agent-0", "0xhash-a"),
            "agent-1": self._tx_payload("agent-1", "0xhash-b"),
            "agent-2": self._tx_payload("agent-2", "0xhash-c"),
        }
        round_ = _make_tx_round(payloads)
        result = round_.end_block()
        assert result is not None
        data, event = result
        assert event == Event.NO_MAJORITY
        assert data.done_tasks == []


# ---------------------------------------------------------------------------
# TaskSubmissionAbciApp tests
# ---------------------------------------------------------------------------


class TestTaskSubmissionAbciApp:
    """Tests for the TaskSubmissionAbciApp FSM class attributes."""

    def test_initial_round_cls(self):
        assert TaskSubmissionAbciApp.initial_round_cls is TaskPoolingRound

    def test_initial_states(self):
        assert TaskPoolingRound in TaskSubmissionAbciApp.initial_states

    def test_final_states(self):
        assert FinishedTaskPoolingRound in TaskSubmissionAbciApp.final_states
        assert FinishedWithoutTasksRound in TaskSubmissionAbciApp.final_states
        assert FinishedTaskExecutionWithErrorRound in TaskSubmissionAbciApp.final_states

    def test_event_to_timeout_has_correct_values(self):
        assert TaskSubmissionAbciApp.event_to_timeout[Event.TASK_EXECUTION_ROUND_TIMEOUT] == 60.0
        assert TaskSubmissionAbciApp.event_to_timeout[Event.ROUND_TIMEOUT] == 60.0

    def test_cross_period_keys_include_done_tasks(self):
        assert "done_tasks" in TaskSubmissionAbciApp.cross_period_persisted_keys

    def test_cross_period_keys_include_final_tx_hash(self):
        assert "final_tx_hash" in TaskSubmissionAbciApp.cross_period_persisted_keys

    def test_pooling_round_done_transitions_to_tx_preparation(self):
        transitions = TaskSubmissionAbciApp.transition_function[TaskPoolingRound]
        assert transitions[Event.DONE] is TransactionPreparationRound

    def test_pooling_round_no_tasks_transitions_to_finished_without_tasks(self):
        transitions = TaskSubmissionAbciApp.transition_function[TaskPoolingRound]
        assert transitions[Event.NO_TASKS] is FinishedWithoutTasksRound

    def test_pooling_round_timeout_loops_back(self):
        transitions = TaskSubmissionAbciApp.transition_function[TaskPoolingRound]
        assert transitions[Event.ROUND_TIMEOUT] is TaskPoolingRound

    def test_tx_preparation_done_transitions_to_finished_pooling(self):
        transitions = TaskSubmissionAbciApp.transition_function[TransactionPreparationRound]
        assert transitions[Event.DONE] is FinishedTaskPoolingRound

    def test_tx_preparation_error_transitions_to_finished_error(self):
        transitions = TaskSubmissionAbciApp.transition_function[TransactionPreparationRound]
        assert transitions[Event.ERROR] is FinishedTaskExecutionWithErrorRound

    def test_tx_preparation_no_majority_transitions_to_finished_error(self):
        transitions = TaskSubmissionAbciApp.transition_function[TransactionPreparationRound]
        assert transitions[Event.NO_MAJORITY] is FinishedTaskExecutionWithErrorRound
