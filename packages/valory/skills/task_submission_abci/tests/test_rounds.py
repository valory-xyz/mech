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
import logging
from typing import Any, Union, cast
from unittest.mock import MagicMock

import pytest

from packages.valory.skills.abstract_round_abci.base import AbciAppDB, get_name
from packages.valory.skills.task_submission_abci.payloads import (
    PostTxSettlementPayload,
    TaskPoolingPayload,
    TransactionPayload,
)
from packages.valory.skills.task_submission_abci.rounds import (
    Event,
    FinishedPostTxSettlementRound,
    FinishedTaskExecutionWithErrorRound,
    FinishedTaskPoolingRound,
    FinishedWithoutTasksRound,
    PostTxSettlementRound,
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


def _make_db(**extra: Any) -> AbciAppDB:
    """Create an AbciAppDB with the required fields."""
    data: dict = {
        "participants": [_PARTICIPANTS],
        "consensus_threshold": [_THRESHOLD],
        "all_participants": [_PARTICIPANTS],
        **{k: [v] for k, v in extra.items()},
    }
    return AbciAppDB(data)


def _make_sync_data(**extra: Any) -> SynchronizedData:
    return SynchronizedData(_make_db(**extra))


def _make_task(request_id: Union[int, str]) -> dict:
    # ``Union[int, str]`` so tests can pass both ``str`` (off-chain wire
    # format) and ``int`` (on-chain bytes32 → int conversion) to exercise
    # the mixed-type sort / dedup path — see
    # ``test_mixed_int_and_str_request_ids_do_not_crash``. Narrower than
    # ``Any``: rejects ``None`` / ``float`` / ``list`` which
    # ``request_id`` can't legitimately be.
    return {"request_id": request_id, "result": "ok"}


def _payload_for(address: str, tasks: list) -> TaskPoolingPayload:
    return TaskPoolingPayload(sender=address, content=json.dumps(tasks))


def _make_pooling_round(payloads: dict, **db_extra: Any) -> TaskPoolingRound:
    """Create a TaskPoolingRound with given payloads dict {address: TaskPoolingPayload}."""
    sync_data = _make_sync_data(**db_extra)
    ctx = MagicMock()
    round_ = TaskPoolingRound(synchronized_data=sync_data, context=ctx)
    round_.collection = payloads
    return round_


def _make_tx_round(payloads: dict, **db_extra: Any) -> TransactionPreparationRound:
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

    def test_most_voted_tx_hash(self) -> None:
        """Test most_voted_tx_hash returns the stored value."""
        sd = _make_sync_data(most_voted_tx_hash="0xdeadbeef")
        assert sd.most_voted_tx_hash == "0xdeadbeef"

    def test_most_voted_tx_hash_missing_raises(self) -> None:
        """Test most_voted_tx_hash raises ValueError when missing."""
        sd = _make_sync_data()
        with pytest.raises(ValueError):
            _ = sd.most_voted_tx_hash

    def test_done_tasks_defaults_to_empty_list(self) -> None:
        """Test done_tasks defaults to empty list."""
        sd = _make_sync_data()
        assert sd.done_tasks == []

    def test_done_tasks_returns_stored_value(self) -> None:
        """Test done_tasks returns the stored value."""
        tasks = [{"request_id": "1"}]
        sd = _make_sync_data(done_tasks=tasks)
        assert sd.done_tasks == tasks

    def test_final_tx_hash(self) -> None:
        """Test final_tx_hash returns the stored value."""
        sd = _make_sync_data(final_tx_hash="0xfinal")
        assert sd.final_tx_hash == "0xfinal"

    def test_final_tx_hash_missing_raises(self) -> None:
        """Test final_tx_hash raises ValueError when missing."""
        sd = _make_sync_data()
        with pytest.raises(ValueError):
            _ = sd.final_tx_hash

    def test_submitted_request_ids_defaults_to_empty_list(self) -> None:
        """``submitted_request_ids`` reads ``[]`` when the key is unset."""
        sd = _make_sync_data()
        assert sd.submitted_request_ids == []

    def test_submitted_request_ids_returns_stored_value(self) -> None:
        """``submitted_request_ids`` reads back the stored list."""
        ids = ["req-1", "req-2"]
        sd = _make_sync_data(submitted_request_ids=ids)
        assert sd.submitted_request_ids == ids

    @pytest.mark.parametrize(
        "bad_value",
        [
            pytest.param("not-a-list", id="non-list"),
            pytest.param([1, 2, 3], id="non-str-entries"),
            pytest.param(["ok", 42], id="mixed-str-and-int"),
        ],
    )
    def test_submitted_request_ids_degrades_to_empty_on_bad_shape(
        self,
        bad_value: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A malformed value logs an error and yields ``[]``.

        The property is read at the top of every FSM cycle across
        every participant on the same consensus block. Raising would
        crash-loop the whole fleet with no in-band recovery because
        the value is cross-period-persisted (``db.create`` copies it
        forward). Degrading to ``[]`` keeps the drift detectable via
        the ``error`` log while the FSM continues.
        """
        sd = _make_sync_data(submitted_request_ids=bad_value)
        with caplog.at_level(logging.ERROR):
            assert sd.submitted_request_ids == []
        assert any(
            "submitted_request_ids invariant broken" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# TaskPoolingRound tests
# ---------------------------------------------------------------------------


class TestTaskPoolingRound:
    """Tests for TaskPoolingRound."""

    def test_threshold_not_reached_returns_none(self) -> None:
        """Below threshold → end_block returns None."""
        # Only 1 payload, threshold is 3
        payloads = {"agent-0": _payload_for("agent-0", [_make_task("req-1")])}
        round_ = _make_pooling_round(payloads)
        assert round_.end_block() is None

    def test_threshold_reached_with_tasks_returns_done(self) -> None:
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
        sd = cast(SynchronizedData, data)
        assert len(sd.done_tasks) == 1  # deduplication

    def test_threshold_reached_no_tasks_returns_no_tasks(self) -> None:
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

    def test_submitted_request_ids_cleared_on_done(self) -> None:
        """DONE clears the hand-off from the previous cycle.

        By the time ``end_block`` fires here, every participant's
        ``TaskPoolingBehaviour.async_act`` has already run
        ``handle_submitted_tasks`` and pruned its local
        ``shared_state[DONE_TASKS]``. Clearing the consensus field
        makes the hand-off one-shot: without this, the next cycle
        re-runs the "already submitted" block on stale ids every
        period, and re-swept requests get pruned before they can be
        pooled.
        """
        task = _make_task("req-new")
        payloads = {
            "agent-0": _payload_for("agent-0", [task]),
            "agent-1": _payload_for("agent-1", [task]),
            "agent-2": _payload_for("agent-2", [task]),
        }
        round_ = _make_pooling_round(
            payloads,
            **{
                get_name(SynchronizedData.submitted_request_ids): [
                    "stale-a",
                    "stale-b",
                ]
            },
        )
        result = round_.end_block()
        assert result is not None
        data, event = result
        assert event == Event.DONE
        assert cast(SynchronizedData, data).submitted_request_ids == []

    def test_submitted_request_ids_cleared_on_no_tasks(self) -> None:
        """NO_TASKS also clears the hand-off.

        A cycle that produced no new tasks still needs to consume the
        prior settlement's ids so the next cycle doesn't loop on them.
        """
        payloads = {
            "agent-0": _payload_for("agent-0", []),
            "agent-1": _payload_for("agent-1", []),
            "agent-2": _payload_for("agent-2", []),
        }
        round_ = _make_pooling_round(
            payloads,
            **{
                get_name(SynchronizedData.submitted_request_ids): [
                    "stale-a",
                    "stale-b",
                ]
            },
        )
        result = round_.end_block()
        assert result is not None
        data, event = result
        assert event == Event.NO_TASKS
        assert cast(SynchronizedData, data).submitted_request_ids == []

    def test_deduplication_by_request_id(self) -> None:
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
        sd = cast(SynchronizedData, data)
        assert len(sd.done_tasks) == 1
        assert sd.done_tasks[0]["request_id"] == "req-dup"

    def test_tasks_sorted_by_request_id(self) -> None:
        """Tasks are sorted by request_id in ascending order."""
        tasks_agent0 = [_make_task("req-c"), _make_task("req-a")]
        tasks_agent1 = [_make_task("req-b")]
        tasks_agent2: list = []
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
        sd = cast(SynchronizedData, data)
        request_ids = [t["request_id"] for t in sd.done_tasks]
        assert request_ids == sorted(request_ids)

    def test_mixed_int_and_str_request_ids_do_not_crash(self) -> None:
        """A batch with both ``int`` and ``str`` request_ids sorts + dedups cleanly.

        Regression guard for the crash observed in prod when an off-chain
        (``str`` request_id from the HTTP body) and an on-chain (``int``
        request_id from ``bytes32.from_bytes``) done_task landed in the
        same pooling batch. Pre-fix, ``sorted(..., key=lambda x:
        x["request_id"])`` raised ``TypeError: '<' not supported between
        instances of 'int' and 'str'`` and restarted the aea container.
        The same restart-transition state also let the un-normalized
        dedup set treat ``42`` and ``"42"`` as distinct — pre-fix the
        batch would then have delivered the same request twice in one
        multisend (a costlier failure than the crash itself).

        The ingress fix in :mod:`task_execution.handlers` coerces
        off-chain request_ids to ``int`` so this shape never happens in
        new writes, but a mech resuming with pre-fix mixed
        ``done_tasks`` in shared state at redeploy time would still hit
        both bugs on the first pooling round without the ``str`` cast on
        the dedup + sort keys. This test locks in that safety net.

        Agent-0 emits both ``42`` (int) and ``"100"`` (str) — the mixed
        types that used to crash the sort. Agent-1 emits ``7`` and
        ``"42"`` — a same-value cross-type duplicate of Agent-0's ``42``
        which used to slip through dedup. Agent-2 emits ``"500"``. Post
        the two normalizations we expect exactly 4 canonical rows (not
        5) in a deterministic str-sorted order.
        """
        tasks_agent0 = [_make_task(42), _make_task("100")]
        tasks_agent1 = [_make_task(7), _make_task("42")]
        tasks_agent2 = [_make_task("500")]
        payloads = {
            "agent-0": _payload_for("agent-0", tasks_agent0),
            "agent-1": _payload_for("agent-1", tasks_agent1),
            "agent-2": _payload_for("agent-2", tasks_agent2),
        }
        round_ = _make_pooling_round(payloads)
        # Must not raise TypeError.
        result = round_.end_block()
        assert result is not None
        data, event = result
        assert event == Event.DONE
        sd = cast(SynchronizedData, data)
        # Dedup collapses ``42`` and ``"42"`` into one row (whichever the
        # iteration order encountered first — Agent-0's int ``42``).
        # Total canonical rows: {42, "100", "500", 7} = 4, not 5.
        assert len(sd.done_tasks) == 4, sd.done_tasks
        # Pin the literal str-sorted order so a future change of the
        # sort key (e.g. back to numeric) is observable as a test
        # failure rather than silently passing.
        request_ids = [t["request_id"] for t in sd.done_tasks]
        assert request_ids == ["100", 42, "500", 7], request_ids

    def test_falsy_request_id_is_skipped_and_does_not_collide_with_none(self) -> None:
        """Falsy ``request_id`` rows are dropped, not folded into a shared bucket.

        ``all_done_tasks`` is a bag of every participant's payloads with
        no schema validation, so a misbehaving or out-of-version agent
        could inject a row with a missing / ``None`` / ``""`` ``request_id``.
        Pre-fix, ``str(obj.get("request_id", ""))`` mapped missing to
        ``""`` and explicit ``None`` to ``"None"`` — two entries in either
        bucket would silently drop the second row from ``unique_objects``,
        including a legitimate row from another agent that happened to
        stringify to the same key. Post-fix the falsy rows are dropped
        with a warning and cannot collide.
        """
        agent0 = [{"request_id": 42, "task_result": "cid42"}]  # legit
        # Two agents inject junk; three participants total so consensus can form.
        agent1 = [
            {"task_result": "no-id"},  # missing key
            {"request_id": None, "task_result": "none-id"},  # explicit None
            {"request_id": "", "task_result": "empty-str"},  # explicit empty
        ]
        agent2 = [{"request_id": 100, "task_result": "cid100"}]  # legit
        payloads = {
            "agent-0": _payload_for("agent-0", agent0),
            "agent-1": _payload_for("agent-1", agent1),
            "agent-2": _payload_for("agent-2", agent2),
        }
        round_ = _make_pooling_round(payloads)
        result = round_.end_block()
        assert result is not None
        data, event = result
        assert event == Event.DONE
        sd = cast(SynchronizedData, data)
        # The two legit rows survive; the three falsy rows are dropped.
        request_ids = [t["request_id"] for t in sd.done_tasks]
        assert request_ids == [100, 42], request_ids

    def test_request_id_zero_int_is_kept(self) -> None:
        """``request_id=0`` (a legitimate int) is not treated as falsy."""
        payloads = {
            "agent-0": _payload_for("agent-0", [_make_task(0)]),
            "agent-1": _payload_for("agent-1", [_make_task(1)]),
            "agent-2": _payload_for("agent-2", []),
        }
        round_ = _make_pooling_round(payloads)
        result = round_.end_block()
        assert result is not None
        data, event = result
        assert event == Event.DONE
        sd = cast(SynchronizedData, data)
        request_ids = [t["request_id"] for t in sd.done_tasks]
        # Both survive dedup — 0 is a valid uint256 id and must not be
        # swept into the falsy-skip branch.
        assert sorted(str(r) for r in request_ids) == ["0", "1"]

    def test_collection_threshold_reached_property_true(self) -> None:
        """Test collection_threshold_reached is True when threshold is met."""
        payloads = {
            "agent-0": _payload_for("agent-0", []),
            "agent-1": _payload_for("agent-1", []),
            "agent-2": _payload_for("agent-2", []),
        }
        round_ = _make_pooling_round(payloads)
        assert round_.collection_threshold_reached is True

    def test_collection_threshold_reached_property_false(self) -> None:
        """Test collection_threshold_reached is False when threshold is not met."""
        payloads = {"agent-0": _payload_for("agent-0", [])}
        round_ = _make_pooling_round(payloads)
        assert round_.collection_threshold_reached is False

    def test_unique_tasks_from_different_agents(self) -> None:
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
        sd = cast(SynchronizedData, data)
        assert len(sd.done_tasks) == 3


# ---------------------------------------------------------------------------
# TransactionPreparationRound tests
# ---------------------------------------------------------------------------


class TestTransactionPreparationRound:
    """Tests for TransactionPreparationRound."""

    def _tx_payload(self, sender: str, content: str) -> TransactionPayload:
        """Create a TransactionPayload for the given sender and content."""
        return TransactionPayload(sender=sender, content=content)

    def test_below_threshold_returns_none(self) -> None:
        """Test end_block returns None when below threshold."""
        payloads = {"agent-0": self._tx_payload("agent-0", "0xhash")}
        round_ = _make_tx_round(payloads)
        assert round_.end_block() is None

    def test_threshold_with_valid_hash_returns_done(self) -> None:
        """Test end_block returns DONE event with valid tx hash at threshold."""
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
        sd = cast(SynchronizedData, data)
        assert sd.most_voted_tx_hash == tx_hash

    def test_threshold_with_error_payload_returns_error(self) -> None:
        """Test end_block returns ERROR event when all payloads are error."""
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
        sd = cast(SynchronizedData, data)
        assert sd.done_tasks == []

    def test_no_majority_possible_returns_no_majority(self) -> None:
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
        sd = cast(SynchronizedData, data)
        assert sd.done_tasks == []


# ---------------------------------------------------------------------------
# TaskSubmissionAbciApp tests
# ---------------------------------------------------------------------------


class TestTaskSubmissionAbciApp:
    """Tests for the TaskSubmissionAbciApp FSM class attributes."""

    def test_app_class_attributes(self) -> None:
        """Initial round, states, timeouts, and cross-period keys are configured correctly."""
        assert TaskSubmissionAbciApp.initial_round_cls is TaskPoolingRound
        assert TaskPoolingRound in TaskSubmissionAbciApp.initial_states
        # PostTxSettlementRound is reached from the composed FSM (settlement
        # confirms in a different skill, then routes here), so it must be
        # declared as an initial state even though the FSM never *starts*
        # there. A missing entry crashes the composition validator.
        assert PostTxSettlementRound in TaskSubmissionAbciApp.initial_states
        assert FinishedTaskPoolingRound in TaskSubmissionAbciApp.final_states
        assert FinishedWithoutTasksRound in TaskSubmissionAbciApp.final_states
        assert FinishedTaskExecutionWithErrorRound in TaskSubmissionAbciApp.final_states
        assert FinishedPostTxSettlementRound in TaskSubmissionAbciApp.final_states
        assert (
            TaskSubmissionAbciApp.event_to_timeout[Event.TASK_EXECUTION_ROUND_TIMEOUT]
            == 60.0
        )
        assert TaskSubmissionAbciApp.event_to_timeout[Event.ROUND_TIMEOUT] == 60.0
        # ``done_tasks`` is intentionally NOT cross-period-persisted;
        # the id hand-off rides ``submitted_request_ids`` instead. See
        # :class:`TestCrossPeriodPersistedKeys` for the schema invariant.
        assert (
            "submitted_request_ids" in TaskSubmissionAbciApp.cross_period_persisted_keys
        )
        assert "final_tx_hash" in TaskSubmissionAbciApp.cross_period_persisted_keys
        assert "done_tasks" not in TaskSubmissionAbciApp.cross_period_persisted_keys

    def test_fsm_transitions(self) -> None:
        """All FSM transitions route to expected destination rounds."""
        pooling = TaskSubmissionAbciApp.transition_function[TaskPoolingRound]
        assert pooling[Event.DONE] is TransactionPreparationRound
        assert pooling[Event.NO_TASKS] is FinishedWithoutTasksRound
        assert pooling[Event.ROUND_TIMEOUT] is TaskPoolingRound

        tx_prep = TaskSubmissionAbciApp.transition_function[TransactionPreparationRound]
        assert tx_prep[Event.DONE] is FinishedTaskPoolingRound
        assert tx_prep[Event.ERROR] is FinishedTaskExecutionWithErrorRound
        assert tx_prep[Event.NO_MAJORITY] is FinishedTaskExecutionWithErrorRound

        # Post-settlement leg: DONE leaves the skill via the new degenerate
        # final state, both NO_MAJORITY and ROUND_TIMEOUT loop back so a
        # transient consensus dip doesn't crash the FSM (the predict-api write
        # itself is idempotent, so re-entry is safe).
        post_tx = TaskSubmissionAbciApp.transition_function[PostTxSettlementRound]
        assert post_tx[Event.DONE] is FinishedPostTxSettlementRound
        assert post_tx[Event.NO_MAJORITY] is PostTxSettlementRound
        assert post_tx[Event.ROUND_TIMEOUT] is PostTxSettlementRound


# ---------------------------------------------------------------------------
# TaskPoolingRound deduplication edge cases
# ---------------------------------------------------------------------------


class TestTaskPoolingRoundDeduplication:
    """Edge-case tests for the deduplication and sorting in end_block."""

    def test_partial_overlap_across_agents(self) -> None:
        """Agents submit overlapping but not identical sets → union, deduplicated."""
        payloads = {
            "agent-0": _payload_for("agent-0", [_make_task("r1"), _make_task("r2")]),
            "agent-1": _payload_for("agent-1", [_make_task("r2"), _make_task("r3")]),
            "agent-2": _payload_for("agent-2", [_make_task("r1"), _make_task("r3")]),
        }
        round_ = _make_pooling_round(payloads)
        result = round_.end_block()
        assert result is not None
        sd = cast(SynchronizedData, result[0])
        ids = [t["request_id"] for t in sd.done_tasks]
        assert ids == ["r1", "r2", "r3"]

    def test_task_missing_request_id_is_skipped(self) -> None:
        """Task without ``request_id`` is skipped rather than folded into the ``""`` bucket.

        Round-3: falsy ``request_id`` rows are dropped with a warning so
        that two malformed rows can't collide on the shared ``""`` /
        ``"None"`` key and silently drop a legitimate row from another
        agent. Previously they were sorted with an empty-string fallback.
        """
        task_no_id = {"result": "ok"}
        payloads = {
            "agent-0": _payload_for("agent-0", [_make_task("b"), task_no_id]),
            "agent-1": _payload_for("agent-1", [_make_task("a")]),
            "agent-2": _payload_for("agent-2", []),
        }
        round_ = _make_pooling_round(payloads)
        result = round_.end_block()
        assert result is not None
        sd = cast(SynchronizedData, result[0])
        # Falsy-id row dropped; the two legit rows sort alphabetically.
        ids = [t.get("request_id") for t in sd.done_tasks]
        assert ids == ["a", "b"]

    def test_single_agent_all_unique(self) -> None:
        """One agent submits all tasks; others submit nothing → no dedup needed."""
        payloads = {
            "agent-0": _payload_for(
                "agent-0", [_make_task("x"), _make_task("y"), _make_task("z")]
            ),
            "agent-1": _payload_for("agent-1", []),
            "agent-2": _payload_for("agent-2", []),
        }
        round_ = _make_pooling_round(payloads)
        result = round_.end_block()
        assert result is not None
        sd = cast(SynchronizedData, result[0])
        assert len(sd.done_tasks) == 3
        assert result[1] == Event.DONE

    def test_all_agents_same_single_task(self) -> None:
        """All agents report the same single task → deduplicated to 1."""
        payloads = {
            "agent-0": _payload_for("agent-0", [_make_task("same")]),
            "agent-1": _payload_for("agent-1", [_make_task("same")]),
            "agent-2": _payload_for("agent-2", [_make_task("same")]),
        }
        round_ = _make_pooling_round(payloads)
        result = round_.end_block()
        assert result is not None
        sd = cast(SynchronizedData, result[0])
        assert len(sd.done_tasks) == 1
        assert sd.done_tasks[0]["request_id"] == "same"

    def test_first_occurrence_wins_dedup(self) -> None:
        """When duplicate request_ids have different data, first occurrence is kept."""
        task_v1 = {"request_id": "dup", "result": "first"}
        task_v2 = {"request_id": "dup", "result": "second"}
        payloads = {
            "agent-0": _payload_for("agent-0", [task_v1]),
            "agent-1": _payload_for("agent-1", [task_v2]),
            "agent-2": _payload_for("agent-2", []),
        }
        round_ = _make_pooling_round(payloads)
        result = round_.end_block()
        assert result is not None
        sd = cast(SynchronizedData, result[0])
        assert len(sd.done_tasks) == 1
        assert sd.done_tasks[0]["result"] == "first"


# ---------------------------------------------------------------------------
# PostTxSettlementRound tests
# ---------------------------------------------------------------------------


def _post_tx_payload_for(address: str) -> PostTxSettlementPayload:
    return PostTxSettlementPayload(sender=address, content="done")


def _make_post_tx_round(payloads: dict, **db_extra: Any) -> PostTxSettlementRound:
    sync_data = _make_sync_data(**db_extra)
    ctx = MagicMock()
    round_ = PostTxSettlementRound(synchronized_data=sync_data, context=ctx)
    round_.collection = payloads
    return round_


class TestPostTxSettlementRound:
    """End-to-end behaviour pin for ``PostTxSettlementRound``.

    The round is a CollectSameUntilThresholdRound: it only needs consensus
    on participation (the predict-api POST itself is fire-and-forget,
    idempotent server-side). DONE on threshold, NO_MAJORITY only when
    consensus is impossible.
    """

    def test_below_threshold_returns_none(self) -> None:
        """One vote on a 3-of-3 board: end_block is still waiting."""
        payloads = {"agent-0": _post_tx_payload_for("agent-0")}
        round_ = _make_post_tx_round(payloads)
        assert round_.end_block() is None

    def test_threshold_reached_returns_done(self) -> None:
        """All three agents voted ``"done"`` → DONE."""
        payloads = {p: _post_tx_payload_for(p) for p in _PARTICIPANTS}
        round_ = _make_post_tx_round(payloads)
        result = round_.end_block()
        assert result is not None
        _, event = result
        assert event == Event.DONE

    def test_no_majority_when_impossible(self) -> None:
        """Round emits NO_MAJORITY when consensus on participation is impossible.

        If every agent votes a different value on a CollectSame round, no
        shared payload can ever land threshold; the composition then routes
        NO_MAJORITY back to PostTxSettlementRound for a clean retry.
        """

        # Build payloads that all disagree: each agent sends a unique
        # content string, so no value can ever reach the
        # collect-same-until-threshold majority. The base round helper's
        # ``is_majority_possible`` will return False once the disagreement
        # is locked in.
        payloads = {
            p: PostTxSettlementPayload(sender=p, content=f"done-{p}")
            for p in _PARTICIPANTS
        }
        round_ = _make_post_tx_round(payloads)
        result = round_.end_block()
        assert result is not None
        _, event = result
        assert event == Event.NO_MAJORITY

    def test_done_tasks_preserved_on_done(self) -> None:
        """``done_tasks`` MUST survive DONE untouched within the same period.

        The behaviour side still reads ``done_tasks`` for predict-api
        writes and log emission during this round; clearing it here
        breaks those reads. The id hand-off to the next cycle rides
        ``submitted_request_ids`` (see the sibling regression test),
        so ``done_tasks`` doesn't need to leave this period.
        """
        tasks = [_make_task("req-1"), _make_task("req-2")]
        payloads = {p: _post_tx_payload_for(p) for p in _PARTICIPANTS}
        round_ = _make_post_tx_round(payloads, done_tasks=tasks)
        result = round_.end_block()
        assert result is not None
        new_sync_data, event = result
        assert event == Event.DONE
        assert cast(SynchronizedData, new_sync_data).done_tasks == tasks

    def test_submitted_request_ids_written_on_done(self) -> None:
        """DONE writes the id list from ``done_tasks`` for the next cycle to prune.

        Ids are extracted from this period's ``done_tasks`` and
        exposed via
        :attr:`SynchronizedData.submitted_request_ids`, which
        the next cycle's
        :meth:`TaskPoolingBehaviour.handle_submitted_tasks` reads to
        prune ``shared_state[DONE_TASKS]`` by request_id.
        """
        tasks = [_make_task("req-a"), _make_task("req-b")]
        payloads = {p: _post_tx_payload_for(p) for p in _PARTICIPANTS}
        round_ = _make_post_tx_round(payloads, done_tasks=tasks)
        result = round_.end_block()
        assert result is not None
        new_sync_data, event = result
        assert event == Event.DONE
        assert cast(SynchronizedData, new_sync_data).submitted_request_ids == [
            "req-a",
            "req-b",
        ]

    def test_no_write_when_done_tasks_empty(self) -> None:
        """DONE with no ``done_tasks`` leaves the hand-off untouched.

        ``PostTxSettlementRound`` is reachable via the delivery-rate
        settlement path and the settlement-internal ``ResetRound``
        retry; both arrive with ``done_tasks == []``. Overwriting a
        still-pending ``submitted_request_ids`` from a prior real
        settlement in those cases silently loses the prune and the
        delivered batch is re-pooled next cycle.
        """
        payloads = {p: _post_tx_payload_for(p) for p in _PARTICIPANTS}
        pending_handoff = ["still-pending-a", "still-pending-b"]
        round_ = _make_post_tx_round(
            payloads,
            **{get_name(SynchronizedData.submitted_request_ids): pending_handoff},
        )
        result = round_.end_block()
        assert result is not None
        new_sync_data, event = result
        assert event == Event.DONE
        assert (
            cast(SynchronizedData, new_sync_data).submitted_request_ids
            == pending_handoff
        )

    def test_submitted_request_ids_normalised_to_str(self) -> None:
        """Ids are ``str``-normalised to match the prune-site lookup key.

        Mixed ``int`` / ``str`` request_id shapes can survive an
        in-place restart. Normalising to ``str`` here keeps parity
        with the downstream match in
        :meth:`TaskExecutionBaseBehaviour.remove_tasks_by_id`.
        """
        tasks = [_make_task(42), _make_task("req-x")]
        payloads = {p: _post_tx_payload_for(p) for p in _PARTICIPANTS}
        round_ = _make_post_tx_round(payloads, done_tasks=tasks)
        result = round_.end_block()
        assert result is not None
        new_sync_data, event = result
        assert event == Event.DONE
        assert cast(SynchronizedData, new_sync_data).submitted_request_ids == [
            "42",
            "req-x",
        ]

    def test_submitted_request_ids_skips_falsy_ids(self) -> None:
        """A missing / empty ``request_id`` is dropped from the id list.

        Mirrors the falsy-id skip in
        :meth:`TaskPoolingRound.end_block`. Legit ``0`` (int) is
        deliberately kept via the explicit ``is None or == ""``
        check rather than a ``not`` truthiness test.
        """
        tasks = [
            {"request_id": 0, "tool": "t0"},  # legit id, keep
            {"request_id": None, "tool": "t-drop"},  # missing, skip
            {"request_id": "", "tool": "t-drop2"},  # empty, skip
            _make_task("req-keep"),
        ]
        payloads = {p: _post_tx_payload_for(p) for p in _PARTICIPANTS}
        round_ = _make_post_tx_round(payloads, done_tasks=tasks)
        result = round_.end_block()
        assert result is not None
        new_sync_data, event = result
        assert event == Event.DONE
        assert cast(SynchronizedData, new_sync_data).submitted_request_ids == [
            "0",
            "req-keep",
        ]


# ---------------------------------------------------------------------------
# TaskSubmissionAbciApp — schema-level invariants
# ---------------------------------------------------------------------------


class TestCrossPeriodPersistedKeys:
    """Schema invariants on which fields ride cross-period consensus carry."""

    def test_submitted_request_ids_is_cross_period_persisted(self) -> None:
        """``submitted_request_ids`` MUST be cross-period-persisted.

        The next cycle's
        :meth:`TaskPoolingBehaviour.handle_submitted_tasks` reads it
        via the framework's cross-period carry to prune
        ``shared_state[DONE_TASKS]``. Removing it from
        ``cross_period_persisted_keys`` would silently drop the id
        hand-off and let already-delivered tasks re-enter the next
        pooling round.
        """
        keys = TaskSubmissionAbciApp.cross_period_persisted_keys
        assert "submitted_request_ids" in keys

    def test_done_tasks_is_not_cross_period_persisted(self) -> None:
        """``done_tasks`` MUST NOT be cross-period-persisted.

        ``done_tasks`` carries per-event request/response payload data.
        Persisting it across periods would inflate DB serialization on
        the next :class:`RegistrationRound` entry. The id hand-off
        rides ``submitted_request_ids`` instead.
        """
        keys = TaskSubmissionAbciApp.cross_period_persisted_keys
        assert "done_tasks" not in keys
