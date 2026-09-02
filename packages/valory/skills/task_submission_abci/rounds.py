# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2026 Valory AG
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

"""This package contains the rounds of TaskSubmissionAbciApp."""

import json
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AppState,
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    CollectionRound,
    DegenerateRound,
    EventToTimeout,
    get_name,
)
from packages.valory.skills.task_submission_abci.payloads import (
    PostTxSettlementPayload,
    TaskPoolingPayload,
    TransactionPayload,
)


def extract_request_ids(tasks: List[Dict[str, Any]]) -> List[str]:
    """Return ``str``-normalised request ids from a task list.

    Falsy ids (missing key, ``None``, empty string) are dropped so
    they don't collide in downstream match sets. Legit ``0`` (int)
    is kept via the explicit ``is None or == ""`` check rather than
    a ``not`` truthiness test.

    :param tasks: task dicts (each expected to carry ``"request_id"``).
    :return: the ``str``-normalised id list, in input order.
    """
    return [
        str(task["request_id"])
        for task in tasks
        if task.get("request_id") is not None and task.get("request_id") != ""
    ]


class Event(Enum):
    """TaskSubmissionAbciApp Events"""

    TASK_EXECUTION_ROUND_TIMEOUT = "task_execution_round_timeout"
    ROUND_TIMEOUT = "round_timeout"
    NO_MAJORITY = "no_majority"
    DONE = "done"
    NO_TASKS = "no_tasks"
    ERROR = "error"


class SynchronizedData(BaseSynchronizedData):
    """
    Class to represent the synchronized data.

    This data is replicated by the tendermint application.
    """

    @property
    def most_voted_tx_hash(self) -> str:
        """Get the most_voted_tx_hash."""
        return cast(str, self.db.get_strict("most_voted_tx_hash"))

    @property
    def done_tasks(self) -> List[Dict[str, Any]]:
        """Done tasks."""
        return cast(List[Dict[str, Any]], self.db.get("done_tasks", []))

    @property
    def submitted_request_ids(self) -> List[str]:
        """Return request ids of tasks submitted in the most recent settlement.

        Written by :class:`PostTxSettlementRound` end_block and read by
        :meth:`TaskPoolingBehaviour.handle_submitted_tasks` next period
        to prune ``shared_state[DONE_TASKS]`` of already-delivered
        entries. Cross-period-persisted because it's small and the next
        cycle needs it; the full ``done_tasks`` list is not carried
        across periods because it holds per-event request/response
        payload data that would inflate DB serialization.

        The value is validated on read: writers must go through
        :func:`extract_request_ids` so that only ``str`` ids land in
        the DB. A future writer that bypasses the helper and stores a
        non-list or non-``str`` entries surfaces as ``TypeError`` here
        rather than as silent type drift at the consumer.

        :return: the list of request ids from the most recent settlement.
        :raises TypeError: if the persisted value is not ``list[str]``.
        """
        value = self.db.get("submitted_request_ids", [])
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise TypeError(
                "submitted_request_ids must be a list[str]; got "
                f"{type(value).__name__}={value!r}"
            )
        return value

    @property
    def final_tx_hash(self) -> str:
        """Get the verified tx hash."""
        return cast(str, self.db.get_strict("final_tx_hash"))


class TaskPoolingRound(CollectionRound):
    """TaskPoolingRound"""

    payload_class = TaskPoolingPayload
    synchronized_data_class = SynchronizedData

    move_forward_payload: Optional[TaskPoolingPayload] = None

    ERROR_PAYLOAD = "ERROR"

    @property
    def collection_threshold_reached(
        self,
    ) -> bool:
        """Check that the collection threshold has been reached."""
        return len(self.collection) >= self.synchronized_data.consensus_threshold

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.collection_threshold_reached:
            all_done_tasks = []
            for payload in self.collection.values():
                done_tasks_str = cast(TaskPoolingPayload, payload).content
                done_tasks = json.loads(done_tasks_str)
                all_done_tasks.extend(done_tasks)

            # Deduplicate and sort by the same ``str``-normalized key so
            # a mixed-type ``done_tasks`` list (an on-chain ``int``
            # ``request_id`` next to an off-chain ``str`` — happens on
            # the old-code → new-code restart transition where an agent
            # resumes with pre-ingress-fix state in shared memory)
            # produces one canonical row per logical request and doesn't
            # raise ``TypeError: '<' not supported between instances of
            # 'int' and 'str'`` on the sort. Post ingress coercion in
            # :mod:`task_execution.handlers` all new writes are ``int``
            # and the cast is a no-op, but leaving the two sites
            # un-normalized would let the same request slip through
            # dedup as both ``42`` and ``"42"`` and get submitted twice
            # in the same multisend — a costlier failure than the crash
            # itself. The dedup key and the sort key MUST use the same
            # normalization, otherwise the two disagree on which entries
            # collide.
            # A falsy ``request_id`` (missing, ``None``, ``""``) would
            # otherwise dedup all such rows to the same ``""`` / ``"None"``
            # bucket and silently drop legitimate rows from other agents —
            # ``all_done_tasks`` is a bag of every participant's payloads
            # with no schema validation, so a single misbehaving or
            # out-of-version agent can introduce one. Skip and warn: this
            # agent's own pipeline should never produce a falsy id (both
            # writers set it explicitly), so hitting this branch is a real
            # protocol violation worth surfacing.
            unique_ids: set = set()
            unique_objects = []
            for obj in all_done_tasks:
                raw_request_id = obj.get("request_id")
                # Skip "genuinely absent" (missing key, None, empty string)
                # without the boolean/float overlap ``not x`` gives:
                # ``not False`` and ``not 0.0`` are both True in Python, so
                # ``if not raw_request_id`` would drop a JSON ``false`` or
                # ``0.0`` id — both malformed but distinguishable from
                # missing. Legit ``0`` (int) is deliberately kept.
                if raw_request_id is None or raw_request_id == "":
                    self.context.logger.warning(
                        "Skipping done_task with missing/empty request_id "
                        "in TaskPoolingRound dedup: %r",
                        obj,
                    )
                    continue
                request_id_key = str(raw_request_id)
                if request_id_key not in unique_ids:
                    unique_ids.add(request_id_key)
                    unique_objects.append(obj)

            # Note on ordering: the ``str`` sort key means int
            # ``request_id``s land in lexicographic order (``[9, 10]``
            # sorts as ``[10, 9]``). This is stable and deterministic
            # *within a uniform fleet* — every agent applies the same
            # rule — so consensus holds. It does NOT hold across a mixed
            # old-code / new-code fleet mid-upgrade: pre-fix agents dedup
            # ``42`` and ``"42"`` as distinct and sort numerically, so an
            # old and new agent computing ``done_tasks`` off the same
            # collected payloads produce different content, different
            # row count, and different order.
            # ``TransactionPreparationRound`` is a
            # ``CollectSameUntilThresholdRound`` keyed on
            # ``most_voted_payload`` and requires 2/3+ byte-identical
            # payloads, so a mixed fleet ``NO_MAJORITY``-stalls until
            # versions converge. This is why the service hash bump has
            # to be atomic across all agents in a service (not rolling)
            # — see the PR's rollout notes. Consumers downstream of the
            # sort do not depend on the specific order (they iterate or
            # index-by-id, not by position).
            unique_done_tasks = sorted(
                unique_objects, key=lambda x: str(x.get("request_id", ""))
            )
            synchronized_data = self.synchronized_data.update(
                synchronized_data_class=SynchronizedData,
                **{
                    get_name(SynchronizedData.done_tasks): unique_done_tasks,
                },
            )
            if len(unique_done_tasks) > 0:
                return synchronized_data, Event.DONE
            return synchronized_data, Event.NO_TASKS

        return None


class TransactionPreparationRound(CollectSameUntilThresholdRound):
    """TransactionPreparationRound"""

    payload_class = TransactionPayload
    payload_attribute = "content"
    synchronized_data_class = SynchronizedData
    extended_requirements = ()

    ERROR_PAYLOAD = "error"

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        if self.threshold_reached:
            if self.most_voted_payload == self.ERROR_PAYLOAD:
                return (
                    self.synchronized_data.update(
                        synchronized_data_class=SynchronizedData,
                        **{
                            get_name(SynchronizedData.done_tasks): [],
                        },
                    ),
                    Event.ERROR,
                )

            state = self.synchronized_data.update(
                synchronized_data_class=self.synchronized_data_class,
                **{
                    get_name(
                        SynchronizedData.most_voted_tx_hash
                    ): self.most_voted_payload,
                },
            )
            return state, Event.DONE
        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            # in case we cant submit this tx, we need to make sure we don't account the tasks as done
            return (
                self.synchronized_data.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.done_tasks): [],
                    },
                ),
                Event.NO_MAJORITY,
            )

        return None


class FinishedTaskPoolingRound(DegenerateRound):
    """FinishedTaskPoolingRound"""


class FinishedTaskExecutionWithErrorRound(DegenerateRound):
    """FinishedTaskExecutionWithErrorRound"""


class FinishedWithoutTasksRound(DegenerateRound):
    """FinishedWithoutTasksRound"""


class PostTxSettlementRound(CollectSameUntilThresholdRound):
    """Runs once on-chain settlement confirms.

    Mirrors the optimus pattern at
    ``liquidity_trader_abci/states/post_tx_settlement.py``: each agent
    runs the matching behaviour, which fires the offchain predict-api data lake
    POST for the round's settled offchain deliveries, then sends back a
    fixed payload so the round can advance on consensus participation.
    The predict-api write is idempotent server-side (PK on ``request_id``),
    so multi-agent services do not need a keeper-elects-one pattern; each
    agent posting its own copy is safe and the per-EOA rate limiter on
    the server scales naturally across agents.

    The round always transitions DONE on threshold: a failed predict-api
    write does NOT block the FSM (the settlement already landed on-chain,
    the analytics row just arrives later via the replay buffer). The
    NO_MAJORITY arm only fires if the agents can't agree on having reached
    this round at all, which is the same shape as every consensus round.

    On DONE the round writes ``submitted_request_ids`` — the id-only
    hand-off used by the next cycle's
    :meth:`TaskPoolingBehaviour.handle_submitted_tasks` to prune
    ``shared_state[DONE_TASKS]``. ``done_tasks`` itself stays in the
    period it was set and is not carried across; the ID hand-off is
    what rides consensus between cycles.

    Do NOT mutate ``done_tasks`` here. It is still read by
    :class:`PostTxSettlementBehaviour` earlier in the same period for
    the predict-api write and log emission, and the behaviour-side
    prune in the next cycle depends on ``shared_state[DONE_TASKS]``
    reflecting only tasks that were not settled — mutating the
    consensus field here breaks that contract.
    """

    payload_class = PostTxSettlementPayload
    payload_attribute = "content"
    synchronized_data_class = SynchronizedData
    extended_requirements = ()

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        if self.threshold_reached:
            done_tasks = cast(SynchronizedData, self.synchronized_data).done_tasks
            submitted_ids = extract_request_ids(done_tasks)
            return (
                self.synchronized_data.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.submitted_request_ids): submitted_ids,
                    },
                ),
                Event.DONE,
            )
        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self.synchronized_data, Event.NO_MAJORITY
        return None


class FinishedPostTxSettlementRound(DegenerateRound):
    """FinishedPostTxSettlementRound"""


class TaskSubmissionAbciApp(AbciApp[Event]):
    """TaskSubmissionAbciApp

    Initial round: TaskPoolingRound

    Initial states: {PostTxSettlementRound, TaskPoolingRound}

    Transition states:
        0. TaskPoolingRound
            - done: 1.
            - no tasks: 4.
            - round timeout: 0.
        1. TransactionPreparationRound
            - done: 2.
            - error: 3.
            - no majority: 3.
            - task execution round timeout: 1.
        2. FinishedTaskPoolingRound
        3. FinishedTaskExecutionWithErrorRound
        4. FinishedWithoutTasksRound
        5. PostTxSettlementRound
            - done: 6.
            - no majority: 5.
            - round timeout: 5.
        6. FinishedPostTxSettlementRound

    Final states: {FinishedPostTxSettlementRound, FinishedTaskExecutionWithErrorRound, FinishedTaskPoolingRound, FinishedWithoutTasksRound}

    Timeouts:
        task execution round timeout: 60.0
        round timeout: 60.0
    """

    initial_round_cls: AppState = TaskPoolingRound
    # ``PostTxSettlementRound`` is reachable from outside the skill via the
    # composition wire (TxSettlementAbci.FinishedTransactionSubmissionRound
    # routes here), so it must be declared as an initial state even though
    # the FSM never *starts* there. Matches the optimus LiquidityTraderAbci
    # pattern where the post-settlement round is one of several initial
    # entry points.
    initial_states: Set[AppState] = {TaskPoolingRound, PostTxSettlementRound}
    transition_function: AbciAppTransitionFunction = {
        TaskPoolingRound: {
            Event.DONE: TransactionPreparationRound,
            Event.NO_TASKS: FinishedWithoutTasksRound,
            Event.ROUND_TIMEOUT: TaskPoolingRound,
        },
        TransactionPreparationRound: {
            Event.DONE: FinishedTaskPoolingRound,
            Event.ERROR: FinishedTaskExecutionWithErrorRound,
            Event.NO_MAJORITY: FinishedTaskExecutionWithErrorRound,
            Event.TASK_EXECUTION_ROUND_TIMEOUT: TransactionPreparationRound,
        },
        FinishedTaskPoolingRound: {},
        FinishedTaskExecutionWithErrorRound: {},
        FinishedWithoutTasksRound: {},
        PostTxSettlementRound: {
            Event.DONE: FinishedPostTxSettlementRound,
            # Tendermint can't reach majority on participation: route back
            # to the post-tx round so a brief disagreement doesn't crash
            # the FSM. The predict-api write is idempotent, so re-entry is safe.
            Event.NO_MAJORITY: PostTxSettlementRound,
            Event.ROUND_TIMEOUT: PostTxSettlementRound,
        },
        FinishedPostTxSettlementRound: {},
    }
    final_states: Set[AppState] = {
        FinishedTaskPoolingRound,
        FinishedWithoutTasksRound,
        FinishedTaskExecutionWithErrorRound,
        FinishedPostTxSettlementRound,
    }
    event_to_timeout: EventToTimeout = {
        Event.TASK_EXECUTION_ROUND_TIMEOUT: 60.0,
        Event.ROUND_TIMEOUT: 60.0,
    }
    cross_period_persisted_keys: FrozenSet[str] = frozenset(
        [
            # Hand-off signal only. ``done_tasks`` is intentionally
            # not cross-period-persisted because its per-entry payload
            # data is large enough to inflate DB serialization on the
            # next registration.
            get_name(SynchronizedData.submitted_request_ids),
            get_name(SynchronizedData.final_tx_hash),
        ]
    )
    db_pre_conditions: Dict[AppState, Set[str]] = {
        TaskPoolingRound: set(),
        # Entered from composition after settlement. Reads
        # ``done_tasks`` from the same FSM cycle's earlier
        # ``TaskPoolingRound`` (present in the current period's DB
        # slot). end_block writes the id-only
        # ``submitted_request_ids`` for the next cycle's prune.
        PostTxSettlementRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedTaskPoolingRound: {"most_voted_tx_hash"},
        FinishedTaskExecutionWithErrorRound: set(),
        FinishedWithoutTasksRound: set(),
        FinishedPostTxSettlementRound: set(),
    }
