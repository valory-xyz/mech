# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 Valory AG
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

"""This package contains the rounds of TaskExecutionAbciApp."""

import json
from enum import Enum
from typing import Dict, FrozenSet, Optional, Set, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp, AbciAppTransitionFunction, AppState, BaseSynchronizedData,
    CollectDifferentUntilAllRound, DegenerateRound, EventToTimeout, get_name, TransactionNotValidError)
from packages.valory.skills.task_execution_abci.payloads import \
    TaskExecutionAbciPayload


class Event(Enum):
    """TaskExecutionAbciApp Events"""

    ROUND_TIMEOUT = "round_timeout"
    NO_MAJORITY = "no_majority"
    DONE = "done"


class SynchronizedData(BaseSynchronizedData):
    """
    Class to represent the synchronized data.

    This data is replicated by the tendermint application.
    """

    @property
    def finished_task_data(self) -> int:
        """Get the finished_task_data."""
        return cast(int, self.db.get_strict("finished_task_data"))

class TaskExecutionRound(CollectDifferentUntilAllRound):
    """TaskExecutionRound"""

    payload_class = TaskExecutionAbciPayload
    synchronized_data_class = SynchronizedData

    def check_payload(self, payload: TaskExecutionAbciPayload) -> None:
        """Check Payload"""
        # new = payload.values
        # existing = [
        #     collection_payload.values
        #     for collection_payload in self.collection.values()
        #     # do not consider empty delegations
        #     if json.loads(collection_payload.json["new_delegations"])
        # ]

        # if payload.sender not in self.collection and new in existing:
        #     raise TransactionNotValidError(
        #         f"`CollectDifferentUntilAllRound` encountered a value {new!r} that already exists. "
        #         f"All values: {existing}"
        #     )

        if payload.round_count != self.synchronized_data.round_count:
            raise TransactionNotValidError(
                f"Expected round count {self.synchronized_data.round_count} and got {payload.round_count}."
            )

        if payload.sender in self.collection:
            raise TransactionNotValidError(
                f"sender {payload.sender} has already sent value for round: {self.round_id}"
            )

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.collection_threshold_reached:

            payloads_json = {
                    "request_id": json.loads(self.collection[list(self.collection.keys())[0]].content)['request_id'],
                    "task_result": [json.loads(f.content)['task_result'] for f in self.collection.values()]
            }

            synchronized_data = self.synchronized_data.update(
                synchronized_data_class=SynchronizedData,
                **{
                    get_name(SynchronizedData.finished_task_data): payloads_json,
                }
            )

            return synchronized_data, Event.DONE

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self.synchronized_data, Event.NO_MAJORITY
        return None


class FinishedTaskExecutionRound(DegenerateRound):
    """FinishedTaskExecutionRound"""


class TaskExecutionAbciApp(AbciApp[Event]):
    """TaskExecutionAbciApp"""

    initial_round_cls: AppState = TaskExecutionRound
    initial_states: Set[AppState] = {TaskExecutionRound}
    transition_function: AbciAppTransitionFunction = {
        TaskExecutionRound: {
            Event.DONE: FinishedTaskExecutionRound,
            Event.NO_MAJORITY: TaskExecutionRound,
            Event.ROUND_TIMEOUT: TaskExecutionRound
        },
        FinishedTaskExecutionRound: {}
    }
    final_states: Set[AppState] = {FinishedTaskExecutionRound}
    event_to_timeout: EventToTimeout = {}
    cross_period_persisted_keys: FrozenSet[str] = frozenset()
    db_pre_conditions: Dict[AppState, Set[str]] = {
        TaskExecutionRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedTaskExecutionRound: set(),
    }
