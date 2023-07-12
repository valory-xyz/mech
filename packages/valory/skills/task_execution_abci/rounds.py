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
from enum import Enum
from typing import Dict, FrozenSet, Optional, Set, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AppState,
    BaseSynchronizedData,
    BaseTxPayload,
    CollectionRound,
    DegenerateRound,
    EventToTimeout,
    get_name,
)
from packages.valory.skills.task_execution_abci.payloads import TaskExecutionAbciPayload


class Event(Enum):
    """TaskExecutionAbciApp Events"""

    TASK_EXECUTION_ROUND_TIMEOUT = "task_execution_round_timeout"
    ROUND_TIMEOUT = "round_timeout"
    NO_MAJORITY = "no_majority"
    DONE = "done"
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


class TaskExecutionRound(CollectionRound):
    """TaskExecutionRound"""

    payload_class = TaskExecutionAbciPayload
    synchronized_data_class = SynchronizedData

    move_forward_payload: Optional[TaskExecutionAbciPayload] = None

    ERROR_PAYLOAD = "ERROR"

    @property
    def collection_threshold_reached(
        self,
    ) -> bool:
        """Check that the collection threshold has been reached."""
        return len(self.collection) >= self.synchronized_data.max_participants

    def process_payload(self, payload: BaseTxPayload) -> None:
        """Process payload."""
        super().process_payload(payload)
        if cast(TaskExecutionAbciPayload, payload).content != self.ERROR_PAYLOAD:
            self.move_forward_payload = cast(TaskExecutionAbciPayload, payload)

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.collection_threshold_reached and self.move_forward_payload is None:
            return self.synchronized_data, Event.ERROR

        if self.move_forward_payload is not None:
            synchronized_data = self.synchronized_data.update(
                synchronized_data_class=SynchronizedData,
                **{
                    get_name(
                        SynchronizedData.most_voted_tx_hash
                    ): self.move_forward_payload.content,
                }
            )
            return synchronized_data, Event.DONE

        return None


class FinishedTaskExecutionRound(DegenerateRound):
    """FinishedTaskExecutionRound"""


class FinishedTaskExecutionWithErrorRound(DegenerateRound):
    """FinishedTaskExecutionWithErrorRound"""


class TaskExecutionAbciApp(AbciApp[Event]):
    """TaskExecutionAbciApp

    Initial round: TaskExecutionRound

    Initial states: {TaskExecutionRound}

    Transition states:
        0. TaskExecutionRound
            - done: 1.
            - round timeout: 0.
            - error: 2.
        1. FinishedTaskExecutionRound
        2. FinishedTaskExecutionWithErrorRound

    Final states: {FinishedTaskExecutionRound, FinishedTaskExecutionWithErrorRound}

    Timeouts:
        round timeout: 30.0
    """

    initial_round_cls: AppState = TaskExecutionRound
    initial_states: Set[AppState] = {TaskExecutionRound}
    transition_function: AbciAppTransitionFunction = {
        TaskExecutionRound: {
            Event.DONE: FinishedTaskExecutionRound,
            Event.TASK_EXECUTION_ROUND_TIMEOUT: TaskExecutionRound,
            Event.ERROR: FinishedTaskExecutionWithErrorRound,
        },
        FinishedTaskExecutionRound: {},
        FinishedTaskExecutionWithErrorRound: {},
    }
    final_states: Set[AppState] = {
        FinishedTaskExecutionRound,
        FinishedTaskExecutionWithErrorRound,
    }
    event_to_timeout: EventToTimeout = {
        Event.TASK_EXECUTION_ROUND_TIMEOUT: 60.0,
    }
    cross_period_persisted_keys: FrozenSet[str] = frozenset()
    db_pre_conditions: Dict[AppState, Set[str]] = {
        TaskExecutionRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedTaskExecutionRound: {"most_voted_tx_hash"},
        FinishedTaskExecutionWithErrorRound: set(),
    }
