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
    AbciApp,
    AbciAppTransitionFunction,
    AppState,
    BaseSynchronizedData,
    CollectDifferentUntilAllRound,
    CollectSameUntilThresholdRound,
    DegenerateRound,
    EventToTimeout,
    get_name,
)
from packages.valory.skills.task_execution_abci.payloads import TaskExecutionAbciPayload


class Event(Enum):
    """TaskExecutionAbciApp Events"""

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


class TaskExecutionRound(CollectSameUntilThresholdRound):
    """TaskExecutionRound"""

    payload_class = TaskExecutionAbciPayload
    synchronized_data_class = SynchronizedData

    ERROR_PAYLOAD = "ERROR"

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            if self.most_voted_payload == TaskExecutionRound.ERROR_PAYLOAD:
                return self.synchronized_data, Event.ERROR

            synchronized_data = self.synchronized_data.update(
                synchronized_data_class=SynchronizedData,
                **{
                    get_name(
                        SynchronizedData.most_voted_tx_hash
                    ): self.most_voted_payload,
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


class FinishedTaskExecutionWithErrorRound(DegenerateRound):
    """FinishedTaskExecutionWithErrorRound"""


class TaskExecutionAbciApp(AbciApp[Event]):
    """TaskExecutionAbciApp"""

    initial_round_cls: AppState = TaskExecutionRound
    initial_states: Set[AppState] = {TaskExecutionRound}
    transition_function: AbciAppTransitionFunction = {
        TaskExecutionRound: {
            Event.DONE: FinishedTaskExecutionRound,
            Event.NO_MAJORITY: TaskExecutionRound,
            Event.ROUND_TIMEOUT: TaskExecutionRound,
            Event.ERROR: FinishedTaskExecutionWithErrorRound,
        },
        FinishedTaskExecutionRound: {},
        FinishedTaskExecutionWithErrorRound: {},
    }
    final_states: Set[AppState] = {
        FinishedTaskExecutionRound,
        FinishedTaskExecutionWithErrorRound,
    }
    event_to_timeout: EventToTimeout = {}
    cross_period_persisted_keys: FrozenSet[str] = frozenset()
    db_pre_conditions: Dict[AppState, Set[str]] = {
        TaskExecutionRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedTaskExecutionRound: {"most_voted_tx_hash"},
        FinishedTaskExecutionWithErrorRound: set(),
    }
