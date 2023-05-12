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

"""This package contains the rounds of MultiplexerAbciApp."""

from enum import Enum
from typing import Dict, FrozenSet, Optional, Set, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp, AbciAppTransitionFunction, AppState, BaseSynchronizedData,
    CollectSameUntilThresholdRound, DegenerateRound, EventToTimeout, get_name)
from packages.valory.skills.multiplexer_abci.payloads import MultiplexerPayload


class Event(Enum):
    """MultiplexerAbciApp Events"""

    ROUND_TIMEOUT = "round_timeout"
    NO_MAJORITY = "no_majority"
    WAIT = "done_post"
    EXECUTE = "execute"
    RESET = "reset"


class SynchronizedData(BaseSynchronizedData):
    """
    Class to represent the synchronized data.

    This data is replicated by the tendermint application.
    """

    @property
    def period_counter(self) -> int:
        """Get the period_counter."""
        return cast(int, self.db.get("period_counter", 0))

class MultiplexerRound(CollectSameUntilThresholdRound):
    """MultiplexerRound"""

    payload_class = MultiplexerPayload
    synchronized_data_class = SynchronizedData

    WAIT_PAYLOAD = "wait"
    RESET_PAYLOAD = "reset"
    EXECUTE_PAYLOAD = "transact"

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:

            period_counter = cast(SynchronizedData, self.synchronized_data).period_counter

            event = Event.WAIT

            if self.most_voted_payload == self.RESET_PAYLOAD:
                period_counter = -1
                event = Event.RESET

            if self.most_voted_payload == self.EXECUTE_PAYLOAD:
                event = Event.EXECUTE

            synchronized_data = self.synchronized_data.update(
                synchronized_data_class=SynchronizedData,
                **{
                    get_name(SynchronizedData.period_counter): period_counter + 1,
                }
            )

            return synchronized_data, event

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self.synchronized_data, Event.NO_MAJORITY
        return None


class FinishedMultiplexerResetRound(DegenerateRound):
    """FinishedDecisionMakingPostRound"""


class FinishedMultiplexerExecuteRound(DegenerateRound):
    """FinishedMultiplexerExecuteRound"""


class MultiplexerAbciApp(AbciApp[Event]):
    """MultiplexerAbciApp"""

    initial_round_cls: AppState = MultiplexerRound
    initial_states: Set[AppState] = {MultiplexerRound}
    transition_function: AbciAppTransitionFunction = {
        MultiplexerRound: {
            Event.WAIT: MultiplexerRound,
            Event.RESET: FinishedMultiplexerResetRound,
            Event.EXECUTE: FinishedMultiplexerExecuteRound,
            Event.NO_MAJORITY: MultiplexerRound,
            Event.ROUND_TIMEOUT: MultiplexerRound,
        },
        FinishedMultiplexerResetRound: {},
        FinishedMultiplexerExecuteRound: {},
    }
    final_states: Set[AppState] = {
        FinishedMultiplexerResetRound,
        FinishedMultiplexerExecuteRound,
    }
    event_to_timeout: EventToTimeout = {
        Event.ROUND_TIMEOUT: 30.0,
    }
    cross_period_persisted_keys: FrozenSet[str] = frozenset()
    db_pre_conditions: Dict[AppState, Set[str]] = {
        MultiplexerRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedMultiplexerResetRound: set(),
        FinishedMultiplexerExecuteRound: set(),
    }
