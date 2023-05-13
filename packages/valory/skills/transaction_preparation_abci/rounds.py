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

"""This package contains the rounds of TransactionPreparationAbciApp."""

import json
from enum import Enum
from typing import Dict, FrozenSet, Optional, Set, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp, AbciAppTransitionFunction, AppState, BaseSynchronizedData,
    CollectSameUntilThresholdRound, DegenerateRound, EventToTimeout, get_name)
from packages.valory.skills.transaction_preparation_abci.payloads import \
    TransactionPreparationAbciPayload


class Event(Enum):
    """TransactionPreparationAbciApp Events"""

    NO_MAJORITY = "no_majority"
    DONE = "done"
    ROUND_TIMEOUT = "round_timeout"
    CONTRACT_ERROR = "contract_error"


class SynchronizedData(BaseSynchronizedData):
    """
    Class to represent the synchronized data.

    This data is replicated by the tendermint application.
    """

    @property
    def finished_task_data(self) -> int:
        """Get the finished_task_data."""
        return cast(int, self.db.get_strict("finished_task_data"))

    @property
    def most_voted_tx_hash(self) -> str:
        """Get the most_voted_tx_hash."""
        return cast(str, self.db.get_strict("most_voted_tx_hash"))


class TransactionPreparationRound(CollectSameUntilThresholdRound):
    """TransactionPreparationRound"""

    payload_class = TransactionPreparationAbciPayload
    synchronized_data_class = SynchronizedData

    ERROR_PAYLOAD = "ERROR"

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:

            payload = json.loads(self.most_voted_payload)

            if payload["tx_hash"] == TransactionPreparationRound.ERROR_PAYLOAD:
                return self.synchronized_data, Event.CONTRACT_ERROR

            synchronized_data = self.synchronized_data.update(
                synchronized_data_class=SynchronizedData,
                **{
                    get_name(SynchronizedData.most_voted_tx_hash): payload["tx_hash"],
                }
            )
            return synchronized_data, Event.DONE
        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self.synchronized_data, Event.NO_MAJORITY
        return None


class FinishedTransactionPreparationRound(DegenerateRound):
    """FinishedTransactionPreparationRound"""


class TransactionPreparationAbciApp(AbciApp[Event]):
    """TransactionPreparationAbciApp"""

    initial_round_cls: AppState = TransactionPreparationRound
    initial_states: Set[AppState] = {TransactionPreparationRound}
    transition_function: AbciAppTransitionFunction = {
        TransactionPreparationRound: {
            Event.DONE: FinishedTransactionPreparationRound,
            Event.NO_MAJORITY: TransactionPreparationRound,
            Event.ROUND_TIMEOUT: TransactionPreparationRound,
            Event.CONTRACT_ERROR: TransactionPreparationRound
        },
        FinishedTransactionPreparationRound: {}
    }
    final_states: Set[AppState] = {FinishedTransactionPreparationRound}
    event_to_timeout: EventToTimeout = {}
    cross_period_persisted_keys: FrozenSet[str] = frozenset()
    db_pre_conditions: Dict[AppState, Set[str]] = {
        TransactionPreparationRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedTransactionPreparationRound: {"most_voted_tx_hash"}
    }
