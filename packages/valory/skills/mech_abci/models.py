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

"""This module contains the shared state for the abci skill of Mech."""

from packages.valory.skills.abstract_round_abci.models import ApiSpecs
from packages.valory.skills.abstract_round_abci.models import \
    BenchmarkTool as BaseBenchmarkTool
from packages.valory.skills.abstract_round_abci.models import \
    Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import \
    SharedState as BaseSharedState
from packages.valory.skills.mech_abci.composition import MechAbciApp
from packages.valory.skills.multiplexer_abci.models import \
    Params as MultiplexerAbciParams
from packages.valory.skills.multiplexer_abci.rounds import \
    Event as MultiplexerEvent
from packages.valory.skills.reset_pause_abci.rounds import \
    Event as ResetPauseEvent
from packages.valory.skills.task_execution_abci.models import \
    Params as TaskExecutionAbciParams
from packages.valory.skills.task_execution_abci.rounds import \
    Event as TaskExecutionEvent
from packages.valory.skills.termination_abci.models import TerminationParams
from packages.valory.skills.transaction_preparation_abci.models import \
    Params as TransactionPreparationAbciParams
from packages.valory.skills.transaction_preparation_abci.rounds import \
    Event as TransactionPreparationEvent

MultiplexerParams = MultiplexerAbciParams
TaskExecutionParams = TaskExecutionAbciParams
TransactionPreparationParams = TransactionPreparationAbciParams

Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool


class RandomnessApi(ApiSpecs):
    """A model that wraps ApiSpecs for randomness api specifications."""


MARGIN = 5


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = MechAbciApp

    def setup(self) -> None:
        """Set up."""
        super().setup()

        MechAbciApp.event_to_timeout[
            MultiplexerEvent.ROUND_TIMEOUT
        ] = self.context.params.round_timeout_seconds

        MechAbciApp.event_to_timeout[
            TaskExecutionEvent.ROUND_TIMEOUT
        ] = self.context.params.round_timeout_seconds

        MechAbciApp.event_to_timeout[
            TransactionPreparationEvent.ROUND_TIMEOUT
        ] = self.context.params.round_timeout_seconds

        MechAbciApp.event_to_timeout[
            ResetPauseEvent.ROUND_TIMEOUT
        ] = self.context.params.round_timeout_seconds

        MechAbciApp.event_to_timeout[ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT] = (
            self.context.params.reset_pause_duration + MARGIN
        )


class Params(MultiplexerParams, TerminationParams, TaskExecutionParams):
    """A model to represent params for multiple abci apps."""

