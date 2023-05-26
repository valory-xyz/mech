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

"""This package contains round behaviours of MechAbciApp."""

import packages.valory.skills.multiplexer_abci.rounds as MultiplexerAbciApp
import packages.valory.skills.registration_abci.rounds as RegistrationAbci
import packages.valory.skills.reset_pause_abci.rounds as ResetAndPauseAbci
import packages.valory.skills.task_execution_abci.rounds as TaskExecutionAbciApp
import packages.valory.skills.transaction_settlement_abci.rounds as TransactionSubmissionAbciApp
from packages.valory.skills.abstract_round_abci.abci_app_chain import (
    AbciAppTransitionMapping, chain)
from packages.valory.skills.termination_abci.rounds import BackgroundRound
from packages.valory.skills.termination_abci.rounds import \
    Event as TerminationEvent
from packages.valory.skills.termination_abci.rounds import TerminationAbciApp

# Here we define how the transition between the FSMs should happen
# more information here: https://docs.autonolas.network/fsm_app_introduction/#composition-of-fsm-apps
abci_app_transition_mapping: AbciAppTransitionMapping = {
    RegistrationAbci.FinishedRegistrationRound: MultiplexerAbciApp.MultiplexerRound,
    MultiplexerAbciApp.FinishedMultiplexerResetRound: ResetAndPauseAbci.ResetAndPauseRound,
    MultiplexerAbciApp.FinishedMultiplexerExecuteRound: TaskExecutionAbciApp.TaskExecutionRound,
    TaskExecutionAbciApp.FinishedTaskExecutionRound: TransactionSubmissionAbciApp.RandomnessTransactionSubmissionRound,  # pylint: disable=C0301
    TaskExecutionAbciApp.FinishedTaskExecutionWithErrorRound: MultiplexerAbciApp.MultiplexerRound,
    TransactionSubmissionAbciApp.FinishedTransactionSubmissionRound: MultiplexerAbciApp.MultiplexerRound,
    TransactionSubmissionAbciApp.FailedRound: TaskExecutionAbciApp.TaskExecutionRound,
    ResetAndPauseAbci.FinishedResetAndPauseRound: MultiplexerAbciApp.MultiplexerRound,
    ResetAndPauseAbci.FinishedResetAndPauseErrorRound: RegistrationAbci.RegistrationRound,
}

MechAbciApp = chain(
    (
        RegistrationAbci.AgentRegistrationAbciApp,
        MultiplexerAbciApp.MultiplexerAbciApp,
        TaskExecutionAbciApp.TaskExecutionAbciApp,
        ResetAndPauseAbci.ResetPauseAbciApp,
        TransactionSubmissionAbciApp.TransactionSubmissionAbciApp,
    ),
    abci_app_transition_mapping,
).add_termination(
    background_round_cls=BackgroundRound,
    termination_event=TerminationEvent.TERMINATE,
    termination_abci_app=TerminationAbciApp,
)
