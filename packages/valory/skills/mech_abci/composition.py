# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2025 Valory AG
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

import packages.valory.skills.delivery_rate_abci.rounds as DeliveryRateUpdateAbciApp
import packages.valory.skills.registration_abci.rounds as RegistrationAbci
import packages.valory.skills.reset_pause_abci.rounds as ResetAndPauseAbci
import packages.valory.skills.task_submission_abci.rounds as TaskSubmissionAbciApp
import packages.valory.skills.transaction_settlement_abci.rounds as TransactionSubmissionAbciApp
from packages.valory.skills.abstract_round_abci.abci_app_chain import (
    AbciAppTransitionMapping,
    chain,
)


# Here we define how the transition between the FSMs should happen
# more information here: https://docs.autonolas.network/fsm_app_introduction/#composition-of-fsm-apps
abci_app_transition_mapping: AbciAppTransitionMapping = {
    RegistrationAbci.FinishedRegistrationRound: DeliveryRateUpdateAbciApp.UpdateDeliveryRateRound,
    DeliveryRateUpdateAbciApp.FinishedWithoutTxRound: TaskSubmissionAbciApp.TaskPoolingRound,
    DeliveryRateUpdateAbciApp.FinishedWithTxRound: TransactionSubmissionAbciApp.RandomnessTransactionSubmissionRound,  # pylint: disable=C0301
    TaskSubmissionAbciApp.FinishedTaskPoolingRound: TransactionSubmissionAbciApp.RandomnessTransactionSubmissionRound,  # pylint: disable=C0301
    TaskSubmissionAbciApp.FinishedTaskExecutionWithErrorRound: ResetAndPauseAbci.ResetAndPauseRound,
    TaskSubmissionAbciApp.FinishedWithoutTasksRound: ResetAndPauseAbci.ResetAndPauseRound,
    TransactionSubmissionAbciApp.FinishedTransactionSubmissionRound: ResetAndPauseAbci.ResetAndPauseRound,
    TransactionSubmissionAbciApp.FailedRound: ResetAndPauseAbci.ResetAndPauseRound,
    ResetAndPauseAbci.FinishedResetAndPauseRound: TaskSubmissionAbciApp.TaskPoolingRound,
    ResetAndPauseAbci.FinishedResetAndPauseErrorRound: RegistrationAbci.RegistrationRound,
}

MechAbciApp = chain(
    (
        RegistrationAbci.AgentRegistrationAbciApp,
        TaskSubmissionAbciApp.TaskSubmissionAbciApp,
        ResetAndPauseAbci.ResetPauseAbciApp,
        TransactionSubmissionAbciApp.TransactionSubmissionAbciApp,
        DeliveryRateUpdateAbciApp.DeliveryRateUpdateAbciApp,
    ),
    abci_app_transition_mapping,
)
