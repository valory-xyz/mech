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
"""Tests for mech_abci.composition."""

import packages.valory.skills.delivery_rate_abci.rounds as DeliveryRateUpdateAbciApp
import packages.valory.skills.registration_abci.rounds as RegistrationAbci
import packages.valory.skills.reset_pause_abci.rounds as ResetAndPauseAbci
import packages.valory.skills.task_submission_abci.rounds as TaskSubmissionAbciApp
import packages.valory.skills.transaction_settlement_abci.rounds as TransactionSubmissionAbciApp
from packages.valory.skills.mech_abci.composition import (
    MechAbciApp,
    abci_app_transition_mapping,
    termination_config,
)
from packages.valory.skills.termination_abci.rounds import (
    BackgroundRound,
    TerminationAbciApp,
)


class TestAbciAppTransitionMapping:
    """Tests for the abci_app_transition_mapping dict."""

    def test_registration_finished_transitions_to_delivery_rate(self) -> None:
        """Test registration finished round transitions to delivery rate update."""
        assert (
            abci_app_transition_mapping[RegistrationAbci.FinishedRegistrationRound]
            is DeliveryRateUpdateAbciApp.UpdateDeliveryRateRound
        )

    def test_delivery_rate_no_tx_transitions_to_task_pooling(self) -> None:
        """Test delivery rate no-tx round transitions to task pooling."""
        assert (
            abci_app_transition_mapping[
                DeliveryRateUpdateAbciApp.FinishedWithoutTxRound
            ]
            is TaskSubmissionAbciApp.TaskPoolingRound
        )

    def test_delivery_rate_with_tx_transitions_to_randomness(self) -> None:
        """Test delivery rate with-tx round transitions to randomness."""
        assert (
            abci_app_transition_mapping[DeliveryRateUpdateAbciApp.FinishedWithTxRound]
            is TransactionSubmissionAbciApp.RandomnessTransactionSubmissionRound
        )

    def test_task_pooling_finished_transitions_to_randomness(self) -> None:
        """Test task pooling finished round transitions to randomness."""
        assert (
            abci_app_transition_mapping[TaskSubmissionAbciApp.FinishedTaskPoolingRound]
            is TransactionSubmissionAbciApp.RandomnessTransactionSubmissionRound
        )

    def test_task_execution_error_transitions_to_reset(self) -> None:
        """Test task execution error round transitions to reset."""
        assert (
            abci_app_transition_mapping[
                TaskSubmissionAbciApp.FinishedTaskExecutionWithErrorRound
            ]
            is ResetAndPauseAbci.ResetAndPauseRound
        )

    def test_finished_without_tasks_transitions_to_reset(self) -> None:
        """Test finished without tasks round transitions to reset."""
        assert (
            abci_app_transition_mapping[TaskSubmissionAbciApp.FinishedWithoutTasksRound]
            is ResetAndPauseAbci.ResetAndPauseRound
        )

    def test_transaction_submission_finished_transitions_to_reset(self) -> None:
        """Test transaction submission finished round transitions to reset."""
        assert (
            abci_app_transition_mapping[
                TransactionSubmissionAbciApp.FinishedTransactionSubmissionRound
            ]
            is ResetAndPauseAbci.ResetAndPauseRound
        )

    def test_transaction_submission_failed_transitions_to_reset(self) -> None:
        """Test transaction submission failed round transitions to reset."""
        assert (
            abci_app_transition_mapping[TransactionSubmissionAbciApp.FailedRound]
            is ResetAndPauseAbci.ResetAndPauseRound
        )

    def test_reset_finished_transitions_to_task_pooling(self) -> None:
        """Test reset finished round transitions to task pooling."""
        assert (
            abci_app_transition_mapping[ResetAndPauseAbci.FinishedResetAndPauseRound]
            is TaskSubmissionAbciApp.TaskPoolingRound
        )

    def test_reset_error_transitions_to_registration(self) -> None:
        """Test reset error round transitions to registration."""
        assert (
            abci_app_transition_mapping[
                ResetAndPauseAbci.FinishedResetAndPauseErrorRound
            ]
            is RegistrationAbci.RegistrationRound
        )


class TestTerminationConfig:
    """Tests for the termination_config BackgroundAppConfig."""

    def test_round_cls_is_background_round(self) -> None:
        """Test termination_config.round_cls is BackgroundRound."""
        assert termination_config.round_cls is BackgroundRound

    def test_abci_app_is_termination_abci_app(self) -> None:
        """Test termination_config.abci_app is TerminationAbciApp."""
        assert termination_config.abci_app is TerminationAbciApp
