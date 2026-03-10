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
"""Tests for mech_abci.behaviours."""

from packages.valory.skills.abstract_round_abci.behaviours import AbstractRoundBehaviour
from packages.valory.skills.delivery_rate_abci.behaviours import (
    UpdateDeliveryRateRoundBehaviour,
)
from packages.valory.skills.mech_abci.behaviours import MechConsensusBehaviour
from packages.valory.skills.mech_abci.composition import MechAbciApp
from packages.valory.skills.registration_abci.behaviours import (
    AgentRegistrationRoundBehaviour,
    RegistrationStartupBehaviour,
)
from packages.valory.skills.reset_pause_abci.behaviours import (
    ResetPauseABCIConsensusBehaviour,
)
from packages.valory.skills.task_submission_abci.behaviours import (
    TaskSubmissionRoundBehaviour,
)
from packages.valory.skills.termination_abci.behaviours import (
    BackgroundBehaviour,
    TerminationAbciBehaviours,
)
from packages.valory.skills.transaction_settlement_abci.behaviours import (
    TransactionSettlementRoundBehaviour,
)


class TestMechConsensusBehaviour:
    """Tests for MechConsensusBehaviour class attributes."""

    def test_is_abstract_round_behaviour(self) -> None:
        """Test MechConsensusBehaviour is a subclass of AbstractRoundBehaviour."""
        assert issubclass(MechConsensusBehaviour, AbstractRoundBehaviour)

    def test_initial_behaviour_is_registration_startup(self) -> None:
        """Test initial_behaviour_cls is RegistrationStartupBehaviour."""
        assert (
            MechConsensusBehaviour.initial_behaviour_cls is RegistrationStartupBehaviour
        )

    def test_abci_app_cls_is_mech_abci_app(self) -> None:
        """Test abci_app_cls is MechAbciApp."""
        assert MechConsensusBehaviour.abci_app_cls is MechAbciApp

    def test_background_behaviours_contains_background_behaviour(self) -> None:
        """Test background_behaviours_cls contains BackgroundBehaviour."""
        assert BackgroundBehaviour in MechConsensusBehaviour.background_behaviours_cls

    def test_behaviours_includes_task_submission(self) -> None:
        """Test behaviours includes all TaskSubmissionRoundBehaviour behaviours."""
        for b in TaskSubmissionRoundBehaviour.behaviours:
            assert b in MechConsensusBehaviour.behaviours

    def test_behaviours_includes_agent_registration(self) -> None:
        """Test behaviours includes all AgentRegistrationRoundBehaviour behaviours."""
        for b in AgentRegistrationRoundBehaviour.behaviours:
            assert b in MechConsensusBehaviour.behaviours

    def test_behaviours_includes_reset_pause(self) -> None:
        """Test behaviours includes all ResetPauseABCIConsensusBehaviour behaviours."""
        for b in ResetPauseABCIConsensusBehaviour.behaviours:
            assert b in MechConsensusBehaviour.behaviours

    def test_behaviours_includes_transaction_settlement(self) -> None:
        """Test behaviours includes all TransactionSettlementRoundBehaviour behaviours."""
        for b in TransactionSettlementRoundBehaviour.behaviours:
            assert b in MechConsensusBehaviour.behaviours

    def test_behaviours_includes_termination(self) -> None:
        """Test behaviours includes all TerminationAbciBehaviours behaviours."""
        for b in TerminationAbciBehaviours.behaviours:
            assert b in MechConsensusBehaviour.behaviours

    def test_behaviours_includes_delivery_rate(self) -> None:
        """Test behaviours includes all UpdateDeliveryRateRoundBehaviour behaviours."""
        for b in UpdateDeliveryRateRoundBehaviour.behaviours:
            assert b in MechConsensusBehaviour.behaviours
