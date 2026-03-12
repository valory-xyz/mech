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
"""Tests for mech_abci.models."""

from unittest.mock import patch

from packages.valory.skills.mech_abci.composition import MechAbciApp
from packages.valory.skills.mech_abci.models import (
    MARGIN,
    SharedState,
)
from packages.valory.skills.mech_abci.tests.conftest import (
    _make_context,
    _make_shared_state,
)
from packages.valory.skills.reset_pause_abci.rounds import Event as ResetPauseEvent
from packages.valory.skills.task_submission_abci.models import (
    SharedState as TaskExecSharedState,
)
from packages.valory.skills.task_submission_abci.rounds import (
    Event as TaskExecutionEvent,
)
from packages.valory.skills.transaction_settlement_abci.rounds import (
    Event as TransactionSettlementEvent,
)


def test_shared_state_init() -> None:
    """SharedState.__init__ sets last_processed_request_block_number to zero."""
    state = _make_shared_state()
    assert state.last_processed_request_block_number == 0


class TestSharedStateSetup:
    """Tests for SharedState.setup."""

    def test_setup_configures_round_timeout(
        self, preserve_event_to_timeout: None
    ) -> None:
        """Test setup configures round timeout on MechAbciApp."""
        state = _make_shared_state(_make_context(round_timeout=10.0))
        with patch.object(TaskExecSharedState, "setup", return_value=None):
            state.setup()
        assert MechAbciApp.event_to_timeout[TaskExecutionEvent.ROUND_TIMEOUT] == 10.0
        assert (
            MechAbciApp.event_to_timeout[
                TaskExecutionEvent.TASK_EXECUTION_ROUND_TIMEOUT
            ]
            == 10.0
        )
        assert MechAbciApp.event_to_timeout[ResetPauseEvent.ROUND_TIMEOUT] == 10.0
        assert (
            MechAbciApp.event_to_timeout[TransactionSettlementEvent.ROUND_TIMEOUT]
            == 10.0
        )

    def test_setup_configures_validate_and_finalize_timeout(
        self, preserve_event_to_timeout: None
    ) -> None:
        """Test setup configures validate and finalize timeouts."""
        state = _make_shared_state(_make_context(validate=20.0, finalize=30.0))
        with patch.object(TaskExecSharedState, "setup", return_value=None):
            state.setup()
        assert (
            MechAbciApp.event_to_timeout[TransactionSettlementEvent.VALIDATE_TIMEOUT]
            == 20.0
        )
        assert (
            MechAbciApp.event_to_timeout[TransactionSettlementEvent.FINALIZE_TIMEOUT]
            == 30.0
        )

    def test_reset_pause_timeout_adds_margin(
        self, preserve_event_to_timeout: None
    ) -> None:
        """Test setup adds MARGIN to reset_pause_duration for timeout."""
        reset_pause = 60.0
        state = _make_shared_state(_make_context(reset_pause=reset_pause))
        with patch.object(TaskExecSharedState, "setup", return_value=None):
            state.setup()
        assert (
            MechAbciApp.event_to_timeout[ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT]
            == reset_pause + MARGIN
        )

    def test_setup_calls_super_setup(self, preserve_event_to_timeout: None) -> None:
        """Test setup calls super().setup()."""
        state = _make_shared_state()
        with patch.object(TaskExecSharedState, "setup") as mock_super_setup:
            state.setup()
        mock_super_setup.assert_called_once()
