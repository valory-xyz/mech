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

from types import SimpleNamespace
from unittest.mock import patch

from packages.valory.skills.abstract_round_abci.models import ApiSpecs
from packages.valory.skills.delivery_rate_abci.models import (
    Params as SubscriptionParams,
)
from packages.valory.skills.mech_abci.composition import MechAbciApp
from packages.valory.skills.mech_abci.models import (
    MARGIN,
    Params,
    RandomnessApi,
    SharedState,
)
from packages.valory.skills.reset_pause_abci.rounds import Event as ResetPauseEvent
from packages.valory.skills.task_submission_abci.models import (
    Params as TaskExecutionParams,
)
from packages.valory.skills.task_submission_abci.models import (
    SharedState as TaskExecSharedState,
)
from packages.valory.skills.task_submission_abci.rounds import (
    Event as TaskExecutionEvent,
)
from packages.valory.skills.termination_abci.models import TerminationParams
from packages.valory.skills.transaction_settlement_abci.rounds import (
    Event as TransactionSettlementEvent,
)


def _make_context(round_timeout=10.0, validate=20.0, finalize=30.0, reset_pause=60.0):  # type: ignore
    return SimpleNamespace(
        params=SimpleNamespace(
            round_timeout_seconds=round_timeout,
            validate_timeout=validate,
            finalize_timeout=finalize,
            reset_pause_duration=reset_pause,
        )
    )


def _make_shared_state(ctx=None) -> SharedState:  # type: ignore
    """Create a SharedState instance bypassing the AEA framework init."""
    with patch.object(TaskExecSharedState, "__init__", return_value=None):
        state = SharedState.__new__(SharedState)
        SharedState.__init__(state)
    # context is a read-only property backed by _context
    object.__setattr__(state, "_context", ctx or _make_context())
    return state


class TestSharedStateInit:
    """Tests for SharedState.__init__."""

    def test_last_processed_block_starts_at_zero(self) -> None:
        """Test last_processed_request_block_number starts at zero."""
        state = _make_shared_state()
        assert state.last_processed_request_block_number == 0

    def test_abci_app_cls_is_mech_abci_app(self) -> None:
        """Test SharedState.abci_app_cls is MechAbciApp."""
        assert SharedState.abci_app_cls is MechAbciApp


class TestSharedStateSetup:
    """Tests for SharedState.setup."""

    def test_setup_configures_round_timeout(self) -> None:
        """Test setup configures round timeout on MechAbciApp."""
        state = _make_shared_state(_make_context(round_timeout=10.0))
        original = dict(MechAbciApp.event_to_timeout)
        try:
            with patch.object(TaskExecSharedState, "setup", return_value=None):
                state.setup()
            assert (
                MechAbciApp.event_to_timeout[TaskExecutionEvent.ROUND_TIMEOUT] == 10.0
            )
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
        finally:
            MechAbciApp.event_to_timeout.clear()
            MechAbciApp.event_to_timeout.update(original)

    def test_setup_configures_validate_and_finalize_timeout(self) -> None:
        """Test setup configures validate and finalize timeouts."""
        state = _make_shared_state(_make_context(validate=20.0, finalize=30.0))
        original = dict(MechAbciApp.event_to_timeout)
        try:
            with patch.object(TaskExecSharedState, "setup", return_value=None):
                state.setup()
            assert (
                MechAbciApp.event_to_timeout[
                    TransactionSettlementEvent.VALIDATE_TIMEOUT
                ]
                == 20.0
            )
            assert (
                MechAbciApp.event_to_timeout[
                    TransactionSettlementEvent.FINALIZE_TIMEOUT
                ]
                == 30.0
            )
        finally:
            MechAbciApp.event_to_timeout.clear()
            MechAbciApp.event_to_timeout.update(original)

    def test_reset_pause_timeout_adds_margin(self) -> None:
        """Test setup adds MARGIN to reset_pause_duration for timeout."""
        reset_pause = 60.0
        state = _make_shared_state(_make_context(reset_pause=reset_pause))
        original = dict(MechAbciApp.event_to_timeout)
        try:
            with patch.object(TaskExecSharedState, "setup", return_value=None):
                state.setup()
            assert (
                MechAbciApp.event_to_timeout[ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT]
                == reset_pause + MARGIN
            )
        finally:
            MechAbciApp.event_to_timeout.clear()
            MechAbciApp.event_to_timeout.update(original)

    def test_setup_calls_super_setup(self) -> None:
        """Test setup calls super().setup()."""
        state = _make_shared_state()
        original = dict(MechAbciApp.event_to_timeout)
        try:
            with patch.object(TaskExecSharedState, "setup") as mock_super_setup:
                state.setup()
            mock_super_setup.assert_called_once()
        finally:
            MechAbciApp.event_to_timeout.clear()
            MechAbciApp.event_to_timeout.update(original)


class TestMarginConstant:
    """Tests for the MARGIN constant."""

    def test_margin_value(self) -> None:
        """Test MARGIN is 5."""
        assert MARGIN == 5


class TestRandomnessApi:
    """Tests for RandomnessApi."""

    def test_is_subclass_of_api_specs(self) -> None:
        """Test RandomnessApi is a subclass of ApiSpecs."""
        assert issubclass(RandomnessApi, ApiSpecs)


class TestParams:
    """Tests for Params."""

    def test_inherits_from_all_three(self) -> None:
        """Test Params inherits from all three base classes."""
        assert issubclass(Params, TaskExecutionParams)
        assert issubclass(Params, SubscriptionParams)
        assert issubclass(Params, TerminationParams)
