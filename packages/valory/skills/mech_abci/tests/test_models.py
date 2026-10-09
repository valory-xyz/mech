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

from typing import Any
from unittest.mock import patch

import pytest

from packages.valory.skills.mech_abci.composition import MechAbciApp
from packages.valory.skills.mech_abci.models import (
    DEFAULT_ERC8004_IDENTITY_REGISTRY,
    MARGIN,
    Params,
    TaskExecutionParams,
    parse_erc8004_agent_id,
    parse_identity_registry_address,
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


SAMPLE_AGENT_ID = 25323
OTHER_REGISTRY = "0x" + "ab" * 20


class TestParseErc8004AgentId:
    """Tests for parse_erc8004_agent_id."""

    @pytest.mark.parametrize("value", [0, 1, SAMPLE_AGENT_ID, 2**64])
    def test_accepts_non_negative_integers(self, value: int) -> None:
        """Any non-negative integer is a valid agent id, including 0."""
        assert parse_erc8004_agent_id(value) == value

    def test_none_means_unset(self) -> None:
        """An unset agent id means the mech publishes no proof."""
        assert parse_erc8004_agent_id(None) is None

    @pytest.mark.parametrize("value", [-1, True, False, "25323", 1.0, "", [], {}])
    def test_rejects_anything_else(self, value: Any) -> None:
        """Negative numbers, bools, strings and floats are configuration errors."""
        with pytest.raises(ValueError, match="erc8004_agent_id"):
            parse_erc8004_agent_id(value)


class TestParseIdentityRegistryAddress:
    """Tests for parse_identity_registry_address."""

    @pytest.mark.parametrize("value", [None, ""])
    def test_unset_falls_back_to_the_default_registry(self, value: Any) -> None:
        """Without an override the shared registry address is used."""
        assert (
            parse_identity_registry_address(value) == DEFAULT_ERC8004_IDENTITY_REGISTRY
        )

    @pytest.mark.parametrize(
        "value",
        [
            DEFAULT_ERC8004_IDENTITY_REGISTRY,
            OTHER_REGISTRY,
            OTHER_REGISTRY.upper().replace("0X", "0x"),
        ],
    )
    def test_accepts_a_hex_address(self, value: str) -> None:
        """A 0x-prefixed 20-byte hex address is returned unchanged."""
        assert parse_identity_registry_address(value) == value

    @pytest.mark.parametrize(
        "value",
        [
            "0x" + "ab" * 19,
            "0x" + "ab" * 21,
            "ab" * 20,
            "0x" + "zz" * 20,
            "eip155:100:" + DEFAULT_ERC8004_IDENTITY_REGISTRY,
            " " + DEFAULT_ERC8004_IDENTITY_REGISTRY,
            123,
        ],
    )
    def test_rejects_anything_else(self, value: Any) -> None:
        """Short, long, unprefixed, non-hex or non-string values are configuration errors."""
        with pytest.raises(ValueError, match="erc8004_identity_registry_address"):
            parse_identity_registry_address(value)


class TestParamsErc8004Settings:
    """Params reads and validates the ERC-8004 settings before the base params."""

    @staticmethod
    def _make_params(**kwargs: Any) -> Params:
        with patch.object(TaskExecutionParams, "__init__", return_value=None):
            return Params(**kwargs)

    def test_reads_the_configured_values(self) -> None:
        """Configured values land on the params object."""
        params = self._make_params(
            erc8004_agent_id=SAMPLE_AGENT_ID,
            erc8004_identity_registry_address=OTHER_REGISTRY,
        )
        assert params.erc8004_agent_id == SAMPLE_AGENT_ID
        assert params.erc8004_identity_registry_address == OTHER_REGISTRY

    def test_defaults_when_absent(self) -> None:
        """Absent keys mean no proof and the shared registry."""
        params = self._make_params()
        assert params.erc8004_agent_id is None
        assert (
            params.erc8004_identity_registry_address
            == DEFAULT_ERC8004_IDENTITY_REGISTRY
        )

    def test_bad_agent_id_fails_at_startup(self) -> None:
        """A misconfigured agent id stops the agent instead of serving a wrong proof."""
        with pytest.raises(ValueError, match="erc8004_agent_id"):
            self._make_params(erc8004_agent_id=-5)
