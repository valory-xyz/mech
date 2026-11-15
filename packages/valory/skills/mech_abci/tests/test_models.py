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

from pathlib import Path
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import pytest
import yaml

from packages.valory.skills.mech_abci.composition import MechAbciApp
from packages.valory.skills.mech_abci.models import (
    MARGIN,
    Params,
    TaskExecutionParams,
    parse_erc8004_agent_id,
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
SAMPLE_CHAIN_ID = 100


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


class TestParamsErc8004Settings:
    """Params reads the agent id and checks it against the chain id."""

    @staticmethod
    def _make_params(**kwargs: Any) -> Params:
        def base_init(self: Params, *args: Any, **kw: Any) -> None:
            self.mech_events_chain_id = int(kw.get("mech_events_chain_id", 0) or 0)

        with patch.object(TaskExecutionParams, "__init__", base_init):
            return Params(**kwargs)

    def test_reads_the_configured_agent_id(self) -> None:
        """A configured agent id with a chain id lands on the params object."""
        params = self._make_params(
            erc8004_agent_id=SAMPLE_AGENT_ID, mech_events_chain_id=SAMPLE_CHAIN_ID
        )
        assert params.erc8004_agent_id == SAMPLE_AGENT_ID

    @pytest.mark.parametrize("chain_id", [0, SAMPLE_CHAIN_ID])
    def test_no_agent_id_needs_no_chain_id(self, chain_id: int) -> None:
        """Without an agent id the mech publishes no proof, whatever the chain."""
        params = self._make_params(mech_events_chain_id=chain_id)
        assert params.erc8004_agent_id is None

    @pytest.mark.parametrize("chain_id", [0, -1, None])
    def test_agent_id_without_chain_id_fails_at_startup(self, chain_id: Any) -> None:
        """An agent id with no chain id would 404 silently, so startup stops instead."""
        with pytest.raises(ValueError, match="mech_events_chain_id"):
            self._make_params(
                erc8004_agent_id=SAMPLE_AGENT_ID, mech_events_chain_id=chain_id
            )

    def test_agent_id_zero_still_needs_a_chain_id(self) -> None:
        """Agent id 0 counts as set, so the chain check applies to it too."""
        with pytest.raises(ValueError, match="mech_events_chain_id"):
            self._make_params(erc8004_agent_id=0)

    def test_bad_agent_id_fails_at_startup(self) -> None:
        """A misconfigured agent id stops the agent instead of serving a wrong proof."""
        with pytest.raises(ValueError, match="erc8004_agent_id"):
            self._make_params(erc8004_agent_id=-5, mech_events_chain_id=SAMPLE_CHAIN_ID)


SKILL_YAML = Path(__file__).parents[1] / "skill.yaml"
SAMPLE_MECH = "0x0000000000000000000000000000000000000001"
SAMPLE_MAX_DELIVERY_RATE = 42
SAMPLE_GAS_LIMIT = 123456
SAMPLE_MULTISEND = "0x0000000000000000000000000000000000000002"

# Keys in this skill's config that no Params in the MRO pops: the ones nothing
# reads, plus ``default_chain_id``, which abstract_round_abci's BaseParams reads
# without popping.
UNREAD_SKILL_CONFIG_KEYS = frozenset(
    {
        "default_chain_id",
        "ipfs_fetch_timeout",
        "polling_interval",
        "reset_period_count",
        "tools_to_package_hash",
        "tools_to_pricing",
        "use_polling",
    }
)


def _make_skill_params() -> Tuple[Params, MagicMock]:
    """Build the real Params from this skill's config, with one mech configured.

    :return: the params and the skill context's logger spy.
    """
    args = yaml.safe_load(SKILL_YAML.read_text())["models"]["params"]["args"]
    args.update(
        mech_to_config={
            SAMPLE_MECH: {"use_dynamic_pricing": False, "is_marketplace_mech": True}
        },
        mech_to_max_delivery_rate={SAMPLE_MECH: SAMPLE_MAX_DELIVERY_RATE},
        manual_gas_limit=SAMPLE_GAS_LIMIT,
        multisend_address=SAMPLE_MULTISEND,
        erc8004_agent_id=SAMPLE_AGENT_ID,
        mech_events_chain_id=SAMPLE_CHAIN_ID,
    )
    skill_context = MagicMock()
    skill_context.skill_id = "valory/mech_abci:0.1.0"
    params = Params(name="params", skill_context=skill_context, **args)
    return params, skill_context.logger


class TestParamsConsumeConfig:
    """The composed Params takes its keys out of kwargs before ``Model`` sees them."""

    def test_params_consume_their_config_keys(self) -> None:
        """Only keys that no class in the MRO pops reach the base ``Model``.

        Without an explicit configuration the framework builds
        ``params.config`` from exactly the kwargs that reach ``Model``.
        """
        params, _ = _make_skill_params()
        assert set(params.config) == UNREAD_SKILL_CONFIG_KEYS

    def test_unread_kwargs_warning_omits_api_keys(self) -> None:
        """``api_keys`` belongs to task_execution, so this skill neither reads nor reports it."""
        _, logger = _make_skill_params()
        messages = [str(call.args[0]) for call in logger.warning.call_args_list]
        assert messages, "expected the framework to report the unread keys"
        assert all("api_keys" not in message for message in messages)

    def test_keys_shared_across_the_mro_reach_every_reader(self) -> None:
        """A key read by several parent Params keeps its value in each of them."""
        params, _ = _make_skill_params()
        assert params.mech_max_delivery_rate == SAMPLE_MAX_DELIVERY_RATE
        assert params.mech_to_max_delivery_rate == {
            SAMPLE_MECH: SAMPLE_MAX_DELIVERY_RATE
        }
        assert params.manual_gas_limit == SAMPLE_GAS_LIMIT
        assert params.multisend_address == SAMPLE_MULTISEND
        assert params.erc8004_agent_id == SAMPLE_AGENT_ID
        assert params.mech_events_chain_id == SAMPLE_CHAIN_ID
        assert params.agent_mech_contract_address == SAMPLE_MECH
