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
"""Tests for task_submission_abci.models."""

from types import SimpleNamespace
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.skills.abstract_round_abci.models import BaseParams
from packages.valory.skills.task_submission_abci.models import (
    MutableParams,
    Params,
    SharedState,
)
from packages.valory.skills.task_submission_abci.rounds import TaskSubmissionAbciApp


# ---------------------------------------------------------------------------
# SharedState
# ---------------------------------------------------------------------------


class TestSharedState:
    def test_abci_app_cls_is_task_submission(self):
        assert SharedState.abci_app_cls is TaskSubmissionAbciApp

    def test_prometheus_metrics_defined(self):
        # These are class-level Prometheus metrics
        assert hasattr(SharedState, "mech_delivery_last_block_number")
        assert hasattr(SharedState, "mech_agent_balance")
        assert hasattr(SharedState, "tool_delivery_time")


# ---------------------------------------------------------------------------
# MutableParams
# ---------------------------------------------------------------------------


class TestMutableParams:
    def test_latest_metadata_hash_defaults_to_none(self):
        p = MutableParams()
        assert p.latest_metadata_hash is None

    def test_latest_metadata_hash_can_be_mutated(self):
        p = MutableParams()
        p.latest_metadata_hash = b"some-hash"
        assert p.latest_metadata_hash == b"some-hash"

    def test_can_be_reset_to_none(self):
        p = MutableParams()
        p.latest_metadata_hash = b"hash"
        p.latest_metadata_hash = None
        assert p.latest_metadata_hash is None


# ---------------------------------------------------------------------------
# Params._ensure_get
# ---------------------------------------------------------------------------

MECH_ADDR = "0xMECH"
MECH_CONFIG = {MECH_ADDR: {"use_dynamic_pricing": False, "is_marketplace_mech": False}}
MECH_RATES = {MECH_ADDR: 10}

_VALID_KWARGS = {
    "task_wait_timeout": 30.0,
    "service_endpoint_base": "http://localhost",
    "multisend_address": "0xMULTI",
    "complementary_service_metadata_address": "0xMETA",
    "metadata_hash": "Qm...",
    "manual_gas_limit": 500000,
    "service_owner_share": 1000,
    "profit_split_balance": 500,
    "mech_to_config": MECH_CONFIG,
    "hash_checkpoint_address": "0xCHECK",
    "mech_marketplace_address": "0xMARKET",
    "mech_staking_instance_address": "0xSTAKE",
    "minimum_agent_balance": 100,
    "agent_funding_amount": 200,
    "default_chain_id": "100",
    "mech_to_max_delivery_rate": MECH_RATES,
}


def _make_skill_ctx():
    ctx = MagicMock()
    ctx.skill_id = "mock/skill:0.1.0"
    return ctx


class TestParamsEnsureGet:
    """Unit tests for the Params._ensure_get classmethod."""

    def test_ensure_get_returns_value_when_present(self):
        kwargs = {"skill_context": _make_skill_ctx(), "my_key": "my_value"}
        result = Params._ensure_get("my_key", kwargs, str)
        assert result == "my_value"
        # Key should NOT be popped
        assert "my_key" in kwargs

    def test_ensure_get_raises_when_key_missing(self):
        kwargs = {"skill_context": _make_skill_ctx()}
        with pytest.raises(Exception):  # enforce raises AEA exception
            Params._ensure_get("missing_key", kwargs, str)

    def test_ensure_get_raises_without_skill_context(self):
        kwargs = {"my_key": "val"}
        with pytest.raises(Exception):
            Params._ensure_get("my_key", kwargs, str)

    def test_ensure_get_does_not_pop_key(self):
        """_ensure_get should NOT pop the key from kwargs (unlike _ensure)."""
        kwargs = {"skill_context": _make_skill_ctx(), "my_key": "my_value"}
        Params._ensure_get("my_key", kwargs, str)
        assert "my_key" in kwargs


class TestParamsInit:
    """Tests for Params.__init__ — validation and attribute assignment."""

    def _make_kwargs(self, **overrides):
        kwargs = dict(_VALID_KWARGS)
        kwargs["skill_context"] = _make_skill_ctx()
        kwargs.update(overrides)
        return kwargs

    def test_agent_mech_contract_address_is_first_in_list(self):
        """agent_mech_contract_address should be the first key in mech_to_config."""
        kwargs = self._make_kwargs()
        with patch.object(BaseParams, "__init__", return_value=None):
            p = Params.__new__(Params)
            Params.__init__(p, **kwargs)
        assert p.agent_mech_contract_address == MECH_ADDR
        assert p.agent_mech_contract_addresses == [MECH_ADDR]

    def test_empty_mech_to_config_raises_value_error(self):
        """No addresses in mech_to_config → should raise ValueError."""
        kwargs = self._make_kwargs(mech_to_config={})
        with patch.object(BaseParams, "__init__", return_value=None):
            p = Params.__new__(Params)
            with pytest.raises(ValueError, match="No mech contract addresses"):
                Params.__init__(p, **kwargs)

    def test_task_wait_timeout_stored(self):
        kwargs = self._make_kwargs(task_wait_timeout=99.0)
        with patch.object(BaseParams, "__init__", return_value=None):
            p = Params.__new__(Params)
            Params.__init__(p, **kwargs)
        assert p.task_wait_timeout == 99.0

    def test_service_owner_share_stored(self):
        kwargs = self._make_kwargs(service_owner_share=500)
        with patch.object(BaseParams, "__init__", return_value=None):
            p = Params.__new__(Params)
            Params.__init__(p, **kwargs)
        assert p.service_owner_share == 500

    def test_mech_max_delivery_rate_from_first_value(self):
        rates = {MECH_ADDR: 42}
        kwargs = self._make_kwargs(mech_to_max_delivery_rate=rates)
        with patch.object(BaseParams, "__init__", return_value=None):
            p = Params.__new__(Params)
            Params.__init__(p, **kwargs)
        assert p.mech_max_delivery_rate == 42

    def test_task_mutable_params_initialized(self):
        kwargs = self._make_kwargs()
        with patch.object(BaseParams, "__init__", return_value=None):
            p = Params.__new__(Params)
            Params.__init__(p, **kwargs)
        assert isinstance(p.task_mutable_params, MutableParams)
        assert p.task_mutable_params.latest_metadata_hash is None

    def test_calls_super_init(self):
        kwargs = self._make_kwargs()
        with patch.object(BaseParams, "__init__") as mock_super:
            mock_super.return_value = None
            p = Params.__new__(Params)
            Params.__init__(p, **kwargs)
        mock_super.assert_called_once()
