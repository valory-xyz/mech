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

"""Tests for the task_execution skill's models."""

from typing import Any, Dict

import pytest
from aea.exceptions import AEAEnforceError

import packages.valory.skills.task_execution.models as m


def test_params_init_derivations(
    params_kwargs: Dict[str, Any], dialogue_skill_context: Any
) -> None:
    """
    Initialize Params and verify derived fields and defaults.

    Ensures:
      - numeric mirrors (polling_interval, task_deadline, cleanup_freq)
      - mech key lowercasing + dataclass conversion
      - agent_mech_contract_addresses derived from keys
      - marketplace flag computed from address
      - defaultdict behaviour for request_id_to_num_timeouts
      - empty callback/deadline maps
      - RequestParams structures
    """
    p: m.Params = m.Params(name="params", **params_kwargs)

    # basic mirrors
    assert p.polling_interval == 12.5
    assert p.task_deadline == 111.0
    assert p.cleanup_freq == 77

    # lower-casing of mech keys + dataclass conversion
    assert set(p.mech_to_config.keys()) == {"0xmech"}
    cfg: m.MechConfig = p.mech_to_config["0xmech"]
    assert isinstance(cfg, m.MechConfig)
    assert cfg.use_dynamic_pricing is True
    assert cfg.is_marketplace_mech is False

    # agent_mech_contract_addresses derived from keys
    assert p.agent_mech_contract_addresses == ["0xmech"]

    # marketplace flag computed from address
    assert p.use_mech_marketplace is False  # ZERO_ADDRESS -> disabled

    # default counters/maps
    assert p.request_id_to_num_timeouts[123] == 0  # defaultdict
    assert p.req_to_callback == {}
    assert p.req_to_deadline == {}

    # request params structure
    assert p.req_params.from_block == {"legacy": None, "marketplace": None}
    assert p.req_params.last_polling == {"legacy": None, "marketplace": None}


def test_mech_config_from_dict_defaults_and_values() -> None:
    """Build MechConfig from dict and verify defaulted flags and overrides."""
    assert m.MechConfig.from_dict({}) == m.MechConfig(False, False)
    assert m.MechConfig.from_dict({"use_dynamic_pricing": True}) == m.MechConfig(
        True, False
    )
    assert m.MechConfig.from_dict({"is_marketplace_mech": True}) == m.MechConfig(
        False, True
    )
    assert m.MechConfig.from_dict(
        {"use_dynamic_pricing": True, "is_marketplace_mech": True}
    ) == m.MechConfig(True, True)


def test_request_params_defaults() -> None:
    """Ensure RequestParams default maps for from_block and last_polling."""
    rp: m.RequestParams = m.RequestParams()
    assert rp.from_block == {"legacy": None, "marketplace": None}
    assert rp.last_polling == {"legacy": None, "marketplace": None}


def test_params_marketplace_flag_true(params_kwargs: Dict[str, Any]) -> None:
    """Set a non-zero marketplace address and ensure use_mech_marketplace is True."""
    params_kwargs["mech_marketplace_address"] = "0xabc123"
    p: m.Params = m.Params(name="params", **params_kwargs)
    assert p.use_mech_marketplace is True


def test_params_tools_pricing_keys_match_ok(params_kwargs: Dict[str, Any]) -> None:
    """When tools_to_pricing keys match tools_to_package_hash, initialization succeeds."""
    params_kwargs["tools_to_package_hash"] = {"sum": "h1", "mul": "h2"}
    params_kwargs["tools_to_pricing"] = {"sum": 10, "mul": 20}
    p: m.Params = m.Params(name="params", **params_kwargs)
    assert p.tools_to_pricing == {"sum": 10, "mul": 20}  # no exception


def test_params_tools_pricing_keys_mismatch_raises(
    params_kwargs: Dict[str, Any]
) -> None:
    """Mismatched pricing/package keys should raise an enforcement error."""
    params_kwargs["tools_to_package_hash"] = {"sum": "h1"}
    params_kwargs["tools_to_pricing"] = {"mul": 20}  # mismatch
    with pytest.raises(AEAEnforceError) as ei:
        m.Params(name="params", **params_kwargs)
    assert "Extra keys" in str(ei.value)


def test_params_missing_required_key_raises(
    dialogue_skill_context: Any, params_kwargs: Dict[str, Any]
) -> None:
    """Missing a required key (e.g., num_agents) should raise an enforcement error."""
    bad: Dict[str, Any] = params_kwargs.copy()
    bad.pop("num_agents")
    with pytest.raises(AEAEnforceError):
        m.Params(name="params", **bad)


def test_params_wrong_type_raises(
    dialogue_skill_context: Any, params_kwargs: Dict[str, Any]
) -> None:
    """Wrong type for a required key (e.g., agent_index as str) should raise an enforcement error."""
    bad: Dict[str, Any] = params_kwargs.copy()
    bad["agent_index"] = "not-an-int"
    with pytest.raises(AEAEnforceError):
        m.Params(name="params", **bad)
