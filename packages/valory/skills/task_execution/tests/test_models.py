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
import pytest
from aea.exceptions import AEAEnforceError

import packages.valory.skills.task_execution.models as m


def test_params_init_derivations(params_kwargs, dialogue_skill_context):
    p = m.Params(name="params", **params_kwargs)

    # basic mirrors
    assert p.polling_interval == 12.5
    assert p.task_deadline == 111.0
    assert p.cleanup_freq == 77

    # lower-casing of mech keys + dataclass conversion
    assert set(p.mech_to_config.keys()) == {"0xmech"}
    cfg = p.mech_to_config["0xmech"]
    assert (
        isinstance(cfg, m.MechConfig)
        and cfg.use_dynamic_pricing is True
        and cfg.is_marketplace_mech is False
    )

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


def test_mech_config_from_dict_defaults_and_values():
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


def test_request_params_defaults():
    rp = m.RequestParams()
    assert rp.from_block == {"legacy": None, "marketplace": None}
    assert rp.last_polling == {"legacy": None, "marketplace": None}


def test_params_marketplace_flag_true(params_kwargs):
    params_kwargs["mech_marketplace_address"] = "0xabc123"
    p = m.Params(name="params", **params_kwargs)
    assert p.use_mech_marketplace is True


def test_params_tools_pricing_keys_match_ok(params_kwargs):
    params_kwargs["tools_to_package_hash"] = {"sum": "h1", "mul": "h2"}
    params_kwargs["tools_to_pricing"] = {"sum": 10, "mul": 20}
    p = m.Params(name="params", **params_kwargs)
    assert p.tools_to_pricing == {"sum": 10, "mul": 20}  # no exception


def test_params_tools_pricing_keys_mismatch_raises(params_kwargs):
    params_kwargs["tools_to_package_hash"] = {"sum": "h1"}
    params_kwargs["tools_to_pricing"] = {"mul": 20}  # mismatch
    with pytest.raises(AEAEnforceError) as ei:
        m.Params(name="params", **params_kwargs)
    # helpful to ensure we fail for mismatched keys
    assert "Extra keys" in str(ei.value)


def test_params_missing_required_key_raises(dialogue_skill_context, params_kwargs):
    bad = params_kwargs.copy()
    bad.pop("num_agents")  # required
    with pytest.raises(AEAEnforceError):
        m.Params(name="params", **bad)


def test_params_wrong_type_raises(dialogue_skill_context, params_kwargs):
    bad = params_kwargs.copy()
    bad["agent_index"] = "not-an-int"
    with pytest.raises(AEAEnforceError):
        m.Params(name="params", **bad)
