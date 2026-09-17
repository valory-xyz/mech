# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2026 Valory AG
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

from typing import Any, Dict, cast

import pytest
from aea.exceptions import AEAEnforceError

import packages.valory.skills.task_execution.models as m


def test_params_init_derivations(
    params_kwargs: Dict[str, Any], dialogue_skill_context: Any
) -> None:
    """
    Initialize Params and verify derived fields and defaults.

    :param params_kwargs: Baseline keyword arguments used to construct Params.
    :type params_kwargs: Dict[str, Any]
    :param dialogue_skill_context: Minimal skill context passed through in kwargs.
    :type dialogue_skill_context: Any
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
    assert p.req_to_error_callback == {}
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
    """
    Set a non-zero marketplace address and ensure use_mech_marketplace is True.

    :param params_kwargs: Baseline keyword arguments used to construct Params.
    :type params_kwargs: Dict[str, Any]
    """
    params_kwargs["mech_marketplace_address"] = "0xabc123"
    p: m.Params = m.Params(name="params", **params_kwargs)
    assert p.use_mech_marketplace is True


def test_params_tools_pricing_keys_match_ok(params_kwargs: Dict[str, Any]) -> None:
    """
    When tools_to_pricing keys match tools_to_package_hash, initialization succeeds.

    :param params_kwargs: Baseline keyword arguments used to construct Params.
    :type params_kwargs: Dict[str, Any]
    """
    params_kwargs["tools_to_package_hash"] = {"sum": "h1", "mul": "h2"}
    params_kwargs["tools_to_pricing"] = {"sum": 10, "mul": 20}
    p: m.Params = m.Params(name="params", **params_kwargs)
    assert p.tools_to_pricing == {"sum": 10, "mul": 20}


def test_params_tools_pricing_keys_mismatch_raises(
    params_kwargs: Dict[str, Any],
) -> None:
    """
    Mismatched pricing/package keys should raise an enforcement error.

    :param params_kwargs: Baseline keyword arguments used to construct Params.
    :type params_kwargs: Dict[str, Any]
    """
    params_kwargs["tools_to_package_hash"] = {"sum": "h1"}
    params_kwargs["tools_to_pricing"] = {"mul": 20}  # mismatch
    with pytest.raises(AEAEnforceError) as ei:
        m.Params(name="params", **params_kwargs)
    assert "Extra keys" in str(ei.value)


def test_params_missing_required_key_raises(
    dialogue_skill_context: Any, params_kwargs: Dict[str, Any]
) -> None:
    """
    Missing a required key (e.g., num_agents) should raise an enforcement error.

    :param dialogue_skill_context: Minimal skill context passed through in kwargs.
    :type dialogue_skill_context: Any
    :param params_kwargs: Baseline keyword arguments used to construct Params.
    :type params_kwargs: Dict[str, Any]
    """
    bad: Dict[str, Any] = params_kwargs.copy()
    bad.pop("num_agents")
    with pytest.raises(AEAEnforceError):
        m.Params(name="params", **bad)


def test_params_wrong_type_raises(
    dialogue_skill_context: Any, params_kwargs: Dict[str, Any]
) -> None:
    """
    Wrong type for a required key (e.g., agent_index as str) should raise an enforcement error.

    :param dialogue_skill_context: Minimal skill context passed through in kwargs.
    :type dialogue_skill_context: Any
    :param params_kwargs: Baseline keyword arguments used to construct Params.
    :type params_kwargs: Dict[str, Any]
    """
    bad: Dict[str, Any] = params_kwargs.copy()
    bad["agent_index"] = "not-an-int"
    with pytest.raises(AEAEnforceError):
        m.Params(name="params", **bad)


def test_params_empty_mech_to_config_raises_value_error(
    params_kwargs: Dict[str, Any],
) -> None:
    """Empty mech_to_config should raise ValueError when no address can be set."""
    params_kwargs["mech_to_config"] = {}
    with pytest.raises(ValueError, match="No mech contract addresses found"):
        m.Params(name="params", **params_kwargs)


@pytest.mark.parametrize(
    "mech_to_config, fallback, expected",
    [
        (
            {
                "0xlegacy": m.MechConfig(
                    use_dynamic_pricing=False, is_marketplace_mech=False
                ),
                "0xmarket": m.MechConfig(
                    use_dynamic_pricing=False, is_marketplace_mech=True
                ),
            },
            "0xlegacy",
            "0xmarket",
        ),
        (
            {
                "0xlegacy": m.MechConfig(
                    use_dynamic_pricing=False, is_marketplace_mech=False
                )
            },
            "0xlegacy",
            "0xlegacy",
        ),
        ({}, "0xonly", "0xonly"),
    ],
    ids=["prefers-marketplace-mech", "falls-back-to-first", "empty-config"],
)
def test_metrics_mech_address_prefers_marketplace_mech(
    mech_to_config: Dict[str, Any], fallback: str, expected: str
) -> None:
    """The label picks the marketplace mech, else the first configured mech.

    :param mech_to_config: the per-mech config map.
    :param fallback: ``agent_mech_contract_address``.
    :param expected: the label value.
    """
    from types import SimpleNamespace

    params = cast(
        m.MetricsParams,
        SimpleNamespace(
            mech_to_config=mech_to_config,
            agent_mech_contract_address=fallback,
            default_chain_id="gnosis",
        ),
    )
    assert m.metrics_mech_address(params) == expected


def test_offchain_metric_labels_pairs_chain_with_marketplace_mech() -> None:
    """One helper feeds every off-chain series so the two skills cannot drift."""
    from types import SimpleNamespace

    params = cast(
        m.MetricsParams,
        SimpleNamespace(
            mech_to_config={
                "0xmarket": m.MechConfig(
                    use_dynamic_pricing=False, is_marketplace_mech=True
                )
            },
            agent_mech_contract_address="0xother",
            default_chain_id=100,
        ),
    )
    assert m.offchain_metric_labels(params) == {
        "chain": "100",
        "mech_address": "0xmarket",
    }


def test_metrics_mech_address_is_lower_cased_regardless_of_config_case() -> None:
    """A checksummed address in config must not split one mech into two label values.

    task_execution lower-cases its mech keys and task_submission_abci keeps the
    configured case; both go through this helper, so it normalises.
    """
    from types import SimpleNamespace

    params = cast(
        m.MetricsParams,
        SimpleNamespace(
            mech_to_config={
                "0xFf82123dFB52ab75C417195c5fDB87630145ae81": m.MechConfig(
                    use_dynamic_pricing=False, is_marketplace_mech=True
                )
            },
            agent_mech_contract_address="0xAbC",
            default_chain_id="gnosis",
        ),
    )
    assert (
        m.metrics_mech_address(params) == "0xff82123dfb52ab75c417195c5fdb87630145ae81"
    )
    fallback = cast(
        m.MetricsParams,
        SimpleNamespace(
            mech_to_config={}, agent_mech_contract_address="0xAbC", default_chain_id="1"
        ),
    )
    assert m.metrics_mech_address(fallback) == "0xabc"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("https://www.valory.xyz/terms/mechs", "https://www.valory.xyz/terms/mechs"),
        (
            "  https://www.valory.xyz/terms/mechs  ",
            "https://www.valory.xyz/terms/mechs",
        ),
        ("", ""),
        ("   ", ""),
        (None, ""),
    ],
    ids=["set", "stripped", "empty", "whitespace", "none"],
)
def test_params_mech_terms_url_is_stripped_and_defaults_to_empty(
    params_kwargs: Dict[str, Any], given: Any, expected: str
) -> None:
    """
    ``mech_terms_url`` is stripped; blank or null means no terms are advertised.

    :param params_kwargs: Baseline keyword arguments used to construct Params.
    :type params_kwargs: Dict[str, Any]
    :param given: the configured value.
    :type given: Any
    :param expected: the value Params must expose.
    :type expected: str
    """
    params_kwargs["mech_terms_url"] = given
    p: m.Params = m.Params(name="params", **params_kwargs)
    assert p.mech_terms_url == expected


def test_params_mech_terms_url_absent_means_none_advertised(
    params_kwargs: Dict[str, Any],
) -> None:
    """
    A skill config without the key advertises no terms.

    :param params_kwargs: Baseline keyword arguments used to construct Params.
    :type params_kwargs: Dict[str, Any]
    """
    params_kwargs.pop("mech_terms_url", None)
    p: m.Params = m.Params(name="params", **params_kwargs)
    assert p.mech_terms_url == ""
