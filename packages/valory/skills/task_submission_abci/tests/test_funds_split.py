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

"""Tests for funds splitting behaviour."""

from types import SimpleNamespace
from typing import Any, Callable, Dict, Generator, Optional, Tuple

import pytest


def test_split_true_at_exact_threshold(
    fs_behaviour: Any,
    fs_ctx: SimpleNamespace,
    patch_mech_info: Any,
    run_to_completion: Any,
) -> None:
    """Return True when the mech balance equals the threshold."""
    fs_ctx.params.profit_split_balance = 10
    fs_ctx.params.agent_mech_contract_addresses = ["0xA"]
    patch_mech_info({"0xA": 10})
    assert run_to_completion(fs_behaviour._should_split_profits()) is True


def test_split_false_below_threshold(
    fs_behaviour: Any,
    fs_ctx: SimpleNamespace,
    patch_mech_info: Any,
    run_to_completion: Any,
) -> None:
    """Return False when the mech balance is below the threshold."""
    fs_ctx.params.profit_split_balance = 10
    fs_ctx.params.agent_mech_contract_addresses = ["0xA"]
    patch_mech_info({"0xA": 9})
    assert run_to_completion(fs_behaviour._should_split_profits()) is False


def test_balance_logic_avoids_old_modulo_flakiness(
    fs_behaviour: Any,
    fs_ctx: SimpleNamespace,
    patch_mech_info: Any,
    run_to_completion: Any,
) -> None:
    """Threshold logic triggers once balance surpasses target (avoids 9→11 miss)."""
    fs_ctx.params.profit_split_balance = 10
    fs_ctx.params.agent_mech_contract_addresses = ["0xA"]
    patch_mech_info({"0xA": 11})
    assert run_to_completion(fs_behaviour._should_split_profits()) is True


def test_error_on_missing_mech_info_returns_false(
    fs_behaviour: Any,
    fs_ctx: SimpleNamespace,
    patch_mech_info: Any,
    run_to_completion: Any,
) -> None:
    """If a mech returns None from _get_mech_info, method returns False."""
    fs_ctx.params.profit_split_balance = 10
    fs_ctx.params.agent_mech_contract_addresses = ["0xMissing"]
    patch_mech_info({})
    assert run_to_completion(fs_behaviour._should_split_profits()) is False


def test_agent_dips_below_minimum_triggers_split_via_deficit(
    fs_behaviour: Any,
    fs_ctx: Any,
    run_to_completion: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When an agent needs funding (deficit present), _should_split_profits()

    returns True immediately, even if mech balance is below the threshold.
    """

    ONE_XDAI = 10**18
    fs_ctx.params.minimum_agent_balance = 10**17
    fs_ctx.params.agent_funding_amount = 2 * 10**17
    fs_ctx.params.profit_split_balance = ONE_XDAI
    fs_ctx.params.agent_mech_contract_addresses = ["0xMECH"]

    calls = {"funds": 0, "mech": 0}

    def _get_agent_funding_amounts_deficit(
        self: Any,
    ) -> Generator[None, None, Optional[Dict[str, int]]]:
        calls["funds"] += 1
        if False:
            yield
        return {"0xAGENT": fs_ctx.params.agent_funding_amount}

    def _get_mech_info_low_balance(
        self: Any, mech: str
    ) -> Generator[None, None, Optional[Tuple[bytes, str, int]]]:
        calls["mech"] += 1
        if False:
            yield
        return b"\x00", "0xTRACKER", 5 * 10**17  # 0.5 xDAI

    monkeypatch.setattr(
        type(fs_behaviour),
        "_get_agent_funding_amounts",
        _get_agent_funding_amounts_deficit,
    )
    monkeypatch.setattr(
        type(fs_behaviour), "_get_mech_info", _get_mech_info_low_balance
    )

    should_split = run_to_completion(fs_behaviour._should_split_profits())
    assert should_split is True
    assert calls["funds"] == 1, "Deficit path not evaluated"


def test_owner_share_rounding_wei_error(
    fs_behaviour: Any,
    fs_ctx: Any,
    run_to_completion: Callable[[Generator[Any, None, Any]], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Expose rounding: int(0.1 * profits) != profits // 10 for large integers.

    With profits = 1_000_000_000_000_003_139:
      - profits // 10 = 100_000_000_000_000_313
      - int(0.1 * profits) becomes 100_000_000_000_000_320 (off by +7 wei)
    """
    profits: int = 1_000_000_000_000_003_139

    fs_ctx.params.service_owner_share = 1000
    fs_ctx.params.on_chain_service_id = 1

    def _no_deficits(self: Any) -> Generator[None, None, Optional[Dict[str, int]]]:
        if False:
            yield
        return {}

    def _owner(self: Any, _sid: int) -> Generator[None, None, Optional[str]]:
        if False:
            yield
        return "0xOWNER"

    def _ops(
        self: Any, operator_share: int
    ) -> Generator[None, None, Optional[Dict[str, int]]]:
        if False:
            yield
        return {"0xOP": operator_share}

    monkeypatch.setattr(type(fs_behaviour), "_get_agent_funding_amounts", _no_deficits)
    monkeypatch.setattr(type(fs_behaviour), "_get_service_owner", _owner)
    monkeypatch.setattr(type(fs_behaviour), "_get_funds_by_operator", _ops)

    split = run_to_completion(fs_behaviour._split_funds(profits))
    assert split is not None

    owner_actual: int = split["0xOWNER"]  # calculated by current code
    owner_expected: int = profits // 10  # integer 10% benchmark

    assert owner_actual == owner_expected


def test_split_exact_allocation_no_deficits_bps(
    fs_behaviour: Any,
    fs_ctx: Any,
    run_to_completion: Callable[[Generator[Any, None, Any]], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    No agent deficits: owner gets bps share, operator gets the remainder.

    Sum of payouts must equal profits exactly (no over-allocation, no dust).
    """
    profits: int = 1_000_000_000_000_003_139

    fs_ctx.params.service_owner_share = 1_000  # 10% in basis points
    fs_ctx.params.on_chain_service_id = 1

    def _no_deficits(self: Any) -> Generator[None, None, Optional[Dict[str, int]]]:
        if False:
            yield
        return {}

    def _owner(self: Any, _sid: int) -> Generator[None, None, Optional[str]]:
        if False:
            yield
        return "0xOWNER"

    # Give 100% of operator share to a single operator for exact equality
    def _ops(
        self: Any, operator_share: int
    ) -> Generator[None, None, Optional[Dict[str, int]]]:
        if False:
            yield
        return {"0xOP": operator_share}

    monkeypatch.setattr(type(fs_behaviour), "_get_agent_funding_amounts", _no_deficits)
    monkeypatch.setattr(type(fs_behaviour), "_get_service_owner", _owner)
    monkeypatch.setattr(type(fs_behaviour), "_get_funds_by_operator", _ops)

    split = run_to_completion(fs_behaviour._split_funds(profits))
    assert split is not None

    owner_expected: int = profits * fs_ctx.params.service_owner_share // 10_000
    operator_expected: int = profits - owner_expected

    assert split == {"0xOWNER": owner_expected, "0xOP": operator_expected}
    assert sum(split.values()) == profits  # exact, proves no over-allocation


def test_split_with_agent_deficits_exact_totals_bps(
    fs_behaviour: Any,
    fs_ctx: Any,
    run_to_completion: Callable[[Generator[Any, None, Any]], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    With agent deficits present (total < profits), we first fund agents,

    then split the remainder between owner (bps) and operator. Total out == profits.
    """
    profits: int = 2_000_000_000_000_000_000  # 2 xDAI
    fs_ctx.params.service_owner_share = 1_000  # 10% in bps
    fs_ctx.params.on_chain_service_id = 1

    # Two agents each need 0.2 xDAI, total deficits = 0.4 xDAI
    agent_funding_amount: int = 200_000_000_000_000_000
    deficits_map: Dict[str, int] = {
        "0xA1": agent_funding_amount,
        "0xA2": agent_funding_amount,
    }
    total_deficits: int = sum(deficits_map.values())

    def _deficits(self: Any) -> Generator[None, None, Optional[Dict[str, int]]]:
        if False:
            yield
        return dict(deficits_map)

    def _owner(self: Any, _sid: int) -> Generator[None, None, Optional[str]]:
        if False:
            yield
        return "0xOWNER"

    # All operator share goes to a single operator for exact equality checks
    def _ops(
        self: Any, operator_share: int
    ) -> Generator[None, None, Optional[Dict[str, int]]]:
        if False:
            yield
        return {"0xOP": operator_share}

    monkeypatch.setattr(type(fs_behaviour), "_get_agent_funding_amounts", _deficits)
    monkeypatch.setattr(type(fs_behaviour), "_get_service_owner", _owner)
    monkeypatch.setattr(type(fs_behaviour), "_get_funds_by_operator", _ops)

    split = run_to_completion(fs_behaviour._split_funds(profits))
    assert split is not None

    # Remainder after funding agents
    remainder: int = profits - total_deficits
    owner_expected: int = remainder * fs_ctx.params.service_owner_share // 10_000
    operator_expected: int = remainder - owner_expected

    # Build expected dict
    expected: Dict[str, int] = dict(deficits_map)
    expected["0xOWNER"] = owner_expected
    expected["0xOP"] = operator_expected

    assert split == expected
    assert sum(split.values()) == profits  # nothing more than available funds


def test_split_when_deficits_exceed_profits_proportional_only_to_agents(
    fs_behaviour: Any,
    fs_ctx: Any,
    run_to_completion: Callable[[Generator[Any, None, Any]], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    If total agent deficits > profits, we split profits proportionally among agents

    and do NOT allocate anything to owner/operators. Sum of payouts <= profits.
    """
    profits: int = 900  # small number to make math obvious

    # Two agents require a total of 1200 > 900
    deficits_map: Dict[str, int] = {"0xA": 600, "0xB": 600}

    def _deficits(self: Any) -> Generator[None, None, Optional[Dict[str, int]]]:
        if False:
            yield
        return dict(deficits_map)

    def _owner(self: Any, _sid: int) -> Generator[None, None, Optional[str]]:
        if False:
            yield
        return "0xOWNER"

    def _ops(
        self: Any, operator_share: int
    ) -> Generator[None, None, Optional[Dict[str, int]]]:
        if False:
            yield
        # Should never be called in this path, but return empty if it is
        return {}

    monkeypatch.setattr(type(fs_behaviour), "_get_agent_funding_amounts", _deficits)
    monkeypatch.setattr(type(fs_behaviour), "_get_service_owner", _owner)
    monkeypatch.setattr(type(fs_behaviour), "_get_funds_by_operator", _ops)

    split = run_to_completion(fs_behaviour._split_funds(profits))
    assert split is not None

    # Proportional shares: floor((need_i * profits) / total_need)
    total_need: int = sum(deficits_map.values())
    expected_a: int = (deficits_map["0xA"] * profits) // total_need
    expected_b: int = (deficits_map["0xB"] * profits) // total_need

    # Owner/operator should not appear when deficits consume all profits
    assert "0xOWNER" not in split
    assert "0xOP" not in split

    # Exact agent results (may leave ≤ 1 wei of dust due to flooring, never > profits)
    assert split == {"0xA": expected_a, "0xB": expected_b}
    assert sum(split.values()) <= profits
