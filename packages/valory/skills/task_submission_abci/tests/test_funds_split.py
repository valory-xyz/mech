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


def test_split_owner_operator_at_t0(
    fs_behaviour: Any,
    fs_ctx: Any,
    run_to_completion: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify T=0 split with service_owner_share = 0.1.

    Asserts:
      - Owner receives exactly 10% of profits (0.2 xDAI).
      - Operator bucket gets the remainder (1.8 xDAI).
    """
    ONE_XDAI: int = 10**18

    # Configure params as per defaults
    fs_ctx.params.profit_split_balance = ONE_XDAI  # 1 xDAI threshold
    fs_ctx.params.service_owner_share = 0.1  # 10%
    fs_ctx.params.on_chain_service_id = 1

    # No agent funding required at T=0 -> all profits go owner/operators.
    def _get_agent_funding_amounts_none(
        self: Any,
    ) -> Generator[None, None, Optional[Dict[str, int]]]:
        if False:
            yield
        return {}

    def _get_service_owner(
        self: Any, service_id: int
    ) -> Generator[None, None, Optional[str]]:
        if False:
            yield
        return "0xOWNER"

    def _get_funds_by_operator(
        self: Any, operator_share: int
    ) -> Generator[None, None, Optional[Dict[str, int]]]:
        if False:
            yield
        return {"0xOP": operator_share}

    monkeypatch.setattr(
        type(fs_behaviour),
        "_get_agent_funding_amounts",
        _get_agent_funding_amounts_none,
    )
    monkeypatch.setattr(type(fs_behaviour), "_get_service_owner", _get_service_owner)
    monkeypatch.setattr(
        type(fs_behaviour), "_get_funds_by_operator", _get_funds_by_operator
    )

    profits: int = 2 * ONE_XDAI  # 2 xDAI
    split = run_to_completion(fs_behaviour._split_funds(profits))

    expected_owner: int = int(fs_ctx.params.service_owner_share * profits)  # 0.2 xDAI
    expected_operator: int = profits - expected_owner  # 1.8 xDAI
    assert split == {"0xOWNER": expected_owner, "0xOP": expected_operator}


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


def test_split_never_over_allocates(
    fs_behaviour: Any,
    fs_ctx: Any,
    run_to_completion: Callable[[Generator[Any, None, Any]], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Guard: total allocated must never exceed profits.

    Catches rounding/allocation bugs leading to sum(payouts) > profits.
    """
    profits: int = 1_000_000_000_000_003_139

    fs_ctx.params.service_owner_share = 0.1
    fs_ctx.params.on_chain_service_id = 1

    def _get_agent_funding_amounts(
        self: Any,
    ) -> Generator[None, None, Optional[Dict[str, int]]]:
        if False:
            yield
        return {}

    def _get_service_owner(
        self: Any, _sid: int
    ) -> Generator[None, None, Optional[str]]:
        if False:
            yield
        return "0xOWNER"

    def _get_funds_by_operator(
        self: Any, operator_share: int
    ) -> Generator[None, None, Optional[Dict[str, int]]]:
        if False:
            yield
        return {"0xOP": operator_share}

    monkeypatch.setattr(
        type(fs_behaviour), "_get_agent_funding_amounts", _get_agent_funding_amounts
    )
    monkeypatch.setattr(type(fs_behaviour), "_get_service_owner", _get_service_owner)
    monkeypatch.setattr(
        type(fs_behaviour), "_get_funds_by_operator", _get_funds_by_operator
    )

    split = run_to_completion(fs_behaviour._split_funds(profits))
    assert split is not None

    total_out: int = sum(split.values())
    assert total_out <= profits  # must not over-allocate


def test_owner_share_rounding_wei_error_xfail(
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

    fs_ctx.params.service_owner_share = 0.1
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

    assert owner_actual > owner_expected
