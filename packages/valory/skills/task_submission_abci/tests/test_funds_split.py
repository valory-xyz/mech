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

from .conftest import (  # noqa: F401
    DummyFundsSplit,
    fs_behaviour,
    fs_ctx,
    patch_mech_info,
)


def test_split_true_at_exact_threshold(
    fs_behaviour: DummyFundsSplit,
    fs_ctx: SimpleNamespace,
    patch_mech_info,
    run_to_completion,
) -> None:
    """Return True when the sum of mech balances equals the threshold."""
    fs_ctx.params.profit_split_balance = 10
    fs_ctx.params.agent_mech_contract_addresses = ["0xA"]
    patch_mech_info({"0xA": 10})
    assert run_to_completion(fs_behaviour._should_split_profits()) is True


def test_split_false_below_threshold(
    fs_behaviour: DummyFundsSplit,
    fs_ctx: SimpleNamespace,
    patch_mech_info,
    run_to_completion,
) -> None:
    """Return False when the sum of mech balances is below the threshold."""
    fs_ctx.params.profit_split_balance = 10
    fs_ctx.params.agent_mech_contract_addresses = ["0xA"]
    patch_mech_info({"0xA": 9})
    assert run_to_completion(fs_behaviour._should_split_profits()) is False


def test_split_true_with_multiple_mechs_sum_over_threshold(
    fs_behaviour: DummyFundsSplit,
    fs_ctx: SimpleNamespace,
    patch_mech_info,
    run_to_completion,
) -> None:
    """Sum balances across mechs; if total >= threshold, return True."""
    fs_ctx.params.profit_split_balance = 10
    fs_ctx.params.agent_mech_contract_addresses = ["0xA", "0xB"]
    patch_mech_info({"0xA": 6, "0xB": 5})  # total = 11
    assert run_to_completion(fs_behaviour._should_split_profits()) is True


def test_balance_logic_avoids_old_modulo_flakiness(
    fs_behaviour: DummyFundsSplit,
    fs_ctx: SimpleNamespace,
    patch_mech_info,
    run_to_completion,
) -> None:
    """
    Old modulo logic could miss splitting on a jump 9→11.
    Balance threshold triggers once value surpasses target.
    """
    fs_ctx.params.profit_split_balance = 10
    fs_ctx.params.agent_mech_contract_addresses = ["0xA"]
    patch_mech_info({"0xA": 11})
    assert run_to_completion(fs_behaviour._should_split_profits()) is True


def test_error_on_missing_mech_info_returns_none(
    fs_behaviour: DummyFundsSplit,
    fs_ctx: SimpleNamespace,
    patch_mech_info,
    run_to_completion,
) -> None:
    """If a mech returns None from _get_mech_info, method returns None (and logs error)."""
    fs_ctx.params.profit_split_balance = 10
    fs_ctx.params.agent_mech_contract_addresses = ["0xA", "0xMissing"]
    patch_mech_info({"0xA": 10})  # "0xMissing" intentionally absent
    assert run_to_completion(fs_behaviour._should_split_profits()) is None
