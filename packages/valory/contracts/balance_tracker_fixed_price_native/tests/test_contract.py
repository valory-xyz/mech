# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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

"""Encode-assertion tests for the balance_tracker_fixed_price_native contract package."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from web3 import Web3

from packages.valory.contracts.balance_tracker_fixed_price_native.contract import (
    BalanceTrackerFixedPriceNativeContract,
)

_ABI_PATH = (
    Path(__file__).parent.parent / "build" / "BalanceTrackerFixedPriceNative.json"
)
_ACCOUNT = "0x1000000000000000000000000000000000000001"
_AMOUNT = 12345
_DEPOSIT_FOR_SELECTOR = "aa67c919"


def _web3_contract() -> object:
    """Build a Web3 contract instance from the packaged ABI."""
    with open(_ABI_PATH) as f:
        artifact = json.load(f)
    return Web3().eth.contract(abi=artifact["abi"])


class TestBuildDepositForData:
    """`build_deposit_for_data` produces well-formed `depositFor(account)` calldata plus a positive `value`."""

    def test_calldata_starts_with_deposit_for_selector(self) -> None:
        """The 4-byte selector matches keccak256("depositFor(address)")[:4]."""
        with patch.object(
            BalanceTrackerFixedPriceNativeContract,
            "get_instance",
            return_value=_web3_contract(),
        ):
            result = BalanceTrackerFixedPriceNativeContract.build_deposit_for_data(
                ledger_api=None,
                contract_address="0x0000000000000000000000000000000000000000",
                account=_ACCOUNT,
                amount=_AMOUNT,
            )
        assert result["data"][:4].hex() == _DEPOSIT_FOR_SELECTOR

    def test_calldata_has_expected_length(self) -> None:
        """4-byte selector + 32-byte address (no amount in calldata) = 36 bytes."""
        with patch.object(
            BalanceTrackerFixedPriceNativeContract,
            "get_instance",
            return_value=_web3_contract(),
        ):
            result = BalanceTrackerFixedPriceNativeContract.build_deposit_for_data(
                ledger_api=None,
                contract_address="0x0000000000000000000000000000000000000000",
                account=_ACCOUNT,
                amount=_AMOUNT,
            )
        assert len(result["data"]) == 36

    def test_value_matches_amount(self) -> None:
        """`value` is returned so the caller cannot omit tx.value on the payable call."""
        with patch.object(
            BalanceTrackerFixedPriceNativeContract,
            "get_instance",
            return_value=_web3_contract(),
        ):
            result = BalanceTrackerFixedPriceNativeContract.build_deposit_for_data(
                ledger_api=None,
                contract_address="0x0000000000000000000000000000000000000000",
                account=_ACCOUNT,
                amount=_AMOUNT,
            )
        assert result["value"] == _AMOUNT

    @pytest.mark.parametrize("bad_amount", [0, -1, -100])
    def test_zero_or_negative_amount_raises(self, bad_amount: int) -> None:
        """On-chain depositFor has no zero-value guard; caller-side guard prevents silent no-op credits."""
        with patch.object(
            BalanceTrackerFixedPriceNativeContract,
            "get_instance",
            return_value=_web3_contract(),
        ):
            with pytest.raises(ValueError, match="requires amount > 0"):
                BalanceTrackerFixedPriceNativeContract.build_deposit_for_data(
                    ledger_api=None,
                    contract_address="0x0000000000000000000000000000000000000000",
                    account=_ACCOUNT,
                    amount=bad_amount,
                )
