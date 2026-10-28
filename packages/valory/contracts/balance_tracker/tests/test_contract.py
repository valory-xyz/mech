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

"""Encode-assertion tests for the balance_tracker contract package."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

import pytest
from web3 import Web3
from web3.exceptions import ContractLogicError, Web3RPCError

from packages.valory.contracts.balance_tracker.contract import BalanceTrackerContract
from packages.valory.contracts.olas_mech.contract import (
    SIMULATION_OK_KEY,
    SIMULATION_TIMED_OUT_KEY,
)

_ABI_PATH = Path(__file__).parent.parent / "build" / "BalanceTracker.json"
_ACCOUNT = "0x1000000000000000000000000000000000000001"
_AMOUNT = 12345
_DEPOSIT_FOR_SELECTOR = "2f4f21e2"


def _web3_contract() -> object:
    """Build a Web3 contract instance from the packaged ABI."""
    with open(_ABI_PATH) as f:
        artifact = json.load(f)
    return Web3().eth.contract(abi=artifact["abi"])


class TestBuildDepositForData:
    """`build_deposit_for_data` produces well-formed `depositFor(account, amount)` calldata."""

    def test_calldata_starts_with_deposit_for_selector(self) -> None:
        """The 4-byte selector matches keccak256("depositFor(address,uint256)")[:4]."""
        with patch.object(
            BalanceTrackerContract, "get_instance", return_value=_web3_contract()
        ):
            result = BalanceTrackerContract.build_deposit_for_data(
                ledger_api=None,  # patched away
                contract_address="0x0000000000000000000000000000000000000000",
                account=_ACCOUNT,
                amount=_AMOUNT,
            )
        assert result["data"][:4].hex() == _DEPOSIT_FOR_SELECTOR

    def test_calldata_has_expected_length(self) -> None:
        """4-byte selector + 32-byte address + 32-byte uint256 = 68 bytes."""
        with patch.object(
            BalanceTrackerContract, "get_instance", return_value=_web3_contract()
        ):
            result = BalanceTrackerContract.build_deposit_for_data(
                ledger_api=None,
                contract_address="0x0000000000000000000000000000000000000000",
                account=_ACCOUNT,
                amount=_AMOUNT,
            )
        assert len(result["data"]) == 68

    def test_calldata_encodes_amount_argument(self) -> None:
        """The last 32 bytes decode to the supplied amount."""
        with patch.object(
            BalanceTrackerContract, "get_instance", return_value=_web3_contract()
        ):
            result = BalanceTrackerContract.build_deposit_for_data(
                ledger_api=None,
                contract_address="0x0000000000000000000000000000000000000000",
                account=_ACCOUNT,
                amount=_AMOUNT,
            )
        assert int.from_bytes(result["data"][-32:], "big") == _AMOUNT


def _ledger(eth_call: Callable[[Any], object]) -> SimpleNamespace:
    """Return an ``EthereumApi`` stand-in whose ``eth.call`` runs ``eth_call``."""
    return SimpleNamespace(
        api=SimpleNamespace(
            eth=SimpleNamespace(call=eth_call),
            to_checksum_address=lambda address: address,
        )
    )


def _raise(error: Exception) -> Callable[[Any], object]:
    """Return an ``eth.call`` stand-in that raises ``error``."""

    def _call(_tx: Any) -> object:
        raise error

    return _call


class TestGetProcessPaymentTx:
    """`get_process_payment_tx` returns the calldata with a bounded simulation verdict."""

    @pytest.mark.parametrize(
        ("eth_call", "ok", "timed_out"),
        [
            (lambda _tx: b"", True, False),
            (_raise(ContractLogicError("execution reverted")), False, False),
            (_raise(TimeoutError("read timed out")), False, True),
            (_raise(Web3RPCError("rate limit exceeded")), False, True),
        ],
    )
    def test_reports_the_simulation_verdict(
        self, eth_call: Callable[[Any], object], ok: bool, timed_out: bool
    ) -> None:
        """A revert is a conclusive failure; timeouts and RPC errors are inconclusive."""
        with patch.object(
            BalanceTrackerContract, "get_instance", return_value=_web3_contract()
        ):
            result = BalanceTrackerContract.get_process_payment_tx(
                ledger_api=_ledger(eth_call),  # type: ignore[arg-type]
                contract_address=_ACCOUNT,
                sender_address=_ACCOUNT,
                mech_address=_ACCOUNT,
            )
        assert result[SIMULATION_OK_KEY] is ok
        assert result[SIMULATION_TIMED_OUT_KEY] is timed_out
        assert isinstance(result["data"], bytes)
