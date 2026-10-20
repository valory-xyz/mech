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

"""Encode-assertion tests for the olas_mech contract package."""

import json
from pathlib import Path
from unittest.mock import patch

from eth_abi import decode as abi_decode
from web3 import Web3

from packages.valory.contracts.olas_mech.contract import OlasMechContract

_ABI_PATH = Path(__file__).parent.parent / "build" / "AgentMech.json"
_REQUESTER = "0x1111111111111111111111111111111111111111"
_SENDER = "0x3333333333333333333333333333333333333333"
_MECH = "0x2222222222222222222222222222222222222222"
_CAP_OVERLOAD_SIGNATURE = (
    "deliverMarketplaceWithSignatures("
    "address,(bytes,bytes,bytes)[],uint256[],uint256[],bytes)"
)
_LEGACY_OVERLOAD_SIGNATURE = (
    "deliverMarketplaceWithSignatures(address,(bytes,bytes,bytes)[],uint256[],bytes)"
)


def _web3_contract() -> object:
    """Build a Web3 contract instance from the packaged ABI."""
    with open(_ABI_PATH) as f:
        artifact = json.load(f)
    return Web3().eth.contract(abi=artifact["abi"])


def _expected_cap_overload_selector() -> str:
    """keccak256(signature)[:4] in hex, no 0x."""
    return Web3().keccak(text=_CAP_OVERLOAD_SIGNATURE).hex()[:8]


class TestGetOffchainDeliverDataWithCap:
    """`get_offchain_deliver_data_with_cap` encodes the 5-arg overload against the packaged ABI."""

    def _call(self) -> dict:
        """Invoke the wrapper with a sample single-request batch, patching simulate_tx."""
        deliver_with_sigs = [
            {
                "requestData": b"\xaa" * 32,
                "signature": b"\xbb" * 65,
                "deliveryData": b"\xcc" * 32,
            }
        ]
        with (
            patch.object(
                OlasMechContract, "get_instance", return_value=_web3_contract()
            ),
            patch.object(OlasMechContract, "simulate_tx", return_value={"data": True}),
        ):
            return OlasMechContract.get_offchain_deliver_data_with_cap(
                ledger_api=None,  # patched away via get_instance / simulate_tx
                contract_address=_MECH,
                sender=_SENDER,
                requester=_REQUESTER,
                deliverWithSignatures=deliver_with_sigs,
                deliveryRates=[99],
                maxRates=[100],
                paymentData=b"",
            )

    def test_calldata_selects_the_five_arg_overload(self) -> None:
        """First 4 bytes match the explicit 5-arg signature's selector.

        Catches a regression that would otherwise silently route to the
        legacy 4-arg overload (which lacks the cap check).
        """
        result = self._call()
        actual_selector = result["data"][:4].hex()
        assert actual_selector == _expected_cap_overload_selector()

    def test_calldata_round_trips_delivery_rates_and_max_rates(self) -> None:
        """The deliveryRates and maxRates arrays decode back to what we passed in.

        Catches a regression that would swap the two arrays or drop either.
        """
        result = self._call()
        types = ["address", "(bytes,bytes,bytes)[]", "uint256[]", "uint256[]", "bytes"]
        decoded = abi_decode(types, result["data"][4:])
        assert decoded[0].lower() == _REQUESTER.lower()
        assert list(decoded[2]) == [99]
        assert list(decoded[3]) == [100]


class TestGetOffchainDeliverDataLegacy:
    """`get_offchain_deliver_data` keeps resolving the 4-arg overload against the refreshed ABI."""

    def test_calldata_selects_the_four_arg_overload(self) -> None:
        """First 4 bytes match the legacy 4-arg signature's selector.

        The refreshed ABI carries three overloads; the bare-name lookup the
        flag-off path relies on must keep picking the 4-arg one.
        """
        deliver_with_sigs = [
            {
                "requestData": b"\xaa" * 32,
                "signature": b"\xbb" * 65,
                "deliveryData": b"\xcc" * 32,
            }
        ]
        with (
            patch.object(
                OlasMechContract, "get_instance", return_value=_web3_contract()
            ),
            patch.object(OlasMechContract, "simulate_tx", return_value={"data": True}),
        ):
            result = OlasMechContract.get_offchain_deliver_data(
                ledger_api=None,  # patched away via get_instance / simulate_tx
                contract_address=_MECH,
                sender=_SENDER,
                requester=_REQUESTER,
                deliverWithSignatures=deliver_with_sigs,
                deliveryRates=[100],
                paymentData=b"",
            )
        expected = Web3().keccak(text=_LEGACY_OVERLOAD_SIGNATURE).hex()[:8]
        assert result["data"][:4].hex() == expected
