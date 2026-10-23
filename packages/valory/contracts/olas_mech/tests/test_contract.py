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
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, patch

import pytest
from aea_ledger_ethereum import EthereumApi
from eth_abi import decode as abi_decode
from web3 import Web3

from packages.valory.contracts.olas_mech.contract import (
    MARKETPLACE_DELIVER_WITH_CAP_SIGNATURE,
    MARKETPLACE_DELIVER_WITH_NONCES_SIGNATURE,
    OlasMechContract,
)

_ABI_PATH = Path(__file__).parent.parent / "build" / "AgentMech.json"
_REQUESTER = "0x1111111111111111111111111111111111111111"
_SENDER = "0x3333333333333333333333333333333333333333"
_MECH = "0x2222222222222222222222222222222222222222"
_MARKETPLACE = "0x4444444444444444444444444444444444444444"
_MARKETPLACE_CHECKSUM_INPUT = "0x343f2b005cf6d70ba610cd9f1f1927049414b582"
# Selectors of the marketplace's own settlement overloads (MechMarketplace ABI).
_MARKETPLACE_CAP_SELECTOR = "c2ee2841"
_MARKETPLACE_NONCES_SELECTOR = "39a6aac7"
_EXEC_TYPES = ["address", "uint256", "bytes", "uint8", "uint256"]
_CAP_TYPES = ["address", "(bytes,bytes,bytes)[]", "uint256[]", "uint256[]", "bytes"]
_NONCES_TYPES = [
    "address",
    "(bytes,bytes,bytes)[]",
    "uint256[]",
    "uint256[]",
    "uint256[]",
    "uint256[]",
    "bytes",
]
_DELIVER_WITH_SIGS = [
    {
        "requestData": b"\xaa" * 32,
        "signature": b"\xbb" * 65,
        "deliveryData": b"\xcc" * 32,
    },
    {
        "requestData": b"\xdd" * 32,
        "signature": b"\xee" * 65,
        "deliveryData": b"\xff" * 32,
    },
]
_CAP_OVERLOAD_SIGNATURE = (
    "deliverMarketplaceWithSignatures("
    "address,(bytes,bytes,bytes)[],uint256[],uint256[],bytes)"
)
_LEGACY_OVERLOAD_SIGNATURE = (
    "deliverMarketplaceWithSignatures(address,(bytes,bytes,bytes)[],uint256[],bytes)"
)
_NONCES_OVERLOAD_SIGNATURE = (
    "deliverMarketplaceWithSignatures("
    "address,(bytes,bytes,bytes)[],uint256[],uint256[],uint256[],uint256[],bytes)"
)


def _web3_contract() -> object:
    """Build a Web3 contract instance from the packaged ABI."""
    with open(_ABI_PATH) as f:
        artifact = json.load(f)
    return Web3().eth.contract(abi=artifact["abi"])


def _offline_ledger_api() -> EthereumApi:
    """An EthereumApi that never connects: contract reads and simulation are patched."""
    return EthereumApi(address="http://127.0.0.1:9")


def _call_through_exec(method: str, **kwargs: Any) -> Tuple[Dict, MagicMock]:
    """Invoke a settlement builder with the packaged ABI, patching only the simulation."""
    with (
        patch.object(OlasMechContract, "get_instance", return_value=_web3_contract()),
        patch.object(
            OlasMechContract, "simulate_tx", return_value={"data": True}
        ) as simulate,
    ):
        result = getattr(OlasMechContract, method)(
            ledger_api=_offline_ledger_api(),
            contract_address=_MECH,
            marketplace_address=_MARKETPLACE,
            sender=_SENDER,
            requester=_REQUESTER,
            deliverWithSignatures=_DELIVER_WITH_SIGS,
            paymentData=b"",
            **kwargs,
        )
    return result, simulate


def _split_exec(calldata: bytes) -> Tuple[str, Tuple]:
    """Return the outer selector and the decoded ``exec`` arguments."""
    return calldata[:4].hex(), abi_decode(_EXEC_TYPES, calldata[4:])


_CAP_KWARGS = {"deliveryRates": [99, 150], "maxRates": [100, 200]}
_NONCES_KWARGS = {
    "deliveryRates": [99, 150],
    "maxRates": [100, 200],
    "nonces": [4243, 4242],
    "deadlines": [1_900_000_000, 1_900_000_060],
}


@pytest.mark.parametrize(
    "method, kwargs, inner_selector",
    [
        ("get_offchain_deliver_data_with_cap", _CAP_KWARGS, _MARKETPLACE_CAP_SELECTOR),
        (
            "get_offchain_deliver_data_with_nonces",
            _NONCES_KWARGS,
            _MARKETPLACE_NONCES_SELECTOR,
        ),
    ],
)
class TestSettlementThroughExec:
    """The 5-/7-arg deliveries are marketplace calls run by the mech's ``exec``."""

    def test_outer_call_is_a_plain_exec_to_the_marketplace(
        self, method: str, kwargs: Dict, inner_selector: str
    ) -> None:
        """exec(marketplace, 0, inner, CALL, 0): no value, no delegatecall, all gas."""
        result, _ = _call_through_exec(method, **kwargs)
        selector, (to, value, _inner, operation, tx_gas) = _split_exec(result["data"])
        exec_selector = Web3.keccak(text="exec(address,uint256,bytes,uint8,uint256)")
        assert selector == exec_selector.hex()[:8]
        assert to.lower() == _MARKETPLACE.lower()
        assert (value, operation, tx_gas) == (0, 0, 0)

    def test_inner_call_selects_the_marketplace_overload(
        self, method: str, kwargs: Dict, inner_selector: str
    ) -> None:
        """The wrapped call's first 4 bytes are the marketplace overload's selector."""
        result, _ = _call_through_exec(method, **kwargs)
        _, (_to, _value, inner, _op, _gas) = _split_exec(result["data"])
        assert inner[:4].hex() == inner_selector

    def test_simulation_runs_the_exec_call_from_the_safe_on_the_mech(
        self, method: str, kwargs: Dict, inner_selector: str
    ) -> None:
        """The simulated call is exactly the returned calldata, sent to the mech."""
        result, simulate = _call_through_exec(method, **kwargs)
        (_ledger, address, sender, data), _ = simulate.call_args
        assert (address, sender) == (_MECH, _SENDER)
        assert data == "0x" + result["data"].hex()
        assert result["simulation_ok"] is True


def test_cap_inner_call_round_trips_its_arguments() -> None:
    """The wrapped 5-arg call decodes back to the requester, items and both rate arrays."""
    result, _ = _call_through_exec("get_offchain_deliver_data_with_cap", **_CAP_KWARGS)
    _, (_to, _value, inner, _op, _gas) = _split_exec(result["data"])
    requester, items, rates, caps, payment = abi_decode(_CAP_TYPES, inner[4:])
    assert requester.lower() == _REQUESTER.lower()
    assert [item[0] for item in items] == [b"\xaa" * 32, b"\xdd" * 32]
    assert [item[2] for item in items] == [b"\xcc" * 32, b"\xff" * 32]
    assert (list(rates), list(caps), payment) == ([99, 150], [100, 200], b"")


def test_nonces_inner_call_round_trips_its_arguments_in_order() -> None:
    """The wrapped 7-arg call keeps rates, caps, nonces and deadlines aligned."""
    result, _ = _call_through_exec(
        "get_offchain_deliver_data_with_nonces", **_NONCES_KWARGS
    )
    _, (_to, _value, inner, _op, _gas) = _split_exec(result["data"])
    decoded = abi_decode(_NONCES_TYPES, inner[4:])
    assert decoded[0].lower() == _REQUESTER.lower()
    assert [item[1] for item in decoded[1]] == [b"\xbb" * 65, b"\xee" * 65]
    assert list(decoded[2]) == [99, 150]
    assert list(decoded[3]) == [100, 200]
    assert list(decoded[4]) == [4243, 4242]
    assert list(decoded[5]) == [1_900_000_000, 1_900_000_060]


@pytest.mark.parametrize(
    "signature, selector",
    [
        (_CAP_OVERLOAD_SIGNATURE, _MARKETPLACE_CAP_SELECTOR),
        (_NONCES_OVERLOAD_SIGNATURE, _MARKETPLACE_NONCES_SELECTOR),
        (MARKETPLACE_DELIVER_WITH_CAP_SIGNATURE, _MARKETPLACE_CAP_SELECTOR),
        (MARKETPLACE_DELIVER_WITH_NONCES_SIGNATURE, _MARKETPLACE_NONCES_SELECTOR),
    ],
)
def test_marketplace_signatures_hash_to_the_marketplace_selectors(
    signature: str, selector: str
) -> None:
    """The overload signatures used for encoding hash to the marketplace's selectors."""
    assert Web3.keccak(text=signature).hex()[:8] == selector


def test_failed_simulation_is_reported() -> None:
    """A reverting exec simulation comes back as simulation_ok False."""
    with (
        patch.object(OlasMechContract, "get_instance", return_value=_web3_contract()),
        patch.object(OlasMechContract, "simulate_tx", return_value={"data": False}),
    ):
        result = OlasMechContract.get_offchain_deliver_data_with_cap(
            ledger_api=_offline_ledger_api(),
            contract_address=_MECH,
            marketplace_address=_MARKETPLACE,
            sender=_SENDER,
            requester=_REQUESTER,
            deliverWithSignatures=_DELIVER_WITH_SIGS[:1],
            deliveryRates=[99],
            maxRates=[100],
            paymentData=b"",
        )
    assert result["simulation_ok"] is False


class TestGetOffchainDeliverDataLegacy:
    """`get_offchain_deliver_data` keeps resolving the 4-arg overload against the refreshed ABI."""

    def test_calldata_selects_the_four_arg_overload(self) -> None:
        """First 4 bytes match the legacy 4-arg signature's selector.

        The bare-name lookup the flag-off path relies on must keep picking the
        4-arg overload as the ABI gains more.
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


def test_mech_marketplace_is_returned_checksummed() -> None:
    """The mech's ``mechMarketplace()`` comes back in checksum form."""
    instance = Web3().eth.contract(abi=_web3_contract().abi)
    instance.functions = MagicMock()
    instance.functions.mechMarketplace.return_value.call.return_value = (
        _MARKETPLACE_CHECKSUM_INPUT
    )
    with patch.object(OlasMechContract, "get_instance", return_value=instance):
        result = OlasMechContract.get_mech_marketplace(
            ledger_api=_offline_ledger_api(), contract_address=_MECH
        )
    assert result == {
        "mech_marketplace": Web3.to_checksum_address(_MARKETPLACE_CHECKSUM_INPUT)
    }
