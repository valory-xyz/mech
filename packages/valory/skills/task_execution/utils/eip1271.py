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

"""EIP-1271 signature check against a contract sender.

Wraps a single web3 view call. Kept in the task_execution skill (rather than
against the third-party ``GnosisSafeContract`` wrapper) so the minimal ABI
fragment lives alongside the caller: the check is a one-selector view that
does not warrant its own contract-package build tree.

Also fetches the marketplace ``getDomainSeparator`` view at boot time — a
``bytes32`` getter that the request-id derivation needs but that the
packaged ``MechMarketplace`` wrapper does not expose today. Colocated with
the sig-verify boot path so the ABI fragment and call site live in one
file; the mech ``paymentType`` is read through the packaged
``OlasMechContract.get_mech_type`` wrapper (both call the same on-chain
``paymentType()`` selector).
"""

from __future__ import annotations

from typing import Any, Dict

from aea_ledger_ethereum import EthereumApi
from requests.exceptions import RequestException
from web3.exceptions import BadFunctionCallOutput, ContractLogicError, Web3RPCError

# bytes4 of ``keccak256("isValidSignature(bytes32,bytes)")``. A contract
# sender must return exactly this value from its ``isValidSignature`` view
# for the presented signature to be accepted per EIP-1271.
EIP1271_MAGIC_VALUE: bytes = b"\x16\x26\xba\x7e"


_IS_VALID_SIGNATURE_ABI: Dict[str, Any] = {
    "inputs": [
        {"internalType": "bytes32", "name": "_dataHash", "type": "bytes32"},
        {"internalType": "bytes", "name": "_signature", "type": "bytes"},
    ],
    "name": "isValidSignature",
    "outputs": [{"internalType": "bytes4", "name": "", "type": "bytes4"}],
    "stateMutability": "view",
    "type": "function",
}


_DOMAIN_SEPARATOR_ABI: Dict[str, Any] = {
    "inputs": [],
    "name": "getDomainSeparator",
    "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
    "stateMutability": "view",
    "type": "function",
}


# Marketplace ``getRequestId`` view. Colocated with the rest of the
# boot-time verification helpers because the packaged ``MechMarketplace``
# contract wrapper does not expose it today and this repo carries that
# wrapper as a third-party package (git-ignored, upstream hash pinned).
# Extending the wrapper would drift the local file off its upstream hash
# and fail ``check-third-party-hashes`` on every CI run; keeping the ABI
# fragment here in the skill's utils module colocates it with the sig
# verifier that consumes it and keeps the third-party package untouched.
_GET_REQUEST_ID_ABI: Dict[str, Any] = {
    "inputs": [
        {"internalType": "address", "name": "mech", "type": "address"},
        {"internalType": "address", "name": "requester", "type": "address"},
        {"internalType": "bytes", "name": "data", "type": "bytes"},
        {"internalType": "uint256", "name": "deliveryRate", "type": "uint256"},
        {"internalType": "bytes32", "name": "paymentType", "type": "bytes32"},
        {"internalType": "uint256", "name": "nonce", "type": "uint256"},
    ],
    "name": "getRequestId",
    "outputs": [{"internalType": "bytes32", "name": "requestId", "type": "bytes32"}],
    "stateMutability": "view",
    "type": "function",
}


def check_eip1271_signature(
    ledger_api: EthereumApi,
    contract_address: str,
    message_hash: bytes,
    signature: bytes,
) -> bool:
    """Return True iff the contract at ``contract_address`` accepts the signature.

    Calls the sender contract's ``isValidSignature(bytes32,bytes)`` view; any
    return value other than the EIP-1271 magic value — including a revert or
    an ABI-decode failure — is flattened to False so the caller can treat the
    outcome as a simple boolean.

    :param ledger_api: the ledger API object.
    :param contract_address: the contract sender to query.
    :param message_hash: the 32-byte message hash the caller signed.
    :param signature: the packed signature bytes.
    :return: True on magic-value match; False otherwise (including on revert).
    """
    contract_instance = ledger_api.api.eth.contract(
        address=ledger_api.api.to_checksum_address(contract_address),
        abi=[_IS_VALID_SIGNATURE_ABI],
    )
    try:
        returned = contract_instance.functions.isValidSignature(
            bytes(message_hash), bytes(signature)
        ).call()
    except (
        ContractLogicError,
        BadFunctionCallOutput,
        Web3RPCError,
        ValueError,
        RequestException,
    ):
        # Web3 raises ``BadFunctionCallOutput`` when the target address
        # holds no code and ``Web3RPCError`` when the node returns a
        # JSON-RPC error; ``requests`` transport failures (connection
        # error, read timeout — HTTPError is one subclass) surface as
        # ``RequestException``. Any of them means the contract did not
        # unambiguously return the magic value, so the signature is
        # not accepted.
        return False
    return bytes(returned) == EIP1271_MAGIC_VALUE


def get_marketplace_domain_separator(
    ledger_api: EthereumApi, marketplace_address: str
) -> bytes:
    """Return the marketplace EIP-712 domain separator as raw bytes32.

    Calls ``getDomainSeparator()`` (not the storage getter
    ``domainSeparator``): the view returns
    ``block.chainid == chainId ? domainSeparator : _computeDomainSeparator()``,
    so the value stays correct after a chain fork where the raw
    storage getter would return the pre-fork value.

    :param ledger_api: the ledger API object.
    :param marketplace_address: the mech marketplace contract address.
    :return: the 32-byte domain separator.
    :raises ValueError: only from the explicit 32-byte length guard.
        Web3 raises ``BadFunctionCallOutput`` when ``eth_call`` returns
        undecodable output, and ``ContractLogicError`` on a revert;
        both propagate to the caller.
    """
    contract_instance = ledger_api.api.eth.contract(
        address=ledger_api.api.to_checksum_address(marketplace_address),
        abi=[_DOMAIN_SEPARATOR_ABI],
    )
    domain_separator = contract_instance.functions.getDomainSeparator().call()
    result = bytes(domain_separator)
    if len(result) != 32:
        raise ValueError(f"domain_separator length is {len(result)}, expected 32")
    return result


def get_marketplace_request_id_view(
    ledger_api: EthereumApi,
    marketplace_address: str,
    mech: str,
    requester: str,
    data: bytes,
    delivery_rate: int,
    payment_type: bytes,
    nonce: int,
) -> bytes:
    """Call the marketplace ``getRequestId`` view and return its bytes32 output.

    Used at handler boot to cross-check the locally-reimplemented
    ``compute_request_id`` against the marketplace contract. Any
    mismatch means the marketplace was upgraded to a layout the local
    formula does not mirror; the caller flips the boot-cached
    constants off so subsequent offchain requests are refused rather
    than being accepted under a mis-derived request id.

    :param ledger_api: the ledger API object.
    :param marketplace_address: the marketplace contract address.
    :param mech: the mech contract address.
    :param requester: the requester address (EOA or Safe).
    :param data: the raw request-data blob.
    :param delivery_rate: the delivery rate uint256.
    :param payment_type: the mech's ``paymentType`` bytes32.
    :param nonce: the requester nonce uint256.
    :return: the 32-byte request id returned by the view.
    :raises ValueError: only from the explicit 32-byte length guard.
        Web3 raises ``BadFunctionCallOutput`` when ``eth_call`` returns
        undecodable output, and ``ContractLogicError`` on a revert;
        both propagate to the caller.
    """
    contract_instance = ledger_api.api.eth.contract(
        address=ledger_api.api.to_checksum_address(marketplace_address),
        abi=[_GET_REQUEST_ID_ABI],
    )
    request_id = contract_instance.functions.getRequestId(
        ledger_api.api.to_checksum_address(mech),
        ledger_api.api.to_checksum_address(requester),
        data,
        delivery_rate,
        payment_type,
        nonce,
    ).call()
    result = bytes(request_id)
    if len(result) != 32:
        raise ValueError(f"request_id length is {len(result)}, expected 32")
    return result
