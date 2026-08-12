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

import logging
from enum import Enum
from typing import Any, Dict, Optional

from aea_ledger_ethereum import EthereumApi
from requests.exceptions import RequestException, Timeout
from web3.exceptions import BadFunctionCallOutput, ContractLogicError, Web3Exception

# Prefix web3 uses on ``BadFunctionCallOutput`` when the decode failure
# comes from calling into an address that has no code (i.e. the sender
# is a plain EOA rather than a contract with an ``isValidSignature``
# view). See ``web3.contract.utils.call_contract_function``'s
# ``is_missing_code_error`` branch — it substitutes this message when
# both the return data and the contract's code are empty. The other
# ``BadFunctionCallOutput`` shape ("Could not decode contract function
# call to ... with return data: ...") fires when the target contract
# has code but the ``eth_call`` response cannot be decoded, which is
# what a pruned or lagging node returns.
_MISSING_CODE_MESSAGE_PREFIX = "Could not transact with/call contract function"

# bytes4 of ``keccak256("isValidSignature(bytes32,bytes)")``. A contract
# sender must return exactly this value from its ``isValidSignature`` view
# for the presented signature to be accepted per EIP-1271.
EIP1271_MAGIC_VALUE: bytes = b"\x16\x26\xba\x7e"


class Eip1271Verdict(str, Enum):
    """Tri-state outcome of an ``isValidSignature`` view call.

    * ``VALID`` — the contract returned the EIP-1271 magic value; the
      presented signature verifies. Accept.
    * ``DECLINED`` — the contract returned a different bytes4 value,
      reverted with :class:`ContractLogicError`, or the target address
      has no code (``BadFunctionCallOutput`` with the missing-code
      message shape). Semantically "the signature does not verify".
      The caller should reject with 401.
    * ``CALL_FAILED`` — infra-side failure: ``Web3RPCError``,
      ``BadResponseFormat``, ``Web3ValidationError``, ``RequestException``,
      ``BadFunctionCallOutput`` with the decode-failure shape (empty
      or malformed ``eth_call`` return from a pruned or lagging node
      against a target that DOES have code), ``ValueError``, or the
      wall-clock deadline expired. The caller should reject with
      503; the sender's credentials are not at fault, an RPC or
      transport error is.

    A single ``bool`` return flattened all three of the ``CALL_FAILED``
    reasons into the same outcome as a genuine ``DECLINED``, so a
    Safe requester saw 401 "unauthorized" during an RPC outage.
    """

    VALID = "valid"
    DECLINED = "declined"
    CALL_FAILED = "call_failed"


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
    gas: int,
    logger: Optional[logging.Logger] = None,
) -> Eip1271Verdict:
    """Verify the contract's ``isValidSignature`` view and return a tri-state.

    Calls the sender contract's ``isValidSignature(bytes32,bytes)``
    view. The outcome is one of the three :class:`Eip1271Verdict`
    values so the caller can route "the RPC failed" separately from
    "the signature does not verify" — flattening those two to the same
    boolean surfaced RPC outages as 401 "unauthorized" to well-
    behaved Safe callers.

    The call carries an explicit ``gas`` cap in the ``eth_call``
    transaction payload so a sender contract whose ``isValidSignature``
    view loops to the block gas limit is capped at a bounded amount
    of node work per request; an out-of-gas revert then surfaces as
    :class:`ContractLogicError` and returns ``DECLINED`` like any
    other revert.

    Transport-level read timeouts propagate as :class:`TimeoutError`
    so the caller can distinguish a slow / hostile ``isValidSignature``
    from either a plain rejection or an RPC infra failure.

    :param ledger_api: the ledger API object.
    :param contract_address: the contract sender to query.
    :param message_hash: the 32-byte message hash the caller signed.
    :param signature: the packed signature bytes.
    :param gas: the ``gas`` cap applied to the ``eth_call``. Bounds the
        amount of work a sender's ``isValidSignature`` view can force
        the RPC node to do per request.
    :param logger: optional logger used to record which of the three
        branches fired (including the exception type on ``DECLINED`` /
        ``CALL_FAILED``). Kept optional so callers with no logger in
        scope can still call the helper; recommended for the accept
        path so operator triage on 503s has the exception type
        visible.
    :return: an :class:`Eip1271Verdict`. ``VALID`` on magic-value
        match; ``DECLINED`` on a non-magic return value or a revert;
        ``CALL_FAILED`` on an infra-side failure (RPC error, transport
        error, ABI-decode failure, empty/undecodable ``eth_call``
        return from a pruned or lagging node).
    :raises TimeoutError: when the underlying HTTP read times out (the
        provider-level ``timeout`` on the ``EthereumApi`` fires). The
        caller distinguishes this from a signature rejection.
    """
    contract_instance = ledger_api.api.eth.contract(
        address=ledger_api.api.to_checksum_address(contract_address),
        abi=[_IS_VALID_SIGNATURE_ABI],
    )
    try:
        returned = contract_instance.functions.isValidSignature(
            bytes(message_hash), bytes(signature)
        ).call({"gas": gas})
    except Timeout as exc:
        # Read/connect timeouts surface here when the sender's
        # ``isValidSignature`` view is slow enough that the provider's
        # transport timeout fires. Re-raise as the built-in
        # ``TimeoutError`` so the caller can return a distinct verdict
        # (503 with the timeout reason) instead of the generic
        # signature-rejected outcome that a revert or bad output would
        # produce.
        raise TimeoutError(str(exc)) from exc
    except ContractLogicError as exc:
        # Revert (including the out-of-gas revert triggered by the
        # ``gas`` cap above). Semantically "the contract did not accept
        # this signature"; 401 at the caller.
        if logger is not None:
            logger.debug(
                "check_eip1271_signature DECLINED for %s: %s",
                contract_address,
                exc,
            )
        return Eip1271Verdict.DECLINED
    except BadFunctionCallOutput as exc:
        # Two distinct producers, disambiguated by the message shape
        # web3 sets on the raise: the ``eth_call`` target has no code
        # (the sender is a plain EOA whose ``ecrecover`` did not match
        # the request id) versus the target has code but the return
        # data does not decode (a pruned or lagging node responded
        # with empty or malformed data for a genuine contract call).
        # The first is a credential-side outcome and belongs in 401;
        # the second is infra-side and belongs in 503.
        message = str(exc)
        if message.startswith(_MISSING_CODE_MESSAGE_PREFIX):
            if logger is not None:
                logger.debug(
                    "check_eip1271_signature DECLINED for %s: no code at "
                    "target (%s)",
                    contract_address,
                    exc,
                )
            return Eip1271Verdict.DECLINED
        if logger is not None:
            logger.warning(
                "check_eip1271_signature CALL_FAILED for %s: undecodable "
                "return data (%s)",
                contract_address,
                exc,
            )
        return Eip1271Verdict.CALL_FAILED
    except (Web3Exception, RequestException, ValueError) as exc:
        # Infra-side failure classes at the JSON-RPC / transport
        # boundary: ``Web3RPCError`` on a JSON-RPC error,
        # ``BadResponseFormat`` / ``Web3ValidationError`` from the
        # ``web3.manager.formatted_response`` layer (outside the
        # ``RotatingHTTPProvider.make_request`` boundary),
        # ``RequestException`` for connection / HTTP / non-timeout
        # transport failures, and ``ValueError`` for ABI decode
        # failures the web3 layer surfaces without a
        # ``BadFunctionCallOutput`` wrap. All belong in the
        # retry-later bucket — the sender's credentials are not at
        # fault. ``BadFunctionCallOutput`` is handled above so the
        # codeless-target shape can route to ``DECLINED``.
        if logger is not None:
            logger.warning(
                "check_eip1271_signature CALL_FAILED for %s: %r",
                contract_address,
                exc,
            )
        return Eip1271Verdict.CALL_FAILED
    if bytes(returned) == EIP1271_MAGIC_VALUE:
        return Eip1271Verdict.VALID
    if logger is not None:
        logger.debug(
            "check_eip1271_signature DECLINED for %s: non-magic return %r",
            contract_address,
            bytes(returned).hex() if returned is not None else None,
        )
    return Eip1271Verdict.DECLINED


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
