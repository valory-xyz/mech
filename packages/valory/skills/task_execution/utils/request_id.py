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

"""Marketplace request-id derivation and EOA signature recovery helpers.

Pure functions. No I/O, no ledger calls. Mirrors the on-chain
``MechMarketplace.getRequestId`` formula so an offchain request-id and its
requester signature can be validated locally against the same bytes32 the
marketplace would produce.
"""

from __future__ import annotations

from typing import Optional

from eth_account import Account
from eth_utils import keccak, to_checksum_address

# EIP-1271 magic value returned by a contract's isValidSignature when it
# accepts the presented (message_hash, signature) pair. bytes4 of
# keccak256("isValidSignature(bytes32,bytes)").
EIP1271_MAGIC_VALUE: bytes = b"\x16\x26\xba\x7e"

# Length of a canonical secp256k1 signature: r (32) + s (32) + v (1).
SIGNATURE_BYTES_LENGTH: int = 65

# Upper bound on the low-s half of the secp256k1 order. A signature whose
# s value is above this bound is malleable and must be rejected before
# recovery — mirrors the EIP-2 / marketplace ``_verifySignedHash`` guard.
SECP256K1_HALF_ORDER: int = int(
    "0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0",
    16,
)


def _encode_address(address: str) -> bytes:
    """Return the 32-byte left-padded encoding of an Ethereum address."""
    address_bytes = bytes.fromhex(to_checksum_address(address)[2:])
    return b"\x00" * 12 + address_bytes


def _encode_uint256(value: int) -> bytes:
    """Return the 32-byte big-endian encoding of a uint256."""
    if value < 0 or value >= 2**256:
        raise ValueError(f"value {value} out of uint256 range")
    return value.to_bytes(32, "big")


def _encode_bytes32(value: bytes) -> bytes:
    """Return the 32-byte value, right-padded with zeros if shorter."""
    if len(value) > 32:
        raise ValueError(f"bytes32 value longer than 32: len={len(value)}")
    return value + b"\x00" * (32 - len(value))


def compute_request_id(
    marketplace: str,
    mech: str,
    requester: str,
    request_data: bytes,
    delivery_rate: int,
    payment_type: bytes,
    nonce: int,
    domain_separator: bytes,
) -> bytes:
    r"""Compute the marketplace request_id from its constituent fields.

    Mirrors ``MechMarketplace.getRequestId`` byte-for-byte::

        requestId = keccak256(
            abi.encodePacked(
                "\x19\x01",
                domainSeparator,
                keccak256(abi.encode(
                    marketplace, mech, requester,
                    keccak256(requestData),
                    deliveryRate, paymentType, nonce
                ))
            )
        )

    :param marketplace: mech marketplace contract address (0x-prefixed).
    :param mech: mech contract address (0x-prefixed).
    :param requester: requester address, EOA or Safe (0x-prefixed).
    :param request_data: raw request data blob signed on-chain.
    :param delivery_rate: uint256 delivery rate.
    :param payment_type: bytes32 payment type identifier.
    :param nonce: uint256 requester nonce as tracked by
        ``MechMarketplace.mapNonces[requester]`` at the time of signing.
    :param domain_separator: 32-byte EIP-712 domain separator, as returned
        by ``MechMarketplace.domainSeparator()``.
    :return: 32-byte request_id.
    """
    if len(domain_separator) != 32:
        raise ValueError(
            f"domain_separator must be 32 bytes, got {len(domain_separator)}"
        )
    if len(payment_type) > 32:
        raise ValueError(
            f"payment_type must be at most 32 bytes, got {len(payment_type)}"
        )

    inner = (
        _encode_address(marketplace)
        + _encode_address(mech)
        + _encode_address(requester)
        + keccak(request_data)
        + _encode_uint256(delivery_rate)
        + _encode_bytes32(payment_type)
        + _encode_uint256(nonce)
    )
    inner_hash = keccak(inner)

    outer = b"\x19\x01" + domain_separator + inner_hash
    return keccak(outer)


def recover_eoa_signer(message_hash: bytes, signature: bytes) -> Optional[str]:
    r"""Recover the signer address from a raw-hash EOA signature.

    The marketplace signs the raw ``request_id`` bytes32 directly (no
    ``\x19Ethereum Signed Message:\n32`` prefix) because the request_id
    is itself an EIP-712 signable digest. This helper mirrors the
    marketplace's ``ecrecover`` path: normalise a Ledger-style
    ``v ∈ {0, 1}`` to ``{27, 28}`` and reject an upper-half ``s``
    (malleability).

    :param message_hash: the 32-byte request_id being signed.
    :param signature: the 65-byte ``r || s || v`` signature.
    :return: the checksummed signer address, or ``None`` if the signature
        is malformed, malleable, or otherwise unrecoverable.
    """
    if len(message_hash) != 32:
        return None
    if len(signature) != SIGNATURE_BYTES_LENGTH:
        return None

    v = signature[64]
    if v < 4:
        v += 27
    if v not in (27, 28):
        return None

    s = int.from_bytes(signature[32:64], "big")
    if s > SECP256K1_HALF_ORDER:
        return None

    normalised = signature[:64] + bytes([v])

    try:
        recovered = Account._recover_hash(message_hash, signature=normalised)
    except Exception:  # pylint: disable=broad-exception-caught
        # Any recovery failure — bad r/s, curve point at infinity, decode
        # error — is treated as a failed recovery so the caller can fall
        # through to the EIP-1271 path.
        return None
    return to_checksum_address(recovered)
