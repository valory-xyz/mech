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

"""Tests for the marketplace request-id derivation and EOA recovery helpers.

``eth_abi.encode`` here is the *independent* reference implementation
that pins the production ``compute_request_id`` byte-for-byte against
the marketplace's Solidity ``getRequestId`` layout. Production never
imports ``eth_abi``, only this test does. ``check-dependencies`` still
forces the declaration in the skill's ``dependencies:`` because the
scanner walks the test tree too, so the version pin lives in
``skill.yaml`` (currently ``eth-abi==6.0.0b1``, the pre-release that
resolves transitively via the ``web3<8,>=7.15.0`` pin — track upstream
and switch to ``==6.0.0`` once the stable release ships).
"""

from __future__ import annotations

import pytest
from eth_abi import encode as abi_encode
from eth_account import Account
from eth_utils import keccak

from packages.valory.skills.task_execution.utils.request_id import (
    SECP256K1_HALF_ORDER,
    compute_request_id,
    recover_eoa_signer,
)

# --- request-id derivation ---------------------------------------------------


def _reference_request_id(
    marketplace: str,
    mech: str,
    requester: str,
    request_data: bytes,
    delivery_rate: int,
    payment_type: bytes,
    nonce: int,
    domain_separator: bytes,
) -> bytes:
    """Independent reference implementation using ``eth_abi.encode``.

    Written from the marketplace's ``getRequestId`` Solidity source, not
    by copying the helper. If the two diverge the test fails.

    :param marketplace: mech marketplace address.
    :param mech: mech contract address.
    :param requester: requester address.
    :param request_data: raw request data bytes.
    :param delivery_rate: uint256 delivery rate.
    :param payment_type: bytes32 payment type identifier.
    :param nonce: uint256 requester nonce.
    :param domain_separator: 32-byte EIP-712 domain separator.
    :return: 32-byte reference request_id.
    """
    inner = abi_encode(
        [
            "address",
            "address",
            "address",
            "bytes32",
            "uint256",
            "bytes32",
            "uint256",
        ],
        [
            marketplace,
            mech,
            requester,
            keccak(request_data),
            delivery_rate,
            payment_type,
            nonce,
        ],
    )
    inner_hash = keccak(inner)
    outer = b"\x19\x01" + domain_separator + inner_hash
    return keccak(outer)


@pytest.mark.parametrize(
    "request_data,delivery_rate,payment_type,nonce",
    [
        pytest.param(b"", 0, b"\x00" * 32, 0, id="all_zero_edges"),
        pytest.param(b"foo", 100, b"\xff" * 32, 5, id="small_bytes"),
        pytest.param(
            bytes(range(32)),
            2**200,
            (42).to_bytes(32, "big"),
            2**64,
            id="mid_range_values",
        ),
        pytest.param(
            bytes(range(64)),
            2**256 - 1,
            b"\xaa" * 32,
            2**256 - 1,
            id="uint256_upper_bound",
        ),
    ],
)
def test_compute_request_id_matches_abi_reference(
    request_data: bytes,
    delivery_rate: int,
    payment_type: bytes,
    nonce: int,
) -> None:
    """The helper produces the same bytes32 as the independent reference."""
    marketplace = "0x" + "11" * 20
    mech = "0x" + "22" * 20
    requester = "0x" + "33" * 20
    domain_separator = b"\xab" * 32

    actual = compute_request_id(
        marketplace=marketplace,
        mech=mech,
        requester=requester,
        request_data=request_data,
        delivery_rate=delivery_rate,
        payment_type=payment_type,
        nonce=nonce,
        domain_separator=domain_separator,
    )
    expected = _reference_request_id(
        marketplace=marketplace,
        mech=mech,
        requester=requester,
        request_data=request_data,
        delivery_rate=delivery_rate,
        payment_type=payment_type,
        nonce=nonce,
        domain_separator=domain_separator,
    )
    assert actual == expected


def test_compute_request_id_rejects_wrong_domain_separator_length() -> None:
    """A domain separator not exactly 32 bytes is a caller bug and raises."""
    with pytest.raises(ValueError, match="domain_separator"):
        compute_request_id(
            marketplace="0x" + "11" * 20,
            mech="0x" + "22" * 20,
            requester="0x" + "33" * 20,
            request_data=b"foo",
            delivery_rate=1,
            payment_type=b"\x00" * 32,
            nonce=0,
            domain_separator=b"\x00" * 31,
        )


def test_compute_request_id_rejects_oversized_payment_type() -> None:
    """A payment type longer than 32 bytes is a caller bug and raises."""
    with pytest.raises(ValueError, match="payment_type"):
        compute_request_id(
            marketplace="0x" + "11" * 20,
            mech="0x" + "22" * 20,
            requester="0x" + "33" * 20,
            request_data=b"foo",
            delivery_rate=1,
            payment_type=b"\x00" * 33,
            nonce=0,
            domain_separator=b"\x00" * 32,
        )


def test_compute_request_id_rejects_out_of_range_uint256() -> None:
    """A uint256 field above ``2**256 - 1`` is caller bug and raises."""
    with pytest.raises(ValueError, match="uint256"):
        compute_request_id(
            marketplace="0x" + "11" * 20,
            mech="0x" + "22" * 20,
            requester="0x" + "33" * 20,
            request_data=b"foo",
            delivery_rate=2**256,
            payment_type=b"\x00" * 32,
            nonce=0,
            domain_separator=b"\x00" * 32,
        )


def test_compute_request_id_short_payment_type_is_right_padded() -> None:
    """A short payment_type is right-padded to 32 bytes (Solidity bytes32 layout)."""
    short = b"\xde\xad"
    padded = short + b"\x00" * 30
    marketplace = "0x" + "11" * 20
    mech = "0x" + "22" * 20
    requester = "0x" + "33" * 20
    domain_separator = b"\xab" * 32
    a = compute_request_id(
        marketplace=marketplace,
        mech=mech,
        requester=requester,
        request_data=b"foo",
        delivery_rate=1,
        payment_type=short,
        nonce=0,
        domain_separator=domain_separator,
    )
    b = compute_request_id(
        marketplace=marketplace,
        mech=mech,
        requester=requester,
        request_data=b"foo",
        delivery_rate=1,
        payment_type=padded,
        nonce=0,
        domain_separator=domain_separator,
    )
    assert a == b


# --- EOA signer recovery -----------------------------------------------------


PRIVATE_KEY = "0x" + "01" * 32


def _sign_hash(request_id: bytes) -> bytes:
    """Sign a raw 32-byte hash with the fixed test key. Returns 65 bytes."""
    account = Account.from_key(PRIVATE_KEY)
    return account.unsafe_sign_hash(request_id).signature


def _test_signer_address() -> str:
    """Return the checksummed address of the fixed test key."""
    return Account.from_key(PRIVATE_KEY).address


def test_recover_eoa_signer_valid_signature_recovers_signer() -> None:
    """A signature produced by the private key recovers to its address."""
    request_id = keccak(b"some request id preimage")
    signature = _sign_hash(request_id)

    recovered = recover_eoa_signer(request_id, signature)
    assert recovered == _test_signer_address()


def test_recover_eoa_signer_ledger_style_v_is_normalised() -> None:
    """A v byte of 0 / 1 (Ledger convention) is normalised to 27 / 28."""
    request_id = keccak(b"ledger style request")
    signature = _sign_hash(request_id)
    # Downshift v from 27/28 to 0/1 so the helper's normalise branch fires.
    v_byte = signature[64]
    assert v_byte in (27, 28)
    ledger_sig = signature[:64] + bytes([v_byte - 27])

    recovered = recover_eoa_signer(request_id, ledger_sig)
    assert recovered == _test_signer_address()


def test_recover_eoa_signer_rejects_high_s_malleable_signature() -> None:
    """A signature with s above secp256k1n/2 is rejected as malleable."""
    request_id = keccak(b"malleability check")
    signature = _sign_hash(request_id)
    # Force s into the upper half of the curve order.
    high_s = SECP256K1_HALF_ORDER + 1
    malleable = signature[:32] + high_s.to_bytes(32, "big") + signature[64:]

    assert recover_eoa_signer(request_id, malleable) is None


@pytest.mark.parametrize(
    "signature",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"\x00" * 64, id="one_byte_short"),
        pytest.param(b"\x00" * 66, id="one_byte_long"),
    ],
)
def test_recover_eoa_signer_rejects_wrong_signature_length(
    signature: bytes,
) -> None:
    """A signature that is not exactly 65 bytes recovers to ``None``."""
    request_id = keccak(b"length check")
    assert recover_eoa_signer(request_id, signature) is None


def test_recover_eoa_signer_rejects_wrong_hash_length() -> None:
    """A hash that is not exactly 32 bytes recovers to ``None``."""
    signature = _sign_hash(keccak(b"anything"))
    short_hash = b"\x00" * 31
    assert recover_eoa_signer(short_hash, signature) is None


def test_recover_eoa_signer_rejects_out_of_range_v() -> None:
    """A v byte outside {0,1,27,28} (after normalisation) recovers to ``None``."""
    request_id = keccak(b"v range check")
    signature = _sign_hash(request_id)
    # v = 29 is not the ledger convention (< 4) and not the canonical (27/28).
    bad_v = signature[:64] + bytes([29])
    assert recover_eoa_signer(request_id, bad_v) is None


def test_recover_eoa_signer_wrong_signature_recovers_different_address() -> None:
    """A signature for a different hash recovers to a different address."""
    signed_hash = keccak(b"actually-signed")
    signature = _sign_hash(signed_hash)
    other_hash = keccak(b"different-hash")

    recovered = recover_eoa_signer(other_hash, signature)
    # Recovery may succeed with a spurious address or fail entirely; either
    # way it must NOT return the true signer's address.
    assert recovered != _test_signer_address()
