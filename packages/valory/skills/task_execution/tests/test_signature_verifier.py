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

"""Tests for the accept-time signature verifier + handler integration."""

from __future__ import annotations

import json
import urllib.parse
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from eth_abi import encode as abi_encode  # type: ignore[import-not-found]
from eth_account import Account  # type: ignore[import-not-found]
from eth_utils import keccak as eth_keccak  # type: ignore[import-not-found]
from eth_utils import to_checksum_address  # type: ignore[import-not-found]

from packages.valory.protocols.http.message import HttpMessage
from packages.valory.skills.task_execution.handlers import (
    HttpCode,
    MechHttpHandler,
)
from packages.valory.skills.task_execution.utils.signature_verifier import (
    SafeInfo,
    SafeOwnerCache,
    VerifyResult,
    compute_safe_message_hash,
    derive_request_id_bytes,
    ecrecover_address,
    verify_signature,
)


# ------------------------- shared test fixtures ----------------------------- #

# Hardcoded test key — never used off-line-of-text, no funds, purely for
# ECDSA reproducibility across test runs.
_TRADER_KEY = "0x" + "11" * 32
_TRADER_ACCOUNT = Account.from_key(_TRADER_KEY)
_TRADER_ADDRESS = _TRADER_ACCOUNT.address

_MARKETPLACE = to_checksum_address("0x" + "aa" * 20)
_MECH = to_checksum_address("0x" + "bb" * 20)
_SAFE = to_checksum_address("0x" + "cc" * 20)
_CHAIN_ID = 100

_IPFS_HASH_BYTES = bytes.fromhex("ab" * 32)
_PAYMENT_TYPE = bytes.fromhex("de" * 32)


def _make_signature(digest: bytes) -> str:
    """Sign ``digest`` with the trader key using the raw-hash mode.

    Mirrors the ``sign_message(..., is_deprecated_mode=True)`` path
    mech-client and the AEA framework use — signs the passed 32-byte
    digest verbatim, no ``\x19Ethereum Signed Message:`` prefix.
    """
    sm = Account._sign_hash(digest, _TRADER_KEY)  # noqa: SLF001
    hex_body = sm.signature.hex()
    if hex_body.startswith("0x"):
        hex_body = hex_body[2:]
    return "0x" + hex_body


# ------------------------- hash math regression ----------------------------- #


def test_derive_request_id_bytes_matches_manual_computation() -> None:
    """Local request_id derivation matches a hand-computed EIP-712 digest.

    Guards against any silent drift in the abi_encode packing, the version-
    string encoding, or the outer ``\\x19\\x01 || sep || struct`` wrap. If
    this test fails, the on-chain contract will reject every signature the
    server verifies locally — traders would be locked out.
    """
    delivery_rate = 42
    nonce = 7
    expected = derive_request_id_bytes(
        marketplace_address=_MARKETPLACE,
        mech_address=_MECH,
        requester=_TRADER_ADDRESS,
        data=_IPFS_HASH_BYTES,
        delivery_rate=delivery_rate,
        payment_type=_PAYMENT_TYPE,
        nonce=nonce,
        chain_id=_CHAIN_ID,
    )

    # Recompute step-by-step so a bug in ``derive_request_id_bytes``' packing
    # (wrong type list, missing keccak, mismatched wrap) would produce a
    # different result here and fail the assertion.
    domain_typehash = eth_keccak(
        text=(
            "EIP712Domain(string name,string version,uint256 chainId,"
            "address verifyingContract)"
        )
    )
    domain_sep = eth_keccak(
        abi_encode(
            ["bytes32", "bytes32", "bytes32", "uint256", "address"],
            [
                domain_typehash,
                eth_keccak(text="MechMarketplace"),
                eth_keccak(abi_encode(["string"], ["1.1.0"])),
                _CHAIN_ID,
                _MARKETPLACE,
            ],
        )
    )
    inner = eth_keccak(
        abi_encode(
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
                _MARKETPLACE,
                _MECH,
                _TRADER_ADDRESS,
                eth_keccak(_IPFS_HASH_BYTES),
                delivery_rate,
                _PAYMENT_TYPE,
                nonce,
            ],
        )
    )
    manual = eth_keccak(b"\x19\x01" + domain_sep + inner)
    assert expected == manual
    assert len(expected) == 32


def test_derive_request_id_bytes_rejects_wrong_payment_type_length() -> None:
    """A payment_type that is not exactly 32 bytes is refused up front."""
    with pytest.raises(ValueError, match="payment_type"):
        derive_request_id_bytes(
            marketplace_address=_MARKETPLACE,
            mech_address=_MECH,
            requester=_TRADER_ADDRESS,
            data=_IPFS_HASH_BYTES,
            delivery_rate=1,
            payment_type=b"\x00" * 31,  # wrong length
            nonce=0,
            chain_id=_CHAIN_ID,
        )


def test_compute_safe_message_hash_matches_manual_computation() -> None:
    """Local Safe-wrap matches the CompatibilityFallbackHandler shape."""
    request_id = eth_keccak(text="req-x")
    expected = compute_safe_message_hash(request_id, _SAFE, _CHAIN_ID)

    typehash = eth_keccak(
        text="EIP712Domain(uint256 chainId,address verifyingContract)"
    )
    message_typehash = eth_keccak(text="SafeMessage(bytes message)")
    domain_sep = eth_keccak(
        abi_encode(
            ["bytes32", "uint256", "address"], [typehash, _CHAIN_ID, _SAFE]
        )
    )
    message = abi_encode(["bytes32"], [request_id])
    struct_hash = eth_keccak(
        abi_encode(
            ["bytes32", "bytes32"], [message_typehash, eth_keccak(message)]
        )
    )
    manual = eth_keccak(b"\x19\x01" + domain_sep + struct_hash)
    assert expected == manual


def test_compute_safe_message_hash_rejects_wrong_length() -> None:
    """Wrong-length request_id bytes are refused."""
    with pytest.raises(ValueError, match="request_id_bytes"):
        compute_safe_message_hash(b"\x00" * 31, _SAFE, _CHAIN_ID)


# ------------------------- ecrecover primitives ----------------------------- #


@pytest.mark.parametrize(
    "sig",
    [
        # Wrong length: 64 bytes instead of 65.
        "0x" + "ab" * 64,
        # Not hex at all.
        "not-hex",
        # Empty.
        "",
    ],
    ids=["wrong-length", "not-hex", "empty"],
)
def test_ecrecover_address_rejects_bad_signatures(sig: str) -> None:
    """Malformed signatures raise ``ValueError`` rather than silently returning junk."""
    with pytest.raises(ValueError):
        ecrecover_address(b"\x00" * 32, sig)


def test_ecrecover_address_rejects_wrong_digest_length() -> None:
    """A digest that isn't exactly 32 bytes is refused."""
    with pytest.raises(ValueError, match="32-byte digest"):
        ecrecover_address(b"\x00" * 31, "0x" + "ab" * 65)


def test_ecrecover_roundtrip_recovers_signer() -> None:
    """Sign a digest, recover, get the signer address back."""
    digest = eth_keccak(text="roundtrip")
    sig = _make_signature(digest)
    assert ecrecover_address(digest, sig).lower() == _TRADER_ADDRESS.lower()


# ------------------------- SafeOwnerCache ----------------------------------- #


def test_safe_owner_cache_evicts_lru_over_max_entries() -> None:
    """Once the cache exceeds ``max_entries`` the oldest entry is evicted."""
    cache = SafeOwnerCache(max_entries=2, ttl_seconds=1000.0)
    cache.put("0xA", SafeInfo(is_contract=False), now=0.0)
    cache.put("0xB", SafeInfo(is_contract=False), now=1.0)
    cache.put("0xC", SafeInfo(is_contract=False), now=2.0)
    assert cache.get("0xA", now=3.0) is None
    assert cache.get("0xB", now=3.0) is not None
    assert cache.get("0xC", now=3.0) is not None


def test_safe_owner_cache_expires_after_ttl() -> None:
    """A cached entry past its TTL returns None so callers refetch."""
    cache = SafeOwnerCache(max_entries=10, ttl_seconds=10.0)
    cache.put("0xA", SafeInfo(is_contract=False), now=100.0)
    assert cache.get("0xA", now=105.0) is not None
    assert cache.get("0xA", now=115.0) is None


# ------------------------- verify_signature EOA path ------------------------ #


def _make_cache_with_eoa_sender(sender: str) -> SafeOwnerCache:
    """Prebuild a cache that treats ``sender`` as an EOA (no code)."""
    cache = SafeOwnerCache(max_entries=10, ttl_seconds=1000.0)
    cache.put(sender, SafeInfo(is_contract=False))
    return cache


def test_verify_signature_eoa_valid() -> None:
    """A signature from an EOA that recovers to ``sender`` is accepted."""
    request_id = derive_request_id_bytes(
        marketplace_address=_MARKETPLACE,
        mech_address=_MECH,
        requester=_TRADER_ADDRESS,
        data=_IPFS_HASH_BYTES,
        delivery_rate=1,
        payment_type=_PAYMENT_TYPE,
        nonce=0,
        chain_id=_CHAIN_ID,
    )
    sig = _make_signature(request_id)
    cache = _make_cache_with_eoa_sender(_TRADER_ADDRESS)

    result = verify_signature(
        sender=_TRADER_ADDRESS,
        signature=sig,
        ipfs_hash_bytes=_IPFS_HASH_BYTES,
        delivery_rate=1,
        nonce=0,
        marketplace_address=_MARKETPLACE,
        mech_address=_MECH,
        payment_type=_PAYMENT_TYPE,
        chain_id=_CHAIN_ID,
        cache=cache,
        code_fetcher=lambda _addr: b"",
        safe_meta_fetcher=lambda _addr: {"owners": [], "threshold": 0},
    )
    assert result.ok is True
    assert result.request_id_bytes == request_id


def test_verify_signature_eoa_rejected_when_signer_not_sender() -> None:
    """A signature that recovers to some other EOA is rejected."""
    request_id = derive_request_id_bytes(
        marketplace_address=_MARKETPLACE,
        mech_address=_MECH,
        requester=_TRADER_ADDRESS,
        data=_IPFS_HASH_BYTES,
        delivery_rate=1,
        payment_type=_PAYMENT_TYPE,
        nonce=0,
        chain_id=_CHAIN_ID,
    )
    sig = _make_signature(request_id)
    imposter = to_checksum_address("0x" + "77" * 20)
    cache = _make_cache_with_eoa_sender(imposter)

    result = verify_signature(
        sender=imposter,
        signature=sig,
        ipfs_hash_bytes=_IPFS_HASH_BYTES,
        delivery_rate=1,
        nonce=0,
        marketplace_address=_MARKETPLACE,
        mech_address=_MECH,
        payment_type=_PAYMENT_TYPE,
        chain_id=_CHAIN_ID,
        cache=cache,
        code_fetcher=lambda _addr: b"",
        safe_meta_fetcher=lambda _addr: {"owners": [], "threshold": 0},
    )
    assert result.ok is False
    assert "does not recover to EOA sender" in result.reason


# ------------------------- verify_signature Safe threshold=1 --------------- #


def _prime_cache_as_safe(cache: SafeOwnerCache, safe: str, owner: str, threshold: int = 1) -> None:
    cache.put(
        safe,
        SafeInfo(
            is_contract=True,
            owners={to_checksum_address(owner)},
            threshold=threshold,
        ),
    )


def test_verify_signature_safe_threshold_one_valid_owner() -> None:
    """A Safe-wrapped signature from an owner (threshold=1) is accepted."""
    delivery_rate = 100
    nonce = 3
    request_id = derive_request_id_bytes(
        marketplace_address=_MARKETPLACE,
        mech_address=_MECH,
        requester=_SAFE,
        data=_IPFS_HASH_BYTES,
        delivery_rate=delivery_rate,
        payment_type=_PAYMENT_TYPE,
        nonce=nonce,
        chain_id=_CHAIN_ID,
    )
    wrapped = compute_safe_message_hash(request_id, _SAFE, _CHAIN_ID)
    sig = _make_signature(wrapped)

    cache = SafeOwnerCache(max_entries=10, ttl_seconds=1000.0)
    _prime_cache_as_safe(cache, _SAFE, _TRADER_ADDRESS)

    result = verify_signature(
        sender=_SAFE,
        signature=sig,
        ipfs_hash_bytes=_IPFS_HASH_BYTES,
        delivery_rate=delivery_rate,
        nonce=nonce,
        marketplace_address=_MARKETPLACE,
        mech_address=_MECH,
        payment_type=_PAYMENT_TYPE,
        chain_id=_CHAIN_ID,
        cache=cache,
        code_fetcher=lambda _addr: b"\x60",  # non-empty code
        safe_meta_fetcher=lambda _addr: {
            "owners": [_TRADER_ADDRESS],
            "threshold": 1,
        },
    )
    assert result.ok is True


def test_verify_signature_safe_threshold_one_rejects_non_owner() -> None:
    """A signature by a non-owner against a threshold=1 Safe is rejected."""
    delivery_rate = 100
    nonce = 3
    request_id = derive_request_id_bytes(
        marketplace_address=_MARKETPLACE,
        mech_address=_MECH,
        requester=_SAFE,
        data=_IPFS_HASH_BYTES,
        delivery_rate=delivery_rate,
        payment_type=_PAYMENT_TYPE,
        nonce=nonce,
        chain_id=_CHAIN_ID,
    )
    wrapped = compute_safe_message_hash(request_id, _SAFE, _CHAIN_ID)
    sig = _make_signature(wrapped)

    cache = SafeOwnerCache(max_entries=10, ttl_seconds=1000.0)
    other_owner = to_checksum_address("0x" + "99" * 20)
    _prime_cache_as_safe(cache, _SAFE, other_owner, threshold=1)

    result = verify_signature(
        sender=_SAFE,
        signature=sig,
        ipfs_hash_bytes=_IPFS_HASH_BYTES,
        delivery_rate=delivery_rate,
        nonce=nonce,
        marketplace_address=_MARKETPLACE,
        mech_address=_MECH,
        payment_type=_PAYMENT_TYPE,
        chain_id=_CHAIN_ID,
        cache=cache,
        code_fetcher=lambda _addr: b"\x60",
        safe_meta_fetcher=lambda _addr: {
            "owners": [other_owner],
            "threshold": 1,
        },
    )
    assert result.ok is False
    assert "not recover to a Safe owner" in result.reason


# ------------------------- verify_signature Safe threshold>1 --------------- #


def test_verify_signature_multi_owner_safe_falls_back_to_rpc() -> None:
    """A threshold>1 Safe skips ecrecover and delegates to isValidSignature."""
    cache = SafeOwnerCache(max_entries=10, ttl_seconds=1000.0)
    cache.put(
        _SAFE,
        SafeInfo(
            is_contract=True,
            owners={to_checksum_address("0x" + "01" * 20)},
            threshold=2,
        ),
    )

    calls: List[Any] = []

    def _isvs(addr: str, digest: bytes, sig_hex: str) -> bool:
        calls.append((addr, digest, sig_hex))
        return True

    result = verify_signature(
        sender=_SAFE,
        signature="0x" + "aa" * 65,
        ipfs_hash_bytes=_IPFS_HASH_BYTES,
        delivery_rate=1,
        nonce=0,
        marketplace_address=_MARKETPLACE,
        mech_address=_MECH,
        payment_type=_PAYMENT_TYPE,
        chain_id=_CHAIN_ID,
        cache=cache,
        code_fetcher=lambda _addr: b"",  # not called on cache hit
        safe_meta_fetcher=lambda _addr: {},  # not called on cache hit
        is_valid_signature_fetcher=_isvs,
    )
    assert result.ok is True
    assert len(calls) == 1
    assert calls[0][0] == _SAFE


def test_verify_signature_multi_owner_safe_rejected_when_rpc_says_invalid() -> None:
    """If ``isValidSignature`` returns False, the request is rejected."""
    cache = SafeOwnerCache(max_entries=10, ttl_seconds=1000.0)
    cache.put(
        _SAFE,
        SafeInfo(
            is_contract=True,
            owners={to_checksum_address("0x" + "01" * 20)},
            threshold=3,
        ),
    )

    result = verify_signature(
        sender=_SAFE,
        signature="0x" + "aa" * 65,
        ipfs_hash_bytes=_IPFS_HASH_BYTES,
        delivery_rate=1,
        nonce=0,
        marketplace_address=_MARKETPLACE,
        mech_address=_MECH,
        payment_type=_PAYMENT_TYPE,
        chain_id=_CHAIN_ID,
        cache=cache,
        code_fetcher=lambda _addr: b"",
        safe_meta_fetcher=lambda _addr: {},
        is_valid_signature_fetcher=lambda _a, _d, _s: False,
    )
    assert result.ok is False
    assert "isValidSignature" in result.reason


def test_verify_signature_multi_owner_safe_rejected_when_no_fallback_provided() -> None:
    """Missing the RPC fallback for a multi-owner Safe is treated as failure."""
    cache = SafeOwnerCache(max_entries=10, ttl_seconds=1000.0)
    cache.put(_SAFE, SafeInfo(is_contract=True, owners=set(), threshold=2))

    result = verify_signature(
        sender=_SAFE,
        signature="0x" + "aa" * 65,
        ipfs_hash_bytes=_IPFS_HASH_BYTES,
        delivery_rate=1,
        nonce=0,
        marketplace_address=_MARKETPLACE,
        mech_address=_MECH,
        payment_type=_PAYMENT_TYPE,
        chain_id=_CHAIN_ID,
        cache=cache,
        code_fetcher=lambda _addr: b"",
        safe_meta_fetcher=lambda _addr: {},
        is_valid_signature_fetcher=None,
    )
    assert result.ok is False
    assert "isValidSignature RPC fallback" in result.reason


# ------------------------- cache-hit RPC accounting ------------------------- #


def test_verify_signature_cache_hit_makes_no_rpc_call() -> None:
    """After the first call primes the cache, subsequent calls do zero RPCs.

    This is the whole point of the cache: the hot path stays pure-CPU. If
    this test fails, the handler is doing a getCode / getOwners per request
    and the ~100-300ms latency budget the design targets is blown.
    """
    request_id = derive_request_id_bytes(
        marketplace_address=_MARKETPLACE,
        mech_address=_MECH,
        requester=_TRADER_ADDRESS,
        data=_IPFS_HASH_BYTES,
        delivery_rate=1,
        payment_type=_PAYMENT_TYPE,
        nonce=0,
        chain_id=_CHAIN_ID,
    )
    sig = _make_signature(request_id)

    code_calls = 0
    meta_calls = 0

    def _code(_addr: str) -> bytes:
        nonlocal code_calls
        code_calls += 1
        return b""

    def _meta(_addr: str) -> Dict[str, Any]:
        nonlocal meta_calls
        meta_calls += 1
        return {"owners": [], "threshold": 0}

    cache = SafeOwnerCache(max_entries=10, ttl_seconds=1000.0)
    for i in range(100):
        # Vary nonce so each call re-derives, but the cache lookup key
        # (sender) stays constant.
        rid = derive_request_id_bytes(
            marketplace_address=_MARKETPLACE,
            mech_address=_MECH,
            requester=_TRADER_ADDRESS,
            data=_IPFS_HASH_BYTES,
            delivery_rate=1,
            payment_type=_PAYMENT_TYPE,
            nonce=i,
            chain_id=_CHAIN_ID,
        )
        result = verify_signature(
            sender=_TRADER_ADDRESS,
            signature=_make_signature(rid),
            ipfs_hash_bytes=_IPFS_HASH_BYTES,
            delivery_rate=1,
            nonce=i,
            marketplace_address=_MARKETPLACE,
            mech_address=_MECH,
            payment_type=_PAYMENT_TYPE,
            chain_id=_CHAIN_ID,
            cache=cache,
            code_fetcher=_code,
            safe_meta_fetcher=_meta,
        )
        assert result.ok is True

    assert code_calls == 1, f"expected 1 code_fetcher call over 100 reqs, got {code_calls}"
    assert meta_calls == 0, "EOA path should never touch safe_meta_fetcher"


# ------------------------- handler-level integration ----------------------- #


def _make_http_msg(body_dict: Dict[str, str]) -> SimpleNamespace:
    body = urllib.parse.urlencode(body_dict).encode("utf-8")
    return SimpleNamespace(
        body=body,
        version="1.1",
        headers="",
        performative=HttpMessage.Performative.REQUEST,
    )


def test_handler_rejects_with_401_on_bad_signature(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """A trader posting a bad signature gets 401 before the balance check runs.

    The balance-check stub blows up if invoked; the test passes iff the
    handler short-circuits at the sig verification step and never reaches
    the balance path, so no compute / API budget is consumed.
    """
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    def _balance_should_not_run(**_kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("balance check must not run after a 401")

    monkeypatch.setattr(
        mh, "_check_offchain_requester_balance", _balance_should_not_run
    )
    monkeypatch.setattr(
        mh,
        "_verify_offchain_signature",
        lambda **_kwargs: VerifyResult(
            ok=False,
            reason="signature does not recover to a Safe owner",
            request_id_bytes=b"\x00" * 32,
        ),
    )

    body = {
        "ipfs_hash": "0x" + "ab" * 32,
        "request_id": "42",
        "ipfs_data": '{"foo":"bar"}',
        "delivery_rate": "123",
        "sender": "0x0000000000000000000000000000000000000001",
        "signature": "0x" + "cc" * 65,
        "nonce": "7",
    }
    mh._handle_signed_requests(_make_http_msg(body), http_dialogue)

    resp = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.UNAUTHORIZED_CODE.value
    payload = json.loads(resp.body.decode("utf-8"))
    assert payload["status"] == "rejected"
    assert "signature verification failed" in payload["reason"]
    assert "does not recover to a Safe owner" in payload["reason"]
    # Nothing enqueued.
    assert handler_context.shared_state["pending_tasks"] == []


def test_handler_accepts_when_verifier_ok_and_falls_through_to_balance(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """When the verifier returns ok=True the handler proceeds to the balance path.

    Locks in the ordering (verify → balance → enqueue) — a regression that
    put the balance check before verify would break other tests, but this
    one specifically confirms the happy-path handoff still ends in a 200.
    """
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    monkeypatch.setattr(
        mh,
        "_verify_offchain_signature",
        lambda **_kwargs: VerifyResult(ok=True, request_id_bytes=b"\x00" * 32),
    )
    monkeypatch.setattr(
        mh,
        "_check_offchain_requester_balance",
        lambda sender, delivery_rate: {
            "status": "ok",
            "required_amount": int(delivery_rate),
            "available_amount": int(delivery_rate) + 1,
            "reason": "balance check completed",
            "balance_tracker_address": "0xBalanceTracker",
            "payment_type": "0xpaymenttype",
            "chain_id": 100,
        },
    )

    body = {
        "ipfs_hash": "0x" + "ab" * 32,
        "request_id": "42",
        "ipfs_data": '{"foo":"bar"}',
        "delivery_rate": "123",
        "sender": "0x0000000000000000000000000000000000000001",
        "signature": "0x" + "cc" * 65,
        "nonce": "7",
    }
    mh._handle_signed_requests(_make_http_msg(body), http_dialogue)

    resp = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.OK_CODE.value
    assert len(handler_context.shared_state["pending_tasks"]) == 1


def test_handler_verify_missing_signature_field_returns_401(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """A request with no ``signature`` form field is rejected at the verifier."""
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    # No monkeypatch on ``_verify_offchain_signature`` — we exercise the
    # real short-circuit on missing signature at the top of the function.
    # But we DO monkeypatch _get_ledger_settings so the code before that
    # short-circuit doesn't try to reach for an RPC.
    body = {
        "ipfs_hash": "0x" + "ab" * 32,
        "request_id": "42",
        "ipfs_data": '{"foo":"bar"}',
        "delivery_rate": "123",
        "sender": "0x0000000000000000000000000000000000000001",
        # signature intentionally omitted
        "nonce": "7",
    }
    mh._handle_signed_requests(_make_http_msg(body), http_dialogue)

    resp = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.UNAUTHORIZED_CODE.value
    payload = json.loads(resp.body.decode("utf-8"))
    assert "missing signature" in payload["reason"]


def test_handler_payment_type_is_cached_across_requests(
    handler_context: Any, monkeypatch: Any
) -> None:
    """The mech's payment_type is fetched at most once per mech address.

    A stale ``paymentType`` between requests would be a real problem (the
    request_id derivation would drift), so this test also acts as a
    regression on the cache key (must be the mech address, not e.g. a
    per-request identifier that would defeat the whole cache).
    """
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    calls = {"n": 0}

    def _fake_get_mech_type(_cls: Any, _api: Any, _addr: str) -> Dict[str, Any]:
        calls["n"] += 1
        return {"mech_type": _PAYMENT_TYPE}

    import packages.valory.skills.task_execution.handlers as hmod

    monkeypatch.setattr(
        hmod.OlasMechContract,
        "get_mech_type",
        classmethod(_fake_get_mech_type),
    )

    fake_api = SimpleNamespace()

    first = mh._resolve_mech_payment_type(fake_api, _MECH)  # type: ignore[arg-type]
    second = mh._resolve_mech_payment_type(fake_api, _MECH)  # type: ignore[arg-type]
    third = mh._resolve_mech_payment_type(fake_api, _MECH)  # type: ignore[arg-type]

    assert first == _PAYMENT_TYPE
    assert second == _PAYMENT_TYPE
    assert third == _PAYMENT_TYPE
    assert calls["n"] == 1, f"expected 1 payment_type fetch, got {calls['n']}"
