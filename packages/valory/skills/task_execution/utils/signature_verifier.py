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

r"""Accept-time signature verification for off-chain mech requests.

The off-chain path accepts a signed request over HTTP, checks the requester's
prepaid balance and enqueues the task for execution. The signature the trader
posts is what the marketplace validates on-chain at settlement time via
``Safe.isValidSignature(request_id, sig)`` (for Safe requesters) or
``ecrecover(request_id, sig) == sender`` (for EOAs). If the trader posts a bad
signature, the mech does the work, spends API + compute, then eats the gas of
a settlement transaction that reverts with ``GS026``.

This module lets the server reject bad signatures at HTTP accept time so no
tool ever runs. The verification is local ecrecover against a cached set of
Safe owners — the first request from a new Safe pays one RPC for
``getOwners`` / ``getThreshold`` / ``eth_getCode``, all later requests from
the same Safe are pure CPU (microseconds). Multi-owner Safes fall back to a
per-request RPC ``isValidSignature`` call because reconstructing the packed
multi-sig blob locally is not worth the code complexity for the current
requester distribution (overwhelming majority are threshold=1 agent Safes).

The hashing helpers here (``derive_request_id_bytes``,
``compute_safe_message_hash``) are byte-for-byte mirrors of the on-chain
formulas — see the mech-interact ``offchain_request.py`` docstrings for the
matching Solidity references. If either drifts from the contract, verification
would either false-reject valid signatures (traders locked out) or false-accept
invalid ones (the exact bug this module exists to prevent). Do not change
either without cross-checking against the marketplace deployment.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set

from eth_abi import encode as abi_encode  # type: ignore[import-not-found]
from eth_account import Account  # type: ignore[import-not-found]
from eth_utils import keccak as eth_keccak  # type: ignore[import-not-found]
from eth_utils import to_checksum_address  # type: ignore[import-not-found]


_LOGGER = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# request_id derivation — local mirror of MechMarketplace.getRequestId
# (contracts/MechMarketplace.sol:883-908). Computing client-side avoids an RPC
# hop per request; the same hash is what the contract validates at settlement
# so byte-for-byte parity matters.
# ----------------------------------------------------------------------------

_MARKETPLACE_NAME = "MechMarketplace"
_MARKETPLACE_VERSION = "1.1.0"
_DOMAIN_TYPEHASH = eth_keccak(
    text=(
        "EIP712Domain(string name,string version,uint256 chainId,"
        "address verifyingContract)"
    )
)


# The name+version hashes and the domain separator per ``(chain_id,
# marketplace_address)`` are process-level constants: the marketplace
# contract does not rotate its EIP-712 name/version, and a mech is bound to
# one marketplace deployment. Caching lets the accept-time hot path skip an
# abi_encode + keccak256 per request.
_MARKETPLACE_NAME_HASH = eth_keccak(text=_MARKETPLACE_NAME)
_MARKETPLACE_VERSION_HASH = eth_keccak(
    abi_encode(["string"], [_MARKETPLACE_VERSION])
)
_DOMAIN_SEPARATOR_CACHE: Dict[tuple, bytes] = {}
_DOMAIN_SEPARATOR_CACHE_LOCK = threading.Lock()


def _compute_domain_separator(chain_id: int, marketplace_address: str) -> bytes:
    """Reproduce ``MechMarketplace._computeDomainSeparator`` in Python.

    The contract hashes the version through ``abi.encode`` not as raw bytes
    (``MechMarketplace.sol:160``), so the Python side must encode-then-hash
    the version string too. A raw ``keccak256(b"1.1.0")`` would not match.

    Result is cached per ``(chain_id, marketplace_address)`` so the hot
    verification path skips the ``abi_encode`` + ``keccak256`` cost on every
    request after the first.

    :param chain_id: EIP-155 integer chain id (settlement chain).
    :param marketplace_address: Checksummed marketplace contract address.
    :return: The 32-byte EIP-712 domain separator.
    """
    key = (chain_id, marketplace_address)
    cached = _DOMAIN_SEPARATOR_CACHE.get(key)
    if cached is not None:
        return cached
    sep = eth_keccak(
        abi_encode(
            ["bytes32", "bytes32", "bytes32", "uint256", "address"],
            [
                _DOMAIN_TYPEHASH,
                _MARKETPLACE_NAME_HASH,
                _MARKETPLACE_VERSION_HASH,
                chain_id,
                marketplace_address,
            ],
        )
    )
    with _DOMAIN_SEPARATOR_CACHE_LOCK:
        _DOMAIN_SEPARATOR_CACHE[key] = sep
    return sep


def derive_request_id_bytes(  # noqa: D417
    marketplace_address: str,
    mech_address: str,
    requester: str,
    data: bytes,
    delivery_rate: int,
    payment_type: bytes,
    nonce: int,
    chain_id: int,
) -> bytes:
    """Local mirror of ``MechMarketplace.getRequestId``.

    :param marketplace_address: ``address(this)`` on the settlement chain.
    :param mech_address: The mech the request targets.
    :param requester: The Safe (or EOA) that owns the prepaid balance.
    :param data: The 32-byte ipfs multihash the mech will submit as
        ``requestData`` at settlement (``bytes.fromhex(ipfs_hash[2:])``,
        NOT the JSON body carried on the ``ipfs_data`` form field).
    :param delivery_rate: Per-request charge (matches ``deliveryRate``).
    :param payment_type: 32-byte ``paymentType`` for the mech's payment model.
    :param nonce: The requester's current on-chain ``mapNonces`` value.
    :param chain_id: Settlement chain id (for the EIP-712 domain).
    :return: The 32-byte ``request_id`` the contract computes at settlement.
    :raises ValueError: if ``payment_type`` is not 32 bytes.
    """
    if len(payment_type) != 32:
        raise ValueError("payment_type must be 32 bytes")
    marketplace_address = to_checksum_address(marketplace_address)
    mech_address = to_checksum_address(mech_address)
    requester = to_checksum_address(requester)
    domain_separator = _compute_domain_separator(chain_id, marketplace_address)
    inner_hash = eth_keccak(
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
                marketplace_address,
                mech_address,
                requester,
                eth_keccak(data),
                delivery_rate,
                payment_type,
                nonce,
            ],
        )
    )
    return eth_keccak(b"\x19\x01" + domain_separator + inner_hash)


# ----------------------------------------------------------------------------
# Safe EIP-1271 message wrapping — mirrors ``CompatibilityFallbackHandler``
# (Safe v1.3.0 and v1.4.1) ``getMessageHashForSafe`` / ``isValidSignature``.
# When the marketplace validates delivery via ``Safe.isValidSignature``, the
# fallback handler rehashes the raw ``request_id`` into a ``SafeMessage``
# EIP-712 struct bound to ``(chainId, safeAddress)`` and runs
# ``checkSignatures`` against that wrapped digest. A raw ECDSA signature over
# the unwrapped ``request_id`` reverts with ``GS026``, so a Safe owner sig must
# be produced over the wrapped hash. Pre-v1.3.0 Safes used a domain that
# lacked ``chainId`` and are out of scope (Autonolas services deploy v1.3.0+).
# ----------------------------------------------------------------------------

_SAFE_DOMAIN_TYPEHASH = eth_keccak(
    text="EIP712Domain(uint256 chainId,address verifyingContract)"
)
_SAFE_MESSAGE_TYPEHASH = eth_keccak(text="SafeMessage(bytes message)")


def compute_safe_message_hash(
    request_id_bytes: bytes,
    safe_address: str,
    chain_id: int,
) -> bytes:
    """Wrap ``request_id`` in the Safe ``SafeMessage`` EIP-712 digest.

    :param request_id_bytes: The 32-byte ``request_id``.
    :param safe_address: The Safe contract acting as the requester.
    :param chain_id: The settlement chain id (bound into the EIP-712 domain).
    :return: The 32-byte digest the fallback handler computes for the message.
    :raises ValueError: if ``request_id_bytes`` is not 32 bytes.
    """
    if len(request_id_bytes) != 32:
        raise ValueError("request_id_bytes must be 32 bytes")
    safe_address = to_checksum_address(safe_address)
    domain_separator = eth_keccak(
        abi_encode(
            ["bytes32", "uint256", "address"],
            [_SAFE_DOMAIN_TYPEHASH, chain_id, safe_address],
        )
    )
    message = abi_encode(["bytes32"], [request_id_bytes])
    struct_hash = eth_keccak(
        abi_encode(
            ["bytes32", "bytes32"],
            [_SAFE_MESSAGE_TYPEHASH, eth_keccak(message)],
        )
    )
    return eth_keccak(b"\x19\x01" + domain_separator + struct_hash)


# ----------------------------------------------------------------------------
# ecrecover primitives
# ----------------------------------------------------------------------------


def _normalize_signature_bytes(signature: str) -> bytes:
    """Decode a 65-byte ``r || s || v`` signature from hex.

    ``eth_account`` accepts ``v`` in ``{0, 1, 27, 28}``; both mech-client
    (``sign_message(..., is_deprecated_mode=True)``) and mech-interact emit
    ``v`` in ``{27, 28}`` for the raw-hash signing path, so no normalisation is
    needed here — this only decodes the hex to bytes.

    :param signature: Hex string (with or without ``0x`` prefix).
    :return: 65 raw bytes.
    :raises ValueError: on malformed hex or wrong length.
    """
    s = signature.strip()
    if s.startswith("0x") or s.startswith("0X"):
        s = s[2:]
    raw = bytes.fromhex(s)
    if len(raw) != 65:
        raise ValueError(f"expected 65-byte signature, got {len(raw)}")
    return raw


def ecrecover_address(digest: bytes, signature: str) -> str:
    """Recover the signer address for ``digest`` and ``signature``.

    :param digest: 32-byte hash the signer signed (raw, not eth-prefixed).
    :param signature: Hex-encoded 65-byte ECDSA signature.
    :return: Checksummed signer address.
    :raises ValueError: on any failure (bad hex, wrong length, bad recovery id).
    """
    if len(digest) != 32:
        raise ValueError(f"expected 32-byte digest, got {len(digest)}")
    sig_bytes = _normalize_signature_bytes(signature)
    # ``_recover_hash`` is the raw-hash recovery path that skips the
    # ``\x19Ethereum Signed Message:\n32`` prefix. The trader signs the raw
    # request_id (EOA) or the Safe-wrapped hash (Safe owner) with
    # ``is_deprecated_mode=True`` in mech-client / ``get_signature`` in the
    # AEA framework, so we must recover with the same non-prefixed path.
    return Account._recover_hash(  # noqa: SLF001
        message_hash=digest, signature=sig_bytes
    )


# ----------------------------------------------------------------------------
# Cache of Safe metadata
# ----------------------------------------------------------------------------


@dataclass
class SafeInfo:
    """Cached on-chain metadata for a requester address.

    ``is_contract`` distinguishes EOA senders (plain ecrecover against
    ``sender``) from Safe senders (Safe-wrapped hash + owner-set membership).
    ``owners`` and ``threshold`` are populated only when ``is_contract``; for
    EOAs they stay empty / 0.
    """

    is_contract: bool
    owners: Set[str] = field(default_factory=set)
    threshold: int = 0
    expires_at: float = 0.0


class SafeOwnerCache:
    """Thread-safe LRU of Safe owner sets.

    Bounded size prevents unbounded growth if hostile clients spam distinct
    ``sender`` addresses to blow up memory. Entries older than ``ttl_seconds``
    are treated as misses so a Safe that changes owners eventually re-fetches
    (an owner add/remove without cache refresh would false-reject valid sigs
    from the new owner until the TTL rolls over).
    """

    def __init__(self, max_entries: int = 1000, ttl_seconds: float = 3600.0) -> None:
        """Initialize an empty cache.

        :param max_entries: LRU cap on distinct sender addresses tracked.
        :param ttl_seconds: How long a cached ``SafeInfo`` stays valid.
        """
        self._max = max(1, int(max_entries))
        self._ttl = float(ttl_seconds)
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, SafeInfo]" = OrderedDict()

    def get(self, address: str, now: Optional[float] = None) -> Optional[SafeInfo]:
        """Return the cached entry for ``address`` if fresh, else None.

        :param address: Requester address (case-normalised internally).
        :param now: Optional injected clock for tests.
        :return: The cached ``SafeInfo`` on a fresh hit, else ``None``.
        """
        key = address.lower()
        ts = time.time() if now is None else now
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.expires_at <= ts:
                # Expired: drop so callers see a clean miss and refetch.
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return entry

    def put(self, address: str, info: SafeInfo, now: Optional[float] = None) -> None:
        """Insert or refresh ``address`` -> ``info`` in the cache.

        :param address: Requester address (case-normalised internally).
        :param info: The metadata to store; its ``expires_at`` is set here.
        :param now: Optional injected clock for tests.
        """
        key = address.lower()
        ts = time.time() if now is None else now
        info.expires_at = ts + self._ttl
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = info
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def size(self) -> int:
        """Return the current number of cached entries."""
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        """Drop all cached entries (used by tests)."""
        with self._lock:
            self._data.clear()


# ----------------------------------------------------------------------------
# Verification result + orchestration
# ----------------------------------------------------------------------------


@dataclass
class VerifyResult:
    """Outcome of a single accept-time signature verification.

    :ivar ok: True iff the signature is valid for the derived request_id.
    :ivar reason: Short human-readable reason (empty on success).
    :ivar request_id_bytes: The 32-byte request_id the verifier computed
        locally, useful for logging / correlation with settlement.
    """

    ok: bool
    reason: str = ""
    request_id_bytes: Optional[bytes] = None


# Type aliases for injected callables so callers do not have to import web3
# here (the module stays test-friendly under monkeypatch).
CodeFetcher = Callable[[str], bytes]  # eth_getCode(address) -> code bytes
SafeMetaFetcher = Callable[[str], Dict[str, Any]]  # {owners: list, threshold: int}
IsValidSignatureFetcher = Callable[[str, bytes, str], bool]
# (safe_address, request_id_bytes, signature_hex) -> valid


def _lookup_or_fetch_safe_info(
    sender: str,
    cache: SafeOwnerCache,
    code_fetcher: CodeFetcher,
    safe_meta_fetcher: SafeMetaFetcher,
) -> SafeInfo:
    """Return ``SafeInfo`` for ``sender``, hitting the RPC only on miss.

    :param sender: Requester address (Safe or EOA, any case).
    :param cache: Shared owner cache.
    :param code_fetcher: Callable that returns ``eth_getCode`` bytes.
    :param safe_meta_fetcher: Callable that returns ``{owners, threshold}``
        for a Safe (only invoked if ``code_fetcher`` returns non-empty).
    :return: A ``SafeInfo`` describing the sender (cached on the way out).
    """
    cached = cache.get(sender)
    if cached is not None:
        return cached
    checksum_sender = to_checksum_address(sender)
    code = code_fetcher(checksum_sender)
    if not code:
        info = SafeInfo(is_contract=False)
        cache.put(sender, info)
        return info
    meta = safe_meta_fetcher(checksum_sender)
    owners_iterable = meta.get("owners") or []
    owners = {to_checksum_address(o) for o in owners_iterable}
    threshold = int(meta.get("threshold") or 0)
    info = SafeInfo(is_contract=True, owners=owners, threshold=threshold)
    cache.put(sender, info)
    return info


def verify_signature(  # noqa: D417
    *,
    sender: str,
    signature: str,
    ipfs_hash_bytes: bytes,
    delivery_rate: int,
    nonce: int,
    marketplace_address: str,
    mech_address: str,
    payment_type: bytes,
    chain_id: int,
    cache: SafeOwnerCache,
    code_fetcher: CodeFetcher,
    safe_meta_fetcher: SafeMetaFetcher,
    is_valid_signature_fetcher: Optional[IsValidSignatureFetcher] = None,
) -> VerifyResult:
    """Verify a trader-posted signature end-to-end.

    Fast path:
      1. Recompute the request_id locally (no RPC).
      2. Look up the sender's cache entry (single RPC on cold miss for
         ``eth_getCode`` + ``getOwners`` + ``getThreshold``).
      3. EOA sender → plain ecrecover against ``sender``.
      4. Safe with threshold=1 → ecrecover against Safe-wrapped hash, check
         the recovered address is in the cached owner set.
      5. Safe with threshold>1 → fall back to ``isValidSignature`` RPC for
         this request (rare, ok to eat the latency).

    Any exception on the crypto path (bad hex, wrong length, key recovery
    failure) is treated as a verification failure. This is the security
    boundary: a permissive fallback would defeat the whole point.

    :return: A ``VerifyResult``. On ``ok=False`` ``reason`` is safe to log
        and to surface in the HTTP rejection body.
    """
    try:
        request_id_bytes = derive_request_id_bytes(
            marketplace_address=marketplace_address,
            mech_address=mech_address,
            requester=sender,
            data=ipfs_hash_bytes,
            delivery_rate=delivery_rate,
            payment_type=payment_type,
            nonce=nonce,
            chain_id=chain_id,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return VerifyResult(ok=False, reason=f"request_id derivation failed: {exc}")

    try:
        info = _lookup_or_fetch_safe_info(
            sender, cache, code_fetcher, safe_meta_fetcher
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return VerifyResult(
            ok=False,
            reason=f"sender metadata lookup failed: {exc}",
            request_id_bytes=request_id_bytes,
        )

    # EOA path: signature must recover to sender exactly.
    if not info.is_contract:
        try:
            recovered = ecrecover_address(request_id_bytes, signature)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return VerifyResult(
                ok=False,
                reason=f"ecrecover failed: {exc}",
                request_id_bytes=request_id_bytes,
            )
        if recovered.lower() != to_checksum_address(sender).lower():
            return VerifyResult(
                ok=False,
                reason="signature does not recover to EOA sender",
                request_id_bytes=request_id_bytes,
            )
        return VerifyResult(ok=True, request_id_bytes=request_id_bytes)

    # Multi-owner Safe: bail out to RPC. Reconstructing checkSignatures
    # locally for arbitrary threshold + signature packing is not worth the
    # complexity given how rare multi-owner requester Safes are.
    if info.threshold > 1:
        if is_valid_signature_fetcher is None:
            return VerifyResult(
                ok=False,
                reason=(
                    "multi-owner Safe requires isValidSignature RPC "
                    "fallback but none was provided"
                ),
                request_id_bytes=request_id_bytes,
            )
        try:
            valid = is_valid_signature_fetcher(
                to_checksum_address(sender), request_id_bytes, signature
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return VerifyResult(
                ok=False,
                reason=f"isValidSignature RPC failed: {exc}",
                request_id_bytes=request_id_bytes,
            )
        if not valid:
            return VerifyResult(
                ok=False,
                reason="isValidSignature returned invalid",
                request_id_bytes=request_id_bytes,
            )
        return VerifyResult(ok=True, request_id_bytes=request_id_bytes)

    # Threshold=1 Safe: ecrecover on the Safe-wrapped hash and check
    # the recovered address is an owner.
    try:
        wrapped = compute_safe_message_hash(
            request_id_bytes=request_id_bytes,
            safe_address=sender,
            chain_id=chain_id,
        )
        recovered = ecrecover_address(wrapped, signature)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return VerifyResult(
            ok=False,
            reason=f"ecrecover failed: {exc}",
            request_id_bytes=request_id_bytes,
        )
    if to_checksum_address(recovered) not in info.owners:
        return VerifyResult(
            ok=False,
            reason="signature does not recover to a Safe owner",
            request_id_bytes=request_id_bytes,
        )
    return VerifyResult(ok=True, request_id_bytes=request_id_bytes)
