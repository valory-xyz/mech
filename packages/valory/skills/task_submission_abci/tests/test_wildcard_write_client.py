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

"""Unit tests for the wildcard write client helpers.

The server-side mirror lives at ``wildcard/server/src/mech_sig.py`` in
predict-api#162; both sides must agree on the canonical-JSON encoding
bit-for-bit. These tests pin the contract:

* ``canonical_json_bytes`` sorts keys at every level, uses compact
  separators, and keeps non-ASCII text as raw UTF-8 (so a prompt with
  emoji or accented characters hashes the same on both sides).
* ``compute_batch_hash`` is order-sensitive: rearranging events in the
  array yields a different hash (the wildcard server iterates left-to-
  right, so the mech can't shuffle the order after signing).
* ``build_typed_data`` produces the exact shape the server expects:
  primary type ``MechEventBatch``, domain pinned to
  ``Olas Mech Events`` / ``1`` / chainId / verifyingContract, message
  carrying the Safe address and the batch hash.
"""

import json

from packages.valory.skills.task_submission_abci.wildcard_write_client import (
    EVENTS_DOMAIN_NAME,
    EVENTS_DOMAIN_VERSION,
    EVENTS_PRIMARY_TYPE,
    build_typed_data,
    canonical_json_bytes,
    compute_batch_hash,
)


# ---------------------------------------------------------------------------
# canonical_json_bytes
# ---------------------------------------------------------------------------


class TestCanonicalJsonBytes:
    def test_keys_sorted_top_level(self) -> None:
        """Same dict, different declaration order → same encoded bytes."""
        a = canonical_json_bytes({"b": 1, "a": 2})
        b = canonical_json_bytes({"a": 2, "b": 1})
        assert a == b
        assert a == b'{"a":2,"b":1}'

    def test_keys_sorted_recursively(self) -> None:
        """Nested dicts also sort; ``zzz`` inside an outer ``inner`` still
        precedes ``aaa`` only by the inner-level alphabetical rule."""
        encoded = canonical_json_bytes({"inner": {"zzz": 1, "aaa": 2}})
        assert encoded == b'{"inner":{"aaa":2,"zzz":1}}'

    def test_compact_separators_no_whitespace(self) -> None:
        """Defaults would inject spaces after commas and colons; the
        canonical encoder strips them so the byte string is minimal."""
        encoded = canonical_json_bytes({"a": [1, 2, 3]})
        assert b" " not in encoded
        assert encoded == b'{"a":[1,2,3]}'

    def test_non_ascii_preserved_as_utf8(self) -> None:
        """``ensure_ascii=False`` keeps the bytes the operator wrote rather
        than escaping to ``\\uXXXX``. A prompt with non-ASCII characters
        must hash to the same value on both sides."""
        encoded = canonical_json_bytes({"prompt": "café ☃"})
        assert "café".encode("utf-8") in encoded
        # Reverse: confirm the explicit \u escape did NOT land in the
        # output (which is what the json default would have produced).
        assert b"\\u" not in encoded

    def test_round_trips_through_json(self) -> None:
        """The canonical bytes are valid JSON; parsing them back yields
        the same logical dict (modulo key ordering, which is what we
        normalised)."""
        original = {"b": 1, "a": {"x": "y", "n": 2}}
        encoded = canonical_json_bytes(original)
        decoded = json.loads(encoded.decode("utf-8"))
        assert decoded == original


# ---------------------------------------------------------------------------
# compute_batch_hash
# ---------------------------------------------------------------------------


class TestComputeBatchHash:
    def test_hash_is_0x_prefixed_32_byte_hex(self) -> None:
        h = compute_batch_hash([{"a": 1}])
        assert h.startswith("0x")
        assert len(h) == 2 + 64  # 0x + 32 bytes hex

    def test_hash_stable_across_dict_key_order(self) -> None:
        """The whole point of canonical encoding: identical content, different
        declaration order → same hash. Without this, a serialization that
        accidentally reordered keys would invalidate the signature on the
        wildcard side."""
        h1 = compute_batch_hash([{"x": 1, "y": 2}])
        h2 = compute_batch_hash([{"y": 2, "x": 1}])
        assert h1 == h2

    def test_hash_changes_if_event_payload_changes(self) -> None:
        """Single-character flip in the payload moves the hash entirely;
        this is the property that makes the signed batch_hash a real
        tamper detector on the wildcard side."""
        h1 = compute_batch_hash([{"a": "real result"}])
        h2 = compute_batch_hash([{"a": "spoofed result"}])
        assert h1 != h2

    def test_hash_is_order_sensitive_across_events(self) -> None:
        """Two events in different order → different hashes. The wildcard
        server iterates left-to-right when inserting, so the mech can't
        shuffle and still get a match."""
        events_a = [{"r": "first"}, {"r": "second"}]
        events_b = [{"r": "second"}, {"r": "first"}]
        assert compute_batch_hash(events_a) != compute_batch_hash(events_b)


# ---------------------------------------------------------------------------
# build_typed_data
# ---------------------------------------------------------------------------


class TestBuildTypedData:
    def test_shape_matches_server_expectation(self) -> None:
        """Mirror the ``SignedMechEventBatch`` Pydantic schema on the server:
        primary type ``MechEventBatch``, domain with the four required
        fields, message with ``mech_service_multisig`` + ``batch_hash``."""
        td = build_typed_data(
            mech_service_multisig="0x" + "ab" * 20,
            chain_id=100,
            verifying_contract="0x" + "cd" * 20,
            batch_hash_hex="0x" + "ef" * 32,
        )
        assert td["primaryType"] == EVENTS_PRIMARY_TYPE
        assert td["domain"] == {
            "name": EVENTS_DOMAIN_NAME,
            "version": EVENTS_DOMAIN_VERSION,
            "chainId": 100,
            "verifyingContract": "0x" + "cd" * 20,
        }
        assert td["message"] == {
            "mech_service_multisig": "0x" + "ab" * 20,
            "batch_hash": "0x" + "ef" * 32,
        }
        # The MechEventBatch type schema must list exactly these two fields
        # in this order; the server's ecrecover hashes the typed-data
        # against the same schema and any mismatch yields a different
        # recovered address.
        assert td["types"][EVENTS_PRIMARY_TYPE] == [
            {"name": "mech_service_multisig", "type": "address"},
            {"name": "batch_hash", "type": "bytes32"},
        ]
