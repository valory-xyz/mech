# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2026 Valory AG
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

"""EIP-712 typed-data builder and batch-hash helper for the wildcard
``POST /mech/events`` write client.

The server side at ``wildcard/server/src/mech_sig.py`` (predict-api#162)
recomputes the same ``batch_hash`` from the parsed events array and
compares against the signed value. Both sides must agree on the
canonical-JSON encoding bit-for-bit:

* keys sorted at every level
* compact separators (no spaces)
* ``ensure_ascii`` off so non-ASCII text in prompts hashes as the bytes
  the operator wrote rather than as ``\\uXXXX`` escapes

Any drift between this module and the server's encoder breaks every
batch. Keep the contract narrow.
"""

import json
from typing import Any, Dict, List

from eth_utils import keccak


# The exact name/version pair that the server's allowlist binds via the
# EIP-712 domain. Treat as a versioned constant; bumping the version
# requires a coordinated change on both sides.
EVENTS_DOMAIN_NAME = "Olas Mech Events"
EVENTS_DOMAIN_VERSION = "1"
EVENTS_PRIMARY_TYPE = "MechEventBatch"


def canonical_json_bytes(value: Any) -> bytes:
    """Encode ``value`` as a deterministic UTF-8 byte string.

    The mirror of ``wildcard/server/src/mech_sig.py::canonical_json_bytes``;
    any divergence here breaks the batch-hash match on the server. Inputs
    must be JSON-native (no ``Decimal``, ``datetime``, etc.) — the caller
    is responsible for normalising before this function sees them.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def compute_batch_hash(events: List[Dict[str, Any]]) -> str:
    """Return ``"0x" + keccak256(canonical_json(events)).hex()``.

    The result is what goes into the signed typed-data's
    ``message.batch_hash`` field. The server recomputes from the parsed
    wire payload and rejects on mismatch.
    """
    return "0x" + keccak(canonical_json_bytes(events)).hex()


def build_typed_data(
    *,
    mech_service_multisig: str,
    chain_id: int,
    verifying_contract: str,
    batch_hash_hex: str,
) -> Dict[str, Any]:
    """Build the EIP-712 typed-data dict for one settled batch.

    The shape mirrors the server's :class:`SignedMechEventBatch` expectation:
    a domain pinned to (Olas Mech Events / 1 / chainId / verifyingContract)
    and a single ``MechEventBatch`` primary type carrying the Safe address
    and the batch hash.

    Returned as a plain dict so callers can stamp it into the POST body
    verbatim; the AEA signing helper hashes it back into bytes via the
    existing ``packages.valory.contracts.gnosis_safe.encode.encode_typed_data``
    path.
    """
    return {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            EVENTS_PRIMARY_TYPE: [
                {"name": "mech_service_multisig", "type": "address"},
                {"name": "batch_hash", "type": "bytes32"},
            ],
        },
        "primaryType": EVENTS_PRIMARY_TYPE,
        "domain": {
            "name": EVENTS_DOMAIN_NAME,
            "version": EVENTS_DOMAIN_VERSION,
            "chainId": chain_id,
            "verifyingContract": verifying_contract,
        },
        "message": {
            "mech_service_multisig": mech_service_multisig,
            "batch_hash": batch_hash_hex,
        },
    }
