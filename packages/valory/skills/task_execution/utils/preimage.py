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

"""Durable preimage buffer for the off-chain delivery path.

In off-chain mode the mech never publishes the request/response to IPFS (the
response stays private behind a locally-computed CID), so there is no public
record an operator can use to prove what was requested and what was delivered.
This module keeps a short-lived, operator-private audit copy — the "preimage" —
of each off-chain ``(request, response)`` pair in the persistent
``valory/kv_store`` connection, keyed by ``request_id``, and prunes entries past
an operator-configurable retention window via a background sweeper.

The functions here are deliberately pure: they take the skill's ``shared_state``
dict and a clock value, so the buffer/sweep bookkeeping is unit-testable without
the AEA runtime. The async kv_store I/O (CREATE_OR_UPDATE / LIST / DELETE) lives
in the behaviour, and its replies are processed by the handler.

Stored value (a JSON string, under key ``f"{prefix}{request_id}"``)::

    {
      "request_id": str,
      "request": str | None,       # the requester's signed request payload
      "response": str | None,      # the delivered response (or failure reason)
      "response_cid": str | None,  # local CID of the response, when delivered
      "accepted_at": int | None,   # epoch seconds the request was accepted
      "settled_at": int | None,    # epoch seconds it was delivered / rejected
      "settlement_status": "processing" | "delivered" | "rejected",
    }
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# shared_state keys (owned here; initialised by the behaviour's setup()).
PREIMAGE_RECORDS = "preimage_records"  # Dict[str, Dict[str, Any]] desired state
PREIMAGE_WRITE_QUEUE = "preimage_write_queue"  # List[str] request_ids to flush
PREIMAGE_DELETE_QUEUE = "preimage_delete_queue"  # List[str] kv keys to delete
PREIMAGE_KV_IN_FLIGHT = "preimage_kv_in_flight"  # bool — one kv op at a time
PREIMAGE_INFLIGHT_WRITE = "preimage_inflight_write"  # Optional[str] request_id
PREIMAGE_LAST_SWEEP = "preimage_last_sweep"  # float epoch seconds

# settlement_status values.
STATUS_PROCESSING = "processing"
STATUS_DELIVERED = "delivered"
STATUS_REJECTED = "rejected"

TERMINAL_STATUSES = (STATUS_DELIVERED, STATUS_REJECTED)


def init_shared_state(shared_state: Dict[str, Any]) -> None:
    """Initialise the preimage shared-state keys in place (idempotent).

    :param shared_state: the skill's shared state dict.
    """
    shared_state.setdefault(PREIMAGE_RECORDS, {})
    shared_state.setdefault(PREIMAGE_WRITE_QUEUE, [])
    shared_state.setdefault(PREIMAGE_DELETE_QUEUE, [])
    shared_state.setdefault(PREIMAGE_KV_IN_FLIGHT, False)
    shared_state.setdefault(PREIMAGE_INFLIGHT_WRITE, None)
    shared_state.setdefault(PREIMAGE_LAST_SWEEP, 0.0)


def preimage_key(prefix: str, request_id: str) -> str:
    """Return the kv_store key for a request's preimage.

    :param prefix: the operator-configured key namespace.
    :param request_id: the off-chain request id.
    :return: the namespaced kv_store key.
    """
    return f"{prefix}{request_id}"


def _enqueue_write(shared_state: Dict[str, Any], request_id: str) -> None:
    """Append a request id to the write queue, de-duplicating.

    :param shared_state: the skill's shared state dict.
    :param request_id: the off-chain request id to flush.
    """
    queue: List[str] = shared_state.setdefault(PREIMAGE_WRITE_QUEUE, [])
    if request_id not in queue:
        queue.append(request_id)


def record_accept(
    shared_state: Dict[str, Any], request_id: str, request_payload: str, now: float
) -> None:
    """Buffer a freshly accepted off-chain request (status=processing).

    :param shared_state: the skill's shared state dict.
    :param request_id: the off-chain request id.
    :param request_payload: the requester's signed request payload.
    :param now: the current epoch time in seconds.
    """
    records = shared_state.setdefault(PREIMAGE_RECORDS, {})
    records[request_id] = {
        "request_id": request_id,
        "request": request_payload,
        "response": None,
        "response_cid": None,
        "accepted_at": int(now),
        "settled_at": None,
        "settlement_status": STATUS_PROCESSING,
    }
    _enqueue_write(shared_state, request_id)


def record_settlement(
    shared_state: Dict[str, Any],
    request_id: str,
    response_payload: str,
    response_cid: Optional[str],
    status: str,
    now: float,
) -> None:
    """Buffer the settled outcome for a request (status=delivered|rejected).

    Tolerates a missing accept record (e.g. the agent restarted between accept
    and settle): a minimal record is created so the response is still captured.

    :param shared_state: the skill's shared state dict.
    :param request_id: the off-chain request id.
    :param response_payload: the delivered response, or the failure reason.
    :param response_cid: the local CID of the response, when delivered.
    :param status: one of STATUS_DELIVERED / STATUS_REJECTED.
    :param now: the current epoch time in seconds.
    """
    records = shared_state.setdefault(PREIMAGE_RECORDS, {})
    record = records.get(request_id) or {
        "request_id": request_id,
        "request": None,
        "accepted_at": None,
    }
    record.update(
        {
            "response": response_payload,
            "response_cid": response_cid,
            "settled_at": int(now),
            "settlement_status": status,
        }
    )
    records[request_id] = record
    _enqueue_write(shared_state, request_id)


def expired_keys(
    list_data: Dict[str, str], now: float, retention_seconds: int
) -> List[str]:
    """Return the kv keys whose preimage is older than the retention window.

    Age is measured from ``settled_at`` when present, else ``accepted_at``.
    Entries with neither timestamp, or whose value can't be parsed, are left
    untouched — the sweeper must never delete a row it doesn't understand.

    :param list_data: the LIST_RESPONSE key -> JSON value map.
    :param now: the current epoch time in seconds.
    :param retention_seconds: the retention window, in seconds.
    :return: the list of keys to delete.
    """
    expired: List[str] = []
    for key, raw in list_data.items():
        try:
            record = json.loads(raw)
            stamp = record.get("settled_at") or record.get("accepted_at")
        except (ValueError, TypeError, AttributeError):
            continue
        if stamp is None:
            continue
        if now - float(stamp) > retention_seconds:
            expired.append(key)
    return expired


def serialize(record: Dict[str, Any]) -> str:
    """Serialize a preimage record to its kv_store string value.

    :param record: the preimage record.
    :return: the JSON string stored in kv_store.
    """
    return json.dumps(record, sort_keys=True)
