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
response stays out of the public IPFS layer behind a locally-computed CID), so
there is no public record an operator can use to prove what was requested and
what was delivered. This module keeps a short-lived audit copy — the
"preimage" — of each off-chain ``(request, response)`` pair in the persistent
``valory/kv_store`` connection, keyed by ``request_id``, and prunes entries past
an operator-configurable retention window via a background sweeper.

**Retention is storage-bound, not cryptographic.** The sweeper's DELETE removes
the row from kv_store queries, so the on-disk footprint plateaus at "peak
retention-window worth of preimages" instead of growing without bound — that
is the property this feature exists to provide. It does NOT zero the bytes
on disk: SQLite's default delete marks pages as free-for-reuse without
overwriting them, and WAL frames retain copies until checkpointed. An
operator with file-level access to the .db / .db-wal files can recover
"deleted" preimage contents until later writes happen to overwrite the
freed pages. Treat retention as "stops appearing in queries after N
seconds," not "wiped from disk after N seconds." For an actual privacy
property, enable ``PRAGMA secure_delete=ON`` on the kv_store connection
(separate concern — that's a kv-store package change) or rely on
volume-level encryption at rest.

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
import logging
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

# shared_state keys (owned here; initialised by the behaviour's setup()).
PREIMAGE_RECORDS = "preimage_records"  # Dict[str, Dict[str, Any]] desired state
PREIMAGE_WRITE_QUEUE = "preimage_write_queue"  # List[str] request_ids to flush
PREIMAGE_DELETE_QUEUE = "preimage_delete_queue"  # List[str] kv keys to delete
PREIMAGE_KV_IN_FLIGHT = "preimage_kv_in_flight"  # bool — one kv op at a time
PREIMAGE_INFLIGHT_WRITE = "preimage_inflight_write"  # Optional[str] request_id
PREIMAGE_INFLIGHT_SENT_AT = "preimage_inflight_sent_at"  # Optional[float] epoch
# What kind of kv op is in flight: "list" / "delete" / "write" / None. Needed
# so the handler can route ERROR replies to the right arm — DELETE and LIST
# both have PREIMAGE_INFLIGHT_WRITE=None, so without this they're
# indistinguishable. Note: LIST has a retry counter (PREIMAGE_LIST_ATTEMPTS),
# WRITE has a per-id counter (PREIMAGE_WRITE_ATTEMPTS), DELETE has neither —
# it only uses INFLIGHT_OP for observability (logging the dropped batch
# size) since the path self-heals via the next sweep.
PREIMAGE_INFLIGHT_OP = "preimage_inflight_op"  # Optional[str]
OP_LIST = "list"
OP_DELETE = "delete"
OP_WRITE = "write"
# Size of the in-flight DELETE batch. The handler's ERROR arm logs it when a
# DELETE fails so a degraded kv_store is visible in operator logs even when
# LIST is still succeeding.
PREIMAGE_INFLIGHT_DELETE_COUNT = "preimage_inflight_delete_count"  # int
# Dialogue reference (tuple-of-strs) of the in-flight kv op. The handler
# compares incoming replies to this value and ignores any reply that doesn't
# match the current in-flight dialogue — closes the late-reply
# misattribution path where a slow reply arrives after the watchdog gave up
# and started the next op (the late reply would otherwise clobber the new
# op's counters and clear its in-flight gate).
PREIMAGE_INFLIGHT_DIALOGUE = "preimage_inflight_dialogue"  # Optional[Tuple[str, str]]
PREIMAGE_LAST_SWEEP = "preimage_last_sweep"  # float epoch seconds
# Non-empty when a multi-page sweep is mid-flight; the next _send_kv_list
# tick passes this back as the LIST cursor. Cleared when a LIST_RESPONSE
# returns an empty next_cursor (final page) — only then is PREIMAGE_LAST_SWEEP
# stamped, so the next sweep_interval starts from a fully-walked namespace.
PREIMAGE_LIST_CURSOR = "preimage_list_cursor"  # Optional[str]
# Per-request_id retry counter for kv_store CREATE_OR_UPDATE failures. Each
# ERROR increments; when the count reaches PREIMAGE_MAX_WRITE_ATTEMPTS the
# record is dropped + WARN'd so a persistently unhealthy kv_store can't
# hot-loop the agent retrying the same record forever.
PREIMAGE_WRITE_ATTEMPTS = "preimage_write_attempts"  # Dict[str, int]
# Consecutive LIST ERROR counter (single int, no per-id slot — LIST has no id).
# Each LIST ERROR increments; at the cap the cursor is cleared, LAST_SWEEP is
# stamped, and a WARN is emitted so the next sweep_interval is the natural
# backoff instead of a per-tick hot loop. Reset to 0 on any LIST_RESPONSE.
PREIMAGE_LIST_ATTEMPTS = "preimage_list_attempts"  # int

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
    shared_state.setdefault(PREIMAGE_INFLIGHT_SENT_AT, None)
    shared_state.setdefault(PREIMAGE_LAST_SWEEP, 0.0)
    shared_state.setdefault(PREIMAGE_LIST_CURSOR, None)
    shared_state.setdefault(PREIMAGE_WRITE_ATTEMPTS, {})
    shared_state.setdefault(PREIMAGE_LIST_ATTEMPTS, 0)
    shared_state.setdefault(PREIMAGE_INFLIGHT_OP, None)
    shared_state.setdefault(PREIMAGE_INFLIGHT_DELETE_COUNT, 0)
    shared_state.setdefault(PREIMAGE_INFLIGHT_DIALOGUE, None)


def preimage_key(prefix: str, request_id: str) -> str:
    """Return the kv_store key for a request's preimage.

    :param prefix: the operator-configured key namespace.
    :param request_id: the off-chain request id.
    :return: the namespaced kv_store key.
    """
    return f"{prefix}{request_id}"


def enqueue_write(shared_state: Dict[str, Any], request_id: str) -> None:
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
    enqueue_write(shared_state, request_id)


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
    # Narrow the contract: settling with STATUS_PROCESSING would write a
    # record the SUCCESS handler never pops (it only pops TERMINAL_STATUSES),
    # leaking the id in PREIMAGE_RECORDS until restart. Both call sites pass
    # a terminal status today; the assertion makes a future regression that
    # forwards a non-terminal status unrepresentable rather than a silent
    # leak.
    assert status in TERMINAL_STATUSES, (
        f"record_settlement called with non-terminal status {status!r}; "
        f"expected one of {TERMINAL_STATUSES}"
    )
    records = shared_state.setdefault(PREIMAGE_RECORDS, {})
    existing = records.get(request_id)
    if existing is not None and existing.get("settlement_status") in TERMINAL_STATUSES:
        # In-memory-only guard: catches a double-settle that happens BEFORE
        # the terminal write is flushed and popped (the record is still in
        # PREIMAGE_RECORDS with a terminal status). Without this we'd flow
        # into the update block below, mutate the already-terminal record,
        # and re-enqueue a redundant write of the same data.
        #
        # NOT covered: the post-pop case, where the terminal write already
        # flushed and the SUCCESS handler popped the record. A second
        # settle there sees existing=None, takes the fallback at line 203
        # (request=None, accepted_at=None), and overwrites the good kv row
        # with a stripped one. We leave that intentionally unguarded —
        # tracking settled ids in a never-cleared set would leak memory,
        # and there's no plausible code path that calls record_settlement
        # twice for the same id after the first one was already persisted.
        # The executor returns immediately after settling and the task slot
        # is single-threaded.
        _logger.warning(
            "Double-settle ignored for request_id=%s (already %s).",
            request_id,
            existing.get("settlement_status"),
        )
        return
    record = existing or {
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
    enqueue_write(shared_state, request_id)


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
    skipped: List[str] = []
    for key, raw in list_data.items():
        try:
            record = json.loads(raw)
            stamp = record.get("settled_at") or record.get("accepted_at")
            if stamp is None:
                skipped.append(key)
                continue
            age = now - float(stamp)
        except (ValueError, TypeError, AttributeError):
            # ValueError / TypeError from float() catch a row with a
            # JSON-parseable but non-numeric timestamp ({"settled_at":
            # "oops"}) — without this guard the exception escapes the
            # sweeper and crashes the handler on every cycle (poison-pill
            # row), permanently stalling retention pruning. Not reachable
            # from current writers (they always store int(now)), but the
            # docstring promises rows we don't understand are left alone.
            skipped.append(key)
            continue
        if age > retention_seconds:
            expired.append(key)
    if skipped:
        # Silently treating unparseable / timestamp-less rows as
        # "leave alone" is the safe choice (don't delete what we don't
        # understand), but it can mask a schema drift where every row
        # starts skipping and retention pruning quietly stops. Surface
        # the count + a sample key so a degraded namespace is visible
        # in operator logs at near-zero cost when the count is zero.
        _logger.warning(
            "Preimage sweep: %d entr%s skipped (unparseable or "
            "timestamp-less). First skipped key: %r",
            len(skipped),
            "y" if len(skipped) == 1 else "ies",
            skipped[0],
        )
    return expired


def serialize(record: Dict[str, Any]) -> str:
    """Serialize a preimage record to its kv_store string value.

    :param record: the preimage record.
    :return: the JSON string stored in kv_store.
    """
    return json.dumps(record, sort_keys=True)
