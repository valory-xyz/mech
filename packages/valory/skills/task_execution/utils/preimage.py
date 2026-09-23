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
      # Replay inputs, captured on the delivered path so a restart can
      # re-run the follow-up steps (on-chain settlement, requester fetch,
      # predict-api row) from disk alone:
      "done_task": dict | None,          # the consensus-ready done_task
      "predict_api_event": dict | None,  # the built predict-api event
      # Per-step stamps. ``settled_tx_hash`` + ``posted_at`` (when the
      # predict-api write is on) make a delivered row *complete*;
      # ``fetched_at`` is visibility only; ``abandoned_at`` marks a
      # deliver the settlement retry cap gave up on.
      "settled_tx_hash": str | None,
      "fetched_at": int | None,
      "posted_at": int | None,
      "abandoned_at": int | None,
    }

Retention treats a row as *complete* (24h retention) when it is rejected,
never answered (``processing``), abandoned, or delivered with every follow-up
stamped. A delivered row missing a follow-up is *incomplete*: the sweeper
keeps it up to ``preimage_incomplete_cap_seconds`` (default 7 days) so the
drainer (``hydrate``, run on every LIST page of the sweep) can replay the
missing step, and only deletes it past the cap, with a WARNING.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

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
# True when the operator enabled preimage retention. Stamps and replay
# bookkeeping are no-ops when this is False, so the settlement skill (which
# has no copy of the retention flag) can call ``record_stamp`` unguarded
# without leaking pending stamps on deployments that opted out.
PREIMAGE_RETENTION_ACTIVE = "preimage_retention_active"  # bool
# Stamps that arrived for a record not held in memory (typically after a
# restart, before the sweep hydrated it). Applied by ``hydrate`` when the
# row is loaded from the kv_store. Dict[request_id, Dict[field, value]].
PREIMAGE_PENDING_STAMPS = "preimage_pending_stamps"
# Epoch seconds each parked entry was created, for ``prune_parked_stamps``.
PREIMAGE_PENDING_STAMPS_AT = "preimage_pending_stamps_at"  # Dict[str, float]
# True when the settlement skill will actually POST delivered events to the
# predict-api (``use_offchain`` and a non-empty events URL on that skill).
# Published by task_submission_abci at setup; read here to decide whether
# ``posted_at`` is required for a delivered row to count as complete.
PREDICT_API_WRITE_CONFIGURED = "predict_api_write_configured"  # bool
# Row counters accumulated across the pages of one sweep walk; the handler
# publishes them to the gauges when the final page arrives.
PREIMAGE_SWEEP_ROW_COUNT = "preimage_sweep_row_count"  # int
PREIMAGE_SWEEP_INCOMPLETE_COUNT = "preimage_sweep_incomplete_count"  # int

# Record field names shared by the writers, the stamps and the drainer.
FIELD_DONE_TASK = "done_task"
FIELD_PREDICT_API_EVENT = "predict_api_event"
FIELD_SETTLED_TX_HASH = "settled_tx_hash"
FIELD_FETCHED_AT = "fetched_at"
FIELD_POSTED_AT = "posted_at"
FIELD_ABANDONED_AT = "abandoned_at"
STAMP_FIELDS = (
    FIELD_SETTLED_TX_HASH,
    FIELD_FETCHED_AT,
    FIELD_POSTED_AT,
    FIELD_ABANDONED_AT,
)
# Stamps worth remembering for a row that is not in memory. ``fetched_at``
# is visibility only, so a fetch of an already-popped row is dropped rather
# than parked forever in ``PREIMAGE_PENDING_STAMPS``.
DURABLE_STAMP_FIELDS = (FIELD_SETTLED_TX_HASH, FIELD_POSTED_AT, FIELD_ABANDONED_AT)

# settlement_status values.
STATUS_PROCESSING = "processing"
STATUS_DELIVERED = "delivered"
STATUS_REJECTED = "rejected"

TERMINAL_STATUSES = (STATUS_DELIVERED, STATUS_REJECTED)


def init_shared_state(
    shared_state: Dict[str, Any], retention_enabled: Optional[bool] = None
) -> None:
    """Initialise the preimage shared-state keys in place (idempotent).

    :param shared_state: the skill's shared state dict.
    :param retention_enabled: when given, records whether the operator
        turned preimage retention on (``PREIMAGE_RETENTION_ACTIVE``).
        ``None`` leaves the existing flag alone (default False).
    """
    shared_state.setdefault(PREIMAGE_RETENTION_ACTIVE, False)
    if retention_enabled is not None:
        shared_state[PREIMAGE_RETENTION_ACTIVE] = bool(retention_enabled)
    shared_state.setdefault(PREIMAGE_PENDING_STAMPS, {})
    shared_state.setdefault(PREIMAGE_PENDING_STAMPS_AT, {})
    shared_state.setdefault(PREIMAGE_SWEEP_ROW_COUNT, 0)
    shared_state.setdefault(PREIMAGE_SWEEP_INCOMPLETE_COUNT, 0)
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
        **_empty_replay_fields(),
    }
    enqueue_write(shared_state, request_id)


def _empty_replay_fields() -> Dict[str, Any]:
    """Return the replay-input and stamp fields, all unset.

    :return: a fresh dict of every replay / stamp field set to ``None``.
    """
    return {
        FIELD_DONE_TASK: None,
        FIELD_PREDICT_API_EVENT: None,
        FIELD_SETTLED_TX_HASH: None,
        FIELD_FETCHED_AT: None,
        FIELD_POSTED_AT: None,
        FIELD_ABANDONED_AT: None,
    }


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
        **_empty_replay_fields(),
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


def record_replay_inputs(
    shared_state: Dict[str, Any],
    request_id: str,
    done_task: Optional[Dict[str, Any]],
    predict_api_event: Optional[Dict[str, Any]],
) -> bool:
    """Attach the replay inputs to a delivered record.

    Called from the done path right after the consensus-ready ``done_task``
    (and, when the predict-api write is on, its event) is built, so the
    record flushed to the kv_store carries everything a restart needs to
    re-run settlement and the predict-api POST. Both values must be JSON
    serialisable; the caller passes JSON round-tripped copies so later
    in-place mutation of the live dicts cannot leak into the record.

    :param shared_state: the skill's shared state dict.
    :param request_id: the off-chain request id.
    :param done_task: the done_task dict, or ``None`` to leave it unset.
    :param predict_api_event: the built event, or ``None`` to leave it unset.
    :return: ``True`` when a delivered record was updated, ``False`` when
        there was no delivered record to attach to (nothing is written).
    """
    records = shared_state.setdefault(PREIMAGE_RECORDS, {})
    record = records.get(request_id)
    if record is None or record.get("settlement_status") != STATUS_DELIVERED:
        _logger.warning(
            "record_replay_inputs: no delivered record for request_id=%s; "
            "replay inputs dropped.",
            request_id,
        )
        return False
    if done_task is not None:
        record[FIELD_DONE_TASK] = done_task
    if predict_api_event is not None:
        record[FIELD_PREDICT_API_EVENT] = predict_api_event
    enqueue_write(shared_state, request_id)
    return True


def record_stamp(
    shared_state: Dict[str, Any],
    request_id: str,
    now: Optional[float] = None,
    **stamps: Any,
) -> None:
    """Stamp one or more follow-up steps on a record.

    No-op unless retention is active (``PREIMAGE_RETENTION_ACTIVE``). When
    the record is held in memory the stamp is applied and a write queued.
    When it is not (the process restarted before the sweep hydrated the
    row), durable stamps are parked in ``PREIMAGE_PENDING_STAMPS`` and
    applied by :func:`hydrate`; ``fetched_at`` is dropped instead, because
    it is visibility only and a fetch of an already-popped row would
    otherwise park an entry forever. Parked entries are timestamped so
    :func:`prune_parked_stamps` can drop the ones no row ever claims.

    Stamps are write-once: an existing non-``None`` value is kept, so a
    round that re-enters (NO_MAJORITY / timeout) cannot move a stamp.

    :param shared_state: the skill's shared state dict.
    :param request_id: the off-chain request id.
    :param now: the current epoch time in seconds (``time.time()`` when
        omitted); only used to timestamp a parked entry.
    :param stamps: ``field=value`` pairs; ``field`` must be in ``STAMP_FIELDS``.
    :raises ValueError: on a field name outside ``STAMP_FIELDS``.
    """
    unknown = set(stamps) - set(STAMP_FIELDS)
    if unknown:
        raise ValueError(f"record_stamp: unknown stamp field(s) {sorted(unknown)}")
    if not shared_state.get(PREIMAGE_RETENTION_ACTIVE):
        return
    record = shared_state.setdefault(PREIMAGE_RECORDS, {}).get(request_id)
    if record is None:
        durable = {k: v for k, v in stamps.items() if k in DURABLE_STAMP_FIELDS}
        if durable:
            pending = shared_state.setdefault(PREIMAGE_PENDING_STAMPS, {})
            slot = pending.setdefault(request_id, {})
            for field, value in durable.items():
                slot.setdefault(field, value)
            shared_state.setdefault(PREIMAGE_PENDING_STAMPS_AT, {}).setdefault(
                request_id, now if now is not None else time.time()
            )
        return
    if _apply_stamps(record, stamps):
        enqueue_write(shared_state, request_id)


def prune_parked_stamps(
    shared_state: Dict[str, Any], now: float, max_age_seconds: float
) -> int:
    """Drop parked stamps older than ``max_age_seconds``.

    A parked stamp is claimed by :func:`hydrate` when its row turns up in
    this agent's store. A stamp whose row never turns up (the id was not
    an off-chain request of this agent, or the row was already retired)
    would otherwise sit in memory for the life of the process. The
    callers already gate stamps to this agent's off-chain ids; this is
    the bound behind that gate.

    :param shared_state: the skill's shared state dict.
    :param now: the current epoch time in seconds.
    :param max_age_seconds: how long a parked entry may wait for its row.
    :return: the number of entries dropped.
    """
    pending: Dict[str, Dict[str, Any]] = shared_state.setdefault(
        PREIMAGE_PENDING_STAMPS, {}
    )
    pending_at: Dict[str, float] = shared_state.setdefault(
        PREIMAGE_PENDING_STAMPS_AT, {}
    )
    stale = [
        request_id
        for request_id in pending
        if now - pending_at.get(request_id, now) > max_age_seconds
    ]
    for request_id in stale:
        pending.pop(request_id, None)
        pending_at.pop(request_id, None)
    return len(stale)


def _apply_stamps(record: Dict[str, Any], stamps: Dict[str, Any]) -> bool:
    """Apply write-once stamps to ``record`` in place.

    :param record: the preimage record.
    :param stamps: ``field=value`` pairs.
    :return: ``True`` when at least one field changed.
    """
    changed = False
    for field, value in stamps.items():
        if value is None or record.get(field) is not None:
            continue
        record[field] = value
        changed = True
    return changed


def is_complete(record: Dict[str, Any], require_posted: bool) -> bool:
    """Return whether every follow-up step for ``record`` is done.

    Only a delivered row can be incomplete: rejected rows, rows that were
    never answered (``processing``), rows the settlement retry cap gave up
    on, and rows of an unknown status (older schema) have nothing left to
    replay. A delivered row is complete once the on-chain settlement is
    stamped and, when the predict-api write is on, the POST is stamped.

    :param record: the parsed preimage record.
    :param require_posted: whether the predict-api write is enabled, i.e.
        whether ``posted_at`` is required for completeness.
    :return: ``True`` when nothing is left to replay.
    """
    if record.get("settlement_status") != STATUS_DELIVERED:
        return True
    if record.get(FIELD_ABANDONED_AT) is not None:
        return True
    if record.get(FIELD_SETTLED_TX_HASH) is None:
        # Unsettled: replayable only with a done_task to re-queue. A row
        # without one (older schema, or replay inputs that failed to
        # attach) can never be settled from here, so it is retired on
        # the normal window rather than held to the cap.
        return record.get(FIELD_DONE_TASK) is None
    if not require_posted or record.get(FIELD_POSTED_AT) is not None:
        return True
    # Settled but not posted: replayable only with an event to send.
    return record.get(FIELD_PREDICT_API_EVENT) is None


def fetch_payload(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Rebuild the ``/fetch_offchain_info`` body for a settled record.

    Mirrors the two shapes the writers put in
    ``shared_state[OFFCHAIN_REQUEST_RESPONSES]``: the delivered shape (the
    committed response object verbatim under ``response`` plus the CID
    envelope) and the rejection shape (``status`` + ``reason``).

    :param record: the parsed preimage record.
    :return: the body to serve, or ``None`` when the record is not settled
        or its stored response cannot be decoded.
    """
    status = record.get("settlement_status")
    request_id = record.get("request_id")
    if request_id is None:
        return None
    if status == STATUS_REJECTED:
        return {
            "request_id": str(request_id),
            "status": "rejected",
            "reason": record.get("response") or "rejected",
        }
    if status != STATUS_DELIVERED:
        return None
    raw = record.get("response")
    if not isinstance(raw, str):
        return None
    try:
        response = json.loads(raw)
    except ValueError:
        return None
    return {
        "request_id": str(request_id),
        "status": "ok",
        "content_cid": record.get("response_cid"),
        "response": response,
    }


def _parse_record(raw: str) -> Optional[Dict[str, Any]]:
    """Parse one kv_store value into a record dict.

    :param raw: the JSON string stored in the kv_store.
    :return: the record, or ``None`` when it is not a JSON object.
    """
    try:
        record = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return record if isinstance(record, dict) else None


def _record_age(record: Dict[str, Any], now: float) -> Optional[float]:
    """Return the record's age in seconds, measured like the sweeper does.

    :param record: the parsed preimage record.
    :param now: the current epoch time in seconds.
    :return: the age, or ``None`` when the record carries no usable stamp.
    """
    stamp = record.get("settled_at")
    if stamp is None:
        stamp = record.get("accepted_at")
    if stamp is None:
        return None
    try:
        return now - float(stamp)
    except (ValueError, TypeError):
        return None


def hydrate(
    shared_state: Dict[str, Any],
    list_data: Dict[str, str],
    now: float,
    retention_seconds: int,
    incomplete_cap_seconds: int,
    require_posted: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load one LIST page back into memory and pick the rows to replay.

    This is the drainer's pure half. For every parseable row it:

    * counts it toward the sweep's row / incomplete-row counters;
    * returns a fetch payload for every settled row still inside the
      retention window (the caller restores the in-memory response map
      from these, so a requester polling after a restart is served);
    * for a delivered row that is incomplete, not yet in memory and not
      past the cap, applies any parked stamps, loads it into
      ``PREIMAGE_RECORDS`` and returns it for replay.

    Rows already held in memory belong to the live flow and are never
    replayed: their follow-ups are in progress and their stamps land on
    the in-memory copy. Complete rows are not loaded, only served.

    :param shared_state: the skill's shared state dict.
    :param list_data: the LIST_RESPONSE key -> JSON value map.
    :param now: the current epoch time in seconds.
    :param retention_seconds: the retention window for complete rows.
    :param incomplete_cap_seconds: the hard cap for incomplete rows.
    :param require_posted: whether ``posted_at`` counts toward completeness.
    :return: ``(fetch_payloads, replay_records)``.
    """
    records: Dict[str, Dict[str, Any]] = shared_state.setdefault(PREIMAGE_RECORDS, {})
    pending: Dict[str, Dict[str, Any]] = shared_state.setdefault(
        PREIMAGE_PENDING_STAMPS, {}
    )
    pending_at: Dict[str, float] = shared_state.setdefault(
        PREIMAGE_PENDING_STAMPS_AT, {}
    )
    fetch_payloads: List[Dict[str, Any]] = []
    replay: List[Dict[str, Any]] = []
    rows = 0
    incomplete_rows = 0
    for raw in list_data.values():
        record = _parse_record(raw)
        if record is None:
            continue
        rows += 1
        request_id = record.get("request_id")
        if request_id is None:
            continue
        request_id = str(request_id)
        age = _record_age(record, now)
        # Parked stamps are applied before the completeness check so a row
        # whose settlement landed just before the restart is not replayed.
        parked = pending.pop(request_id, None)
        pending_at.pop(request_id, None)
        merged = bool(parked) and _apply_stamps(record, parked or {})
        complete = is_complete(record, require_posted)
        if not complete:
            incomplete_rows += 1
        if age is not None and age <= retention_seconds:
            payload = fetch_payload(record)
            if payload is not None:
                fetch_payloads.append(payload)
        if request_id in records:
            # Live row: its follow-ups are in progress on the in-memory copy.
            continue
        if age is not None and age > incomplete_cap_seconds:
            # Past the cap: the sweeper deletes it this pass; do not replay.
            continue
        if merged:
            # The parked stamp changed the row: load it and persist the merge
            # even when the stamp is what completed it, or the stored row
            # would still read incomplete on the next sweep and be replayed
            # on top of work that already happened. A complete row is popped
            # by the SUCCESS handler once the write lands.
            records[request_id] = record
            enqueue_write(shared_state, request_id)
        if complete:
            continue
        records[request_id] = record
        replay.append(record)
    shared_state[PREIMAGE_SWEEP_ROW_COUNT] = (
        shared_state.get(PREIMAGE_SWEEP_ROW_COUNT, 0) + rows
    )
    shared_state[PREIMAGE_SWEEP_INCOMPLETE_COUNT] = (
        shared_state.get(PREIMAGE_SWEEP_INCOMPLETE_COUNT, 0) + incomplete_rows
    )
    return fetch_payloads, replay


def replayable_events(
    shared_state: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any], str]]:
    """Return the settled-but-unposted predict-api events held in memory.

    These are rows the drainer hydrated whose on-chain settlement already
    happened (so the normal post-settlement path, which only POSTs the
    current round's deliveries, will never pick them up). The settlement
    behaviour POSTs them as a separate replay batch and stamps
    ``posted_at`` on a 2xx.

    :param shared_state: the skill's shared state dict.
    :return: ``(request_id, event, settled_tx_hash)`` triples.
    """
    out: List[Tuple[str, Dict[str, Any], str]] = []
    for request_id, record in shared_state.get(PREIMAGE_RECORDS, {}).items():
        if record.get("settlement_status") != STATUS_DELIVERED:
            continue
        if record.get(FIELD_ABANDONED_AT) is not None:
            continue
        tx_hash = record.get(FIELD_SETTLED_TX_HASH)
        event = record.get(FIELD_PREDICT_API_EVENT)
        if not tx_hash or record.get(FIELD_POSTED_AT) is not None:
            continue
        if not isinstance(event, dict):
            continue
        out.append((str(request_id), event, str(tx_hash)))
    return out


def expired_keys(
    list_data: Dict[str, str],
    now: float,
    retention_seconds: int,
    incomplete_cap_seconds: Optional[int] = None,
    require_posted: bool = True,
) -> List[str]:
    """Return the kv keys whose preimage is older than the retention window.

    Age is measured from ``settled_at`` when present, else ``accepted_at``.
    Entries with neither timestamp, or whose value can't be parsed, are left
    untouched — the sweeper must never delete a row it doesn't understand.

    A delivered row with a follow-up step still missing (see
    :func:`is_complete`) is held past ``retention_seconds`` so the drainer
    can replay it: it expires only past ``incomplete_cap_seconds`` (never,
    when that is ``None``), and each such deletion is logged at WARNING
    because it is unpaid / unreported work being dropped.

    :param list_data: the LIST_RESPONSE key -> JSON value map.
    :param now: the current epoch time in seconds.
    :param retention_seconds: the retention window, in seconds.
    :param incomplete_cap_seconds: the hard cap for incomplete rows.
    :param require_posted: whether ``posted_at`` counts toward completeness.
    :return: the list of keys to delete.
    """
    expired: List[str] = []
    skipped: List[str] = []
    dropped_incomplete: List[str] = []
    for key, raw in list_data.items():
        try:
            record = json.loads(raw)
            # Explicit None checks rather than ``settled_at or accepted_at``:
            # a numeric 0 (epoch 0, 1970) is falsy, so the ``or`` form would
            # fall through to ``accepted_at`` and then trip the "stamp is
            # None" branch. Absurd in practice but technically a valid
            # timestamp, and easy to write a fuzz test that hits it.
            stamp = record.get("settled_at")
            if stamp is None:
                stamp = record.get("accepted_at")
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
        if isinstance(record, dict) and not is_complete(record, require_posted):
            if incomplete_cap_seconds is not None and age > incomplete_cap_seconds:
                expired.append(key)
                dropped_incomplete.append(key)
            continue
        if age > retention_seconds:
            expired.append(key)
    if dropped_incomplete:
        _logger.warning(
            "Preimage sweep: %d delivered entr%s past the %ds incomplete cap "
            "with a follow-up step still missing; deleting. First key: %r",
            len(dropped_incomplete),
            "y" if len(dropped_incomplete) == 1 else "ies",
            incomplete_cap_seconds,
            dropped_incomplete[0],
        )
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
