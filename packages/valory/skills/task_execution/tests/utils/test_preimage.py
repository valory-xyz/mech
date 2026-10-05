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

"""Tests for the off-chain preimage buffer helpers.

These cover the pure bookkeeping the buffer relies on (record accept/settle,
write-queue de-duplication, and the sweeper's expiry computation), which is the
part most likely to drift independently of the AEA async plumbing.
"""

import json
from typing import Any

import pytest

from packages.valory.skills.task_execution.utils import preimage


def _new_state() -> dict:
    """Return a freshly-initialised shared_state dict."""
    state: dict = {}
    preimage.init_shared_state(state)
    return state


def test_init_shared_state_is_idempotent() -> None:
    """init_shared_state sets defaults once and never clobbers existing data."""
    state = _new_state()
    state[preimage.PREIMAGE_RECORDS]["r1"] = {"x": 1}
    preimage.init_shared_state(state)  # second call must not reset
    assert state[preimage.PREIMAGE_RECORDS] == {"r1": {"x": 1}}
    assert state[preimage.PREIMAGE_WRITE_QUEUE] == []
    assert state[preimage.PREIMAGE_KV_IN_FLIGHT] is False
    assert state[preimage.PREIMAGE_INFLIGHT_WRITE] is None


def test_preimage_key() -> None:
    """Keys are the prefix concatenated with the request id."""
    assert preimage.preimage_key("mech_preimage/", "abc") == "mech_preimage/abc"


def test_record_accept_buffers_processing_record_and_queues_write() -> None:
    """A fresh accept stores a processing record and queues exactly one write."""
    state = _new_state()
    preimage.record_accept(state, "r1", "the-request", now=1000.0)
    record = state[preimage.PREIMAGE_RECORDS]["r1"]
    assert record["request"] == "the-request"
    assert record["response"] is None
    assert record["accepted_at"] == 1000
    assert record["settled_at"] is None
    assert record["settlement_status"] == preimage.STATUS_PROCESSING
    assert state[preimage.PREIMAGE_WRITE_QUEUE] == ["r1"]


def test_write_queue_dedupes() -> None:
    """Accept then settle for the same id enqueues the id only once."""
    state = _new_state()
    preimage.record_accept(state, "r1", "req", now=1.0)
    preimage.record_settlement(
        state, "r1", "resp", "cidv1", preimage.STATUS_DELIVERED, now=2.0
    )
    assert state[preimage.PREIMAGE_WRITE_QUEUE] == ["r1"]


def test_record_settlement_merges_onto_accept() -> None:
    """Settling an accepted request keeps the request and adds the outcome."""
    state = _new_state()
    preimage.record_accept(state, "r1", "the-request", now=1000.0)
    preimage.record_settlement(
        state, "r1", "the-response", "cidv1", preimage.STATUS_DELIVERED, now=1500.0
    )
    record = state[preimage.PREIMAGE_RECORDS]["r1"]
    assert record["request"] == "the-request"  # preserved
    assert record["response"] == "the-response"
    assert record["response_cid"] == "cidv1"
    assert record["accepted_at"] == 1000
    assert record["settled_at"] == 1500
    assert record["settlement_status"] == preimage.STATUS_DELIVERED


def test_record_settlement_without_accept_creates_minimal_record() -> None:
    """A settle with no prior accept (e.g. restart) still captures the response."""
    state = _new_state()
    preimage.record_settlement(
        state, "r9", "reason", None, preimage.STATUS_REJECTED, now=42.0
    )
    record = state[preimage.PREIMAGE_RECORDS]["r9"]
    assert record["request"] is None
    assert record["accepted_at"] is None
    assert record["response"] == "reason"
    assert record["settlement_status"] == preimage.STATUS_REJECTED
    assert record["settled_at"] == 42


def test_serialize_round_trips() -> None:
    """The serialize helper produces a JSON string that parses back to the record."""
    state = _new_state()
    preimage.record_accept(state, "r1", "req", now=1.0)
    record = state[preimage.PREIMAGE_RECORDS]["r1"]
    assert json.loads(preimage.serialize(record)) == record


# --- sweeper expiry --------------------------------------------------------


def _value(**kwargs: Any) -> str:
    """Build a serialized preimage value with the given fields."""
    return json.dumps(kwargs)


def test_expired_keys_by_settled_at() -> None:
    """An entry older than the window measured from settled_at is expired."""
    now = 100_000.0
    data = {
        "mech_preimage/old": _value(settled_at=now - 200, accepted_at=now - 300),
        "mech_preimage/fresh": _value(settled_at=now - 10, accepted_at=now - 50),
    }
    assert preimage.expired_keys(data, now, retention_seconds=100) == [
        "mech_preimage/old"
    ]


def test_expired_keys_falls_back_to_accepted_at() -> None:
    """An in-flight (no settled_at) entry expires off accepted_at."""
    now = 100_000.0
    data = {"mech_preimage/stuck": _value(accepted_at=now - 500, settled_at=None)}
    assert preimage.expired_keys(data, now, retention_seconds=100) == [
        "mech_preimage/stuck"
    ]


def test_expired_keys_skips_unparseable_and_timestampless() -> None:
    """Malformed or timestamp-free values are never selected for deletion."""
    now = 100_000.0
    data = {
        "mech_preimage/bad": "not-json",
        "mech_preimage/empty": _value(request="x"),  # no timestamps
        "mech_preimage/old": _value(settled_at=now - 999),
    }
    assert preimage.expired_keys(data, now, retention_seconds=100) == [
        "mech_preimage/old"
    ]


def test_expired_keys_empty() -> None:
    """An empty LIST response yields nothing to delete."""
    assert preimage.expired_keys({}, 1.0, retention_seconds=100) == []


# --- replay inputs / stamps / completeness ---------------------------------

REQUEST_ID = "r1"
SETTLED_TX_HASH = "0x" + "ab" * 32
SAMPLE_DONE_TASK = {"request_id": 1, "sender": "0xSENDER", "nonce": 7}
SAMPLE_EVENT = {"response": {"request_id": "1", "delivery_tx_hash": None}}
NOW = 1_000.0
_DELIVERED_WITH_INPUTS = {
    "settlement_status": preimage.STATUS_DELIVERED,
    preimage.FIELD_DONE_TASK: SAMPLE_DONE_TASK,
    preimage.FIELD_PREDICT_API_EVENT: SAMPLE_EVENT,
}


def _active_state() -> dict:
    """Return an initialised shared_state with retention switched on."""
    state: dict = {}
    preimage.init_shared_state(state, retention_enabled=True)
    return state


def _delivered_state(**stamps: Any) -> dict:
    """Return an active state holding one delivered record with ``stamps``."""
    state = _active_state()
    preimage.record_accept(state, REQUEST_ID, "req", now=NOW)
    preimage.record_settlement(
        state, REQUEST_ID, '{"result": "ok"}', "cid", preimage.STATUS_DELIVERED, NOW
    )
    state[preimage.PREIMAGE_RECORDS][REQUEST_ID].update(stamps)
    state[preimage.PREIMAGE_WRITE_QUEUE].clear()
    return state


def test_init_shared_state_records_retention_flag_only_when_given() -> None:
    """The retention flag defaults False, is set when passed, and is kept otherwise."""
    state: dict = {}
    preimage.init_shared_state(state)
    assert state[preimage.PREIMAGE_RETENTION_ACTIVE] is False
    preimage.init_shared_state(state, retention_enabled=True)
    assert state[preimage.PREIMAGE_RETENTION_ACTIVE] is True
    preimage.init_shared_state(state)
    assert state[preimage.PREIMAGE_RETENTION_ACTIVE] is True


def test_record_replay_inputs_attaches_and_queues_write() -> None:
    """A delivered record gains done_task + event and is re-queued for flush."""
    state = _delivered_state()
    ok = preimage.record_replay_inputs(
        state, REQUEST_ID, SAMPLE_DONE_TASK, SAMPLE_EVENT
    )
    record = state[preimage.PREIMAGE_RECORDS][REQUEST_ID]
    assert ok is True
    assert record[preimage.FIELD_DONE_TASK] == SAMPLE_DONE_TASK
    assert record[preimage.FIELD_PREDICT_API_EVENT] == SAMPLE_EVENT
    assert state[preimage.PREIMAGE_WRITE_QUEUE] == [REQUEST_ID]


@pytest.mark.parametrize(
    "status", [preimage.STATUS_PROCESSING, preimage.STATUS_REJECTED]
)
def test_record_replay_inputs_ignores_non_delivered_records(status: str) -> None:
    """Only a delivered record can carry replay inputs; others are left untouched."""
    state = _active_state()
    if status == preimage.STATUS_PROCESSING:
        preimage.record_accept(state, REQUEST_ID, "req", now=NOW)
    else:
        preimage.record_settlement(state, REQUEST_ID, "reason", None, status, NOW)
    state[preimage.PREIMAGE_WRITE_QUEUE].clear()
    ok = preimage.record_replay_inputs(
        state, REQUEST_ID, SAMPLE_DONE_TASK, SAMPLE_EVENT
    )
    assert ok is False
    assert (
        state[preimage.PREIMAGE_RECORDS][REQUEST_ID][preimage.FIELD_DONE_TASK] is None
    )
    assert state[preimage.PREIMAGE_WRITE_QUEUE] == []


def test_record_replay_inputs_without_record_is_a_noop() -> None:
    """No record at all: nothing is created and nothing is queued."""
    state = _active_state()
    assert (
        preimage.record_replay_inputs(state, "ghost", SAMPLE_DONE_TASK, None) is False
    )
    assert state[preimage.PREIMAGE_RECORDS] == {}
    assert state[preimage.PREIMAGE_WRITE_QUEUE] == []


def test_record_stamp_is_noop_when_retention_inactive() -> None:
    """With retention off nothing is stamped, queued or parked."""
    state: dict = {}
    preimage.init_shared_state(state)
    preimage.record_settlement(
        state, REQUEST_ID, "resp", "cid", preimage.STATUS_DELIVERED, NOW
    )
    state[preimage.PREIMAGE_WRITE_QUEUE].clear()
    preimage.record_stamp(state, REQUEST_ID, settled_tx_hash=SETTLED_TX_HASH)
    preimage.record_stamp(state, "ghost", settled_tx_hash=SETTLED_TX_HASH)
    record = state[preimage.PREIMAGE_RECORDS][REQUEST_ID]
    assert record[preimage.FIELD_SETTLED_TX_HASH] is None
    assert state[preimage.PREIMAGE_WRITE_QUEUE] == []
    assert state[preimage.PREIMAGE_PENDING_STAMPS] == {}


def test_record_stamp_applies_write_once_and_queues_write() -> None:
    """A stamp lands once, is not moved by a second call, and queues a flush."""
    state = _delivered_state()
    preimage.record_stamp(state, REQUEST_ID, settled_tx_hash=SETTLED_TX_HASH)
    preimage.record_stamp(state, REQUEST_ID, settled_tx_hash="0xlater")
    record = state[preimage.PREIMAGE_RECORDS][REQUEST_ID]
    assert record[preimage.FIELD_SETTLED_TX_HASH] == SETTLED_TX_HASH
    assert state[preimage.PREIMAGE_WRITE_QUEUE] == [REQUEST_ID]


def test_record_stamp_with_none_value_changes_nothing() -> None:
    """A ``None`` stamp is ignored and does not queue a write."""
    state = _delivered_state()
    preimage.record_stamp(state, REQUEST_ID, settled_tx_hash=None)
    assert state[preimage.PREIMAGE_WRITE_QUEUE] == []


def test_record_stamp_parks_durable_stamps_for_missing_record() -> None:
    """Settlement / post stamps for an unloaded row wait for hydrate; fetched is dropped."""
    state = _active_state()
    preimage.record_stamp(
        state, "r9", settled_tx_hash=SETTLED_TX_HASH, fetched_at=int(NOW)
    )
    preimage.record_stamp(state, "r9", settled_tx_hash="0xlater", posted_at=int(NOW))
    assert state[preimage.PREIMAGE_PENDING_STAMPS] == {
        "r9": {
            preimage.FIELD_SETTLED_TX_HASH: SETTLED_TX_HASH,
            preimage.FIELD_POSTED_AT: int(NOW),
        }
    }
    assert state[preimage.PREIMAGE_WRITE_QUEUE] == []


def test_record_stamp_rejects_unknown_field() -> None:
    """A typo in a stamp name is a programming error, not a silent no-op."""
    state = _delivered_state()
    with pytest.raises(ValueError, match="unknown stamp field"):
        preimage.record_stamp(state, REQUEST_ID, settled_tx=SETTLED_TX_HASH)


@pytest.mark.parametrize(
    "record, require_posted, expected",
    [
        ({"settlement_status": preimage.STATUS_REJECTED}, True, True),
        ({"settlement_status": preimage.STATUS_PROCESSING}, True, True),
        ({"settlement_status": None}, True, True),  # older schema / unknown
        # Unsettled with a done_task to re-queue: incomplete either way.
        ({**_DELIVERED_WITH_INPUTS}, True, False),
        ({**_DELIVERED_WITH_INPUTS}, False, False),
        # Unsettled without a done_task (pre-replay schema): nothing to do.
        ({"settlement_status": preimage.STATUS_DELIVERED}, True, True),
        (
            {
                "settlement_status": preimage.STATUS_DELIVERED,
                preimage.FIELD_PREDICT_API_EVENT: SAMPLE_EVENT,
            },
            True,
            True,
        ),
        # Settled, not posted, with an event: incomplete only when a post is required.
        (
            {**_DELIVERED_WITH_INPUTS, preimage.FIELD_SETTLED_TX_HASH: SETTLED_TX_HASH},
            True,
            False,
        ),
        (
            {**_DELIVERED_WITH_INPUTS, preimage.FIELD_SETTLED_TX_HASH: SETTLED_TX_HASH},
            False,
            True,
        ),
        # Settled, not posted, no event to send: complete.
        (
            {
                "settlement_status": preimage.STATUS_DELIVERED,
                preimage.FIELD_DONE_TASK: SAMPLE_DONE_TASK,
                preimage.FIELD_SETTLED_TX_HASH: SETTLED_TX_HASH,
            },
            True,
            True,
        ),
        (
            {
                **_DELIVERED_WITH_INPUTS,
                preimage.FIELD_SETTLED_TX_HASH: SETTLED_TX_HASH,
                preimage.FIELD_POSTED_AT: 1,
            },
            True,
            True,
        ),
        (
            {
                "settlement_status": preimage.STATUS_DELIVERED,
                preimage.FIELD_ABANDONED_AT: 1,
            },
            True,
            True,
        ),
    ],
)
def test_is_complete(record: dict, require_posted: bool, expected: bool) -> None:
    """Completeness depends on status, the settle stamp and (optionally) the post stamp."""
    assert preimage.is_complete(record, require_posted) is expected


# --- expiry with incomplete rows -------------------------------------------

RETENTION = 100
CAP = 1_000


def _delivered_value(age: float, now: float = 100_000.0, **extra: Any) -> str:
    """Serialize a delivered row with replay inputs, settled ``age`` seconds before ``now``."""
    fields: dict = {
        "settlement_status": preimage.STATUS_DELIVERED,
        "settled_at": now - age,
        preimage.FIELD_DONE_TASK: SAMPLE_DONE_TASK,
        preimage.FIELD_PREDICT_API_EVENT: SAMPLE_EVENT,
    }
    fields.update(extra)
    return _value(**fields)


def test_expired_keys_retires_pre_replay_delivered_rows_on_the_normal_window() -> None:
    """A delivered row from before replay inputs existed has nothing to replay."""
    now = 100_000.0
    data = {
        "k/legacy": _value(
            settlement_status=preimage.STATUS_DELIVERED, settled_at=now - RETENTION - 1
        )
    }
    assert preimage.expired_keys(data, now, RETENTION, incomplete_cap_seconds=CAP) == [
        "k/legacy"
    ]


def test_expired_keys_keeps_incomplete_delivered_row_past_retention() -> None:
    """An unsettled delivered row outlives the retention window."""
    now = 100_000.0
    data = {"k/unsettled": _delivered_value(RETENTION + 50, now)}
    assert preimage.expired_keys(data, now, RETENTION, incomplete_cap_seconds=CAP) == []


def test_expired_keys_deletes_incomplete_delivered_row_past_cap() -> None:
    """Past the hard cap an unsettled row is deleted after all."""
    now = 100_000.0
    data = {"k/stuck": _delivered_value(CAP + 1, now)}
    assert preimage.expired_keys(data, now, RETENTION, incomplete_cap_seconds=CAP) == [
        "k/stuck"
    ]


def test_expired_keys_without_cap_never_deletes_incomplete_rows() -> None:
    """``incomplete_cap_seconds=None`` (the legacy call shape) holds them forever."""
    now = 100_000.0
    data = {"k/stuck": _delivered_value(10 * CAP, now)}
    assert preimage.expired_keys(data, now, RETENTION) == []


def test_expired_keys_treats_settled_and_posted_row_as_complete() -> None:
    """A fully stamped delivered row expires on the normal window."""
    now = 100_000.0
    data = {
        "k/done": _delivered_value(
            RETENTION + 1, now, settled_tx_hash=SETTLED_TX_HASH, posted_at=now
        )
    }
    assert preimage.expired_keys(data, now, RETENTION, incomplete_cap_seconds=CAP) == [
        "k/done"
    ]


def test_expired_keys_ignores_post_stamp_when_not_required() -> None:
    """With the predict-api write off, settled alone makes the row complete."""
    now = 100_000.0
    data = {"k/done": _delivered_value(RETENTION + 1, now, settled_tx_hash="0x1")}
    assert preimage.expired_keys(
        data, now, RETENTION, incomplete_cap_seconds=CAP, require_posted=False
    ) == ["k/done"]


# --- fetch payload ----------------------------------------------------------


def test_fetch_payload_rebuilds_delivered_shape() -> None:
    """A delivered row serves the committed response verbatim under ``response``."""
    record = {
        "request_id": 42,
        "settlement_status": preimage.STATUS_DELIVERED,
        "response": '{"result": "p_yes=0.7"}',
        "response_cid": "cid",
    }
    assert preimage.fetch_payload(record) == {
        "request_id": "42",
        "status": "ok",
        "content_cid": "cid",
        "response": {"result": "p_yes=0.7"},
    }


def test_fetch_payload_rebuilds_rejected_shape() -> None:
    """A rejected row serves the stored reason."""
    record = {
        "request_id": "42",
        "settlement_status": preimage.STATUS_REJECTED,
        "response": "insufficient balance",
    }
    assert preimage.fetch_payload(record) == {
        "request_id": "42",
        "status": "rejected",
        "reason": "insufficient balance",
    }


@pytest.mark.parametrize(
    "record",
    [
        {"request_id": "1", "settlement_status": preimage.STATUS_PROCESSING},
        {"settlement_status": preimage.STATUS_DELIVERED, "response": "{}"},
        {"request_id": "1", "settlement_status": preimage.STATUS_DELIVERED},
        {
            "request_id": "1",
            "settlement_status": preimage.STATUS_DELIVERED,
            "response": "not-json",
        },
    ],
)
def test_fetch_payload_returns_none_for_unservable_rows(record: dict) -> None:
    """Unanswered, id-less, response-less or undecodable rows are not served."""
    assert preimage.fetch_payload(record) is None


# --- hydrate ------------------------------------------------------------------


def _row(request_id: str, age: float, now: float = 100_000.0, **extra: Any) -> str:
    """Serialize a delivered row with replay inputs, settled ``age`` seconds ago."""
    fields = {
        "request_id": request_id,
        "settlement_status": preimage.STATUS_DELIVERED,
        "settled_at": now - age,
        "response": '{"result": "x"}',
        "response_cid": "cid-" + request_id,
        preimage.FIELD_DONE_TASK: {"request_id": int(request_id)},
        preimage.FIELD_PREDICT_API_EVENT: {"response": {"request_id": request_id}},
    }
    fields.update(extra)
    return json.dumps(fields)


def _hydrate(state: dict, data: dict, now: float = 100_000.0) -> tuple:
    """Run hydrate with the module's test-wide retention / cap constants."""
    return preimage.hydrate(state, data, now, RETENTION, CAP, require_posted=True)


def test_hydrate_restores_fetch_payloads_within_retention_only() -> None:
    """Served answers come back for rows inside the window, not for older ones."""
    state = _active_state()
    now = 100_000.0
    data = {
        "k/fresh": _row("1", 10, now, settled_tx_hash="0x1", posted_at=1),
        "k/old": _row("2", RETENTION + 1, now, settled_tx_hash="0x1", posted_at=1),
    }
    payloads, replay = _hydrate(state, data, now)
    assert [p["request_id"] for p in payloads] == ["1"]
    assert payloads[0]["status"] == "ok" and payloads[0]["content_cid"] == "cid-1"
    assert replay == []


def test_hydrate_loads_incomplete_rows_into_memory_for_replay() -> None:
    """An unsettled delivered row is loaded and returned for replay."""
    state = _active_state()
    _, replay = _hydrate(state, {"k/1": _row("1", 10)})
    assert [r["request_id"] for r in replay] == ["1"]
    assert state[preimage.PREIMAGE_RECORDS]["1"][preimage.FIELD_DONE_TASK] == {
        "request_id": 1
    }
    assert state[preimage.PREIMAGE_WRITE_QUEUE] == []  # unchanged row, no re-flush


def test_hydrate_skips_rows_already_in_memory() -> None:
    """A row the live flow still holds is never replayed on top of itself."""
    state = _active_state()
    state[preimage.PREIMAGE_RECORDS]["1"] = {"request_id": "1", "live": True}
    _, replay = _hydrate(state, {"k/1": _row("1", 10)})
    assert replay == []
    assert state[preimage.PREIMAGE_RECORDS]["1"] == {"request_id": "1", "live": True}


def test_hydrate_skips_complete_and_past_cap_rows() -> None:
    """Complete rows and rows the sweeper is about to delete are not loaded."""
    state = _active_state()
    data = {
        "k/done": _row("1", 10, settled_tx_hash="0x1", posted_at=1),
        "k/stuck": _row("2", CAP + 1),
    }
    _, replay = _hydrate(state, data)
    assert replay == []
    assert state[preimage.PREIMAGE_RECORDS] == {}


def test_hydrate_applies_parked_stamps_before_deciding() -> None:
    """Parked stamps merge in first: a fully stamped row is not replayed; a partial one is."""
    state = _active_state()
    state[preimage.PREIMAGE_PENDING_STAMPS] = {
        "1": {preimage.FIELD_SETTLED_TX_HASH: "0x1", preimage.FIELD_POSTED_AT: 5},
        "2": {preimage.FIELD_SETTLED_TX_HASH: "0x2"},
    }
    state[preimage.PREIMAGE_PENDING_STAMPS_AT] = {"1": 1.0, "2": 1.0}
    _, replay = _hydrate(state, {"k/1": _row("1", 10), "k/2": _row("2", 10)})
    assert [r["request_id"] for r in replay] == ["2"]
    assert replay[0][preimage.FIELD_SETTLED_TX_HASH] == "0x2"
    # Both merged rows are written back, including the one the stamp completed:
    # otherwise the stored row would still read incomplete on the next sweep.
    assert state[preimage.PREIMAGE_WRITE_QUEUE] == ["1", "2"]
    assert state[preimage.PREIMAGE_RECORDS]["1"][preimage.FIELD_POSTED_AT] == 5
    assert state[preimage.PREIMAGE_PENDING_STAMPS] == {}
    assert state[preimage.PREIMAGE_PENDING_STAMPS_AT] == {}


def test_hydrate_completed_by_parked_stamp_is_popped_after_its_write_lands() -> None:
    """The written-back complete row is a terminal, complete record: nothing is replayed twice."""
    state = _active_state()
    state[preimage.PREIMAGE_PENDING_STAMPS] = {
        "1": {preimage.FIELD_SETTLED_TX_HASH: "0x1", preimage.FIELD_POSTED_AT: 5}
    }
    _hydrate(state, {"k/1": _row("1", 10)})
    record = state[preimage.PREIMAGE_RECORDS]["1"]
    assert preimage.is_complete(record, require_posted=True) is True
    # A second sweep sees the row already in memory and leaves it alone.
    _, replay = _hydrate(state, {"k/1": _row("1", 10)})
    assert replay == []


def test_record_stamp_timestamps_parked_entries_and_prune_drops_old_ones() -> None:
    """Parked stamps carry a timestamp and are dropped once older than the max age."""
    state = _active_state()
    preimage.record_stamp(state, "old", now=100.0, settled_tx_hash="0x1")
    preimage.record_stamp(state, "new", now=900.0, settled_tx_hash="0x2")
    preimage.record_stamp(
        state, "old", now=950.0, posted_at=1
    )  # keeps first stamp time
    dropped = preimage.prune_parked_stamps(state, now=1_000.0, max_age_seconds=500)
    assert dropped == 1
    assert set(state[preimage.PREIMAGE_PENDING_STAMPS]) == {"new"}
    assert set(state[preimage.PREIMAGE_PENDING_STAMPS_AT]) == {"new"}


def test_hydrate_counts_rows_and_incomplete_rows_across_pages() -> None:
    """Row counters accumulate over pages; unparseable values are not counted."""
    state = _active_state()
    _hydrate(state, {"k/1": _row("1", 10), "k/bad": "not-json"})
    _hydrate(state, {"k/2": _row("2", 10, settled_tx_hash="0x1", posted_at=1)})
    assert state[preimage.PREIMAGE_SWEEP_ROW_COUNT] == 2
    assert state[preimage.PREIMAGE_SWEEP_INCOMPLETE_COUNT] == 1


def test_replayable_events_returns_settled_unposted_rows_with_an_event() -> None:
    """Only delivered, settled, unposted, non-abandoned rows carrying an event qualify."""
    state = _active_state()
    state[preimage.PREIMAGE_RECORDS] = {
        "1": json.loads(_row("1", 1, settled_tx_hash="0xok")),  # qualifies
        "2": json.loads(_row("2", 1)),  # unsettled
        "3": json.loads(_row("3", 1, settled_tx_hash="0x1", posted_at=1)),  # posted
        "4": json.loads(_row("4", 1, settled_tx_hash="0x1", abandoned_at=1)),
        "5": json.loads(_row("5", 1, settled_tx_hash="0x1", predict_api_event=None)),
        "6": {"settlement_status": preimage.STATUS_REJECTED, "settled_tx_hash": "0x1"},
    }
    assert preimage.replayable_events(state) == [
        ("1", {"response": {"request_id": "1"}}, "0xok")
    ]


def test_replayable_events_honours_limit_oldest_settled_first() -> None:
    """A limit returns the oldest settled rows and leaves the rest for the next round."""
    state = _active_state()
    for rid, settled_at in (("3", 300.0), ("1", 100.0), ("2", 200.0)):
        state[preimage.PREIMAGE_RECORDS][rid] = json.loads(
            _row(rid, 1, settled_tx_hash="0x" + rid, settled_at=settled_at)
        )
    assert [rid for rid, _, _ in preimage.replayable_events(state, limit=2)] == [
        "1",
        "2",
    ]
    assert len(preimage.replayable_events(state)) == 3


def test_evict_expired_records_drops_only_rows_past_the_cap() -> None:
    """In-memory rows older than the cap are dropped, along with their queued writes."""
    state = _active_state()
    now = 100_000.0
    state[preimage.PREIMAGE_RECORDS] = {
        "1": json.loads(_row("1", CAP + 1, now)),  # past the cap
        "2": json.loads(_row("2", 10, now)),  # fresh
        "3": {"request_id": "3", "settlement_status": "delivered"},  # no stamp
    }
    state[preimage.PREIMAGE_WRITE_QUEUE] = ["1", "2"]
    state[preimage.PREIMAGE_WRITE_ATTEMPTS] = {"1": 2}
    assert preimage.evict_expired_records(state, now, CAP) == 1
    assert set(state[preimage.PREIMAGE_RECORDS]) == {"2", "3"}
    assert state[preimage.PREIMAGE_WRITE_QUEUE] == ["2"]
    assert state[preimage.PREIMAGE_WRITE_ATTEMPTS] == {}
