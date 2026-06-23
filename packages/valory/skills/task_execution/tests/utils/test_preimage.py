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
