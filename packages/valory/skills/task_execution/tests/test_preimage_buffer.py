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

"""Tests for the async wiring of the off-chain preimage buffer.

The pure bookkeeping is covered in ``tests/utils/test_preimage.py``; here we
exercise the behaviour's flush/sweep loop and the handler's reply handling
against the stub skill context.
"""

import json
import time
from types import SimpleNamespace
from typing import Any, Dict, Generator, List

import pytest
from prometheus_client import REGISTRY

from packages.valory.protocols.kv_store.message import KvStoreMessage
from packages.valory.skills.task_execution.handlers import KvStoreHandler
from packages.valory.skills.task_execution.utils import preimage


@pytest.fixture(autouse=True)
def _clear_prometheus_registry() -> Generator[None, None, None]:
    """Unregister prometheus collectors after each test.

    The behaviour fixture builds a TaskExecutionBehaviour, which registers
    global metrics; without this the second instantiation in the session trips
    a duplicate-timeseries error (mirrors test_behaviours.py).

    :yields: control back to the test, then sweeps the registry on return.
    """
    yield
    for collector in list(REGISTRY._names_to_collectors.values()):
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass


class _CaptureOutbox:
    """Outbox that records messages so a test can inspect what was sent."""

    def __init__(self) -> None:
        """Initialize the capture list."""
        self.sent: List[Any] = []

    def put_message(self, message: Any = None, **_kwargs: Any) -> None:
        """Record an outgoing message."""
        self.sent.append(message)


# --- behaviour: flush / sweep loop -----------------------------------------


def test_process_buffer_noop_when_disabled(behaviour: Any) -> None:
    """With retention off (the default), nothing is ever sent."""
    behaviour.context.params.preimage_retention_enabled = False
    out = _CaptureOutbox()
    behaviour.context.outbox = out
    behaviour._process_preimage_buffer()
    assert out.sent == []


def test_process_buffer_in_flight_blocks(behaviour: Any) -> None:
    """A kv op already in flight prevents issuing another."""
    behaviour.context.params.preimage_retention_enabled = True
    behaviour.context.shared_state[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    out = _CaptureOutbox()
    behaviour.context.outbox = out
    behaviour._process_preimage_buffer()
    assert out.sent == []


def test_first_tick_sweeps(behaviour: Any) -> None:
    """First tick issues a LIST_REQUEST (sweep due, last_sweep defaults to 0)."""
    behaviour.context.params.preimage_retention_enabled = True
    out = _CaptureOutbox()
    behaviour.context.outbox = out
    behaviour._process_preimage_buffer()
    assert len(out.sent) == 1
    assert out.sent[0].performative == KvStoreMessage.Performative.LIST_REQUEST
    assert out.sent[0].key_prefix == "mech_preimage/"
    assert behaviour.context.shared_state[preimage.PREIMAGE_KV_IN_FLIGHT] is True


def test_flushes_queued_write_when_sweep_not_due(behaviour: Any) -> None:
    """With the sweep not due, a queued preimage is upserted into the kv_store."""
    ss = behaviour.context.shared_state
    behaviour.context.params.preimage_retention_enabled = True
    ss[preimage.PREIMAGE_LAST_SWEEP] = time.time()
    preimage.record_accept(ss, "r1", "the-request", now=time.time())
    out = _CaptureOutbox()
    behaviour.context.outbox = out
    behaviour._process_preimage_buffer()
    assert out.sent[0].performative == (
        KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST
    )
    assert "mech_preimage/r1" in out.sent[0].data
    assert ss[preimage.PREIMAGE_INFLIGHT_WRITE] == "r1"
    assert ss[preimage.PREIMAGE_KV_IN_FLIGHT] is True


def test_flushes_queued_deletes_before_writes(behaviour: Any) -> None:
    """Expired-key deletes drain (batched) ahead of pending writes."""
    ss = behaviour.context.shared_state
    behaviour.context.params.preimage_retention_enabled = True
    ss[preimage.PREIMAGE_LAST_SWEEP] = time.time()
    ss[preimage.PREIMAGE_DELETE_QUEUE].extend(["mech_preimage/a", "mech_preimage/b"])
    preimage.record_accept(ss, "r1", "req", now=time.time())
    out = _CaptureOutbox()
    behaviour.context.outbox = out
    behaviour._process_preimage_buffer()
    assert out.sent[0].performative == KvStoreMessage.Performative.DELETE_REQUEST
    assert set(out.sent[0].keys) == {"mech_preimage/a", "mech_preimage/b"}
    assert ss[preimage.PREIMAGE_DELETE_QUEUE] == []
    # the write is left queued for the next tick
    assert ss[preimage.PREIMAGE_WRITE_QUEUE] == ["r1"]


# --- handler: reply handling -----------------------------------------------


def _handler(handler_context: Any) -> KvStoreHandler:
    """Build a KvStoreHandler on the (preimage-initialised) stub context."""
    preimage.init_shared_state(handler_context.shared_state)
    return KvStoreHandler(name="kv_store", skill_context=handler_context)


def test_handler_list_response_queues_expired(handler_context: Any) -> None:
    """LIST_RESPONSE queues expired keys for deletion and clears in-flight."""
    handler_context.params.preimage_retention_seconds = 100
    handler = _handler(handler_context)
    handler_context.shared_state[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    now = time.time()
    msg = SimpleNamespace(
        performative=KvStoreMessage.Performative.LIST_RESPONSE,
        data={
            "mech_preimage/old": json.dumps({"settled_at": now - 9999}),
            "mech_preimage/new": json.dumps({"settled_at": now}),
        },
        message="",
    )
    handler.handle(msg)
    ss = handler_context.shared_state
    assert ss[preimage.PREIMAGE_DELETE_QUEUE] == ["mech_preimage/old"]
    assert ss[preimage.PREIMAGE_KV_IN_FLIGHT] is False


def test_handler_success_pops_terminal_record(handler_context: Any) -> None:
    """A successful write of a settled record drops the in-process copy."""
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    preimage.record_settlement(
        ss, "r1", "resp", "cid", preimage.STATUS_DELIVERED, now=1.0
    )
    # Simulate the behaviour loop having already popped "r1" off the write
    # queue when it sent the kv write — that's the state the SUCCESS handler
    # actually runs against.
    ss[preimage.PREIMAGE_WRITE_QUEUE].remove("r1")
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r1"
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.SUCCESS, message="ok")
    )
    assert "r1" not in ss[preimage.PREIMAGE_RECORDS]
    assert ss[preimage.PREIMAGE_INFLIGHT_WRITE] is None
    assert ss[preimage.PREIMAGE_KV_IN_FLIGHT] is False


def test_handler_success_keeps_processing_record(handler_context: Any) -> None:
    """A successful processing-write keeps the record for the settle merge."""
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    preimage.record_accept(ss, "r1", "req", now=1.0)
    ss[preimage.PREIMAGE_WRITE_QUEUE].remove("r1")
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r1"
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.SUCCESS, message="ok")
    )
    assert "r1" in ss[preimage.PREIMAGE_RECORDS]


def test_handler_success_keeps_record_if_settle_raced_with_write(
    handler_context: Any,
) -> None:
    """A settle DURING an in-flight processing write keeps the terminal record."""
    # The Critical race: if we pop the record on SUCCESS while a re-enqueued
    # terminal write is sitting in the queue, the next flush hits
    # ``record is None`` and the terminal preimage is silently lost. The
    # guard is ``inflight in PREIMAGE_WRITE_QUEUE`` — true exactly when
    # ``record_settlement`` re-enqueued the id after the processing write
    # snapshot was already sent.
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    # Step 1: accept the request. Enqueues "r1" with status=processing.
    preimage.record_accept(ss, "r1", "req", now=1.0)
    # Step 2: the behaviour loop popped "r1" off the queue and sent it.
    ss[preimage.PREIMAGE_WRITE_QUEUE].remove("r1")
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r1"
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    # Step 3: BEFORE the SUCCESS for the processing write arrives, the task
    # settles. record_settlement mutates the record to terminal and
    # re-enqueues "r1".
    preimage.record_settlement(
        ss, "r1", "resp", "cid", preimage.STATUS_DELIVERED, now=2.0
    )
    assert "r1" in ss[preimage.PREIMAGE_WRITE_QUEUE]
    # Step 4: the SUCCESS for the processing snapshot finally arrives. It must
    # leave the (now terminal) record alone so the next tick flushes it.
    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.SUCCESS, message="ok")
    )
    assert "r1" in ss[preimage.PREIMAGE_RECORDS]
    assert (
        ss[preimage.PREIMAGE_RECORDS]["r1"]["settlement_status"]
        == preimage.STATUS_DELIVERED
    )
    assert "r1" in ss[preimage.PREIMAGE_WRITE_QUEUE]


def test_handler_error_requeues_write(handler_context: Any) -> None:
    """A write ERROR re-queues the request id so the preimage isn't lost."""
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    preimage.record_accept(ss, "r1", "req", now=1.0)
    ss[preimage.PREIMAGE_WRITE_QUEUE].clear()
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r1"
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.ERROR, message="boom")
    )
    assert ss[preimage.PREIMAGE_WRITE_QUEUE] == ["r1"]
    assert ss[preimage.PREIMAGE_KV_IN_FLIGHT] is False


def test_handler_error_requeue_dedupes_via_enqueue_write(
    handler_context: Any,
) -> None:
    """A write ERROR re-queues through enqueue_write — duplicate id is collapsed."""
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    # Simulate "r1" already in the queue (e.g. a settle re-enqueued it after
    # the in-flight processing write was sent). An ERROR for the in-flight
    # write must not double-add: enqueue_write is dedup'd.
    preimage.record_accept(ss, "r1", "req", now=1.0)
    ss[preimage.PREIMAGE_WRITE_QUEUE].remove("r1")
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r1"
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    ss[preimage.PREIMAGE_WRITE_QUEUE].append("r1")
    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.ERROR, message="boom")
    )
    assert ss[preimage.PREIMAGE_WRITE_QUEUE] == ["r1"]


def test_handler_delete_or_list_error_self_heals(handler_context: Any) -> None:
    """A failed DELETE / LIST (PREIMAGE_INFLIGHT_WRITE is None) doesn't re-queue."""
    # DELETE/LIST aren't writes so there's nothing in PREIMAGE_INFLIGHT_WRITE
    # to re-queue. The next sweep tick re-issues the LIST and picks up the
    # same expired keys (they're still in kv_store), so the path self-heals
    # without retrying. What we DO assert is that the in-flight flag clears
    # so the next tick can fire.
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = None
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.ERROR, message="boom")
    )
    assert ss[preimage.PREIMAGE_WRITE_QUEUE] == []
    assert ss[preimage.PREIMAGE_KV_IN_FLIGHT] is False
    assert ss[preimage.PREIMAGE_INFLIGHT_WRITE] is None


def test_handler_list_response_advances_sweep_timestamp(
    handler_context: Any,
) -> None:
    """LIST_RESPONSE stamps PREIMAGE_LAST_SWEEP; an ERROR LIST does not."""
    # Stamping on send (which we used to do) made a transient LIST ERROR
    # indistinguishable from "nothing expired" — sweeping stalled for a
    # full preimage_sweep_interval. Now the stamp moves only on a successful
    # response, so a failed LIST keeps the next tick eligible to retry.
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    ss[preimage.PREIMAGE_LAST_SWEEP] = 0.0
    handler.handle(
        SimpleNamespace(
            performative=KvStoreMessage.Performative.LIST_RESPONSE, data={}, message=""
        )
    )
    assert ss[preimage.PREIMAGE_LAST_SWEEP] > 0.0
    # An ERROR (e.g. transient connection) doesn't move the clock.
    ss[preimage.PREIMAGE_LAST_SWEEP] = 1.0
    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.ERROR, message="boom")
    )
    assert ss[preimage.PREIMAGE_LAST_SWEEP] == 1.0


def test_process_buffer_resets_stuck_kv_in_flight(behaviour: Any) -> None:
    """The watchdog clears PREIMAGE_KV_IN_FLIGHT after the timeout."""
    # Without this, a lost SUCCESS/ERROR (connection drop, dropped envelope,
    # exception in the connection's send path) wedges the flag True forever
    # and the loop never drains again — PREIMAGE_RECORDS / PREIMAGE_WRITE_QUEUE
    # grow unbounded until restart, silently.
    behaviour.context.params.preimage_retention_enabled = True
    ss = behaviour.context.shared_state
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r1"
    # Stamp the in-flight time well in the past to trigger the watchdog.
    ss[preimage.PREIMAGE_INFLIGHT_SENT_AT] = time.time() - 999.0
    # Pin the sweep clock to recent so the watchdog reset doesn't fall
    # straight through into a fresh LIST (which would set in_flight=True
    # again and mask the reset).
    ss[preimage.PREIMAGE_LAST_SWEEP] = time.time()

    out = _CaptureOutbox()
    behaviour.context.outbox = out
    behaviour._process_preimage_buffer()

    assert ss[preimage.PREIMAGE_KV_IN_FLIGHT] is False
    assert ss[preimage.PREIMAGE_INFLIGHT_WRITE] is None
    assert ss[preimage.PREIMAGE_INFLIGHT_SENT_AT] is None
    # And no kv op was sent — the loop returned after the reset because no
    # work was queued.
    assert out.sent == []


def test_process_buffer_skips_none_record_and_continues(behaviour: Any) -> None:
    """A queued id whose record has been pruned is skipped, next id is flushed."""
    # Pinned because the same code path is the tail of the Critical race: if
    # the record were ever ``None``, the loop would silently move on, and a
    # bug here would lose preimages without surfacing.
    behaviour.context.params.preimage_retention_enabled = True
    ss = behaviour.context.shared_state
    ss[preimage.PREIMAGE_LAST_SWEEP] = time.time()
    # "r1" is queued but has no record (pruned). "r2" is real.
    ss[preimage.PREIMAGE_WRITE_QUEUE].extend(["r1", "r2"])
    preimage.record_accept(ss, "r2", "req2", now=time.time())
    # record_accept enqueued "r2"; remove the second entry to leave the
    # original ordering as ["r1", "r2"].
    ss[preimage.PREIMAGE_WRITE_QUEUE].remove("r2")
    ss[preimage.PREIMAGE_WRITE_QUEUE].append("r2")

    out = _CaptureOutbox()
    behaviour.context.outbox = out
    behaviour._process_preimage_buffer()

    # Exactly one write fires (for r2). r1 was skipped without sending.
    assert len(out.sent) == 1
    assert ss[preimage.PREIMAGE_INFLIGHT_WRITE] == "r2"


def test_buffer_settlement_is_noop_when_retention_disabled() -> None:
    """The settlement helper is a hard no-op when preimage retention is off."""
    # The two call sites in behaviours.py (off-chain success path + rejection
    # path) MUST not mutate shared_state when retention is disabled — a
    # regression that removed the guard would pollute the buffer on
    # retention-off deployments undetected.
    # Build a minimal behaviour-shaped object instead of pulling in the
    # whole TaskExecutionBehaviour fixture: the helper only touches
    # self.params and self.context.shared_state.
    state: Dict[str, Any] = {}
    behaviour = SimpleNamespace(
        params=SimpleNamespace(preimage_retention_enabled=False),
        context=SimpleNamespace(shared_state=state),
    )
    from packages.valory.skills.task_execution.behaviours import (  # noqa: PLC0415
        TaskExecutionBehaviour,
    )

    TaskExecutionBehaviour._buffer_preimage_settlement(
        behaviour,  # type: ignore[arg-type]
        "r1",
        "resp",
        "cid",
        preimage.STATUS_DELIVERED,
    )
    assert state == {}
