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
        next_cursor="",
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
            performative=KvStoreMessage.Performative.LIST_RESPONSE,
            data={},
            next_cursor="",
            message="",
        )
    )
    assert ss[preimage.PREIMAGE_LAST_SWEEP] > 0.0
    # An ERROR (e.g. transient connection) doesn't move the clock.
    ss[preimage.PREIMAGE_LAST_SWEEP] = 1.0
    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.ERROR, message="boom")
    )
    assert ss[preimage.PREIMAGE_LAST_SWEEP] == 1.0


def test_process_buffer_resets_stuck_kv_in_flight_and_resends(
    behaviour: Any,
) -> None:
    """Watchdog clears PREIMAGE_KV_IN_FLIGHT, re-enqueues the write, retries it."""
    # Without this, a lost SUCCESS/ERROR (connection drop, dropped envelope,
    # exception in the connection's send path) wedges the flag True forever
    # and the loop never drains again — PREIMAGE_RECORDS / PREIMAGE_WRITE_QUEUE
    # grow unbounded until restart, silently. AND a terminal in-flight write
    # would be lost on reset (the double-settle guard blocks the only other
    # re-enqueue path), so the watchdog must put it back in the queue. The
    # observable post-reset state is: a fresh CREATE_OR_UPDATE for the same
    # id was sent and the in-flight stamp is now recent (not 999s in the
    # past).
    behaviour.context.params.preimage_retention_enabled = True
    ss = behaviour.context.shared_state
    preimage.record_settlement(
        ss, "r1", "resp", "cid", preimage.STATUS_DELIVERED, now=1.0
    )
    # Simulate "behaviour sent it, then nothing came back": the id is no
    # longer in the queue and is sitting in the in-flight slot.
    ss[preimage.PREIMAGE_WRITE_QUEUE].remove("r1")
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r1"
    stale = time.time() - 999.0
    ss[preimage.PREIMAGE_INFLIGHT_SENT_AT] = stale
    # Pin the sweep clock to recent so the post-reset fall-through goes to
    # the write queue (not into a fresh LIST that would mask the retry).
    ss[preimage.PREIMAGE_LAST_SWEEP] = time.time()

    out = _CaptureOutbox()
    behaviour.context.outbox = out
    behaviour._process_preimage_buffer()

    # Fall-through after the watchdog reset picks the re-enqueued id off the
    # queue and re-sends it. End state: fresh in-flight, new stamp, queue
    # drained.
    assert len(out.sent) == 1
    assert out.sent[0].performative == (
        KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST
    )
    assert "mech_preimage/r1" in out.sent[0].data
    assert ss[preimage.PREIMAGE_INFLIGHT_WRITE] == "r1"
    assert ss[preimage.PREIMAGE_KV_IN_FLIGHT] is True
    assert ss[preimage.PREIMAGE_INFLIGHT_SENT_AT] > stale
    assert ss[preimage.PREIMAGE_WRITE_QUEUE] == []
    # And the record is still in memory — not lost.
    assert "r1" in ss[preimage.PREIMAGE_RECORDS]


def test_process_buffer_watchdog_resets_when_inflight_is_none(
    behaviour: Any,
) -> None:
    """Watchdog handles the LIST/DELETE case + preserves PREIMAGE_INFLIGHT_DIALOGUE."""
    # LIST/DELETE don't populate PREIMAGE_INFLIGHT_WRITE; a lost reply
    # on those should still clear the gate but has nothing to re-queue.
    # Also pins that PREIMAGE_INFLIGHT_DIALOGUE is NOT cleared by the
    # watchdog: a future cleanup that mirrors the other clears alongside
    # PREIMAGE_INFLIGHT_OP and PREIMAGE_INFLIGHT_SENT_AT would look
    # obviously correct, pass every other test, and silently hollow out
    # the late-reply guard for whatever op runs next (no stored ref ⇒
    # guard skips ⇒ any reply gets attributed to the new op).
    behaviour.context.params.preimage_retention_enabled = True
    ss = behaviour.context.shared_state
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = None
    ss[preimage.PREIMAGE_INFLIGHT_SENT_AT] = time.time() - 999.0
    ss[preimage.PREIMAGE_INFLIGHT_DIALOGUE] = ("stuck-op-nonce", "")
    ss[preimage.PREIMAGE_LAST_SWEEP] = time.time()

    out = _CaptureOutbox()
    behaviour.context.outbox = out
    behaviour._process_preimage_buffer()

    assert ss[preimage.PREIMAGE_KV_IN_FLIGHT] is False
    assert ss[preimage.PREIMAGE_INFLIGHT_SENT_AT] is None
    assert ss[preimage.PREIMAGE_WRITE_QUEUE] == []
    assert out.sent == []
    # Critical: the watchdog must preserve the dialogue ref so a slow
    # arriving reply for the timed-out op is still rejected by the
    # late-reply guard until the next _send_kv_* overwrites it.
    assert ss[preimage.PREIMAGE_INFLIGHT_DIALOGUE] == ("stuck-op-nonce", "")


def test_process_buffer_keeps_paging_until_cursor_clears(behaviour: Any) -> None:
    """A non-empty PREIMAGE_LIST_CURSOR forces _send_kv_list on the next tick."""
    # If the loop only fired LIST on the sweep_interval timer, a namespace
    # bigger than one page would only get the first page walked per sweep
    # window — retention pruning would lag forever on a busy mech. The
    # cursor short-circuit forces back-to-back LISTs until the namespace is
    # fully walked.
    behaviour.context.params.preimage_retention_enabled = True
    ss = behaviour.context.shared_state
    # Sweep ran very recently AND the cursor is set: the time gate would
    # block, but the cursor must override.
    ss[preimage.PREIMAGE_LAST_SWEEP] = time.time()
    ss[preimage.PREIMAGE_LIST_CURSOR] = "page-2-cursor"

    out = _CaptureOutbox()
    behaviour.context.outbox = out
    behaviour._process_preimage_buffer()

    assert len(out.sent) == 1
    assert out.sent[0].performative == KvStoreMessage.Performative.LIST_REQUEST
    assert out.sent[0].cursor == "page-2-cursor"
    assert out.sent[0].limit == behaviour.context.params.preimage_list_page_size


def test_handler_list_response_mid_page_saves_cursor_only(
    handler_context: Any,
) -> None:
    """A LIST_RESPONSE with non-empty next_cursor saves cursor, leaves clock alone."""
    # Stamping LAST_SWEEP mid-walk would let the time gate elide later pages
    # the very next tick after the cursor finally clears (since "now" is
    # already > now+0); the sweep would also re-LIST page 1 immediately
    # rather than continuing. Both paths break retention. Only the final
    # page (next_cursor == "") advances the clock.
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    ss[preimage.PREIMAGE_LAST_SWEEP] = 0.0

    handler.handle(
        SimpleNamespace(
            performative=KvStoreMessage.Performative.LIST_RESPONSE,
            data={},
            next_cursor="page-2-cursor",
            message="",
        )
    )

    assert ss[preimage.PREIMAGE_LIST_CURSOR] == "page-2-cursor"
    # Clock untouched: the sweep is still in progress.
    assert ss[preimage.PREIMAGE_LAST_SWEEP] == 0.0

    # Final page: cursor clears AND clock advances together.
    handler.handle(
        SimpleNamespace(
            performative=KvStoreMessage.Performative.LIST_RESPONSE,
            data={},
            next_cursor="",
            message="",
        )
    )
    assert ss[preimage.PREIMAGE_LIST_CURSOR] is None
    assert ss[preimage.PREIMAGE_LAST_SWEEP] > 0.0


def test_handler_error_drops_record_after_max_attempts(
    handler_context: Any,
) -> None:
    """A persistently failing kv_store write is dropped after the configured cap."""
    # An unhealthy kv_store would otherwise hot-loop the agent retrying the
    # same record forever, pinning the in-flight gate and starving sweeps +
    # new writes. After preimage_max_write_attempts ERRORs we drop the
    # record + WARN. The buffer is a best-effort audit copy, so a bounded
    # loss beats an unbounded stall.
    handler_context.params.preimage_max_write_attempts = 3
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    preimage.record_accept(ss, "r1", "req", now=1.0)

    # First two ERRORs keep the record around and re-queue it.
    for _ in range(2):
        ss[preimage.PREIMAGE_WRITE_QUEUE].remove("r1")
        ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r1"
        ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
        handler.handle(
            SimpleNamespace(
                performative=KvStoreMessage.Performative.ERROR, message="boom"
            )
        )
        assert "r1" in ss[preimage.PREIMAGE_RECORDS]
        assert ss[preimage.PREIMAGE_WRITE_QUEUE] == ["r1"]

    # Third ERROR hits the cap: record + attempts entry both purged, queue
    # left empty (no re-queue this round).
    ss[preimage.PREIMAGE_WRITE_QUEUE].remove("r1")
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r1"
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.ERROR, message="boom")
    )
    assert "r1" not in ss[preimage.PREIMAGE_RECORDS]
    assert ss[preimage.PREIMAGE_WRITE_QUEUE] == []
    assert "r1" not in ss[preimage.PREIMAGE_WRITE_ATTEMPTS]


def test_watchdog_counts_toward_write_attempts_and_drops_at_cap(
    behaviour: Any,
) -> None:
    """Watchdog timeouts increment PREIMAGE_WRITE_ATTEMPTS and drop at the cap."""
    # Without this, a kv_store stuck in the no-reply failure mode would
    # bypass the ERROR-branch retry cap (the watchdog re-enqueues but never
    # increments) and retry the same write forever — exactly the unbounded
    # hot-loop the cap was added to close, just via timeout instead of ERROR.
    behaviour.context.params.preimage_retention_enabled = True
    behaviour.context.params.preimage_max_write_attempts = 2
    ss = behaviour.context.shared_state
    preimage.record_settlement(
        ss, "r1", "resp", "cid", preimage.STATUS_DELIVERED, now=1.0
    )

    # Tick 1: watchdog fires for the first time, attempts -> 1 (< cap=2),
    # so the record is re-enqueued, fall-through re-sends it.
    ss[preimage.PREIMAGE_WRITE_QUEUE].remove("r1")
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r1"
    ss[preimage.PREIMAGE_INFLIGHT_SENT_AT] = time.time() - 999.0
    ss[preimage.PREIMAGE_LAST_SWEEP] = time.time()

    out = _CaptureOutbox()
    behaviour.context.outbox = out
    behaviour._process_preimage_buffer()
    assert ss[preimage.PREIMAGE_WRITE_ATTEMPTS]["r1"] == 1
    assert "r1" in ss[preimage.PREIMAGE_RECORDS]
    assert len(out.sent) == 1

    # Tick 2: watchdog fires again, attempts -> 2 (== cap), record + counter
    # purged, no fresh send because the queue is empty.
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r1"
    ss[preimage.PREIMAGE_INFLIGHT_SENT_AT] = time.time() - 999.0
    ss[preimage.PREIMAGE_LAST_SWEEP] = time.time()
    out2 = _CaptureOutbox()
    behaviour.context.outbox = out2
    behaviour._process_preimage_buffer()
    assert "r1" not in ss[preimage.PREIMAGE_RECORDS]
    assert "r1" not in ss[preimage.PREIMAGE_WRITE_ATTEMPTS]
    assert out2.sent == []


def test_handler_list_error_caps_consecutive_failures(handler_context: Any) -> None:
    """N consecutive LIST ERRORs clear cursor + stamp LAST_SWEEP + reset counter."""
    # Pre-cap: a persistently failing kv_store would re-LIST every act()
    # tick (no in-flight write to retry, so the write-counter doesn't fire).
    # At the cap the cursor is cleared and LAST_SWEEP is stamped so the
    # next sweep_interval is the natural backoff. The counter must reset
    # after the cap so future sweeps start fresh.
    handler_context.params.preimage_max_list_attempts = 3
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    ss[preimage.PREIMAGE_LIST_CURSOR] = "page-2-cursor"
    ss[preimage.PREIMAGE_LAST_SWEEP] = 0.0

    # First two LIST ERRORs only bump the counter — cursor and clock left alone
    # so the next tick keeps trying.
    for expected in (1, 2):
        ss[preimage.PREIMAGE_INFLIGHT_OP] = preimage.OP_LIST
        ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
        handler.handle(
            SimpleNamespace(
                performative=KvStoreMessage.Performative.ERROR, message="boom"
            )
        )
        assert ss[preimage.PREIMAGE_LIST_ATTEMPTS] == expected
        assert ss[preimage.PREIMAGE_LIST_CURSOR] == "page-2-cursor"
        assert ss[preimage.PREIMAGE_LAST_SWEEP] == 0.0

    # Third LIST ERROR hits the cap: cursor cleared, clock advanced, counter
    # reset so the next sweep window starts from a clean slate.
    ss[preimage.PREIMAGE_INFLIGHT_OP] = preimage.OP_LIST
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.ERROR, message="boom")
    )
    assert ss[preimage.PREIMAGE_LIST_CURSOR] is None
    assert ss[preimage.PREIMAGE_LAST_SWEEP] > 0.0
    assert ss[preimage.PREIMAGE_LIST_ATTEMPTS] == 0


def test_handler_list_response_resets_attempt_counter(handler_context: Any) -> None:
    """A successful LIST_RESPONSE resets PREIMAGE_LIST_ATTEMPTS to 0."""
    # Counters must be consecutive, not cumulative — a transient failure
    # early in a long-running deployment shouldn't push the sweeper one
    # ERROR away from the cap forever after.
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    ss[preimage.PREIMAGE_LIST_ATTEMPTS] = 4
    handler.handle(
        SimpleNamespace(
            performative=KvStoreMessage.Performative.LIST_RESPONSE,
            data={},
            next_cursor="",
            message="",
        )
    )
    assert ss[preimage.PREIMAGE_LIST_ATTEMPTS] == 0


def test_delete_queue_chunked_to_page_size_per_tick(behaviour: Any) -> None:
    """A backlog larger than page_size drains in slices, not one giant DELETE."""
    # An unbounded DELETE would exceed SQLite's bound-parameter cap
    # (SQLITE_MAX_VARIABLE_NUMBER, typically 999/32766) once the backlog is
    # large — an agent down longer than the retention window comes back,
    # everything expires at once, and a single bulk DELETE fails outright.
    # Failed deletes aren't retried, so the next sweep would rebuild the
    # identical oversized batch: retention pruning would permanently stall
    # and the db would grow without bound. Slicing prevents that.
    behaviour.context.params.preimage_retention_enabled = True
    behaviour.context.params.preimage_list_page_size = 3
    ss = behaviour.context.shared_state
    ss[preimage.PREIMAGE_LAST_SWEEP] = time.time()
    backlog = [f"mech_preimage/k{i}" for i in range(7)]
    ss[preimage.PREIMAGE_DELETE_QUEUE].extend(backlog)

    out = _CaptureOutbox()
    behaviour.context.outbox = out
    behaviour._process_preimage_buffer()

    assert len(out.sent) == 1
    assert out.sent[0].performative == KvStoreMessage.Performative.DELETE_REQUEST
    # First slice fires; remaining ids stay queued in original order for
    # subsequent ticks to drain.
    assert set(out.sent[0].keys) == set(backlog[:3])
    assert ss[preimage.PREIMAGE_DELETE_QUEUE] == backlog[3:]


def test_handler_success_clears_attempt_counter(handler_context: Any) -> None:
    """A successful terminal write clears the per-id retry counter."""
    # Without this, the counter persists past the record being popped —
    # the next re-use of the same request_id would start with a stale
    # count and could hit the cap on the very first ERROR. Keep counters
    # bounded by the live record set, not by lifetime traffic.
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    ss[preimage.PREIMAGE_WRITE_ATTEMPTS]["r1"] = 4

    preimage.record_settlement(
        ss, "r1", "resp", "cid", preimage.STATUS_DELIVERED, now=1.0
    )
    ss[preimage.PREIMAGE_WRITE_QUEUE].remove("r1")
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r1"
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.SUCCESS, message="ok")
    )
    assert "r1" not in ss[preimage.PREIMAGE_WRITE_ATTEMPTS]


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


def test_watchdog_caps_list_no_reply(behaviour: Any) -> None:
    """A LIST that times out N times in a row clears cursor + stamps LAST_SWEEP."""
    # Without this, a hung kv_store (replies stop coming back, doesn't
    # error) would loop _send_kv_list every PREIMAGE_KV_REQUEST_TIMEOUT
    # forever — the LIST ERROR cap closed the same hole for the ERROR
    # path; this closes it for the no-reply path.
    behaviour.context.params.preimage_retention_enabled = True
    behaviour.context.params.preimage_max_list_attempts = 2
    ss = behaviour.context.shared_state
    ss[preimage.PREIMAGE_LIST_CURSOR] = "page-2-cursor"
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    ss[preimage.PREIMAGE_INFLIGHT_OP] = preimage.OP_LIST
    ss[preimage.PREIMAGE_INFLIGHT_SENT_AT] = time.time() - 999.0
    ss[preimage.PREIMAGE_INFLIGHT_DIALOGUE] = ("stuck-list-nonce", "")
    ss[preimage.PREIMAGE_LAST_SWEEP] = 0.0

    out = _CaptureOutbox()
    behaviour.context.outbox = out
    behaviour._process_preimage_buffer()
    # First timeout: counter bumps to 1 (< cap=2). Cursor still set,
    # clock untouched, fall-through re-fires LIST.
    assert ss[preimage.PREIMAGE_LIST_ATTEMPTS] == 1
    assert ss[preimage.PREIMAGE_LIST_CURSOR] == "page-2-cursor"
    assert ss[preimage.PREIMAGE_LAST_SWEEP] == 0.0
    assert len(out.sent) == 1
    assert out.sent[0].performative == KvStoreMessage.Performative.LIST_REQUEST

    # Second timeout: counter hits cap → cursor cleared + LAST_SWEEP
    # stamped → next tick falls through the time gate, no LIST fires.
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    ss[preimage.PREIMAGE_INFLIGHT_OP] = preimage.OP_LIST
    ss[preimage.PREIMAGE_INFLIGHT_SENT_AT] = time.time() - 999.0
    ss[preimage.PREIMAGE_INFLIGHT_DIALOGUE] = ("stuck-list-nonce-2", "")
    out2 = _CaptureOutbox()
    behaviour.context.outbox = out2
    behaviour._process_preimage_buffer()
    assert ss[preimage.PREIMAGE_LIST_CURSOR] is None
    assert ss[preimage.PREIMAGE_LAST_SWEEP] > 0.0
    assert ss[preimage.PREIMAGE_LIST_ATTEMPTS] == 0
    assert out2.sent == []
    # The watchdog must NOT clear PREIMAGE_INFLIGHT_DIALOGUE — clearing
    # it would let a slow-arriving reply for the timed-out op match the
    # next op's guard (no stored ref ⇒ guard skips). Keeping the stale
    # ref means the dialogue-mismatch check in KvStoreHandler still
    # rejects late replies until the next _send_kv_* overwrites it.
    assert ss[preimage.PREIMAGE_INFLIGHT_DIALOGUE] == ("stuck-list-nonce-2", "")


def test_expired_keys_treats_non_numeric_stamp_as_skipped() -> None:
    """A row with a non-numeric timestamp goes to skipped, not raised."""
    # Without the try-wrapped float(), a poison-pill row with
    # {"settled_at": "oops"} crashes the sweeper on every cycle —
    # retention pruning permanently stalls and the agent throws on
    # every act() tick. Defensive: not reachable from current writers,
    # but the docstring promises rows we don't understand are left
    # alone, and a future writer regression must not bypass that.
    keys = preimage.expired_keys(
        {
            "mech_preimage/poison": '{"settled_at": "oops"}',
            "mech_preimage/old": '{"settled_at": 0}',
        },
        now=999999.0,
        retention_seconds=10,
    )
    assert keys == ["mech_preimage/old"]


def test_handler_delete_error_logs_dropped_count_and_keeps_list_counter(
    handler_context: Any,
) -> None:
    """A DELETE ERROR logs its batch size and doesn't borrow the LIST counter."""
    # DELETE failures self-heal via the next sweep (keys are still in
    # kv, will be re-LISTed and re-queued), but without the OP_DELETE
    # arm the only signal is a generic "kv_store error" WARN — operators
    # can't see DELETE failures distinctly from WRITE/LIST. Also pins
    # that DELETE doesn't accidentally bump PREIMAGE_LIST_ATTEMPTS,
    # which would let DELETE failures push the LIST sweeper toward its
    # own cap and stall an otherwise-healthy LIST path.
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    ss[preimage.PREIMAGE_INFLIGHT_OP] = preimage.OP_DELETE
    ss[preimage.PREIMAGE_INFLIGHT_DELETE_COUNT] = 7
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    pre_list_attempts = ss[preimage.PREIMAGE_LIST_ATTEMPTS]

    warnings: List[str] = []
    handler_context.logger.warning = lambda msg, *a, **k: warnings.append(msg % a)

    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.ERROR, message="boom")
    )
    # LIST counter not bumped — DELETE has its own arm, not the LIST cap.
    assert ss[preimage.PREIMAGE_LIST_ATTEMPTS] == pre_list_attempts
    # Dropped-count is in the WARN somewhere.
    assert any("7 expired key(s)" in w for w in warnings)
    # In-flight cleared as usual.
    assert ss[preimage.PREIMAGE_KV_IN_FLIGHT] is False
    assert ss[preimage.PREIMAGE_INFLIGHT_DELETE_COUNT] == 0


def test_record_settlement_rejects_non_terminal_status() -> None:
    """Settling with STATUS_PROCESSING fails the assert (would leak in PREIMAGE_RECORDS)."""
    # Without the assert, callers that accidentally pass
    # STATUS_PROCESSING write a non-terminal record that the SUCCESS
    # handler never pops (it only pops TERMINAL_STATUSES), leaking the
    # id in memory until restart. Make that unrepresentable rather than
    # a silent leak.
    state: Dict[str, Any] = {}
    preimage.init_shared_state(state)
    with pytest.raises(AssertionError):
        preimage.record_settlement(
            state, "r1", "resp", "cid", preimage.STATUS_PROCESSING, now=1.0
        )


def test_handler_ignores_reply_from_previously_timed_out_op(
    handler_context: Any,
) -> None:
    """A late reply whose initiator nonce doesn't match the in-flight op is dropped."""
    # Scenario the guard exists for: kv_store stalls past the watchdog,
    # the watchdog clears the gate and the next op starts; the OLD
    # reply finally lands. Without the guard, it'd be processed as if
    # for the new op — wrong counter incremented, gate cleared while
    # the new op is still in flight. Mismatch on initiator nonce →
    # ignore the reply + leave all current op state untouched.
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    # The new in-flight op has initiator "new-op-nonce" (empty responder
    # slot in the send-time stamp, as real open-aea produces).
    ss[preimage.PREIMAGE_INFLIGHT_DIALOGUE] = ("new-op-nonce", "")
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    ss[preimage.PREIMAGE_INFLIGHT_OP] = preimage.OP_WRITE
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r-new"

    # The conftest _KvStoreDLG.update() returns ("nonce-1", "responder-ref"),
    # initiator "nonce-1" — doesn't match "new-op-nonce", so the reply is
    # for a previously-timed-out op and must be ignored.
    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.SUCCESS, message="ok")
    )
    assert ss[preimage.PREIMAGE_INFLIGHT_DIALOGUE] == ("new-op-nonce", "")
    assert ss[preimage.PREIMAGE_KV_IN_FLIGHT] is True
    assert ss[preimage.PREIMAGE_INFLIGHT_OP] == preimage.OP_WRITE
    assert ss[preimage.PREIMAGE_INFLIGHT_WRITE] == "r-new"


def test_handler_accepts_reply_when_initiator_matches(handler_context: Any) -> None:
    """A reply for the current in-flight op is processed normally."""
    # Regression test for the bug where the guard compared full
    # dialogue tuples — the send-time stamp is ``(nonce, "")`` but the
    # incoming reply carries ``(nonce, responder_ref)`` once open-aea
    # completes the responder slot. A full-tuple comparison rejected
    # every real reply, hollowing out retention pruning entirely. The
    # guard must compare ONLY the initiator nonce.
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    preimage.record_settlement(
        ss, "r1", "resp", "cid", preimage.STATUS_DELIVERED, now=1.0
    )
    ss[preimage.PREIMAGE_WRITE_QUEUE].remove("r1")
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r1"
    ss[preimage.PREIMAGE_INFLIGHT_OP] = preimage.OP_WRITE
    # Send-time stamp: matches what _send_kv_* would stamp via
    # ``dlg.dialogue_label.dialogue_reference`` from the create() stub
    # (responder slot empty).
    ss[preimage.PREIMAGE_INFLIGHT_DIALOGUE] = ("nonce-1", "")
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True

    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.SUCCESS, message="ok")
    )
    # The reply was accepted and processed: terminal record popped,
    # in-flight gate cleared.
    assert "r1" not in ss[preimage.PREIMAGE_RECORDS]
    assert ss[preimage.PREIMAGE_KV_IN_FLIGHT] is False
    assert ss[preimage.PREIMAGE_INFLIGHT_DIALOGUE] is None


def test_handler_drops_unrecognised_message_without_touching_state(
    handler_context: Any, monkeypatch: Any
) -> None:
    """A message whose dialogue can't be matched returns early, no cleanup."""
    # Pinned because the alternative — letting cleanup run on a dropped
    # message — would zero PREIMAGE_INFLIGHT_DIALOGUE and hollow out
    # the late-reply guard for the next real op (no expected ref ⇒
    # guard skips ⇒ any reply gets attributed to the current op).
    handler = _handler(handler_context)
    ss = handler_context.shared_state
    ss[preimage.PREIMAGE_INFLIGHT_DIALOGUE] = ("real-nonce", "")
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True

    # Force dialogues.update() to return None (open-aea's signal for
    # "no matching dialogue"), the exact case the early-return guards.
    monkeypatch.setattr(handler_context.kv_store_dialogues, "update", lambda _msg: None)

    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.SUCCESS, message="ok")
    )
    assert ss[preimage.PREIMAGE_INFLIGHT_DIALOGUE] == ("real-nonce", "")
    assert ss[preimage.PREIMAGE_KV_IN_FLIGHT] is True
