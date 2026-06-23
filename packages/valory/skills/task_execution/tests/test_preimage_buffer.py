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
from typing import Any, Generator, List

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
    ss[preimage.PREIMAGE_INFLIGHT_WRITE] = "r1"
    ss[preimage.PREIMAGE_KV_IN_FLIGHT] = True
    handler.handle(
        SimpleNamespace(performative=KvStoreMessage.Performative.SUCCESS, message="ok")
    )
    assert "r1" in ss[preimage.PREIMAGE_RECORDS]


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
