# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2025 Valory AG
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
import json
import time
import urllib.parse
from types import SimpleNamespace

import packages.valory.skills.task_execution.handlers as hmod
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.http.message import HttpMessage
from packages.valory.protocols.ipfs import IpfsMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.task_execution.handlers import (
    ContractHandler,
    HttpCode,
    IpfsHandler,
    LedgerHandler,
    MechHttpHandler,
)


def test_ipfs_handler_error_sets_flags(handler_context):
    handler = IpfsHandler(name="ipfs", skill_context=handler_context)
    handler.setup()
    # simulate in-flight
    handler_context.params.in_flight_req = True
    # message with ERROR performative
    msg = SimpleNamespace(performative=IpfsMessage.Performative.ERROR)
    handler.handle(msg)
    assert handler_context.params.in_flight_req is False


def test_ipfs_handler_calls_callback_and_clears(handler_context, monkeypatch):
    handler = IpfsHandler(name="ipfs", skill_context=handler_context)
    handler.setup()

    called = {"ok": False}
    # map nonce -> callback and deadline in the future
    handler_context.params.req_to_callback[
        "nonce-1"
    ] = lambda msg, dlg: called.__setitem__("ok", True)
    handler_context.params.req_to_deadline["nonce-1"] = time.time() + 999
    handler_context.params.is_cold_start = True
    handler_context.params.in_flight_req = True

    msg = SimpleNamespace(performative=IpfsMessage.Performative.GET_FILES)
    handler.handle(msg)

    assert called["ok"] is True
    assert handler_context.params.in_flight_req is False
    assert handler_context.params.is_cold_start is False
    assert "nonce-1" not in handler_context.params.req_to_callback
    assert "nonce-1" not in handler_context.params.req_to_deadline


def test_ipfs_handler_deadline_expired_skips_callback(handler_context):
    handler = IpfsHandler(name="ipfs", skill_context=handler_context)
    handler.setup()

    called = {"ok": False}
    handler_context.params.req_to_callback["nonce-1"] = lambda *_: called.__setitem__(
        "ok", True
    )
    handler_context.params.req_to_deadline["nonce-1"] = (
        time.time() - 1
    )  # already expired
    handler_context.params.in_flight_req = True
    handler_context.params.is_cold_start = True

    msg = SimpleNamespace(performative=IpfsMessage.Performative.STORE_FILES)
    handler.handle(msg)

    assert called["ok"] is False
    assert handler_context.params.in_flight_req is False
    assert handler_context.params.is_cold_start is False
    # entries are popped even on timeout
    assert "nonce-1" not in handler_context.params.req_to_callback
    assert "nonce-1" not in handler_context.params.req_to_deadline


def test_contract_handler_setup_initializes_shared(handler_context):
    ch = ContractHandler(name="contract", skill_context=handler_context)
    ch.setup()
    ss = handler_context.shared_state
    assert hmod.PENDING_TASKS in ss and isinstance(ss[hmod.PENDING_TASKS], list)
    assert hmod.DONE_TASKS in ss and isinstance(ss[hmod.DONE_TASKS], list)
    assert hmod.DONE_TASKS_LOCK in ss
    assert hmod.REQUEST_ID_TO_DELIVERY_RATE_INFO in ss and isinstance(
        ss[hmod.REQUEST_ID_TO_DELIVERY_RATE_INFO], dict
    )


def test_contract_handler_state_enqueues_and_updates_from_block(handler_context):
    params = handler_context.params
    params.in_flight_req = True
    params.num_agents = 2
    params.agent_index = 1
    params.req_type = "legacy"
    params.req_params.from_block["legacy"] = 0

    ch = ContractHandler(name="contract", skill_context=handler_context)
    ch.setup()

    # Two reqs; only block_number % 2 == 1 should be kept (agent_index=1)
    reqs = [
        {"block_number": 10, "requestId": 1},
        {"block_number": 11, "requestId": 2},
    ]
    body = {"data": reqs}
    msg = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=body),
    )

    ch.handle(msg)

    # from_block updated to max+1
    assert params.req_params.from_block["legacy"] == 12
    # Only one req enqueued (block 11)
    assert len(ch.pending_tasks) == 1
    assert ch.pending_tasks[0]["requestId"] == 2
    assert params.in_flight_req is False


def test_contract_handler_non_state_sets_flag(handler_context):
    params = handler_context.params
    params.in_flight_req = True
    ch = ContractHandler(name="contract", skill_context=handler_context)
    ch.setup()
    msg = SimpleNamespace(performative=ContractApiMessage.Performative.GET_STATE)
    ch.handle(msg)
    assert params.in_flight_req is False


def test_ledger_handler_updates_from_block(handler_context):
    params = handler_context.params
    params.in_flight_req = True
    params.req_type = "legacy"
    params.from_block_range = 500

    lh = LedgerHandler(name="ledger", skill_context=handler_context)
    lh.setup()

    msg = SimpleNamespace(
        performative=LedgerApiMessage.Performative.STATE,
        state=SimpleNamespace(body={"number": 12345}),
    )
    lh.handle(msg)

    assert params.req_params.from_block["legacy"] == 12345 - 500
    assert params.in_flight_req is False


def test_ledger_handler_non_state_sets_flag(handler_context):
    params = handler_context.params
    params.in_flight_req = True
    lh = LedgerHandler(name="ledger", skill_context=handler_context)
    lh.setup()
    msg = SimpleNamespace(performative=LedgerApiMessage.Performative.GET_STATE)
    lh.handle(msg)
    assert params.in_flight_req is False


def make_http_msg(body_dict: dict, headers=""):
    # They do parse_qs on the decoded body
    body = urllib.parse.urlencode(body_dict).encode("utf-8")
    return SimpleNamespace(
        body=body,
        version="1.1",
        headers=headers,
        performative=HttpMessage.Performative.REQUEST,
    )


def test_signed_requests_success(handler_context, http_dialogue):
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    mh.setup()

    # minimal valid fields
    ipfs_hash = "0x" + "ab" * 64  # even-length hex
    body = {
        "ipfs_hash": ipfs_hash,
        "request_id": "req-1",
        "ipfs_data": '{"foo":"bar"}',
        "delivery_rate": "123",
    }
    http_msg = make_http_msg(body)
    mh._handle_signed_requests(http_msg, http_dialogue)  # call directly

    # pending & ipfs_tasks updated
    pend = handler_context.shared_state["pending_tasks"]
    ipfsq = handler_context.shared_state["ipfs_tasks"]
    assert len(pend) == 1 and len(ipfsq) == 1
    assert pend[0]["is_offchain"] is True
    assert pend[0]["requestId"] == "req-1"
    assert ipfsq[0]["request_id"] == "req-1"

    # response sent
    assert handler_context.outbox.sent, "no HTTP response sent"
    resp = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.OK_CODE.value
    data = json.loads(resp.body.decode("utf-8"))
    assert data["request_id"] == "req-1"


def test_signed_requests_bad_request(handler_context, http_dialogue):
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    mh.setup()
    # missing fields → KeyError → handled as bad request
    http_msg = make_http_msg({"only": "one"})
    mh._handle_signed_requests(http_msg, http_dialogue)

    resp = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.BAD_REQUEST_CODE.value


def test_fetch_offchain_request_info_found(handler_context, http_dialogue):
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    mh.setup()
    # Seed a done_task with string request_id (handler compares as string)
    handler_context.shared_state["ready_tasks"].append(
        {"request_id": "abc", "value": 7}
    )
    http_msg = make_http_msg({"request_id": "abc"})
    mh._handle_offchain_request_info(http_msg, http_dialogue)

    resp = handler_context.outbox.sent[-1]
    payload = json.loads(resp.body.decode("utf-8"))
    assert payload["request_id"] == "abc" and payload["value"] == 7


def test_fetch_offchain_request_info_not_found(handler_context, http_dialogue):
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    mh.setup()
    http_msg = make_http_msg({"request_id": "missing"})
    mh._handle_offchain_request_info(http_msg, http_dialogue)

    resp = handler_context.outbox.sent[-1]
    payload = json.loads(resp.body.decode("utf-8"))
    assert payload == {}


def test_on_message_handled_triggers_cleanup(handler_context, monkeypatch):
    # Set cleanup every request
    handler_context.params.cleanup_freq = 1

    # Track cleanup calls on dialogues created from handlers listing
    class Handlers:
        pass

    handler_context.handlers = Handlers()
    handler_context.handlers.ipfs_handler = object()

    cleaned = {"ipfs": 0}
    handler_context.ipfs_dialogues = SimpleNamespace(
        cleanup=lambda: cleaned.__setitem__("ipfs", cleaned["ipfs"] + 1)
    )

    h = IpfsHandler(name="ipfs", skill_context=handler_context)
    h.setup()
    h.on_message_handled(SimpleNamespace())  # should trigger cleanup

    assert cleaned["ipfs"] == 1
