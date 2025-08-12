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

"""This package contains the tests for the handlers."""

import json
import time
import urllib.parse
from types import SimpleNamespace
from typing import Any, Dict, List

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


def test_ipfs_handler_error_sets_flags(handler_context: Any) -> None:
    """Clear `in_flight_req` when the IPFS handler receives an ERROR."""
    handler: IpfsHandler = IpfsHandler(name="ipfs", skill_context=handler_context)
    handler.setup()

    handler_context.params.in_flight_req = True
    msg: SimpleNamespace = SimpleNamespace(performative=IpfsMessage.Performative.ERROR)

    handler.handle(msg)

    assert handler_context.params.in_flight_req is False


def test_ipfs_handler_calls_callback_and_clears(
    handler_context: Any, monkeypatch: Any
) -> None:
    """Invoke the stored IPFS callback and clear bookkeeping afterward."""
    handler: IpfsHandler = IpfsHandler(name="ipfs", skill_context=handler_context)
    handler.setup()

    called = {"ok": False}
    handler_context.params.req_to_callback[
        "nonce-1"
    ] = lambda _msg, _dlg: called.__setitem__("ok", True)
    handler_context.params.req_to_deadline["nonce-1"] = time.time() + 999.0
    handler_context.params.is_cold_start = True
    handler_context.params.in_flight_req = True

    msg: SimpleNamespace = SimpleNamespace(
        performative=IpfsMessage.Performative.GET_FILES
    )

    handler.handle(msg)

    assert called["ok"] is True
    assert handler_context.params.in_flight_req is False
    assert handler_context.params.is_cold_start is False
    assert "nonce-1" not in handler_context.params.req_to_callback
    assert "nonce-1" not in handler_context.params.req_to_deadline


def test_ipfs_handler_deadline_expired_skips_callback(handler_context: Any) -> None:
    """Skip the stored IPFS callback when its deadline has already expired."""
    handler: IpfsHandler = IpfsHandler(name="ipfs", skill_context=handler_context)
    handler.setup()

    called = {"ok": False}
    handler_context.params.req_to_callback["nonce-1"] = lambda *_: called.__setitem__(
        "ok", True
    )
    handler_context.params.req_to_deadline["nonce-1"] = time.time() - 1.0
    handler_context.params.in_flight_req = True
    handler_context.params.is_cold_start = True

    msg: SimpleNamespace = SimpleNamespace(
        performative=IpfsMessage.Performative.STORE_FILES
    )
    handler.handle(msg)

    assert called["ok"] is False
    assert handler_context.params.in_flight_req is False
    assert handler_context.params.is_cold_start is False
    assert "nonce-1" not in handler_context.params.req_to_callback
    assert "nonce-1" not in handler_context.params.req_to_deadline


def test_contract_handler_setup_initializes_shared(
    handler_context: SimpleNamespace,
) -> None:
    """Initialize `shared_state` collections on setup."""
    ch: ContractHandler = ContractHandler(
        name="contract", skill_context=handler_context
    )
    ch.setup()

    ss: Dict[str, Any] = handler_context.shared_state
    assert hmod.PENDING_TASKS in ss and isinstance(ss[hmod.PENDING_TASKS], list)
    assert hmod.DONE_TASKS in ss and isinstance(ss[hmod.DONE_TASKS], list)
    assert hmod.DONE_TASKS_LOCK in ss
    assert hmod.REQUEST_ID_TO_DELIVERY_RATE_INFO in ss and isinstance(
        ss[hmod.REQUEST_ID_TO_DELIVERY_RATE_INFO], dict
    )


def test_contract_handler_state_enqueues_and_updates_from_block(
    handler_context: SimpleNamespace,
) -> None:
    """Enqueue filtered requests and update `from_block` on STATE."""
    params: Any = handler_context.params
    params.in_flight_req = True
    params.num_agents = 2
    params.agent_index = 1
    params.req_type = "legacy"
    params.req_params.from_block["legacy"] = 0

    ch: ContractHandler = ContractHandler(
        name="contract", skill_context=handler_context
    )
    ch.setup()

    reqs: List[Dict[str, int]] = [
        {"block_number": 10, "requestId": 1},
        {"block_number": 11, "requestId": 2},
    ]
    body: Dict[str, List[Dict[str, int]]] = {"data": reqs}
    msg: SimpleNamespace = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=body),
    )

    ch.handle(msg)

    assert params.req_params.from_block["legacy"] == 12
    assert len(ch.pending_tasks) == 1
    assert ch.pending_tasks[0]["requestId"] == 2
    assert params.in_flight_req is False


def test_contract_handler_non_state_sets_flag(handler_context: SimpleNamespace) -> None:
    """Clear `in_flight_req` on non-STATE performative."""
    params: Any = handler_context.params
    params.in_flight_req = True

    ch: ContractHandler = ContractHandler(
        name="contract", skill_context=handler_context
    )
    ch.setup()

    msg: SimpleNamespace = SimpleNamespace(
        performative=ContractApiMessage.Performative.GET_STATE
    )
    ch.handle(msg)

    assert params.in_flight_req is False


def test_ledger_handler_updates_from_block(handler_context: SimpleNamespace) -> None:
    """Update `from_block` from a ledger STATE message."""
    params: Any = handler_context.params
    params.in_flight_req = True
    params.req_type = "legacy"
    params.from_block_range = 500

    lh: LedgerHandler = LedgerHandler(name="ledger", skill_context=handler_context)
    lh.setup()

    msg: SimpleNamespace = SimpleNamespace(
        performative=LedgerApiMessage.Performative.STATE,
        state=SimpleNamespace(body={"number": 12345}),
    )
    lh.handle(msg)

    assert params.req_params.from_block["legacy"] == 12345 - 500
    assert params.in_flight_req is False


def test_ledger_handler_non_state_sets_flag(handler_context: SimpleNamespace) -> None:
    """Clear `in_flight_req` when a non-STATE ledger message is handled."""
    params: SimpleNamespace = handler_context.params
    params.in_flight_req = True

    lh: LedgerHandler = LedgerHandler(name="ledger", skill_context=handler_context)
    lh.setup()

    msg: SimpleNamespace = SimpleNamespace(
        performative=LedgerApiMessage.Performative.GET_STATE
    )
    lh.handle(msg)

    assert params.in_flight_req is False


def make_http_msg(body_dict: Dict[str, str], headers: str = "") -> SimpleNamespace:
    """
    Build a minimal HttpMessage-like object for handler tests.

    :param body_dict: Key–value form fields to be URL-encoded into the request body.
    :type body_dict: Dict[str, str]
    :param headers: Optional raw headers string to attach to the message.
    :type headers: str
    :returns: An object with the fields used by `MechHttpHandler`.
    :rtype: SimpleNamespace
    """
    body = urllib.parse.urlencode(body_dict).encode("utf-8")
    return SimpleNamespace(
        body=body,
        version="1.1",
        headers=headers,
        performative=HttpMessage.Performative.REQUEST,
    )


def test_signed_requests_success(handler_context: Any, http_dialogue: Any) -> None:
    """Enqueue off-chain request & respond 200 on valid signed POST."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    mh.setup()

    ipfs_hash: str = "0x" + "ab" * 64
    body: Dict[str, str] = {
        "ipfs_hash": ipfs_hash,
        "request_id": "req-1",
        "ipfs_data": '{"foo":"bar"}',
        "delivery_rate": "123",
    }
    http_msg: Any = make_http_msg(body)

    mh._handle_signed_requests(http_msg, http_dialogue)

    pend = handler_context.shared_state["pending_tasks"]
    ipfsq = handler_context.shared_state["ipfs_tasks"]
    assert len(pend) == 1 and len(ipfsq) == 1
    assert pend[0]["is_offchain"] is True
    assert pend[0]["requestId"] == "req-1"
    assert ipfsq[0]["request_id"] == "req-1"

    assert handler_context.outbox.sent, "no HTTP response sent"
    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.OK_CODE.value
    data = json.loads(resp.body.decode("utf-8"))
    assert data["request_id"] == "req-1"


def test_signed_requests_bad_request(handler_context: Any, http_dialogue: Any) -> None:
    """Return HTTP 400 when required POST fields are missing."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    mh.setup()

    http_msg: Any = make_http_msg({"only": "one"})
    mh._handle_signed_requests(http_msg, http_dialogue)

    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.BAD_REQUEST_CODE.value


def test_fetch_offchain_request_info_found(
    handler_context: Any, http_dialogue: Any
) -> None:
    """Return stored result for an off-chain request when present."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    mh.setup()

    handler_context.shared_state["ready_tasks"].append(
        {"request_id": "abc", "value": 7}
    )

    http_msg: Any = make_http_msg({"request_id": "abc"})
    mh._handle_offchain_request_info(http_msg, http_dialogue)

    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    payload: Dict[str, Any] = json.loads(resp.body.decode("utf-8"))
    assert payload["request_id"] == "abc" and payload["value"] == 7


def test_fetch_offchain_request_info_not_found(
    handler_context: Any, http_dialogue: Any
) -> None:
    """Return empty JSON when no off-chain result exists for the given request_id."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    mh.setup()

    http_msg: Any = make_http_msg({"request_id": "missing"})
    mh._handle_offchain_request_info(http_msg, http_dialogue)

    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    payload: Dict[str, Any] = json.loads(resp.body.decode("utf-8"))
    assert payload == {}


def test_on_message_handled_triggers_cleanup(
    handler_context: Any, monkeypatch: Any
) -> None:
    """Invoke dialogues cleanup when the cleanup threshold is reached."""
    handler_context.params.cleanup_freq = 1

    class Handlers:
        ipfs_handler: Any

    handler_context.handlers = Handlers()
    handler_context.handlers.ipfs_handler = object()

    cleaned = {"ipfs": 0}
    handler_context.ipfs_dialogues = SimpleNamespace(
        cleanup=lambda: cleaned.__setitem__("ipfs", cleaned["ipfs"] + 1)
    )

    h: IpfsHandler = IpfsHandler(name="ipfs", skill_context=handler_context)
    h.setup()
    h.on_message_handled(SimpleNamespace())

    assert cleaned["ipfs"] == 1
