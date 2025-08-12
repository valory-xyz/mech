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
    """
    Clear `in_flight_req` when the IPFS handler receives an ERROR.

    Given:
        - `params.in_flight_req` is True.
    When:
        - `IpfsHandler.handle()` is called with a message whose performative is ERROR.
    Then:
        - `params.in_flight_req` is set to False.
    """
    handler: IpfsHandler = IpfsHandler(name="ipfs", skill_context=handler_context)
    handler.setup()

    handler_context.params.in_flight_req = True
    msg: SimpleNamespace = SimpleNamespace(performative=IpfsMessage.Performative.ERROR)

    handler.handle(msg)

    assert handler_context.params.in_flight_req is False


def test_ipfs_handler_calls_callback_and_clears(
    handler_context: Any, monkeypatch: Any
) -> None:
    """
    Invoke the stored IPFS callback and clear bookkeeping afterward.

    Given:
        - `req_to_callback["nonce-1"]` points to a callable.
        - `req_to_deadline["nonce-1"]` is in the future.
        - `in_flight_req` is True and `is_cold_start` is True.
    When:
        - `IpfsHandler.handle()` receives a non-error IPFS message (e.g., GET_FILES).
    Then:
        - The stored callback is executed.
        - `in_flight_req` becomes False and `is_cold_start` becomes False.
        - The nonce is removed from `req_to_callback` and `req_to_deadline`.
    """
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

    # Act
    handler.handle(msg)

    # Assert
    assert called["ok"] is True
    assert handler_context.params.in_flight_req is False
    assert handler_context.params.is_cold_start is False
    assert "nonce-1" not in handler_context.params.req_to_callback
    assert "nonce-1" not in handler_context.params.req_to_deadline


def test_ipfs_handler_deadline_expired_skips_callback(handler_context: Any) -> None:
    """
    Skip the stored IPFS callback when its deadline has already expired.

    Given:
        - `req_to_callback["nonce-1"]` is set.
        - `req_to_deadline["nonce-1"]` is in the past.
        - `in_flight_req` and `is_cold_start` are True.
    When:
        - `IpfsHandler.handle()` receives a non-error IPFS message.
    Then:
        - The callback is NOT executed.
        - `in_flight_req` is set to False.
        - `is_cold_start` is set to False.
        - The nonce is removed from `req_to_callback` and `req_to_deadline`.
    """
    handler: IpfsHandler = IpfsHandler(name="ipfs", skill_context=handler_context)
    handler.setup()

    called = {"ok": False}
    handler_context.params.req_to_callback["nonce-1"] = lambda *_: called.__setitem__(
        "ok", True
    )
    handler_context.params.req_to_deadline["nonce-1"] = time.time() - 1.0  # expired
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
    """
    Initialize `shared_state` collections on setup.

    When:
        - `ContractHandler.setup()` is invoked.
    Then:
        - `shared_state` contains the expected keys with the correct container types.
    """
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
    """
    Enqueue filtered requests and update `from_block` on STATE.

    Given:
        - Two requests with block numbers 10 and 11.
        - `num_agents = 2`, `agent_index = 1` (keep only blocks where block_number % 2 == 1).
        - `req_type = "legacy"` and starting `from_block["legacy"] = 0`.
    When:
        - ContractHandler.handle() receives a STATE with those requests.
    Then:
        - `from_block["legacy"]` becomes max(block)+1 (i.e., 12).
        - Only the request for block 11 is enqueued to pending_tasks.
        - `in_flight_req` is set to False.
    """
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
    """
    Clear `in_flight_req` on non-STATE performative.

    Given:
        - `params.in_flight_req` is True.
    When:
        - ContractHandler receives a ContractApiMessage with performative != STATE
          (e.g., GET_STATE).
    Then:
        - `params.in_flight_req` is set to False.
    """
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
    """
    Update `from_block` from a ledger STATE message.

    Given:
        - `params.req_type` is "legacy"
        - `params.from_block_range` is 500
    When:
        - `LedgerHandler.handle()` receives a STATE with block number 12345
    Then:
        - `params.req_params.from_block["legacy"]` becomes 12345 - 500
        - `params.in_flight_req` is set to False
    """
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
    """
    Clear `in_flight_req` when a non-STATE ledger message is handled.

    Given:
        - `params.in_flight_req` is True.
    When:
        - `LedgerHandler.handle()` receives a message whose performative is not STATE
          (e.g., GET_STATE).
    Then:
        - `params.in_flight_req` is set to False.
    """
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

    Args:
        body_dict: Key–value form fields to be URL-encoded into the request body.
        headers: Optional raw headers string to attach to the message.

    Returns:
        SimpleNamespace: An object with the fields used by `MechHttpHandler`:
            - body (bytes): URL-encoded form body.
            - version (str): HTTP version (e.g., "1.1").
            - headers (str): Raw headers string.
            - performative: `HttpMessage.Performative.REQUEST`.
    """
    body = urllib.parse.urlencode(body_dict).encode("utf-8")
    return SimpleNamespace(
        body=body,
        version="1.1",
        headers=headers,
        performative=HttpMessage.Performative.REQUEST,
    )


def test_signed_requests_success(handler_context: Any, http_dialogue: Any) -> None:
    """
    Enqueue off-chain request & respond 200 on valid signed POST.

    Given:
        - A well-formed body containing `ipfs_hash`, `request_id`,
          `ipfs_data`, and `delivery_rate`.
    When:
        - `MechHttpHandler._handle_signed_requests` is invoked.
    Then:
        - A pending task is appended with `is_offchain=True` and the request id.
        - An IPFS upload task is queued with the same request id.
        - An HTTP 200 response is sent with JSON `{"request_id": <id>}`.
    """
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
    """
    Return HTTP 400 when required POST fields are missing.

    Given:
        - A POST body missing required keys (e.g., only "only" provided).
    When:
        - `MechHttpHandler._handle_signed_requests` is invoked.
    Then:
        - The handler responds with HTTP 400 (Bad Request).
    """
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    mh.setup()

    http_msg: Any = make_http_msg({"only": "one"})
    mh._handle_signed_requests(http_msg, http_dialogue)

    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.BAD_REQUEST_CODE.value


def test_fetch_offchain_request_info_found(
    handler_context: Any, http_dialogue: Any
) -> None:
    """
    Return stored result for an off-chain request when present.

    Given:
        - `ready_tasks` contains an entry with `request_id="abc"`.
    When:
        - `_handle_offchain_request_info` is called with `request_id=abc`.
    Then:
        - The handler responds 200 with the JSON body of the matching done task.
    """
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
    """
    Return empty JSON when no off-chain result exists for the given request_id.

    Given:
        - `ready_tasks` does not contain an entry with the requested id.
    When:
        - `_handle_offchain_request_info` is invoked with that `request_id`.
    Then:
        - The handler responds 200 with an empty JSON object `{}`.
    """
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
    """
    Invoke dialogues cleanup when the cleanup threshold is reached.

    Given:
        - `cleanup_freq = 1` so every handled message triggers cleanup.
        - `handler_context.handlers` contains an `ipfs_handler`, so the
          corresponding `ipfs_dialogues.cleanup()` will be called by
          `BaseHandler.cleanup_dialogues()`.
    When:
        - `on_message_handled()` is called.
    Then:
        - `ipfs_dialogues.cleanup()` is invoked exactly once.
    """
    # Ensure every on_message_handled triggers cleanup
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
