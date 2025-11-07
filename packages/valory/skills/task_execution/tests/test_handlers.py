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
    handler_context.params.req_to_callback["nonce-1"] = (
        lambda _msg, _dlg: called.__setitem__("ok", True)
    )
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
    handler_context: Any,
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
    """Enqueue filtered requests and update `from_block` on STATE (marketplace shape)."""
    params: Any = handler_context.params
    params.in_flight_req = True
    params.num_agents = 2
    params.agent_index = 1
    params.req_type = "marketplace"
    params.req_params.from_block["marketplace"] = 0

    # Make priorityMech match our mech so it goes to pending_tasks (not wait list)
    my_mech = params.agent_mech_contract_addresses[0]

    # Build marketplace-shaped body: each item has arrays requestIds/requestDatas
    reqs: List[Dict[str, Any]] = [
        {
            "tx_hash": "0xaaa",
            "block_number": 10,  # 10 % 2 == 0 -> filtered out by shard
            "priorityMech": my_mech,
            "requester": "0xR1",
            "numRequests": 1,
            "requestIds": [b"\x01" * 32],
            "requestDatas": [b"\x02" * 32],
            "status": 2,
            "request_delivery_rate": 100,
        },
        {
            "tx_hash": "0xbbb",
            "block_number": 11,  # 11 % 2 == 1 -> kept by shard
            "priorityMech": my_mech,
            "requester": "0xR2",
            "numRequests": 1,
            "requestIds": [b"\x03" * 32],
            "requestDatas": [b"\x04" * 32],
            "status": 2,
            "request_delivery_rate": 100,
        },
    ]
    body: Dict[str, Any] = {"data": reqs}

    ch = ContractHandler(name="contract", skill_context=handler_context)
    ch.setup()

    msg = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=body),
    )

    ch.handle(msg)

    # from_block should advance to max(block_number)+1 == 12
    assert params.req_params.from_block["marketplace"] == 12

    # Sharding keeps only block 11; and priorityMech == our mech puts it into pending_tasks
    assert len(ch.pending_tasks) == 1
    kept = ch.pending_tasks[0]
    assert kept["block_number"] == 11
    assert kept["priorityMech"] == my_mech
    assert kept["request_delivery_rate"] == 100

    # in_flight flag must be cleared
    assert params.in_flight_req is False


def test_contract_handler_other_mech_goes_to_wait_list(
    handler_context: SimpleNamespace,
) -> None:
    """Requests for a different mech go to wait_for_timeout_tasks (status == 2)."""
    params: Any = handler_context.params
    params.in_flight_req = True
    params.req_type = "marketplace"
    params.req_params.from_block["marketplace"] = 0

    other_mech: str = "0xDEAD"  # different from our mech
    my_mech: str = params.agent_mech_contract_addresses[0]
    assert other_mech != my_mech

    body: Dict[str, List[Dict[str, Any]]] = {
        "data": [
            {
                "tx_hash": "0xccc",
                "block_number": 21,
                "priorityMech": other_mech,
                "requester": "0xR3",
                "numRequests": 1,
                "requestIds": [b"\xaa" * 32],
                "requestDatas": [b"\xbb" * 32],
                "status": 2,  # kept in wait_for_timeout_tasks
                "request_delivery_rate": 50,
            }
        ]
    }

    ch = ContractHandler(name="contract", skill_context=handler_context)
    ch.setup()

    msg: SimpleNamespace = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=body),
    )

    ch.handle(msg)

    assert params.req_params.from_block["marketplace"] == 22
    assert len(ch.pending_tasks) == 0
    assert len(ch.timed_out_tasks) == 1
    assert ch.timed_out_tasks[0]["priorityMech"] == other_mech
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


def test_contract_handler_my_mech_delivered_not_enqueued(
    handler_context: SimpleNamespace,
) -> None:
    """A delivered request for my mech (status=3) is ignored (not enqueued)."""
    params: Any = handler_context.params
    params.in_flight_req = True
    params.req_type = "marketplace"
    params.req_params.from_block["marketplace"] = 0

    my_mech: str = params.agent_mech_contract_addresses[0]

    body: Dict[str, List[Dict[str, Any]]] = {
        "data": [
            {
                "tx_hash": "0xdelivered",
                "block_number": 50,
                "priorityMech": my_mech,
                "requester": "0xReq",
                "numRequests": 1,
                "requestIds": [b"\x11" * 32],
                "requestDatas": [b"\x22" * 32],
                "status": hmod.DELIVERED_STATUS,  # == 3
                "request_delivery_rate": 100,
            }
        ]
    }

    ch = ContractHandler(name="contract", skill_context=handler_context)
    ch.setup()

    msg: SimpleNamespace = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=body),
    )
    ch.handle(msg)

    # from_block should advance to 51 (50+1)
    assert params.req_params.from_block["marketplace"] == 51

    # Nothing should be enqueued anywhere
    assert len(ch.pending_tasks) == 0
    assert len(ch.wait_for_timeout_tasks) == 0
    assert len(ch.timed_out_tasks) == 0

    # in-flight cleared
    assert params.in_flight_req is False


def test_contract_handler_timed_out_then_delivered_updates_list(
    handler_context: SimpleNamespace,
) -> None:
    """A timed-out request later delivered is removed from the timeout list."""

    params: Any = handler_context.params
    params.in_flight_req = True
    params.req_type = "marketplace"
    params.req_params.from_block["marketplace"] = 0

    # Use a non-matching mech (so it won't go to pending), but status=2 should land in timed_out_tasks.
    other_mech: str = "0xBEEF"
    my_mech: str = params.agent_mech_contract_addresses[0]
    assert other_mech.lower() != my_mech.lower()

    rid = b"\x33" * 32
    data = b"\x44" * 32

    ch = ContractHandler(name="contract", skill_context=handler_context)
    ch.setup()

    # Round 1: status=2 -> goes to timed_out_tasks
    body_round1: Dict[str, List[Dict[str, Any]]] = {
        "data": [
            {
                "tx_hash": "0xround1",
                "block_number": 100,
                "priorityMech": other_mech,
                "requester": "0xReq1",
                "numRequests": 1,
                "requestIds": [rid],
                "requestDatas": [data],
                "status": hmod.TIMED_OUT_STATUS,  # == 2
                "request_delivery_rate": 10,
            }
        ]
    }
    msg1: SimpleNamespace = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=body_round1),
    )
    ch.handle(msg1)

    assert params.req_params.from_block["marketplace"] == 101
    assert len(ch.pending_tasks) == 0
    assert len(ch.wait_for_timeout_tasks) == 0
    assert len(ch.timed_out_tasks) == 1
    assert ch.timed_out_tasks[0]["tx_hash"] == "0xround1"
    assert params.in_flight_req is False

    # Round 2: same request turns into delivered (status=3) -> timed_out_tasks should be cleared
    params.in_flight_req = True
    body_round2: Dict[str, List[Dict[str, Any]]] = {
        "data": [
            {
                "tx_hash": "0xround2",
                "block_number": 105,
                "priorityMech": other_mech,
                "requester": "0xReq1",
                "numRequests": 1,
                "requestIds": [rid],
                "requestDatas": [data],
                "status": hmod.DELIVERED_STATUS,  # == 3
                "request_delivery_rate": 10,
            }
        ]
    }
    msg2: SimpleNamespace = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=body_round2),
    )
    ch.handle(msg2)

    # from_block advances to 106; timed_out list is cleared (handler resets and doesn't re-add delivered)
    assert params.req_params.from_block["marketplace"] == 106
    assert len(ch.timed_out_tasks) == 0
    assert len(ch.pending_tasks) == 0
    assert len(ch.wait_for_timeout_tasks) == 0
    assert params.in_flight_req is False
