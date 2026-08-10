# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2026 Valory AG
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
from unittest.mock import MagicMock, patch

import pytest

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


@pytest.mark.parametrize(
    "reason",
    [
        "protobuf: (PBNode) invalid wireType, expected 2, got 3",
        "Failed to download: bafybeiabc123",
        "custom IPFS node error: connection refused to /ip4/127.0.0.1",
    ],
)
def test_ipfs_handler_error_triggers_error_callback(
    handler_context: Any, reason: str
) -> None:
    """IPFS ERROR triggers error callback with the reason preserved verbatim."""
    handler: IpfsHandler = IpfsHandler(name="ipfs", skill_context=handler_context)
    handler.setup()

    captured: Dict[str, Any] = {"called": False, "reason": None}

    def error_cb(r: str) -> None:
        captured["called"] = True
        captured["reason"] = r

    handler_context.params.in_flight_req = True
    handler_context.params.req_to_callback["nonce-1"] = lambda *_: None
    handler_context.params.req_to_error_callback["nonce-1"] = error_cb
    handler_context.params.req_to_deadline["nonce-1"] = 999.0

    msg: SimpleNamespace = SimpleNamespace(
        performative=IpfsMessage.Performative.ERROR,
        reason=reason,
    )

    handler.handle(msg)

    assert handler_context.params.in_flight_req is False
    assert "nonce-1" not in handler_context.params.req_to_callback
    assert "nonce-1" not in handler_context.params.req_to_error_callback
    assert "nonce-1" not in handler_context.params.req_to_deadline
    assert captured["called"] is True
    assert captured["reason"] == reason


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
    handler_context.params.req_to_error_callback["nonce-1"] = lambda *_: None
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
    assert "nonce-1" not in handler_context.params.req_to_error_callback
    assert "nonce-1" not in handler_context.params.req_to_deadline


def test_ipfs_handler_deadline_expired_still_invokes_callback(
    handler_context: Any,
) -> None:
    """Invoke the callback even when the deadline has expired, so task state is cleaned up."""
    handler: IpfsHandler = IpfsHandler(name="ipfs", skill_context=handler_context)
    handler.setup()

    called = {"ok": False}
    handler_context.params.req_to_callback["nonce-1"] = lambda *_: called.__setitem__(
        "ok", True
    )
    handler_context.params.req_to_error_callback["nonce-1"] = lambda *_: None
    handler_context.params.req_to_deadline["nonce-1"] = time.time() - 1.0
    handler_context.params.in_flight_req = True
    handler_context.params.is_cold_start = True

    msg: SimpleNamespace = SimpleNamespace(
        performative=IpfsMessage.Performative.STORE_FILES
    )
    handler.handle(msg)

    assert called["ok"] is True  # callback must run for cleanup
    assert not handler_context.params.in_flight_req  # flag cleared via normal path
    assert not handler_context.params.is_cold_start  # set False via normal path
    assert "nonce-1" not in handler_context.params.req_to_callback
    assert "nonce-1" not in handler_context.params.req_to_error_callback
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
    assert len(ch.unprocessed_timed_out_tasks) == 1
    assert ch.unprocessed_timed_out_tasks[0]["priorityMech"] == other_mech
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


@pytest.mark.parametrize(
    "balance_status,available_offset,request_id,expected_http_code,enqueued,"
    "expected_resp_status,expected_resp_reason",
    [
        pytest.param(
            "ok",
            1,
            # Numeric decimal string — matches the wire format mech-client
            # sends (``getRequestId`` returns a uint256; mech-client
            # form-encodes it to decimal). Post the ingress coercion in
            # ``_enqueue_offchain_request`` non-numeric placeholders like
            # ``"req-1"`` no longer round-trip: ``int("req-1")`` raises.
            "1",
            HttpCode.OK_CODE.value,
            True,
            None,
            None,
            id="success",
        ),
        pytest.param(
            "ok",
            -1,
            "2",
            HttpCode.PAYMENT_REQUIRED_CODE.value,
            False,
            "rejected",
            "insufficient balance",
            id="insufficient_balance",
        ),
        pytest.param(
            "unavailable",
            -123,
            "3",
            HttpCode.SERVICE_UNAVAILABLE_CODE.value,
            False,
            "rejected",
            "balance check unavailable",
            id="balance_check_unavailable",
        ),
    ],
)
def test_signed_requests_balance_scenarios(
    handler_context: Any,
    http_dialogue: Any,
    monkeypatch: Any,
    balance_status: str,
    available_offset: int,
    request_id: str,
    expected_http_code: int,
    enqueued: bool,
    expected_resp_status: Any,
    expected_resp_reason: Any,
) -> None:
    """Test balance-check outcomes: success, insufficient balance, and unavailable."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    monkeypatch.setattr(
        mh,
        "_check_offchain_requester_balance",
        lambda sender, delivery_rate: {
            "status": balance_status,
            "required_amount": int(delivery_rate),
            "available_amount": int(delivery_rate) + available_offset,
            "reason": (
                "balance check completed"
                if balance_status == "ok"
                else "rpc unavailable"
            ),
            "balance_tracker_address": "0xBalanceTracker",
            "payment_type": "0xpaymenttype",
            "chain_id": 100,
        },
    )
    mh.setup()

    ipfs_hash: str = "0x" + "ab" * 32
    body: Dict[str, str] = {
        "ipfs_hash": ipfs_hash,
        "request_id": request_id,
        "ipfs_data": '{"foo":"bar"}',
        "delivery_rate": "123",
        "sender": "0x0000000000000000000000000000000000000001",
    }
    http_msg: Any = make_http_msg(body)
    mh._handle_signed_requests(http_msg, http_dialogue)

    pend = handler_context.shared_state["pending_tasks"]
    ipfsq = handler_context.shared_state["ipfs_tasks"]
    in_mem = handler_context.shared_state["in_memory_requests"]
    if enqueued:
        # Off-chain path: pending task is queued and the request metadata is
        # buffered in memory. The IPFS task queue stays empty by design — the
        # off-chain flow no longer publishes request metadata to IPFS.
        assert len(pend) == 1 and ipfsq == [] and len(in_mem) == 1
        assert pend[0]["is_offchain"] is True
        # ``requestId`` is stored as ``int`` post the ingress coercion
        # in ``_enqueue_offchain_request``; the ``in_memory_requests``
        # dict is still str-keyed since the local ``request_id`` variable
        # in the handler stays str for the HTTP-state lookups.
        assert pend[0]["requestId"] == int(request_id)
        assert in_mem[request_id] == '{"foo":"bar"}'
    else:
        assert pend == [] and ipfsq == [] and in_mem == {}

    assert handler_context.outbox.sent, "no HTTP response sent"
    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    assert resp.status_code == expected_http_code

    if expected_resp_status is not None:
        payload = json.loads(resp.body.decode("utf-8"))
        assert payload["status"] == expected_resp_status
        assert payload["reason"] == expected_resp_reason


def test_signed_requests_bad_request(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """Return HTTP 400 when required POST fields are missing."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    http_msg: Any = make_http_msg({"only": "one"})
    mh._handle_signed_requests(http_msg, http_dialogue)

    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.BAD_REQUEST_CODE.value


@pytest.mark.parametrize(
    "bad_request_id",
    [
        "req-1",
        "",
        "0x123",
        "42abc",
        "42.0",
        " 42",
        # Non-canonical decimals ``str.isdigit()`` would accept but the
        # coercion + writer contract would silently break: Unicode digits
        # (Thai, Arabic-Indic, superscript) that ``int()`` may re-stringify
        # to something else, and leading zeros that would desynchronise
        # the wire ``request_id`` from the writer's ``str(int(...))`` key.
        "๔๒",
        "٤٢",
        "²²",
        "042",
        "01",
        # Python's ``$`` matches immediately before a trailing ``\n``, so
        # ``re.match(r"...$", "42\n")`` would pass while ``int("42\n") == 42``
        # — the same roundtrip divergence that lets ``in_memory_requests``
        # and ``preimage_buffer`` accept records leak on a paid delivery.
        # ``fullmatch`` (or the ``\Z`` end-anchor on the pattern) closes it.
        "42\n",
        "\n42",
        # A magnitude above uint256 must also 400 up front (documented
        # promise: the ingress guard means ``int(request_id)`` cannot raise
        # deep in the enqueue helper). ``2**256`` is one over the bound.
        str(2**256),
        # CPython 3.11+ raises ``ValueError`` on ``int()`` for a decimal
        # string longer than ``sys.get_int_max_str_digits()`` (4300
        # default). The length check on ``MAX_REQUEST_ID`` MUST
        # short-circuit before ``int()`` runs, otherwise a 5000-digit
        # canonical decimal blows up outside the handler's ``try/except``
        # and skips the 400 path entirely. This case pins that.
        "9" * 5000,
    ],
    ids=[
        "alpha",
        "empty",
        "hex",
        "trailing_alpha",
        "dotted",
        "whitespace",
        "thai_digits",
        "arabic_indic_digits",
        "superscript_digits",
        "leading_zero_padded",
        "leading_zero_two_char",
        "trailing_newline",
        "leading_newline",
        "above_uint256",
        "above_cpython_int_str_digits_limit",
    ],
)
def test_signed_requests_rejects_non_numeric_request_id(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any, bad_request_id: str
) -> None:
    """Non-canonical ``request_id`` is rejected with 400 before any state mutation.

    The ingress coercion in ``_enqueue_offchain_request`` requires
    ``int(request_id)`` to parse AND ``str(int(request_id)) == request_id``
    to hold — the writer keys ``OFFCHAIN_REQUEST_RESPONSES`` /
    ``in_memory_requests`` / preimage buffer by ``str(int(request_id))``,
    so any input where the round trip diverges silently splits
    ``/fetch_offchain_info`` from the writer and reopens the
    poll-forever-while-charged path.

    ``REQUEST_ID_RE`` (``^(0|[1-9][0-9]*)$``) enforces canonical ASCII
    decimal, which is stricter than both ``int()`` (accepts whitespace,
    ``0x`` prefix, underscores, non-ASCII Unicode digits) and
    ``str.isdigit()`` (accepts Thai / Arabic-Indic / superscript
    digits). This test pins that stricter contract with cases from
    all three failure modes.

    :param handler_context: pytest fixture, mech HTTP handler test context.
    :param http_dialogue: pytest fixture, HTTP dialogue stub.
    :param monkeypatch: pytest fixture, per-test monkeypatch helper.
    :param bad_request_id: parametrized non-canonical ``request_id`` string.
    """
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    _install_balance_ok(mh, monkeypatch)
    mh.setup()

    body = _make_signed_request_body(request_id=bad_request_id)
    http_msg: Any = make_http_msg(body)
    mh._handle_signed_requests(http_msg, http_dialogue)

    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.BAD_REQUEST_CODE.value
    # No partial state mutation.
    assert handler_context.shared_state["pending_tasks"] == []
    assert handler_context.shared_state["in_memory_requests"] == {}


def test_signed_requests_offchain_disabled_returns_503(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """use_offchain=False (the Phase 1 default) makes the off-chain endpoint 503."""
    handler_context.params.use_offchain = False
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    http_msg: Any = make_http_msg(
        {
            "ipfs_hash": "0x" + "ab" * 32,
            "sender": "0xSender",
            "delivery_rate": "1",
            "request_id": "req-disabled",
            "ipfs_data": "{}",
        }
    )
    mh._handle_signed_requests(http_msg, http_dialogue)

    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.SERVICE_UNAVAILABLE_CODE.value
    assert mh.pending_tasks == []


def test_fetch_offchain_request_info_returns_insufficient_balance_response(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """Return stored insufficient balance payload via fetch_offchain_info."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    monkeypatch.setattr(
        mh,
        "_check_offchain_requester_balance",
        lambda sender, delivery_rate: {
            "status": "ok",
            "required_amount": int(delivery_rate),
            "available_amount": int(delivery_rate) - 1,
            "reason": "balance check completed",
            "balance_tracker_address": "0xBalanceTracker",
            "payment_type": "0xpaymenttype",
            "chain_id": 100,
        },
    )
    mh.setup()

    request_id = "1003"
    ipfs_hash: str = "0x" + "ab" * 32
    send_msg: Any = make_http_msg(
        {
            "ipfs_hash": ipfs_hash,
            "request_id": request_id,
            "ipfs_data": '{"foo":"bar"}',
            "delivery_rate": "123",
            "sender": "0x0000000000000000000000000000000000000001",
        }
    )
    mh._handle_signed_requests(send_msg, http_dialogue)

    fetch_msg: Any = make_http_msg({"request_id": request_id})
    mh._handle_offchain_request_info(fetch_msg, http_dialogue)

    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    payload = json.loads(resp.body.decode("utf-8"))
    assert payload["status"] == "rejected"
    assert payload["reason"] == "insufficient balance"
    assert payload["request_id"] == request_id


def test_signed_requests_rollback_partial_enqueue(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """Rollback queue entries when enqueue fails after partial append."""
    # Simulates a failure mid-enqueue (the buffered metadata write blows up
    # after pending_tasks has already accepted the entry). The handler must
    # roll the pending entry back and surface 400, not leave dangling state.
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    monkeypatch.setattr(
        mh,
        "_check_offchain_requester_balance",
        lambda sender, delivery_rate: {
            "status": "ok",
            "required_amount": int(delivery_rate),
            "available_amount": int(delivery_rate),
            "reason": "balance check completed",
            "balance_tracker_address": "0xBalanceTracker",
            "payment_type": "0xpaymenttype",
            "chain_id": 100,
        },
    )
    mh.setup()

    class FailingDict(dict):
        """Dict that raises on assignment to simulate partial buffer write."""

        def __setitem__(self, key: Any, value: Any) -> None:
            raise RuntimeError("simulated buffer write failure")

    handler_context.shared_state["in_memory_requests"] = FailingDict()

    ipfs_hash: str = "0x" + "ab" * 32
    body: Dict[str, str] = {
        "ipfs_hash": ipfs_hash,
        # Numeric — the ingress coercion in ``_enqueue_offchain_request``
        # runs before the buffer-write attempt this test simulates.
        "request_id": "6",
        "ipfs_data": '{"foo":"bar"}',
        "delivery_rate": "123",
        "sender": "0x0000000000000000000000000000000000000001",
    }
    http_msg: Any = make_http_msg(body)
    mh._handle_signed_requests(http_msg, http_dialogue)

    assert handler_context.shared_state["pending_tasks"] == []
    assert dict(handler_context.shared_state["in_memory_requests"]) == {}

    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.BAD_REQUEST_CODE.value


def _patched_handler_for_balance(
    handler_context: Any,
    monkeypatch: Any,
    available_offset: int,
    payment_type: str = "0xpaymenttype",
    balance_tracker_address: str = "0xBalanceTracker",
    chain_id: int = 100,
) -> MechHttpHandler:
    """Build a MechHttpHandler with the balance check patched to a fixed result."""
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    monkeypatch.setattr(
        mh,
        "_check_offchain_requester_balance",
        lambda sender, delivery_rate: {
            "status": "ok",
            "required_amount": int(delivery_rate),
            "available_amount": int(delivery_rate) + available_offset,
            "reason": "balance check completed",
            "balance_tracker_address": balance_tracker_address,
            "payment_type": payment_type,
            "chain_id": chain_id,
        },
    )
    mh.setup()
    return mh


def _send_signed_request(
    mh: MechHttpHandler, http_dialogue: Any, request_id: str
) -> SimpleNamespace:
    """Drive a signed offchain request and return the final HTTP response."""
    http_msg: Any = make_http_msg(
        {
            "ipfs_hash": "0x" + "ab" * 32,
            "request_id": request_id,
            "ipfs_data": '{"foo":"bar"}',
            "delivery_rate": "123",
            "sender": "0x0000000000000000000000000000000000000001",
        }
    )
    mh._handle_signed_requests(http_msg, http_dialogue)
    return mh.context.outbox.sent[-1]


def test_402_body_includes_structured_challenge(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """402 body carries the structured challenge fields alongside the legacy keys."""
    handler_context.params.payment_type_to_asset_address = {
        "0xpaymenttype": "0xUSDCToken",
    }
    mh = _patched_handler_for_balance(
        handler_context, monkeypatch, available_offset=-50
    )

    resp = _send_signed_request(mh, http_dialogue, request_id="402")
    assert resp.status_code == HttpCode.PAYMENT_REQUIRED_CODE.value

    payload = json.loads(resp.body.decode("utf-8"))
    # Legacy keys still present so clients ignoring the new schema keep working.
    assert payload["status"] == "rejected"
    assert payload["reason"] == "insufficient balance"
    # Structured challenge fields.
    assert payload["scheme"] == "olas-prepay"
    assert payload["payTo"] == "0xBalanceTracker"
    assert payload["asset"] == "0xUSDCToken"
    assert payload["chainId"] == 100
    assert payload["currentBalance"] == "73"  # 123 + (-50) = 73
    assert payload["required"] == "123"
    assert payload["depositInstructions"] == {
        "contract": "0xBalanceTracker",
        "abi": "depositFor(address requester, uint256 amount)",
    }
    assert payload["error"] == "insufficient balance"


def test_402_emits_www_authenticate_header(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """402 response carries the WWW-Authenticate header keyed by mech address."""
    mh = _patched_handler_for_balance(handler_context, monkeypatch, available_offset=-1)
    resp = _send_signed_request(mh, http_dialogue, request_id="4021")

    assert resp.status_code == HttpCode.PAYMENT_REQUIRED_CODE.value
    assert (
        'WWW-Authenticate: Payment scheme="olas-prepay" '
        'realm="0xMechAddr"' in resp.headers
    )


def test_402_native_asset_when_payment_type_unmapped(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """Unmapped payment types surface the zero address as the asset."""
    # Clients seeing the zero address must skip the ERC20 approve step and
    # treat the payment as a native-asset deposit.
    handler_context.params.payment_type_to_asset_address = {}
    mh = _patched_handler_for_balance(handler_context, monkeypatch, available_offset=-1)
    resp = _send_signed_request(mh, http_dialogue, request_id="4022")

    payload = json.loads(resp.body.decode("utf-8"))
    assert payload["asset"] == "0x0000000000000000000000000000000000000000"


def test_200_emits_payment_receipt_header(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """200 response carries a Payment-Receipt header with valid base64-encoded JSON."""
    mh = _patched_handler_for_balance(handler_context, monkeypatch, available_offset=10)
    resp = _send_signed_request(mh, http_dialogue, request_id="200")

    assert resp.status_code == HttpCode.OK_CODE.value
    receipt_line = next(
        line for line in resp.headers.split("\n") if line.startswith("Payment-Receipt:")
    )
    encoded = receipt_line.split(": ", 1)[1].strip()
    import base64 as _b64

    receipt = json.loads(_b64.b64decode(encoded).decode("utf-8"))
    # ``receipt["request_id"]`` is written from the local ``request_id``
    # variable in the handler, which stays ``str`` (the ingress coercion
    # only touches the value that flows into the pending task).
    assert receipt["request_id"] == "200"
    assert receipt["accepted_amount"] == "123"
    assert receipt["settlement_status"] == "pending"
    # ISO 8601 UTC with trailing Z. Loose check that it parses as a date.
    from datetime import datetime as _dt

    _dt.fromisoformat(receipt["accepted_at"].replace("Z", "+00:00"))


def test_200_body_unchanged_for_legacy_clients(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """200 body shape is unchanged. Receipt only goes in the header."""
    mh = _patched_handler_for_balance(handler_context, monkeypatch, available_offset=10)
    resp = _send_signed_request(mh, http_dialogue, request_id="201")

    payload = json.loads(resp.body.decode("utf-8"))
    assert payload == {"request_id": "201"}


def test_offchain_request_buffered_in_memory_not_in_ipfs_tasks(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """The offchain handler buffers metadata in-memory and skips ipfs_tasks."""
    # Locks in Piece 1's behavioural contract: the legacy IPFS upload queue is
    # no longer written from the offchain path. A future refactor that flipped
    # the flag back without us noticing would fail this test.
    mh = _patched_handler_for_balance(handler_context, monkeypatch, available_offset=10)
    _send_signed_request(mh, http_dialogue, request_id="5")

    assert handler_context.shared_state["ipfs_tasks"] == []
    assert handler_context.shared_state["in_memory_requests"] == {
        # ``in_memory_requests`` stays str-keyed (the ingress coercion
        # only touches the pending-task value, not the local dict keys).
        "5": '{"foo":"bar"}'
    }


def test_fetch_offchain_request_info_found(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """Return stored result for an off-chain request when present."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    # patch prometheus server to bypass port in use error and not relevant to these tests
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    handler_context.shared_state["ready_tasks"].append(
        {"request_id": "abc", "value": 7}
    )

    http_msg: Any = make_http_msg({"request_id": "abc"})
    mh._handle_offchain_request_info(http_msg, http_dialogue)

    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    payload: Dict[str, Any] = json.loads(resp.body.decode("utf-8"))
    assert payload["request_id"] == "abc" and payload["value"] == 7


def test_fetch_offchain_request_info_str_key_matches_wire_id(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """The primary lookup path resolves the wire ``str`` against the writer's ``str`` key.

    ``task_execution.behaviours._handle_done_task`` computes the key as
    ``str(req_id)`` where ``req_id`` is the ``int`` from the ingress
    coercion, so a wire-format ``request_id="42"`` posted by the
    requester must resolve to the ``"42"`` key the writer produces.
    Pre-fix (``cast(str, req_id)`` — a runtime no-op), the writer wrote
    an ``int`` key and this lookup returned ``{}``; the requester paid
    on chain via ``deliverWithSignatures`` and polled empty forever.

    :param handler_context: pytest fixture, mech HTTP handler test context.
    :param http_dialogue: pytest fixture, HTTP dialogue stub.
    :param monkeypatch: pytest fixture, per-test monkeypatch helper.
    """
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    stored = {
        "request_id": "42",
        "status": "ok",
        "content_cid": "bafyexamplecid",
        "response": {"result": "computed"},
    }
    # Match what the writer produces: str-keyed under the same value.
    handler_context.shared_state.setdefault(hmod.OFFCHAIN_REQUEST_RESPONSES, {})[
        "42"
    ] = stored

    http_msg: Any = make_http_msg({"request_id": "42"})
    mh._handle_offchain_request_info(http_msg, http_dialogue)

    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.OK_CODE.value
    payload: Dict[str, Any] = json.loads(resp.body.decode("utf-8"))
    assert payload == stored


def test_fetch_offchain_request_info_fallback_scan_normalizes_types(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """The fallback ``done_tasks`` scan normalizes ``str`` vs ``int`` types.

    The response cache miss path scans ``shared_state[DONE_TASKS]``.
    Off-chain ``done_tasks`` carry ``request_id`` as ``int`` post the
    ingress coercion; the GET body has the wire-format ``str``. Without
    the ``str()`` normalization added to the equality check the scan
    silently misses on the very same batch that just delivered the
    task, and the requester never gets a definitive answer.

    :param handler_context: pytest fixture, mech HTTP handler test context.
    :param http_dialogue: pytest fixture, HTTP dialogue stub.
    :param monkeypatch: pytest fixture, per-test monkeypatch helper.
    """
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    # ``OFFCHAIN_REQUEST_RESPONSES`` intentionally empty so we go through
    # the fallback path.
    handler_context.shared_state[hmod.OFFCHAIN_REQUEST_RESPONSES] = {}
    done_task = {
        "request_id": 42,  # int, as the ingress coercion produces
        "is_offchain": True,
        "tool": "prediction-offline-v1",
    }
    handler_context.shared_state[hmod.DONE_TASKS] = [done_task]

    http_msg: Any = make_http_msg({"request_id": "42"})
    mh._handle_offchain_request_info(http_msg, http_dialogue)

    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.OK_CODE.value
    payload = json.loads(resp.body.decode("utf-8"))
    # ``_handle_offchain_request_info`` returns the first matching
    # done_task dict verbatim (or ``{}`` when no match).
    assert payload == done_task, payload
    assert payload != {}, "fallback scan missed — str/int mismatch regressed"


def test_fetch_offchain_request_info_not_found(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """Return empty JSON when no off-chain result exists for the given request_id."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    # patch prometheus server to bypass port in use error and not relevant to these tests
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    http_msg: Any = make_http_msg({"request_id": "missing"})
    mh._handle_offchain_request_info(http_msg, http_dialogue)

    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    payload: Dict[str, Any] = json.loads(resp.body.decode("utf-8"))
    assert payload == {}


@pytest.mark.parametrize(
    "chain,rpc_attr,expected_chain_id",
    [
        ("gnosis", "http://gnosis-rpc", 100),
        ("polygon", "http://polygon-rpc", 137),
        ("base", "http://base-rpc", 8453),
    ],
)
def test_get_ledger_settings_supported_chains(
    handler_context: Any,
    monkeypatch: Any,
    chain: str,
    rpc_attr: str,
    expected_chain_id: int,
) -> None:
    """Return rpc + chain id for supported default_chain_id values."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    handler_context.params.default_chain_id = chain
    setattr(handler_context.params, f"{chain}_ledger_rpc", rpc_attr)

    settings = mh._get_ledger_settings()
    assert settings["status"] == "ok"
    assert settings["rpc_address"] == rpc_attr
    assert settings["chain_id"] == expected_chain_id


def test_get_ledger_settings_unsupported_chain(
    handler_context: Any, monkeypatch: Any
) -> None:
    """Return unavailable when default_chain_id is not mapped."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    handler_context.params.default_chain_id = "arbitrum"
    handler_context.params.arbitrum_ledger_rpc = "http://arbitrum-rpc"
    settings = mh._get_ledger_settings()

    assert settings["status"] == "unavailable"
    assert "Unsupported chain" in str(settings["reason"])


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
    assert len(ch.unprocessed_timed_out_tasks) == 0

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

    # Use a non-matching mech (so it won't go to pending), but status=2 should land in unprocessed_timed_out_tasks.
    other_mech: str = "0xBEEF"
    my_mech: str = params.agent_mech_contract_addresses[0]
    assert other_mech.lower() != my_mech.lower()

    rid = b"\x33" * 32
    data = b"\x44" * 32

    ch = ContractHandler(name="contract", skill_context=handler_context)
    ch.setup()

    # Round 1: status=2 -> goes to unprocessed_timed_out_tasks
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
    assert len(ch.unprocessed_timed_out_tasks) == 1
    assert ch.unprocessed_timed_out_tasks[0]["tx_hash"] == "0xround1"
    assert params.in_flight_req is False

    # Round 2: same request turns into delivered (status=3) -> unprocessed_timed_out_tasks should be cleared
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
    assert len(ch.unprocessed_timed_out_tasks) == 0
    assert len(ch.pending_tasks) == 0
    assert len(ch.wait_for_timeout_tasks) == 0
    assert params.in_flight_req is False


def test_contract_handler_updates_pending_list_based_on_delivered_request_status(
    handler_context: SimpleNamespace,
) -> None:
    """Updates pending list based on request id status based on marketplace contract data"""
    params: Any = handler_context.params
    params.in_flight_req = True
    params.num_agents = 1
    params.agent_index = 0
    params.req_type = "marketplace"
    params.req_params.from_block["marketplace"] = 0

    # Make priorityMech match our mech so it goes to pending_tasks (not wait list)
    my_mech = params.agent_mech_contract_addresses[0]

    # Build marketplace-shaped body: each item has arrays requestIds/requestDatas
    reqs: List[Dict[str, Any]] = [
        {
            "tx_hash": "0xaaa",
            "block_number": 10,
            "priorityMech": my_mech,
            "requester": "0xR1",
            "numRequests": 1,
            "requestId": b"\x01" * 32,
            "requestData": b"\x02" * 32,
            "status": 1,
            "request_delivery_rate": 100,
        },
        {
            "tx_hash": "0xbbb",
            "block_number": 11,
            "priorityMech": my_mech,
            "requester": "0xR2",
            "numRequests": 1,
            "requestId": b"\x03" * 32,
            "requestData": b"\x04" * 32,
            "status": 1,
            "request_delivery_rate": 100,
        },
    ]
    pending_reqs_body: Dict[str, Any] = {"data": reqs}

    ch = ContractHandler(name="contract", skill_context=handler_context)
    ch.setup()

    msg = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=pending_reqs_body),
    )
    ch.handle(msg)

    assert len(ch.pending_tasks) == 2

    status_check_body: Dict[str, Any] = {"request_ids": (b"\x01" * 32,)}
    msg = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=status_check_body),
    )
    ch.handle(msg)

    assert len(ch.pending_tasks) == 1

    # in_flight flag must be cleared
    assert params.in_flight_req is False


def test_contract_handler_doesnot_updates_pending_list_based_on_undelivered_request_status(
    handler_context: SimpleNamespace,
) -> None:
    """Does not update pending list based on request id status based on marketplace contract data"""
    params: Any = handler_context.params
    params.in_flight_req = True
    params.num_agents = 1
    params.agent_index = 0
    params.req_type = "marketplace"
    params.req_params.from_block["marketplace"] = 0

    # Make priorityMech match our mech so it goes to pending_tasks (not wait list)
    my_mech = params.agent_mech_contract_addresses[0]

    # Build marketplace-shaped body: each item has arrays requestIds/requestDatas
    reqs: List[Dict[str, Any]] = [
        {
            "tx_hash": "0xaaa",
            "block_number": 10,
            "priorityMech": my_mech,
            "requester": "0xR1",
            "numRequests": 1,
            "requestId": b"\x01" * 32,
            "requestData": b"\x02" * 32,
            "status": 1,
            "request_delivery_rate": 100,
        },
        {
            "tx_hash": "0xbbb",
            "block_number": 11,
            "priorityMech": my_mech,
            "requester": "0xR2",
            "numRequests": 1,
            "requestId": b"\x03" * 32,
            "requestData": b"\x04" * 32,
            "status": 1,
            "request_delivery_rate": 100,
        },
    ]
    pending_reqs_body: Dict[str, Any] = {"data": reqs}

    ch = ContractHandler(name="contract", skill_context=handler_context)
    ch.setup()

    msg = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=pending_reqs_body),
    )
    ch.handle(msg)

    assert len(ch.pending_tasks) == 2

    status_check_body: Dict[str, Any] = {"request_ids": ()}
    msg = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=status_check_body),
    )
    ch.handle(msg)

    assert len(ch.pending_tasks) == 2

    # in_flight flag must be cleared
    assert params.in_flight_req is False


def test_contract_handler_handles_wait_for_timeout_tasks_properly_from_contract(
    handler_context: SimpleNamespace,
) -> None:
    """Test handler to work fine with data and wait_for_timeout_tasks."""
    params: Any = handler_context.params
    params.in_flight_req = True
    params.num_agents = 1
    params.agent_index = 0
    params.req_type = "marketplace"
    params.req_params.from_block["marketplace"] = 0

    # Make priorityMech match our mech so it goes to pending_tasks (not wait list)
    my_mech = params.agent_mech_contract_addresses[0]

    # Build marketplace-shaped body: each item has arrays requestIds/requestDatas
    reqs: List[Dict[str, Any]] = [
        {
            "tx_hash": "0xaaa",
            "block_number": 10,
            "priorityMech": my_mech,
            "requester": "0xR1",
            "numRequests": 1,
            "requestId": b"\x01" * 32,
            "requestData": b"\x02" * 32,
            "status": 1,
            "request_delivery_rate": 100,
        },
        {
            "tx_hash": "0xbbb",
            "block_number": 11,
            "priorityMech": my_mech,
            "requester": "0xR2",
            "numRequests": 1,
            "requestId": b"\x03" * 32,
            "requestData": b"\x04" * 32,
            "status": 1,
            "request_delivery_rate": 100,
        },
    ]

    wait_for_timeout_tasks = [
        {
            "tx_hash": "0xaaac",
            "block_number": 10,
            "priorityMech": my_mech,
            "requester": "0xR12",
            "numRequests": 1,
            "requestId": b"\x01" * 32,
            "requestData": b"\x02" * 32,
            "status": 1,
            "request_delivery_rate": 100,
        },
        {
            "tx_hash": "0xbbbc2",
            "block_number": 11,
            "priorityMech": my_mech,
            "requester": "0xR2",
            "numRequests": 1,
            "requestId": b"\x03" * 32,
            "requestData": b"\x04" * 32,
            "status": 1,
            "request_delivery_rate": 100,
        },
        {
            "tx_hash": "0xbbbc5",
            "block_number": 11,
            "priorityMech": my_mech,
            "requester": "0xR2",
            "numRequests": 1,
            "requestId": b"\x03" * 32,
            "requestData": b"\x04" * 32,
            "status": 1,
            "request_delivery_rate": 100,
        },
    ]

    # check data is empty and wait_for_timeout_tasks has to be processed well
    pending_reqs_body: Dict[str, Any] = {
        "data": [],
        "wait_for_timeout_tasks": wait_for_timeout_tasks,
    }

    ch = ContractHandler(name="contract", skill_context=handler_context)
    ch.setup()

    msg = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=pending_reqs_body),
    )
    ch.handle(msg)

    assert len(ch.pending_tasks) == 3

    # check data and wait_for_timeout_tasks
    pending_reqs_body2: Dict[str, Any] = {
        "data": reqs,
        "wait_for_timeout_tasks": wait_for_timeout_tasks,
    }

    ch2 = ContractHandler(name="contract", skill_context=handler_context)
    ch2.setup()

    msg2 = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=pending_reqs_body2),
    )
    ch2.handle(msg2)

    assert len(ch2.pending_tasks) == 5


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


def test_base_handler_teardown(handler_context: Any) -> None:
    """Teardown logs without error."""
    h: IpfsHandler = IpfsHandler(name="ipfs", skill_context=handler_context)
    h.setup()
    h.teardown()  # line 179


def test_acn_handler_handle(handler_context: Any) -> None:
    """AcnHandler.handle logs and calls on_message_handled."""

    h = hmod.AcnHandler(name="acn", skill_context=handler_context)
    h.setup()
    h.handle(SimpleNamespace())  # lines 203-204


def test_contract_handler_step_in_list_size(handler_context: Any) -> None:
    """step_in_list_size property returns params.step_in_list_size."""
    ch: ContractHandler = ContractHandler(
        name="contract", skill_context=handler_context
    )
    ch.setup()
    assert ch.step_in_list_size == handler_context.params.step_in_list_size  # line 317


def test_contract_handler_mech_type_and_mech_types_body_keys(
    handler_context: SimpleNamespace,
) -> None:
    """MECH_TYPE and MECH_TYPES body keys update shared_state."""

    params: Any = handler_context.params
    params.in_flight_req = True
    params.req_type = "legacy"

    body: Dict[str, Any] = {
        "mech_type": "token",
        "mech_types": {"token": "0xBT"},  # nosec B105
    }
    ch: ContractHandler = ContractHandler(
        name="contract", skill_context=handler_context
    )
    ch.setup()

    msg: SimpleNamespace = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=body),
    )
    ch.handle(msg)

    assert handler_context.shared_state[hmod.PAYMENT_MODEL] == "token"  # lines 348-349
    assert handler_context.shared_state[hmod.PAYMENT_INFO] == {
        "token": "0xBT"  # nosec B105
    }  # lines 354-355


def test_contract_handler_handle_get_undelivered_reqs_empty(
    handler_context: SimpleNamespace,
) -> None:
    """_handle_get_undelivered_reqs with empty data returns early (reqs_count==0)."""
    params: Any = handler_context.params
    params.req_type = "legacy"

    ch: ContractHandler = ContractHandler(
        name="contract", skill_context=handler_context
    )
    ch.setup()
    # Call directly to bypass the outer `if body.get("data") or ...` guard
    ch._handle_get_undelivered_reqs(
        {"data": [], "wait_for_timeout_tasks": []}
    )  # lines 390-391

    assert len(ch.pending_tasks) == 0


def test_contract_handler_wait_for_timeout_status(
    handler_context: SimpleNamespace,
) -> None:
    """WAIT_FOR_TIMEOUT_STATUS with high delivery_rate puts request in wait_for_timeout_tasks."""

    params: Any = handler_context.params
    params.req_type = "marketplace"
    params.req_params.from_block["marketplace"] = 0
    my_mech: str = params.agent_mech_contract_addresses[0]
    params.mech_to_max_delivery_rate = {my_mech.lower(): 5}

    ch: ContractHandler = ContractHandler(
        name="contract", skill_context=handler_context
    )
    ch.setup()

    body: Dict[str, Any] = {
        "data": [
            {
                "tx_hash": "0xwait",
                "block_number": 30,
                "priorityMech": "0xOtherMech",  # not my mech
                "requester": "0xR",
                "numRequests": 1,
                "requestIds": [b"\x55" * 32],
                "requestDatas": [b"\x66" * 32],
                "status": hmod.WAIT_FOR_TIMEOUT_STATUS,  # 1
                "request_delivery_rate": 10,  # >= 5
            }
        ]
    }
    msg: SimpleNamespace = SimpleNamespace(
        performative=ContractApiMessage.Performative.STATE,
        state=SimpleNamespace(body=body),
    )
    ch.handle(msg)

    assert (
        len(ch.wait_for_timeout_tasks) == 1
    )  # lines 298-302 (property) + 455-460 (branch)


def test_start_prometheus_server_calls_start_http_server(
    handler_context: Any, monkeypatch: Any
) -> None:
    """start_prometheus_server invokes prometheus_client.start_http_server."""

    called: Dict[str, Any] = {}
    monkeypatch.setattr(
        hmod, "start_http_server", lambda port: called.__setitem__("port", port)
    )

    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    mh.start_prometheus_server()  # lines 552-553

    assert "port" in called


def test_fetch_offchain_request_info_bad_request_missing_key(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """_handle_offchain_request_info returns 400 when request_id is absent from body."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    http_msg: Any = make_http_msg({})  # no request_id → KeyError → bad request
    mh._handle_offchain_request_info(http_msg, http_dialogue)  # lines 650-653

    resp: SimpleNamespace = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.BAD_REQUEST_CODE.value


def test_make_unavailable_balance_response_static(
    handler_context: Any, monkeypatch: Any
) -> None:
    """_make_unavailable_balance_response returns well-formed unavailable dict."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    result: Dict[str, Any] = mh._make_unavailable_balance_response(
        500, "test reason"
    )  # line 809

    assert result["status"] == "unavailable"
    assert result["required_amount"] == 500
    assert result["available_amount"] == 0
    assert result["reason"] == "test reason"


def test_get_ledger_settings_missing_rpc(
    handler_context: Any, monkeypatch: Any
) -> None:
    """_get_ledger_settings returns unavailable when RPC address is empty."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    handler_context.params.default_chain_id = "gnosis"
    handler_context.params.gnosis_ledger_rpc = ""  # falsy → line 929

    settings: Dict[str, Any] = mh._get_ledger_settings()
    assert settings["status"] == "unavailable"
    assert "Missing RPC config" in settings["reason"]


def test_get_mech_payment_type(handler_context: Any, monkeypatch: Any) -> None:
    """_get_mech_payment_type returns the mech_type from contract response."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    with patch.object(
        hmod.OlasMechContract, "get_mech_type", return_value={"mech_type": "token"}
    ):
        result = mh._get_mech_payment_type(MagicMock(), "0xMECH")  # lines 894-895

    assert result == "token"


def test_get_balance_tracker_address_for_payment_type(
    handler_context: Any, monkeypatch: Any
) -> None:
    """_get_balance_tracker_address_for_payment_type returns address from contract response."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    with patch.object(
        hmod.MechMarketplaceContract,
        "get_balance_tracker_for_mech_type",
        return_value={"data": "0xBT"},
    ):
        result = mh._get_balance_tracker_address_for_payment_type(  # lines 904-909
            ledger_api=MagicMock(),
            marketplace_address="0xMARKET",
            payment_type="token",
        )

    assert result == "0xBT"


def test_get_requester_balance(handler_context: Any, monkeypatch: Any) -> None:
    """_get_requester_balance returns balance from contract response."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()

    with patch.object(
        hmod.BalanceTrackerContract,
        "get_requester_balance",
        return_value={"requester_balance": 42},
    ):
        result = mh._get_requester_balance(  # lines 915-920
            ledger_api=MagicMock(),
            balance_tracker_address="0xBT",
            requester="0xSENDER",
        )

    assert result == 42


def test_check_offchain_requester_balance_non_ok_ledger(
    handler_context: Any, monkeypatch: Any
) -> None:
    """_check_offchain_requester_balance returns unavailable when ledger settings not OK."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()
    monkeypatch.setattr(
        mh,
        "_get_ledger_settings",
        lambda: {"status": "unavailable", "reason": "no rpc"},
    )

    result: Dict[str, Any] = mh._check_offchain_requester_balance(
        "0xSENDER", 100
    )  # lines 822-827

    assert result["status"] == "unavailable"
    assert result["required_amount"] == 100
    assert result["available_amount"] == 0


def test_check_offchain_requester_balance_payment_type_none(
    handler_context: Any, monkeypatch: Any
) -> None:
    """_check_offchain_requester_balance returns unavailable when payment type is None."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()
    monkeypatch.setattr(
        mh,
        "_get_ledger_settings",
        lambda: {"status": "ok", "rpc_address": "http://rpc", "chain_id": 100},
    )
    monkeypatch.setattr(mh, "_get_mech_payment_type", lambda api, addr: None)

    fake_api = MagicMock()
    fake_api.api.to_checksum_address = lambda x: x

    with patch.object(hmod, "EthereumApi", return_value=fake_api):
        result: Dict[str, Any] = mh._check_offchain_requester_balance(
            "0xSENDER", 100
        )  # lines 844-847

    assert result["status"] == "unavailable"
    assert "payment type" in result["reason"]


def test_check_offchain_requester_balance_no_balance_tracker(
    handler_context: Any, monkeypatch: Any
) -> None:
    """_check_offchain_requester_balance returns unavailable when balance tracker is zero."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()
    monkeypatch.setattr(
        mh,
        "_get_ledger_settings",
        lambda: {"status": "ok", "rpc_address": "http://rpc", "chain_id": 100},
    )
    monkeypatch.setattr(mh, "_get_mech_payment_type", lambda api, addr: "token")
    monkeypatch.setattr(
        mh,
        "_get_balance_tracker_address_for_payment_type",
        lambda **kw: "0x" + "0" * 40,  # int == 0
    )

    fake_api = MagicMock()
    fake_api.api.to_checksum_address = lambda x: x

    with patch.object(hmod, "EthereumApi", return_value=fake_api):
        result: Dict[str, Any] = mh._check_offchain_requester_balance(
            "0xSENDER", 100
        )  # lines 856-860

    assert result["status"] == "unavailable"
    assert "balance tracker" in result["reason"]


def test_check_offchain_requester_balance_success(
    handler_context: Any, monkeypatch: Any
) -> None:
    """_check_offchain_requester_balance returns OK with balance on full success path."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()
    monkeypatch.setattr(
        mh,
        "_get_ledger_settings",
        lambda: {"status": "ok", "rpc_address": "http://rpc", "chain_id": 100},
    )
    monkeypatch.setattr(mh, "_get_mech_payment_type", lambda api, addr: "token")
    monkeypatch.setattr(
        mh,
        "_get_balance_tracker_address_for_payment_type",
        lambda **kw: "0x" + "1" * 40,  # non-zero
    )
    monkeypatch.setattr(mh, "_get_requester_balance", lambda **kw: 500)

    fake_api = MagicMock()
    fake_api.api.to_checksum_address = lambda x: x

    with patch.object(hmod, "EthereumApi", return_value=fake_api):
        result: Dict[str, Any] = mh._check_offchain_requester_balance(
            "0xSENDER", 100
        )  # lines 830-879

    assert result["status"] == "ok"
    assert result["available_amount"] == 500
    assert result["required_amount"] == 100


def test_check_offchain_requester_balance_exception(
    handler_context: Any, monkeypatch: Any
) -> None:
    """_check_offchain_requester_balance returns unavailable when an exception is raised."""
    mh: MechHttpHandler = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()
    monkeypatch.setattr(
        mh,
        "_get_ledger_settings",
        lambda: {"status": "ok", "rpc_address": "http://rpc", "chain_id": 100},
    )

    with patch.object(hmod, "EthereumApi", side_effect=RuntimeError("rpc down")):
        result: Dict[str, Any] = mh._check_offchain_requester_balance(
            "0xSENDER", 100
        )  # lines 886-889

    assert result["status"] == "unavailable"
    assert result["required_amount"] == 100
    assert "rpc down" in result["reason"]


# ---------------------------------------------------------------------------
# filter_requests edge cases
# ---------------------------------------------------------------------------


def _make_contract_handler(handler_context: SimpleNamespace) -> ContractHandler:
    """Create a fresh ContractHandler with the given context."""
    # Ensure mech_to_max_delivery_rate has our mech so the property doesn't KeyError
    my_mech = handler_context.params.agent_mech_contract_addresses[0].lower()
    handler_context.params.mech_to_max_delivery_rate.setdefault(my_mech, 0)
    ch = ContractHandler(name="contract", skill_context=handler_context)
    ch.setup()
    return ch


def _base_req(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal marketplace request dict, merging overrides."""
    d: Dict[str, Any] = {
        "tx_hash": "0xbase",
        "block_number": 1,
        "priorityMech": "0xMechAddr",
        "requester": "0xReq",
        "numRequests": 1,
        "requestIds": [b"\x01" * 32],
        "requestDatas": [b"\x02" * 32],
        "requestId": "rid-1",
        "status": 1,
        "request_delivery_rate": 100,
    }
    d.update(overrides)
    return d


def test_filter_requests_missing_priority_mech_skips(
    handler_context: SimpleNamespace,
) -> None:
    """Request without 'priorityMech' key should be skipped (not crash)."""
    ch = _make_contract_handler(handler_context)
    req = _base_req(status=hmod.DELIVERED_STATUS)
    del req["priorityMech"]

    ch.filter_requests([req])

    # delivered + no matching mech → falls through to else (skipped)
    assert len(ch.pending_tasks) == 0
    assert len(ch.unprocessed_timed_out_tasks) == 0
    assert len(ch.wait_for_timeout_tasks) == 0


def test_filter_requests_missing_delivery_rate_skips(
    handler_context: SimpleNamespace,
) -> None:
    """Request missing 'request_delivery_rate' with WAIT status should be skipped."""
    my_mech: str = handler_context.params.agent_mech_contract_addresses[0].lower()
    handler_context.params.mech_to_max_delivery_rate = {my_mech: 50}
    ch = _make_contract_handler(handler_context)
    req = _base_req(
        priorityMech="0xOther",
        status=hmod.WAIT_FOR_TIMEOUT_STATUS,
    )
    del req["request_delivery_rate"]

    ch.filter_requests([req])

    assert len(ch.pending_tasks) == 0
    assert len(ch.wait_for_timeout_tasks) == 0


def test_filter_requests_none_status_goes_to_else(
    handler_context: SimpleNamespace,
) -> None:
    """Request with missing 'status' key should be skipped (falls to else)."""
    ch = _make_contract_handler(handler_context)
    req = _base_req(priorityMech="0xOther")
    del req["status"]

    ch.filter_requests([req])

    assert len(ch.pending_tasks) == 0
    assert len(ch.unprocessed_timed_out_tasks) == 0
    assert len(ch.wait_for_timeout_tasks) == 0


def test_filter_requests_delivered_for_my_mech_skipped(
    handler_context: SimpleNamespace,
) -> None:
    """Delivered request (status=3) for my mech goes to else branch."""
    ch = _make_contract_handler(handler_context)
    req = _base_req(status=hmod.DELIVERED_STATUS)

    ch.filter_requests([req])

    assert len(ch.pending_tasks) == 0
    assert len(ch.unprocessed_timed_out_tasks) == 0
    assert len(ch.wait_for_timeout_tasks) == 0


def test_filter_requests_wait_below_max_rate_skipped(
    handler_context: SimpleNamespace,
) -> None:
    """WAIT_FOR_TIMEOUT with delivery rate below max is skipped."""
    my_mech: str = handler_context.params.agent_mech_contract_addresses[0].lower()
    handler_context.params.mech_to_max_delivery_rate = {my_mech: 200}
    ch = _make_contract_handler(handler_context)
    req = _base_req(
        priorityMech="0xOther",
        status=hmod.WAIT_FOR_TIMEOUT_STATUS,
        request_delivery_rate=100,
    )

    ch.filter_requests([req])

    assert len(ch.wait_for_timeout_tasks) == 0


def test_filter_requests_wait_at_max_rate_enqueued(
    handler_context: SimpleNamespace,
) -> None:
    """WAIT_FOR_TIMEOUT with delivery rate >= max goes to wait_for_timeout_tasks."""
    my_mech_lower: str = handler_context.params.agent_mech_contract_addresses[0].lower()
    handler_context.params.mech_to_max_delivery_rate = {my_mech_lower: 100}
    ch = _make_contract_handler(handler_context)
    req = _base_req(
        priorityMech="0xOther",
        status=hmod.WAIT_FOR_TIMEOUT_STATUS,
        request_delivery_rate=100,
    )

    ch.filter_requests([req])

    assert len(ch.wait_for_timeout_tasks) == 1


def test_filter_requests_mixed_batch(
    handler_context: SimpleNamespace,
) -> None:
    """Mixed batch: my-mech-pending, timed-out, wait-for-timeout, and delivered."""
    my_mech: str = handler_context.params.agent_mech_contract_addresses[0]
    my_mech_lower: str = my_mech.lower()
    handler_context.params.mech_to_max_delivery_rate = {my_mech_lower: 50}
    ch = _make_contract_handler(handler_context)

    reqs = [
        _base_req(priorityMech=my_mech, status=1),  # → pending
        _base_req(priorityMech="0xOther", status=hmod.TIMED_OUT_STATUS),  # → timed_out
        _base_req(
            priorityMech="0xOther",
            status=hmod.WAIT_FOR_TIMEOUT_STATUS,
            request_delivery_rate=60,
        ),  # → wait (60 >= 50)
        _base_req(priorityMech=my_mech, status=hmod.DELIVERED_STATUS),  # → skipped
    ]

    ch.filter_requests(reqs)

    assert len(ch.pending_tasks) == 1
    assert len(ch.unprocessed_timed_out_tasks) == 1
    assert len(ch.wait_for_timeout_tasks) == 1


def test_filter_requests_empty_list(
    handler_context: SimpleNamespace,
) -> None:
    """Empty request list is a no-op."""
    ch = _make_contract_handler(handler_context)
    ch.filter_requests([])

    assert len(ch.pending_tasks) == 0


# ---------------------------------------------------------------------------
# Offchain HTTP body cap and ipfs_hash format validation
# ---------------------------------------------------------------------------


def _make_signed_request_body(
    ipfs_hash: str = "0x" + "ab" * 32,
    delivery_rate: str = "123",
    request_id: str = "req-hardening",
) -> Dict[str, str]:
    """Build a valid signed-request body used by hardening tests."""
    return {
        "ipfs_hash": ipfs_hash,
        "request_id": request_id,
        "ipfs_data": '{"foo":"bar"}',
        "delivery_rate": delivery_rate,
        "sender": "0x0000000000000000000000000000000000000001",
    }


def _install_balance_ok(mh: Any, monkeypatch: Any) -> None:
    """Make _check_offchain_requester_balance return an OK, sufficient response."""
    monkeypatch.setattr(
        mh,
        "_check_offchain_requester_balance",
        lambda sender, delivery_rate: {
            "status": "ok",
            "required_amount": int(delivery_rate),
            "available_amount": int(delivery_rate) + 1,
            "reason": "ok",
        },
    )


def test_signed_requests_rejects_oversized_http_body(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """HTTP body larger than MAX_HTTP_BODY_BYTES is rejected with 400 before decode."""
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    _install_balance_ok(mh, monkeypatch)
    mh.setup()

    padding_bytes = hmod.MAX_HTTP_BODY_BYTES + 1
    base = _make_signed_request_body(request_id="req-oversized")
    base["ipfs_data"] = '{"foo":"' + "x" * padding_bytes + '"}'
    http_msg: Any = make_http_msg(base)
    assert len(http_msg.body) > hmod.MAX_HTTP_BODY_BYTES

    parse_calls: List[Any] = []
    original_parse_qs = hmod.urllib.parse.parse_qs

    def spy_parse_qs(*args: Any, **kwargs: Any) -> Any:
        parse_calls.append(args)
        return original_parse_qs(*args, **kwargs)

    monkeypatch.setattr(hmod.urllib.parse, "parse_qs", spy_parse_qs)

    mh._handle_signed_requests(http_msg, http_dialogue)

    resp = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.BAD_REQUEST_CODE.value
    assert handler_context.shared_state["pending_tasks"] == []
    assert parse_calls == [], "parse_qs was called despite oversized body"


@pytest.mark.parametrize(
    "bad_hash",
    [
        "0xabc",
        "0x" + "ab" * 33,
        "0x" + "ab" * 64,
        "ab" * 32,
        "0x" + "z" * 64,
    ],
    ids=[
        "too_short",
        "wrong_length_66_byte_hex",
        "far_too_long",
        "missing_0x_prefix",
        "non_hex_chars",
    ],
)
def test_signed_requests_rejects_invalid_ipfs_hash(
    handler_context: Any,
    http_dialogue: Any,
    monkeypatch: Any,
    bad_hash: str,
) -> None:
    """Malformed ipfs_hash strings are rejected with 400, nothing enqueued."""
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    _install_balance_ok(mh, monkeypatch)
    mh.setup()

    # Numeric request_id keeps this test aimed at the ipfs_hash validator.
    # A non-numeric one would short-circuit on the new upfront REQUEST_ID_RE
    # guard at :meth:`_handle_signed_requests` and the ipfs_hash branch
    # would never run — 400 for the wrong reason.
    body = _make_signed_request_body(ipfs_hash=bad_hash, request_id="1")
    http_msg: Any = make_http_msg(body)
    mh._handle_signed_requests(http_msg, http_dialogue)

    resp = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.BAD_REQUEST_CODE.value
    assert handler_context.shared_state["pending_tasks"] == []


@pytest.mark.parametrize(
    "delivery_rate_str",
    [
        "-1",
        str(hmod.MAX_DELIVERY_RATE + 1),
    ],
    ids=["negative", "above_uint256"],
)
def test_signed_requests_rejects_out_of_range_delivery_rate(
    handler_context: Any,
    http_dialogue: Any,
    monkeypatch: Any,
    delivery_rate_str: str,
) -> None:
    """Delivery rate outside [MIN_DELIVERY_RATE, MAX_DELIVERY_RATE] is rejected."""
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    _install_balance_ok(mh, monkeypatch)
    mh.setup()

    # Numeric request_id keeps this aimed at the delivery-rate range check;
    # a non-numeric one would short-circuit at the REQUEST_ID_RE guard.
    body = _make_signed_request_body(delivery_rate=delivery_rate_str, request_id="1")
    http_msg: Any = make_http_msg(body)
    mh._handle_signed_requests(http_msg, http_dialogue)

    resp = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.BAD_REQUEST_CODE.value
    assert handler_context.shared_state["pending_tasks"] == []


def test_signed_requests_accepts_zero_delivery_rate(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """delivery_rate=0 is accepted so free off-chain tasks can be enqueued."""
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    _install_balance_ok(mh, monkeypatch)
    mh.setup()

    body = _make_signed_request_body(delivery_rate="0", request_id="7")
    http_msg: Any = make_http_msg(body)
    mh._handle_signed_requests(http_msg, http_dialogue)

    resp = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.OK_CODE.value
    assert len(handler_context.shared_state["pending_tasks"]) == 1


def test_enqueue_offchain_request_reserved_keys_filter_blocks_body_overrides(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """Client-supplied body keys can not override trusted enqueue-time fields.

    ``_enqueue_offchain_request`` merges the parsed HTTP body via ``**data``
    into the pending task dict. Without the ``reserved_keys`` filter, a
    request carrying an extra ``requestId`` / ``request_id`` /
    ``is_offchain`` / ``data`` / ``request_delivery_rate`` key would
    silently overwrite the trusted values the handler just set. That
    would (a) re-plant the wire-format ``str`` ``request_id`` and defeat
    the ``int(request_id)`` coercion at ingress, (b) let the client route
    an off-chain task down the on-chain IPFS-GET branch by flipping
    ``is_offchain`` to False, (c) replace the trusted content-hash bytes
    with attacker-controlled payload bytes, and (d) plant a
    ``request_delivery_rate`` that bypasses the tool-minimum-price gate
    while the on-chain-signed rate the requester is actually charged
    stays untouched.

    :param handler_context: pytest fixture, mech HTTP handler test context.
    :param http_dialogue: pytest fixture, HTTP dialogue stub.
    :param monkeypatch: pytest fixture, per-test monkeypatch helper.
    """
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    _install_balance_ok(mh, monkeypatch)
    mh.setup()

    body = _make_signed_request_body(delivery_rate="123", request_id="42")
    # ``body["request_id"]`` (snake) IS the source of truth read by
    # ``_handle_signed_requests`` — the reserved_keys filter's job is to
    # stop it from ALSO surviving as a snake-case dupe on the pending
    # task alongside the trusted ``requestId`` (camel) int. So the
    # source-of-truth ``"42"`` stays; camel + everything else is poisoned.
    body["requestId"] = "999999"
    body["is_offchain"] = "false"
    body["data"] = "0x" + "de" * 32
    body["request_delivery_rate"] = "0"
    # Round-3 additions: reserved_keys was extended to cover the four keys
    # ``_handle_done_task`` re-spreads via ``**executing_task`` and the
    # ``tool`` used for pricing / metrics. A body value here would
    # otherwise ride through executing_task → done_task → consensus and
    # reroute delivery, misattribute the executor, desync the on-chain
    # signature, or point the tool-price gate at the wrong tool.
    body["mech_address"] = "0xATTACKERMECH"
    body["task_executor_address"] = "0xATTACKEREXECUTOR"
    body["request_id_nonce"] = "999"
    body["requestIdWithNonce"] = "999"
    body["tool"] = "attacker-tool"
    # Plus a benign key that must survive the filter untouched, so the
    # test also pins that non-reserved keys still flow through.
    body["signature"] = "0xdead"

    http_msg: Any = make_http_msg(body)
    mh._handle_signed_requests(http_msg, http_dialogue)

    resp = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.OK_CODE.value
    pending = handler_context.shared_state["pending_tasks"]
    assert len(pending) == 1
    task = pending[0]
    # Trusted values won on every reserved key.
    assert task[hmod.RequestKey.REQUEST_ID_CAMEL.value] == 42
    assert task[hmod.RequestKey.IS_OFFCHAIN.value] is True
    assert task[hmod.BodyKey.DATA.value] == bytes.fromhex("ab" * 32)
    assert task[hmod.RequestKey.REQUEST_DELIVERY_RATE.value] == 123
    # Snake-case ``request_id`` must never appear on the pending task:
    # ``_handle_done_task`` spreads ``**executing_task`` into ``done_task``
    # and this key would silently overwrite the canonical int, resurrecting
    # the mixed-type ``sorted()`` crash + str/int keying divergence the
    # PR chain closes.
    assert hmod.RequestKey.REQUEST_ID.value not in task
    # Round-3 keys must not leak into the pending task either.
    for reserved in (
        "mech_address",
        "task_executor_address",
        "request_id_nonce",
        "requestIdWithNonce",
        "tool",
    ):
        assert reserved not in task, f"reserved key {reserved!r} leaked from body"
    # Non-reserved key still flows through.
    assert task["signature"] == "0xdead"


# --- guard tests for the defensive paths added in the remediation -----------


def _http_handler(handler_context: Any, monkeypatch: Any) -> MechHttpHandler:
    mh = MechHttpHandler(name="http", skill_context=handler_context)
    monkeypatch.setattr(mh, "start_prometheus_server", MagicMock())
    mh.setup()
    return mh


def test_build_402_challenge_raises_without_balance_tracker_address(
    handler_context: Any, monkeypatch: Any
) -> None:
    """The 402 builder refuses to emit zero-address deposit instructions."""
    mh = _http_handler(handler_context, monkeypatch)
    with pytest.raises(ValueError, match="balance_tracker_address"):
        mh._build_402_challenge(
            {hmod.ResponseKey.PAYMENT_TYPE.value: "0x01"},
            error_msg="insufficient balance",
        )


def test_build_402_challenge_payment_type_case_insensitive(
    handler_context: Any, monkeypatch: Any
) -> None:
    """A checksummed payment_type resolves against the lower-cased asset map."""
    handler_context.params.payment_type_to_asset_address = {"0xabc": "0xAssetAddr"}
    mh = _http_handler(handler_context, monkeypatch)
    challenge = mh._build_402_challenge(
        {
            hmod.ResponseKey.BALANCE_TRACKER_ADDRESS.value: "0xBalanceTracker",
            hmod.ResponseKey.PAYMENT_TYPE.value: "0xABC",
        },
        error_msg="insufficient balance",
    )
    assert challenge["asset"] == "0xAssetAddr"


def test_build_402_challenge_none_payment_type_is_safe(
    handler_context: Any, monkeypatch: Any
) -> None:
    """A present-but-None payment_type must not AttributeError on .lower()."""
    mh = _http_handler(handler_context, monkeypatch)
    challenge = mh._build_402_challenge(
        {
            hmod.ResponseKey.BALANCE_TRACKER_ADDRESS.value: "0xBalanceTracker",
            hmod.ResponseKey.PAYMENT_TYPE.value: None,
        },
        error_msg="insufficient balance",
    )
    assert challenge["asset"] == hmod.ZERO_ADDRESS


def test_send_ok_response_rejects_non_newline_extra_headers(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """A non-newline-terminated header block is rejected (can't merge body)."""
    mh = _http_handler(handler_context, monkeypatch)
    http_msg: Any = make_http_msg({"request_id": "req-x"})
    with pytest.raises(ValueError, match="newline-terminated"):
        mh._send_ok_response(
            http_msg=http_msg,
            http_dialogue=http_dialogue,
            data={"request_id": "req-x"},
            extra_headers="X-No-Newline: y",
        )


def test_send_rejection_response_rejects_non_newline_extra_headers(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """Same newline guard on the rejection sender."""
    mh = _http_handler(handler_context, monkeypatch)
    http_msg: Any = make_http_msg({"request_id": "req-x"})
    with pytest.raises(ValueError, match="newline-terminated"):
        mh._send_rejection_response(
            http_msg,
            http_dialogue,
            "req-x",
            reason="r",
            status_code=HttpCode.PAYMENT_REQUIRED_CODE.value,
            status_text="t",
            extra_headers="X-No-Newline: y",
        )


def test_handler_replies_500_when_402_build_fails(
    handler_context: Any, http_dialogue: Any, monkeypatch: Any
) -> None:
    """If the 402 builder raises, the handler replies 500 instead of hanging."""
    monkeypatch.setattr(
        MechHttpHandler,
        "_check_offchain_requester_balance",
        lambda self, sender, delivery_rate: {
            hmod.ResponseKey.STATUS.value: hmod.ResponseStatus.OK.value,
            hmod.ResponseKey.AVAILABLE_AMOUNT.value: 0,
            hmod.ResponseKey.REQUIRED_AMOUNT.value: int(delivery_rate),
        },
    )
    mh = _http_handler(handler_context, monkeypatch)
    http_msg: Any = make_http_msg(
        _make_signed_request_body(delivery_rate="1", request_id="500")
    )
    mh._handle_signed_requests(http_msg, http_dialogue)

    resp = handler_context.outbox.sent[-1]
    assert resp.status_code == HttpCode.INTERNAL_SERVER_ERROR_CODE.value
