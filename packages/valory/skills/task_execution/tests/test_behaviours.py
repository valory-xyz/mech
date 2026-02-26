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

"""This package contains the tests for the behaviours."""

import json
import time
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, Callable, Dict, Generator, Tuple
from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY

import packages.valory.skills.task_execution.behaviours as beh_mod


def clear_registry() -> None:
    """Clears the Prometheus registry"""
    collectors = list(REGISTRY._names_to_collectors.values())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass


@pytest.fixture(autouse=True)
def cleanup() -> Generator[None, None, None]:
    """Clears the Prometheus registry after each tests"""
    yield
    clear_registry()


def test_happy_path_executes_and_stores(
    behaviour: Any,
    shared_state: Dict[str, Any],
    params_stub: Any,
    fake_dialogue: Any,
    done_future: Callable[[Tuple[Any, ...]], Any],
    monkeypatch: Any,
    patch_ipfs_multihash: Callable[[], None],
    disable_polling: Callable[[], None],
) -> None:
    """Execute a valid task and store the result."""
    patch_ipfs_multihash()
    disable_polling()

    valid_cid: str = "bafybeigdyrzt5u36sq3x7xvaf2h2k6g2r5fpmy7bcxfbcdx7djzn2k2f3u"

    behaviour._all_tools["sum"] = (
        "tool_py_src",
        "run",
        {"params": {"default_model": "gpt-4o-mini"}},
    )
    behaviour._tools_to_package_hash["sum"] = "hashsum"
    params_stub.tools_to_pricing = {"sum": 0}

    req_id: int = 42
    shared_state[beh_mod.PENDING_TASKS].append(
        {
            "requestId": req_id,
            "request_delivery_rate": 100,
            "data": b"fake-ipfs-pointer",
            "contract_address": "0xmech",
        }
    )
    params_stub.request_id_to_num_timeouts[req_id] = 0
    shared_state[beh_mod.REQUEST_ID_TO_DELIVERY_RATE_INFO][req_id] = 100

    monkeypatch.setattr(
        behaviour, "_build_ipfs_get_file_req", lambda *a, **k: (object(), fake_dialogue)
    )
    monkeypatch.setattr(
        behaviour,
        "_build_ipfs_store_file_req",
        lambda files, **k: (object(), fake_dialogue),
    )

    monkeypatch.setattr(behaviour, "_ensure_payment_model", lambda: True)

    call_no: Dict[str, int] = {"n": 0}

    def send_message_stub(
        msg: Any,
        dlg: Any,
        callback: Callable[[Any, Any], None],
        error_callback: Any = None,
    ) -> None:
        """
        Stub send_message to deliver GET/STORE callbacks.

        :param msg: Message-like object passed by the behaviour.
        :type msg: Any
        :param dlg: Dialogue-like object passed to the callback.
        :type dlg: Any
        :param callback: Callback to invoke with a fake IPFS response.
        :type callback: Callable[[Any, Any], None]
        """
        call_no["n"] += 1
        if call_no["n"] == 1:
            task_body: Dict[str, Any] = {"prompt": "add 2+2", "tool": "sum"}
            fake_get_response: Any = type(
                "Msg", (), {"files": {"task.json": json.dumps(task_body)}}
            )()
            callback(fake_get_response, dlg)
        else:
            fake_store_response: Any = type("Msg", (), {"ipfs_hash": valid_cid})()
            callback(fake_store_response, dlg)
        params_stub.in_flight_req = True

    monkeypatch.setattr(behaviour, "send_message", send_message_stub)

    token_cb: Any = type("CB", (), {"cost_dict": {"input": 10, "output": 5}})()
    keychain: object = object()
    result_tuple = ("4", "add 2+2", {"tx": "0xabc"}, token_cb, keychain)
    monkeypatch.setattr(
        behaviour, "_submit_task", lambda *a, **k: done_future(result_tuple)
    )

    params_stub.in_flight_req = False
    behaviour.act()
    params_stub.in_flight_req = False
    behaviour.act()

    assert len(shared_state[beh_mod.DONE_TASKS]) == 1
    done: Dict[str, Any] = shared_state[beh_mod.DONE_TASKS][0]
    assert done["request_id"] == req_id
    assert done["tool"] == "sum"
    assert done["mech_address"] == "0xmech"
    assert done["task_result"] == f"mh:{valid_cid}"

    assert behaviour._executing_task is None
    assert behaviour._async_result is None
    assert behaviour._request_handling_deadline is None


def test_pricing_too_low_marks_invalid_and_stores_stub(
    behaviour: Any,
    shared_state: Dict[str, Any],
    params_stub: Any,
    fake_dialogue: Any,
    monkeypatch: Any,
    done_future: Callable[[Tuple[Any, ...]], Any],
    patch_ipfs_multihash: Callable[[], None],
    disable_polling: Callable[[], None],
) -> None:
    """Reject underpriced task and store invalid response."""
    patch_ipfs_multihash()
    disable_polling()
    behaviour._tools_to_package_hash = {"sum": "fakehash"}
    behaviour._tools_to_pricing["sum"] = 200
    behaviour._all_tools["sum"] = ("tool_py_src", "run", {"params": {}})

    req_id: int = 99
    shared_state[beh_mod.PENDING_TASKS].append(
        {
            "requestId": req_id,
            "request_delivery_rate": 100,
            "data": b"whatever",
            "contract_address": "0xmech",
        }
    )
    params_stub.request_id_to_num_timeouts[req_id] = 0
    shared_state[beh_mod.REQUEST_ID_TO_DELIVERY_RATE_INFO][req_id] = 100

    monkeypatch.setattr(
        behaviour, "_build_ipfs_get_file_req", lambda *a, **k: (object(), fake_dialogue)
    )
    monkeypatch.setattr(
        behaviour,
        "_build_ipfs_store_file_req",
        lambda files, **k: (object(), fake_dialogue),
    )
    monkeypatch.setattr(behaviour, "_ensure_payment_model", lambda: True)

    def _fail_submit(*a: Any, **k: Any) -> None:
        """
        Ensure execution is not attempted for invalid pricing.

        :param a: Ignored positional arguments.
        :type a: Any
        :param k: Ignored keyword arguments.
        :type k: Any
        """
        raise AssertionError("_submit_task must not be called for invalid pricing")

    monkeypatch.setattr(behaviour, "_submit_task", _fail_submit)

    valid_cid: str = "bafybeigdyrzt5u36sq3x7xvaf2h2k6g2r5fpmy7bcxfbcdx7djzn2k2f3u"
    calls: Dict[str, int] = {"n": 0}

    def send_message_stub(
        msg: Any,
        dlg: Any,
        callback: Callable[[Any, Any], None],
        error_callback: Any = None,
    ) -> None:
        """
        Stub send_message to deliver GET/STORE callbacks for invalid pricing.

        :param msg: Message-like object passed by the behaviour.
        :type msg: Any
        :param dlg: Dialogue-like object passed to the callback.
        :type dlg: Any
        :param callback: Callback to invoke with a fake IPFS response.
        :type callback: Callable[[Any, Any], None]
        """
        calls["n"] += 1
        if calls["n"] == 1:
            body: Dict[str, Any] = {"prompt": "add 2+2", "tool": "sum"}
            callback(type("Msg", (), {"files": {"task.json": json.dumps(body)}})(), dlg)
        else:
            callback(type("Msg", (), {"ipfs_hash": valid_cid})(), dlg)
        params_stub.in_flight_req = True

    monkeypatch.setattr(behaviour, "send_message", send_message_stub)

    params_stub.in_flight_req = False
    behaviour.act()
    params_stub.in_flight_req = False
    behaviour.act()

    assert calls["n"] == 2
    assert len(shared_state[beh_mod.DONE_TASKS]) == 1
    done: Dict[str, Any] = shared_state[beh_mod.DONE_TASKS][0]
    assert done["request_id"] == req_id
    assert done.get("tool") is None
    assert "dynamic_tool_cost" not in done
    assert done["task_result"] == f"mh:{valid_cid}"
    assert behaviour._executing_task is None


def test_broken_process_pool_restart(
    behaviour: Any,
    shared_state: Dict[str, Any],
    params_stub: Any,
    fake_dialogue: Any,
    done_future: Callable[[Tuple[Any, ...]], Future],
    monkeypatch: Any,
    patch_ipfs_multihash: Callable[[], None],
    disable_polling: Callable[[], None],
) -> None:
    """Restart Pebble pool and retry when schedule() raises once."""

    patch_ipfs_multihash()
    disable_polling()

    # Arrange: minimal tool wiring
    behaviour._all_tools["sum"] = ("py", "run", {"params": {}})
    behaviour._tools_to_package_hash["sum"] = "fake-package-hash"

    # Queue a single pending task
    req_id: int = 1
    shared_state[beh_mod.PENDING_TASKS].append(
        {
            "requestId": req_id,
            "request_delivery_rate": 100,
            "data": b"x",
            "contract_address": "0xmech",
        }
    )
    params_stub.request_id_to_num_timeouts[req_id] = 0
    shared_state[beh_mod.REQUEST_ID_TO_DELIVERY_RATE_INFO][req_id] = 100

    # Stub IPFS get/store
    monkeypatch.setattr(
        behaviour, "_build_ipfs_get_file_req", lambda *a, **k: (object(), fake_dialogue)
    )
    monkeypatch.setattr(
        behaviour,
        "_build_ipfs_store_file_req",
        lambda files, **k: (object(), fake_dialogue),
    )

    monkeypatch.setattr(behaviour, "_ensure_payment_model", lambda: True)

    calls: Dict[str, int] = {"n": 0}

    class BrokenOncePool:
        """Pebble-like pool stub that raises once, then returns a done future."""

        def schedule(self, *a: Any, **k: Any) -> Future:
            calls["n"] += 1
            if calls["n"] == 1:
                # Simulate a failure during scheduling (e.g., broken pool)
                raise RuntimeError("boom")
            # On second call, return a completed future with the 5-tuple
            return done_future(
                ("ok", "p", {"tx": 1}, type("CB", (), {"cost_dict": {}})(), object())
            )

    # Replace the behaviour's executor with our stub
    monkeypatch.setattr(behaviour, "_executor", BrokenOncePool())

    # Flag that _restart_executor() was called
    restarted: Dict[str, bool] = {"flag": False}
    monkeypatch.setattr(
        behaviour, "_restart_executor", lambda: restarted.__setitem__("flag", True)
    )

    # send_message stub: route to the right handler based on callback identity
    def send_message_stub(
        msg: Any, dlg: Any, cb: Callable[[Any, Any], None], error_cb: Any = None
    ) -> None:
        func = getattr(cb, "__func__", cb)
        if func is beh_mod.TaskExecutionBehaviour._handle_get_task:
            body: Dict[str, Any] = {"prompt": "p", "tool": "sum"}
            cb(SimpleNamespace(files={"task.json": json.dumps(body)}), dlg)
        elif func is beh_mod.TaskExecutionBehaviour._handle_store_response:
            cb(SimpleNamespace(ipfs_hash="bafyok"), dlg)
        else:
            raise AssertionError(f"Unexpected callback: {cb}")
        params_stub.in_flight_req = True

    monkeypatch.setattr(behaviour, "send_message", send_message_stub)

    # Tick the behaviour twice: first time triggers schedule failure + restart,
    # second time schedules successfully and completes the flow.
    params_stub.in_flight_req = False
    behaviour.act()

    params_stub.in_flight_req = False
    behaviour.act()

    # Assertions
    assert restarted[
        "flag"
    ], "executor should have been restarted after schedule() failure"
    assert len(shared_state[beh_mod.DONE_TASKS]) == 1
    done: Dict[str, Any] = shared_state[beh_mod.DONE_TASKS][0]
    assert done["request_id"] == req_id
    assert done["task_result"] == "mh:bafyok"


def test_invalid_tool_is_recorded_and_no_execution(
    behaviour: Any,
    shared_state: Dict[str, Any],
    params_stub: Any,
    fake_dialogue: Any,
    monkeypatch: Any,
    patch_ipfs_multihash: Callable[[], None],
    disable_polling: Callable[[], None],
) -> None:
    """Record invalid tool and store without executing."""
    patch_ipfs_multihash()
    disable_polling()

    req_id: int = 5
    my_mech = params_stub.agent_mech_contract_addresses[0]
    shared_state[beh_mod.PENDING_TASKS].append(
        {
            "requestId": req_id,
            "request_delivery_rate": 100,
            "data": b"x",
            "contract_address": my_mech,
            "priorityMech": my_mech,
        }
    )
    params_stub.request_id_to_num_timeouts[req_id] = 0
    shared_state[beh_mod.REQUEST_ID_TO_DELIVERY_RATE_INFO][req_id] = 100

    monkeypatch.setattr(
        behaviour, "_build_ipfs_get_file_req", lambda *a, **k: (object(), fake_dialogue)
    )
    monkeypatch.setattr(
        behaviour,
        "_build_ipfs_store_file_req",
        lambda *a, **k: (object(), fake_dialogue),
    )

    monkeypatch.setattr(
        behaviour,
        "_submit_task",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not execute")),
    )
    monkeypatch.setattr(behaviour, "_ensure_payment_model", lambda: True)

    def send_message_stub(
        msg: Any, dlg: Any, cb: Callable[[Any, Any], None], error_cb: Any = None
    ) -> None:
        """
        Stub send_message to produce unknown tool on GET and store afterwards.

        :param msg: Message-like object passed by the behaviour.
        :type msg: Any
        :param dlg: Dialogue-like object passed to the callback.
        :type dlg: Any
        :param cb: Callback to invoke with a fake IPFS response.
        :type cb: Callable[[Any, Any], None]
        """
        func = getattr(cb, "__func__", cb)
        if func is beh_mod.TaskExecutionBehaviour._handle_get_task:
            cb(
                SimpleNamespace(
                    files={"task.json": json.dumps({"prompt": "p", "tool": "unknown"})}
                ),
                dlg,
            )
        elif func is beh_mod.TaskExecutionBehaviour._handle_store_response:
            cb(SimpleNamespace(ipfs_hash="bafyinval"), dlg)
        else:
            raise AssertionError(f"Unexpected callback: {cb}")
        params_stub.in_flight_req = True

    monkeypatch.setattr(behaviour, "send_message", send_message_stub)

    params_stub.in_flight_req = False
    behaviour.act()
    params_stub.in_flight_req = False
    behaviour.act()

    assert len(shared_state[beh_mod.DONE_TASKS]) == 1
    done: Dict[str, Any] = shared_state[beh_mod.DONE_TASKS][0]
    assert done["request_id"] == req_id
    assert done["tool"] == "unknown"
    assert done["task_result"] == "mh:bafyinval"


def test_ipfs_aux_task_removed_from_queue(
    behaviour: Any,
    shared_state: Dict[str, Any],
    params_stub: Any,
    fake_dialogue: Any,
    monkeypatch: Any,
    disable_polling: Callable[[], None],
    patch_ipfs_multihash: Callable[[], None],
) -> None:
    """Remove aux IPFS task from queue after successful store."""
    disable_polling()
    patch_ipfs_multihash()
    shared_state[beh_mod.IPFS_TASKS].append(
        {"request_id": "aux-1", "ipfs_data": '{"foo":"bar"}'}
    )
    monkeypatch.setattr(
        behaviour,
        "_build_ipfs_store_file_req",
        lambda *a, **k: (object(), fake_dialogue),
    )
    monkeypatch.setattr(behaviour, "_ensure_payment_model", lambda: True)

    def send_message_stub(
        msg: Any, dlg: Any, cb: Callable[[Any, Any], None], error_cb: Any = None
    ) -> None:
        """
        Stub send_message and mark request as in-flight.

        :param msg: Message-like object passed by the behaviour.
        :type msg: Any
        :param dlg: Dialogue-like object passed to the callback.
        :type dlg: Any
        :param cb: Callback to invoke with a fake STORE_FILES response.
        :type cb: Callable[[Any, Any], None]
        """
        cb(SimpleNamespace(ipfs_hash="bafyaux"), dlg)
        params_stub.in_flight_req = True

    monkeypatch.setattr(behaviour, "send_message", send_message_stub)
    params_stub.in_flight_req = False
    behaviour.act()
    assert shared_state[beh_mod.IPFS_TASKS] == []


def test_behaviour_status_check_and_proper_updates(
    behaviour: Any,
    shared_state: Dict[str, Any],
    params_stub: Any,
    fake_dialogue: Any,
    done_future: Callable[[Tuple[Any, ...]], Any],
    monkeypatch: Any,
    patch_ipfs_multihash: Callable[[], None],
    disable_polling: Callable[[], None],
) -> None:
    """Execute a valid task and store the result."""
    patch_ipfs_multihash()
    disable_polling()

    behaviour._all_tools["sum"] = (
        "tool_py_src",
        "run",
        {"params": {"default_model": "gpt-4o-mini"}},
    )
    behaviour._tools_to_package_hash["sum"] = "hashsum"
    params_stub.tools_to_pricing = {"sum": 0}

    req_id: int = 42
    shared_state[beh_mod.PENDING_TASKS].append(
        {
            "requestId": req_id,
            "request_delivery_rate": 100,
            "data": b"fake-ipfs-pointer",
            "contract_address": "0xmech",
        }
    )
    params_stub.request_id_to_num_timeouts[req_id] = 0
    shared_state[beh_mod.REQUEST_ID_TO_DELIVERY_RATE_INFO][req_id] = 100

    monkeypatch.setattr(
        behaviour, "_build_ipfs_get_file_req", lambda *a, **k: (object(), fake_dialogue)
    )
    monkeypatch.setattr(
        behaviour,
        "_build_ipfs_store_file_req",
        lambda files, **k: (object(), fake_dialogue),
    )
    # patch _execute_task to not work on the pending task
    # to simulate busy/unresponsive mech
    monkeypatch.setattr(
        behaviour,
        "_execute_task",
        MagicMock(),
    )
    monkeypatch.setattr(time, "time", lambda: 1.0)
    monkeypatch.setattr(behaviour, "_ensure_payment_model", lambda: True)

    behaviour.act()

    assert behaviour._executing_task is None
    assert behaviour._async_result is None
    assert behaviour._request_handling_deadline is None

    # patch variable after patching time
    behaviour.last_status_check_time = time.time()

    assert len(shared_state[beh_mod.PENDING_TASKS]) == 1
    assert behaviour.last_status_check_time == 1.0

    # update time to less than status check interval
    monkeypatch.setattr(time, "time", lambda: beh_mod.STATUS_CHECK_INTERVAL - 1)

    behaviour.act()
    # should not update the time
    assert behaviour.last_status_check_time == 1

    # update time to more than status check interval
    monkeypatch.setattr(time, "time", lambda: beh_mod.STATUS_CHECK_INTERVAL + 1)

    behaviour.act()
    # should update the time
    assert behaviour.last_status_check_time == beh_mod.STATUS_CHECK_INTERVAL + 1


# ---------------------------------------------------------------------------
# IPFS error handling: comprehensive scenario tests
# ---------------------------------------------------------------------------


def _enqueue_task(
    shared_state: Dict[str, Any],
    params_stub: Any,
    req_id: int,
    contract_address: str = "0xmech",
    priority_mech: str | None = None,
) -> None:
    """Queue a pending task and seed supporting maps.

    :param shared_state: Shared state dict.
    :param params_stub: Params-like namespace.
    :param req_id: Request ID for the task.
    :param contract_address: Mech contract address.
    :param priority_mech: Optional priorityMech override.
    """
    task: Dict[str, Any] = {
        "requestId": req_id,
        "request_delivery_rate": 100,
        "data": b"ipfs-pointer",
        "contract_address": contract_address,
    }
    if priority_mech is not None:
        task["priorityMech"] = priority_mech
    shared_state[beh_mod.PENDING_TASKS].append(task)
    params_stub.request_id_to_num_timeouts[req_id] = 0
    shared_state[beh_mod.REQUEST_ID_TO_DELIVERY_RATE_INFO][req_id] = 100


def _make_send_stub(
    behaviour: Any,
    params_stub: Any,
    get_action: Callable[[Any, Any, Callable], None],
    store_cid: str = "bafybeigdyrzt5u36sq3x7xvaf2h2k6g2r5fpmy7bcxfbcdx7djzn2k2f3u",
) -> Callable:
    """Build a send_message stub that dispatches GET vs STORE callbacks.

    :param behaviour: The behaviour under test.
    :param params_stub: Params-like namespace.
    :param get_action: Callable invoked on the first (GET) send.
    :param store_cid: CID returned by the STORE callback.
    :returns: The stub function.
    """
    calls: Dict[str, int] = {"n": 0}

    def stub(
        msg: Any, dlg: Any, callback: Callable[[Any, Any], None], error_cb: Any = None
    ) -> None:
        """Route GET vs STORE calls."""
        calls["n"] += 1
        if calls["n"] == 1:
            get_action(msg, dlg, callback)
        else:
            callback(type("Msg", (), {"ipfs_hash": store_cid})(), dlg)
        params_stub.in_flight_req = True

    return stub


def _run_two_act_cycles(behaviour: Any, params_stub: Any) -> None:
    """Run two act cycles, clearing in_flight between them.

    :param behaviour: Behaviour under test.
    :param params_stub: Params-like namespace.
    """
    params_stub.in_flight_req = False
    behaviour.act()
    params_stub.in_flight_req = False
    behaviour.act()


def _assert_clean_state(behaviour: Any) -> None:
    """Assert all per-task state is fully reset (no cascading failure).

    :param behaviour: Behaviour under test.
    """
    assert behaviour._executing_task is None
    assert behaviour._async_result is None
    assert behaviour._request_handling_deadline is None
    assert behaviour._invalid_request is False
    assert behaviour._ipfs_error_reason is None


# Scenario A: IPFS download fails (codec mismatch) → descriptive error on-chain
def test_ipfs_error_codec_mismatch_delivers_descriptive_rejection(
    behaviour: Any,
    shared_state: Dict[str, Any],
    params_stub: Any,
    fake_dialogue: Any,
    monkeypatch: Any,
    patch_ipfs_multihash: Callable[[], None],
    disable_polling: Callable[[], None],
) -> None:
    """IPFS codec mismatch error delivers descriptive error response on-chain."""
    patch_ipfs_multihash()
    disable_polling()

    req_id: int = 77
    _enqueue_task(shared_state, params_stub, req_id)

    monkeypatch.setattr(
        behaviour, "_build_ipfs_get_file_req", lambda *a, **k: (object(), fake_dialogue)
    )

    stored_payloads: list = []

    def capture_store(files: Dict[str, str], **k: Any) -> Tuple[object, Any]:
        """Capture payload stored to IPFS."""
        stored_payloads.append(files)
        return object(), fake_dialogue

    monkeypatch.setattr(behaviour, "_build_ipfs_store_file_req", capture_store)
    monkeypatch.setattr(behaviour, "_ensure_payment_model", lambda: True)

    error_reason = "protobuf: (PBNode) invalid wireType, expected 2, got 3"

    def simulate_ipfs_error(
        msg: Any, dlg: Any, callback: Callable[[Any, Any], None]
    ) -> None:
        """Simulate what IpfsHandler ERROR path does."""
        behaviour._invalid_request = True
        behaviour._ipfs_error_reason = (
            f"Request data could not be retrieved from IPFS (detail: {error_reason})"
        )

    monkeypatch.setattr(
        behaviour,
        "send_message",
        _make_send_stub(behaviour, params_stub, simulate_ipfs_error),
    )

    _run_two_act_cycles(behaviour, params_stub)

    assert len(shared_state[beh_mod.DONE_TASKS]) == 1
    done: Dict[str, Any] = shared_state[beh_mod.DONE_TASKS][0]
    assert done["request_id"] == req_id

    # Verify stored payload contains the exact descriptive error
    assert len(stored_payloads) == 1
    payload = json.loads(stored_payloads[0][str(req_id)])
    assert "Request data could not be retrieved from IPFS" in payload["result"]
    assert "protobuf" in payload["result"]
    assert payload["result"] != "Invalid response"

    _assert_clean_state(behaviour)


# Scenario B: IPFS download fails (generic network error) → error on-chain
def test_ipfs_error_network_failure_delivers_rejection(
    behaviour: Any,
    shared_state: Dict[str, Any],
    params_stub: Any,
    fake_dialogue: Any,
    monkeypatch: Any,
    patch_ipfs_multihash: Callable[[], None],
    disable_polling: Callable[[], None],
) -> None:
    """Generic IPFS network failure delivers error response on-chain."""
    patch_ipfs_multihash()
    disable_polling()

    req_id: int = 78
    _enqueue_task(shared_state, params_stub, req_id)

    monkeypatch.setattr(
        behaviour, "_build_ipfs_get_file_req", lambda *a, **k: (object(), fake_dialogue)
    )

    stored_payloads: list = []

    def capture_store(files: Dict[str, str], **k: Any) -> Tuple[object, Any]:
        """Capture payload."""
        stored_payloads.append(files)
        return object(), fake_dialogue

    monkeypatch.setattr(behaviour, "_build_ipfs_store_file_req", capture_store)
    monkeypatch.setattr(behaviour, "_ensure_payment_model", lambda: True)

    def simulate_network_error(
        msg: Any, dlg: Any, callback: Callable[[Any, Any], None]
    ) -> None:
        """Simulate network-level IPFS failure."""
        behaviour._invalid_request = True
        behaviour._ipfs_error_reason = (
            "Request data could not be retrieved from IPFS"
            " (detail: Failed to download: bafybeiabc123)"
        )

    monkeypatch.setattr(
        behaviour,
        "send_message",
        _make_send_stub(behaviour, params_stub, simulate_network_error),
    )

    _run_two_act_cycles(behaviour, params_stub)

    assert len(shared_state[beh_mod.DONE_TASKS]) == 1
    payload = json.loads(stored_payloads[0][str(req_id)])
    assert "Failed to download" in payload["result"]

    _assert_clean_state(behaviour)


# Scenario C: IPFS download succeeds, but task data is invalid (missing fields)
def test_ipfs_success_invalid_task_data_delivers_invalid_response(
    behaviour: Any,
    shared_state: Dict[str, Any],
    params_stub: Any,
    fake_dialogue: Any,
    monkeypatch: Any,
    patch_ipfs_multihash: Callable[[], None],
    disable_polling: Callable[[], None],
) -> None:
    """IPFS download succeeds but task data missing required fields → 'Invalid response'."""
    patch_ipfs_multihash()
    disable_polling()

    req_id: int = 79
    _enqueue_task(shared_state, params_stub, req_id)

    monkeypatch.setattr(
        behaviour, "_build_ipfs_get_file_req", lambda *a, **k: (object(), fake_dialogue)
    )

    stored_payloads: list = []

    def capture_store(files: Dict[str, str], **k: Any) -> Tuple[object, Any]:
        """Capture payload."""
        stored_payloads.append(files)
        return object(), fake_dialogue

    monkeypatch.setattr(behaviour, "_build_ipfs_store_file_req", capture_store)
    monkeypatch.setattr(behaviour, "_ensure_payment_model", lambda: True)

    def deliver_bad_task_data(
        msg: Any, dlg: Any, callback: Callable[[Any, Any], None]
    ) -> None:
        """Deliver IPFS response with missing 'tool' field."""
        bad_body: Dict[str, Any] = {"prompt": "some prompt"}  # no 'tool' key
        fake_response: Any = type(
            "Msg", (), {"files": {"task.json": json.dumps(bad_body)}}
        )()
        callback(fake_response, dlg)

    monkeypatch.setattr(
        behaviour,
        "send_message",
        _make_send_stub(behaviour, params_stub, deliver_bad_task_data),
    )

    _run_two_act_cycles(behaviour, params_stub)

    assert len(shared_state[beh_mod.DONE_TASKS]) == 1
    payload = json.loads(stored_payloads[0][str(req_id)])
    assert payload["result"] == "Invalid response"

    _assert_clean_state(behaviour)


# Scenario D: IPFS download succeeds, valid data, happy path → successful delivery
def test_ipfs_success_valid_task_delivers_result(
    behaviour: Any,
    shared_state: Dict[str, Any],
    params_stub: Any,
    fake_dialogue: Any,
    done_future: Callable[[Tuple[Any, ...]], Any],
    monkeypatch: Any,
    patch_ipfs_multihash: Callable[[], None],
    disable_polling: Callable[[], None],
) -> None:
    """IPFS download succeeds with valid task → tool executes and result delivered."""
    patch_ipfs_multihash()
    disable_polling()

    behaviour._all_tools["sum"] = (
        "tool_py_src",
        "run",
        {"params": {"default_model": "gpt-4o-mini"}},
    )
    behaviour._tools_to_package_hash["sum"] = "hashsum"
    params_stub.tools_to_pricing = {"sum": 0}

    req_id: int = 80
    _enqueue_task(shared_state, params_stub, req_id)

    monkeypatch.setattr(
        behaviour, "_build_ipfs_get_file_req", lambda *a, **k: (object(), fake_dialogue)
    )

    stored_payloads: list = []

    def capture_store(files: Dict[str, str], **k: Any) -> Tuple[object, Any]:
        """Capture payload."""
        stored_payloads.append(files)
        return object(), fake_dialogue

    monkeypatch.setattr(behaviour, "_build_ipfs_store_file_req", capture_store)
    monkeypatch.setattr(behaviour, "_ensure_payment_model", lambda: True)

    token_cb: Any = type("CB", (), {"cost_dict": {"input": 10}})()
    result_tuple = ("42", "add 2+2", {"tx": "0xabc"}, token_cb, object())
    monkeypatch.setattr(
        behaviour, "_submit_task", lambda *a, **k: done_future(result_tuple)
    )

    valid_cid: str = "bafybeigdyrzt5u36sq3x7xvaf2h2k6g2r5fpmy7bcxfbcdx7djzn2k2f3u"

    def deliver_valid_task(
        msg: Any, dlg: Any, callback: Callable[[Any, Any], None]
    ) -> None:
        """Deliver valid task data via IPFS callback."""
        body: Dict[str, Any] = {"prompt": "add 2+2", "tool": "sum"}
        callback(type("Msg", (), {"files": {"task.json": json.dumps(body)}})(), dlg)

    monkeypatch.setattr(
        behaviour,
        "send_message",
        _make_send_stub(
            behaviour, params_stub, deliver_valid_task, store_cid=valid_cid
        ),
    )

    _run_two_act_cycles(behaviour, params_stub)

    assert len(shared_state[beh_mod.DONE_TASKS]) == 1
    done: Dict[str, Any] = shared_state[beh_mod.DONE_TASKS][0]
    assert done["request_id"] == req_id
    assert done["task_result"] == f"mh:{valid_cid}"

    # Verify the result is the tool's output, not an error
    payload = json.loads(stored_payloads[0][str(req_id)])
    assert payload["result"] == "42"
    assert payload["prompt"] == "add 2+2"

    _assert_clean_state(behaviour)


# Scenario E: After IPFS error, the next task processes normally (no cascading failure)
def test_ipfs_error_does_not_cascade_to_next_task(
    behaviour: Any,
    shared_state: Dict[str, Any],
    params_stub: Any,
    fake_dialogue: Any,
    done_future: Callable[[Tuple[Any, ...]], Any],
    monkeypatch: Any,
    patch_ipfs_multihash: Callable[[], None],
    disable_polling: Callable[[], None],
) -> None:
    """After an IPFS error, the next task executes normally (no stale deadline)."""
    patch_ipfs_multihash()
    disable_polling()

    behaviour._all_tools["sum"] = (
        "tool_py_src",
        "run",
        {"params": {"default_model": "gpt-4o-mini"}},
    )
    behaviour._tools_to_package_hash["sum"] = "hashsum"
    params_stub.tools_to_pricing = {"sum": 0}

    # Queue TWO tasks
    _enqueue_task(shared_state, params_stub, req_id=100)
    _enqueue_task(shared_state, params_stub, req_id=101)

    monkeypatch.setattr(
        behaviour, "_build_ipfs_get_file_req", lambda *a, **k: (object(), fake_dialogue)
    )
    monkeypatch.setattr(
        behaviour,
        "_build_ipfs_store_file_req",
        lambda files, **k: (object(), fake_dialogue),
    )
    monkeypatch.setattr(behaviour, "_ensure_payment_model", lambda: True)

    token_cb: Any = type("CB", (), {"cost_dict": {}})()
    result_tuple = ("ok", "prompt", {"tx": "0x1"}, token_cb, object())
    monkeypatch.setattr(
        behaviour, "_submit_task", lambda *a, **k: done_future(result_tuple)
    )

    task_num: Dict[str, int] = {"n": 0}

    def send_message_stub(
        msg: Any,
        dlg: Any,
        callback: Callable[[Any, Any], None],
        error_callback: Any = None,
    ) -> None:
        """First task: IPFS error. Second task: success. STORE always succeeds."""
        task_num["n"] += 1
        func = getattr(callback, "__func__", callback)
        if func is beh_mod.TaskExecutionBehaviour._handle_get_task:
            if task_num["n"] == 1:
                # Task 100: IPFS download fails
                behaviour._invalid_request = True
                behaviour._ipfs_error_reason = (
                    "Request data could not be retrieved from IPFS"
                    " (detail: content unpinned)"
                )
            else:
                # Task 101: IPFS download succeeds
                body: Dict[str, Any] = {"prompt": "prompt", "tool": "sum"}
                callback(
                    type("Msg", (), {"files": {"t.json": json.dumps(body)}})(), dlg
                )
        elif func is beh_mod.TaskExecutionBehaviour._handle_store_response:
            callback(type("Msg", (), {"ipfs_hash": "bafyok"})(), dlg)
        params_stub.in_flight_req = True

    monkeypatch.setattr(behaviour, "send_message", send_message_stub)

    # Task 100: IPFS error → rejection pipeline
    params_stub.in_flight_req = False
    behaviour.act()
    params_stub.in_flight_req = False
    behaviour.act()

    assert len(shared_state[beh_mod.DONE_TASKS]) == 1
    first_done = shared_state[beh_mod.DONE_TASKS][0]
    assert first_done["request_id"] == 100
    _assert_clean_state(behaviour)

    # Task 101: should execute normally — no cascading failure
    params_stub.in_flight_req = False
    behaviour.act()
    params_stub.in_flight_req = False
    behaviour.act()

    assert len(shared_state[beh_mod.DONE_TASKS]) == 2
    second_done = shared_state[beh_mod.DONE_TASKS][1]
    assert second_done["request_id"] == 101
    assert second_done["task_result"] == "mh:bafyok"
    _assert_clean_state(behaviour)


# Scenario F: IPFS download succeeds but empty task data → invalid response
def test_ipfs_success_empty_task_data_delivers_invalid_response(
    behaviour: Any,
    shared_state: Dict[str, Any],
    params_stub: Any,
    fake_dialogue: Any,
    monkeypatch: Any,
    patch_ipfs_multihash: Callable[[], None],
    disable_polling: Callable[[], None],
) -> None:
    """IPFS download succeeds but task data is empty dict → 'Invalid response'."""
    patch_ipfs_multihash()
    disable_polling()

    req_id: int = 81
    _enqueue_task(shared_state, params_stub, req_id)

    monkeypatch.setattr(
        behaviour, "_build_ipfs_get_file_req", lambda *a, **k: (object(), fake_dialogue)
    )

    stored_payloads: list = []

    def capture_store(files: Dict[str, str], **k: Any) -> Tuple[object, Any]:
        """Capture payload."""
        stored_payloads.append(files)
        return object(), fake_dialogue

    monkeypatch.setattr(behaviour, "_build_ipfs_store_file_req", capture_store)
    monkeypatch.setattr(behaviour, "_ensure_payment_model", lambda: True)

    def deliver_empty_data(
        msg: Any, dlg: Any, callback: Callable[[Any, Any], None]
    ) -> None:
        """Deliver empty task data."""
        callback(type("Msg", (), {"files": {"task.json": json.dumps({})}})(), dlg)

    monkeypatch.setattr(
        behaviour,
        "send_message",
        _make_send_stub(behaviour, params_stub, deliver_empty_data),
    )

    _run_two_act_cycles(behaviour, params_stub)

    assert len(shared_state[beh_mod.DONE_TASKS]) == 1
    payload = json.loads(stored_payloads[0][str(req_id)])
    assert payload["result"] == "Invalid response"
    _assert_clean_state(behaviour)


# Scenario G: _ipfs_error_reason is None for non-IPFS failures (backward compat)
def test_non_ipfs_failure_uses_generic_invalid_response(
    behaviour: Any,
    shared_state: Dict[str, Any],
    params_stub: Any,
    fake_dialogue: Any,
    monkeypatch: Any,
    patch_ipfs_multihash: Callable[[], None],
    disable_polling: Callable[[], None],
) -> None:
    """When _ipfs_error_reason is None, result falls back to 'Invalid response'."""
    patch_ipfs_multihash()
    disable_polling()

    # Use pricing rejection to trigger _invalid_request without _ipfs_error_reason
    behaviour._tools_to_package_hash = {"sum": "fakehash"}
    behaviour._tools_to_pricing["sum"] = 200  # price > delivery_rate of 100
    behaviour._all_tools["sum"] = ("tool_py_src", "run", {"params": {}})

    req_id: int = 82
    _enqueue_task(shared_state, params_stub, req_id)

    monkeypatch.setattr(
        behaviour, "_build_ipfs_get_file_req", lambda *a, **k: (object(), fake_dialogue)
    )

    stored_payloads: list = []

    def capture_store(files: Dict[str, str], **k: Any) -> Tuple[object, Any]:
        """Capture payload."""
        stored_payloads.append(files)
        return object(), fake_dialogue

    monkeypatch.setattr(behaviour, "_build_ipfs_store_file_req", capture_store)
    monkeypatch.setattr(behaviour, "_ensure_payment_model", lambda: True)

    def deliver_valid_task(
        msg: Any, dlg: Any, callback: Callable[[Any, Any], None]
    ) -> None:
        """Deliver valid task data — pricing rejection happens in callback."""
        body: Dict[str, Any] = {"prompt": "add 2+2", "tool": "sum"}
        callback(type("Msg", (), {"files": {"t.json": json.dumps(body)}})(), dlg)

    monkeypatch.setattr(
        behaviour,
        "send_message",
        _make_send_stub(behaviour, params_stub, deliver_valid_task),
    )

    _run_two_act_cycles(behaviour, params_stub)

    assert len(shared_state[beh_mod.DONE_TASKS]) == 1
    payload = json.loads(stored_payloads[0][str(req_id)])
    # Without _ipfs_error_reason set, falls back to "Invalid response"
    assert payload["result"] == "Invalid response"
    _assert_clean_state(behaviour)
