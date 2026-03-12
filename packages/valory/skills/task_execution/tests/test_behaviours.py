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

    task_body: Dict[str, Any] = {"prompt": "add 2+2", "tool": "sum"}
    get_response = SimpleNamespace(files={"task.json": json.dumps(task_body)})
    store_response = SimpleNamespace(ipfs_hash=valid_cid)
    responses = iter([get_response, store_response])

    def send_message_stub(
        msg: Any,
        dlg: Any,
        callback: Callable[[Any, Any], None],
        error_callback: Any = None,
    ) -> None:
        """Stub send_message with sequential GET then STORE responses."""
        callback(next(responses), dlg)
        params_stub.in_flight_req = True

    monkeypatch.setattr(behaviour, "send_message", send_message_stub)

    token_cb = SimpleNamespace(cost_dict={"input": 10, "output": 5}, actual_model=None)
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
    monkeypatch.setattr(
        behaviour,
        "_submit_task",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("_submit_task must not be called for invalid pricing")
        ),
    )

    valid_cid: str = "bafybeigdyrzt5u36sq3x7xvaf2h2k6g2r5fpmy7bcxfbcdx7djzn2k2f3u"
    task_body: Dict[str, Any] = {"prompt": "add 2+2", "tool": "sum"}
    get_response = SimpleNamespace(files={"task.json": json.dumps(task_body)})
    store_response = SimpleNamespace(ipfs_hash=valid_cid)
    send_calls: list = []
    responses = iter([get_response, store_response])

    def send_message_stub(
        msg: Any,
        dlg: Any,
        callback: Callable[[Any, Any], None],
        error_callback: Any = None,
    ) -> None:
        """Stub send_message with sequential GET then STORE responses."""
        send_calls.append(1)
        callback(next(responses), dlg)
        params_stub.in_flight_req = True

    monkeypatch.setattr(behaviour, "send_message", send_message_stub)

    params_stub.in_flight_req = False
    behaviour.act()
    params_stub.in_flight_req = False
    behaviour.act()

    assert len(send_calls) == 2
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

    schedule_results = iter(
        [
            RuntimeError("boom"),  # first call raises
            done_future(
                (
                    "ok",
                    "p",
                    {"tx": 1},
                    SimpleNamespace(cost_dict={}, actual_model=None),
                    object(),
                )
            ),  # second call succeeds
        ]
    )

    class BrokenOncePool:
        """Pebble-like pool stub that raises once, then returns a done future."""

        def schedule(self, *a: Any, **k: Any) -> Future:
            result = next(schedule_results)
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr(behaviour, "_executor", BrokenOncePool())

    restarted: list = []
    monkeypatch.setattr(behaviour, "_restart_executor", lambda: restarted.append(True))

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
    assert restarted, "executor should have been restarted after schedule() failure"
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
        :param error_cb: Optional error callback for IPFS failures.
        :type error_cb: Any
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
        :param error_cb: Optional error callback for IPFS failures.
        :type error_cb: Any
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
    store_response = SimpleNamespace(ipfs_hash=store_cid)
    is_first_call = iter([True, False])

    def stub(
        msg: Any, dlg: Any, callback: Callable[[Any, Any], None], error_cb: Any = None
    ) -> None:
        """Route GET vs STORE calls."""
        if next(is_first_call):
            get_action(msg, dlg, callback)
        else:
            callback(store_response, dlg)
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

    token_cb: Any = SimpleNamespace(cost_dict={"input": 10}, actual_model=None)
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
        callback(SimpleNamespace(files={"task.json": json.dumps(body)}), dlg)

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

    token_cb: Any = SimpleNamespace(cost_dict={}, actual_model=None)
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
                callback(SimpleNamespace(files={"t.json": json.dumps(body)}), dlg)
        elif func is beh_mod.TaskExecutionBehaviour._handle_store_response:
            callback(SimpleNamespace(ipfs_hash="bafyok"), dlg)
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
        callback(SimpleNamespace(files={"task.json": json.dumps({})}), dlg)

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
        callback(SimpleNamespace(files={"t.json": json.dumps(body)}), dlg)

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


def test_restart_executor_calls_join_with_timeout(
    behaviour: Any,
    monkeypatch: Any,
) -> None:
    """Verify _restart_executor passes timeout=10.0 to join() (H-2 fix)."""
    mock_pool = MagicMock()
    monkeypatch.setattr(behaviour, "_executor", mock_pool)

    # Patch ProcessPool so the new executor creation succeeds
    monkeypatch.setattr(beh_mod, "ProcessPool", lambda max_workers: MagicMock())

    behaviour._restart_executor()

    mock_pool.stop.assert_called_once()
    mock_pool.join.assert_called_once_with(timeout=10.0)


# ===========================================================================
# Coverage sweep — unit-testable uncovered lines
# ===========================================================================


# ---------------------------------------------------------------------------
# MechMetrics
# ---------------------------------------------------------------------------


def test_mech_metrics_set_gauge_with_labels(behaviour: Any) -> None:
    """set_gauge with keyword labels should call metric.labels(**labels).set(value)."""
    from prometheus_client import Gauge

    g = Gauge("test_sg_labels_cov", "test", labelnames=["tool"])
    behaviour.mech_metrics.set_gauge(g, 5, tool="sum")
    assert g.labels(tool="sum")._value.get() == 5.0


def test_mech_metrics_observe_histogram_without_labels(behaviour: Any) -> None:
    """observe_histogram without labels should call metric.observe(value)."""
    from prometheus_client import Histogram

    h = Histogram("test_oh_nolabels_cov", "test")
    behaviour.mech_metrics.observe_histogram(h, 2.5)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_payment_model_property(behaviour: Any, shared_state: Dict[str, Any]) -> None:
    """payment_model returns the value stored in shared_state."""
    shared_state[beh_mod.PAYMENT_MODEL] = "fixed"
    assert behaviour.payment_model == "fixed"


def test_payment_info_property_with_value(
    behaviour: Any, shared_state: Dict[str, Any]
) -> None:
    """payment_info returns the dict stored in shared_state."""
    shared_state[beh_mod.PAYMENT_INFO] = {"0xmech": "fixed"}
    assert behaviour.payment_info == {"0xmech": "fixed"}


def test_payment_info_property_default(behaviour: Any) -> None:
    """payment_info returns {} when not set in shared_state."""
    assert behaviour.payment_info == {}


def test_request_id_to_num_timeouts_property(behaviour: Any, params_stub: Any) -> None:
    """request_id_to_num_timeouts returns params.request_id_to_num_timeouts."""
    params_stub.request_id_to_num_timeouts[99] = 3
    assert behaviour.request_id_to_num_timeouts[99] == 3


def test_count_timeout(behaviour: Any, params_stub: Any) -> None:
    """count_timeout increments the timeout count for a request."""
    params_stub.request_id_to_num_timeouts[7] = 0
    behaviour.count_timeout(7)
    assert params_stub.request_id_to_num_timeouts[7] == 1


def test_timeout_limit_reached_true(behaviour: Any, params_stub: Any) -> None:
    """timeout_limit_reached returns True when limit is met."""
    params_stub.timeout_limit = 2
    params_stub.request_id_to_num_timeouts[8] = 2
    assert behaviour.timeout_limit_reached(8) is True


def test_timeout_limit_reached_false(behaviour: Any, params_stub: Any) -> None:
    """timeout_limit_reached returns False when limit is not yet met."""
    params_stub.timeout_limit = 2
    params_stub.request_id_to_num_timeouts[8] = 1
    assert behaviour.timeout_limit_reached(8) is False


def test_unprocessed_timed_out_tasks_setter(
    behaviour: Any, shared_state: Dict[str, Any]
) -> None:
    """Setting unprocessed_timed_out_tasks updates shared_state."""
    behaviour.unprocessed_timed_out_tasks = [{"requestId": 1}]
    assert shared_state[beh_mod.UNPROCESSED_TIMED_OUT_TASKS] == [{"requestId": 1}]


# ---------------------------------------------------------------------------
# _ensure_payment_model
# ---------------------------------------------------------------------------


def test_ensure_payment_model_non_marketplace(behaviour: Any, params_stub: Any) -> None:
    """_ensure_payment_model returns True immediately when not a marketplace."""
    params_stub.use_mech_marketplace = False
    assert behaviour._ensure_payment_model() is True


def test_ensure_payment_model_inflight(behaviour: Any, params_stub: Any) -> None:
    """_ensure_payment_model returns False when there is an inflight request."""
    params_stub.use_mech_marketplace = True
    params_stub.in_flight_req = True
    assert behaviour._ensure_payment_model() is False


def test_ensure_payment_model_payment_model_already_set(
    behaviour: Any, params_stub: Any, shared_state: Dict[str, Any]
) -> None:
    """_ensure_payment_model returns True when payment_model is already cached."""
    params_stub.use_mech_marketplace = True
    params_stub.in_flight_req = False
    shared_state[beh_mod.PAYMENT_MODEL] = "some_model"
    assert behaviour._ensure_payment_model() is True


def test_ensure_payment_model_requests_model_when_missing(
    behaviour: Any, params_stub: Any
) -> None:
    """_ensure_payment_model calls _request_payment_model when no model is set."""
    params_stub.use_mech_marketplace = True
    params_stub.in_flight_req = False
    # payment_model not in shared_state → None
    result = behaviour._ensure_payment_model()
    assert result is False
    assert params_stub.in_flight_req is True  # _request_payment_model sets this


def test_act_returns_when_payment_model_not_ready(
    behaviour: Any, params_stub: Any, disable_polling: Callable
) -> None:
    """act() returns early when _ensure_payment_model returns False."""
    disable_polling()
    params_stub.use_mech_marketplace = True
    params_stub.in_flight_req = True  # triggers _ensure_payment_model → False
    execute_called: list = []
    behaviour._execute_task = lambda: execute_called.append(True)
    behaviour.act()
    assert execute_called == []


# ---------------------------------------------------------------------------
# _should_poll
# ---------------------------------------------------------------------------


def test_should_poll_no_previous_poll(behaviour: Any, params_stub: Any) -> None:
    """_should_poll returns True when last_polling is None."""
    params_stub.req_params.last_polling = {}
    assert behaviour._should_poll("legacy") is True


def test_should_poll_stale(behaviour: Any, params_stub: Any, monkeypatch: Any) -> None:
    """_should_poll returns True when polling interval has elapsed."""
    params_stub.req_params.last_polling = {"legacy": 1000.0}
    params_stub.polling_interval = 30.0
    monkeypatch.setattr(time, "time", lambda: 1031.0)
    assert behaviour._should_poll("legacy") is True


def test_should_poll_fresh(behaviour: Any, params_stub: Any, monkeypatch: Any) -> None:
    """_should_poll returns False when polling interval has not elapsed."""
    params_stub.req_params.last_polling = {"legacy": 1000.0}
    params_stub.polling_interval = 30.0
    monkeypatch.setattr(time, "time", lambda: 1020.0)
    assert behaviour._should_poll("legacy") is False


# ---------------------------------------------------------------------------
# _fetch_deadline
# ---------------------------------------------------------------------------


def test_fetch_deadline_not_cold_start(
    behaviour: Any, params_stub: Any, monkeypatch: Any
) -> None:
    """_fetch_deadline uses SUBSEQUENT_DEADLINE when not cold start."""
    params_stub.is_cold_start = False
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    assert behaviour._fetch_deadline() == 1000.0 + beh_mod.SUBSEQUENT_DEADLINE


# ---------------------------------------------------------------------------
# _has_executing_task_timed_out
# ---------------------------------------------------------------------------


def test_has_executing_task_timed_out_past_deadline(
    behaviour: Any, monkeypatch: Any
) -> None:
    """_has_executing_task_timed_out returns True when deadline is in the past."""
    behaviour._executing_task = {"requestId": 1, "timeout_deadline": 500.0}
    monkeypatch.setattr(time, "time", lambda: 600.0)
    assert behaviour._has_executing_task_timed_out() is True


def test_has_executing_task_timed_out_future_deadline(
    behaviour: Any, monkeypatch: Any
) -> None:
    """_has_executing_task_timed_out returns False when deadline is in the future."""
    behaviour._executing_task = {"requestId": 1, "timeout_deadline": 700.0}
    monkeypatch.setattr(time, "time", lambda: 600.0)
    assert behaviour._has_executing_task_timed_out() is False


def test_has_executing_task_no_timeout_deadline(behaviour: Any) -> None:
    """_has_executing_task_timed_out returns False when no timeout_deadline set."""
    behaviour._executing_task = {"requestId": 1}
    assert behaviour._has_executing_task_timed_out() is False


# ---------------------------------------------------------------------------
# _download_tools
# ---------------------------------------------------------------------------


def test_download_tools_inflight_returns_early(behaviour: Any) -> None:
    """_download_tools returns immediately when inflight_tool_req is set."""
    behaviour._inflight_tool_req = "mytool"
    behaviour._tools_to_package_hash = {"mytool": "h1", "othertool": "h2"}
    behaviour._all_tools = {}
    behaviour._download_tools()
    assert behaviour._inflight_tool_req == "mytool"  # unchanged


def test_download_tools_fetches_missing_tool(
    behaviour: Any, fake_dialogue: Any, monkeypatch: Any
) -> None:
    """_download_tools fetches the first missing tool and sets inflight_tool_req."""
    behaviour._tools_to_package_hash = {"mytool": "hashABC"}
    behaviour._all_tools = {}
    monkeypatch.setattr(
        behaviour,
        "_build_ipfs_get_file_req",
        lambda h, timeout=None: (object(), fake_dialogue),
    )
    sent: list = []
    monkeypatch.setattr(
        behaviour,
        "send_message",
        lambda msg, dlg, cb: sent.append(True),
    )
    behaviour._download_tools()
    assert behaviour._inflight_tool_req == "mytool"
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# _filter_out_incompatible_reqs
# ---------------------------------------------------------------------------


def test_filter_out_incompatible_reqs_non_marketplace(
    behaviour: Any, params_stub: Any
) -> None:
    """_filter_out_incompatible_reqs returns early when not a marketplace."""
    params_stub.use_mech_marketplace = False
    behaviour._filter_out_incompatible_reqs()  # should not raise


# ---------------------------------------------------------------------------
# _ensure_deadline / _execute_task branches
# ---------------------------------------------------------------------------


def test_ensure_deadline_sets_deadline_when_none(
    behaviour: Any, params_stub: Any, monkeypatch: Any
) -> None:
    """_ensure_deadline sets the deadline when it is None."""
    behaviour._executing_task = {"requestId": 1}
    behaviour._request_handling_deadline = None
    params_stub.is_cold_start = True
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    behaviour._ensure_deadline()
    assert behaviour._request_handling_deadline is not None


def test_ensure_deadline_does_not_reset_when_set(behaviour: Any) -> None:
    """_ensure_deadline does not change an already-set deadline."""
    behaviour._request_handling_deadline = 9999.0
    behaviour._ensure_deadline()
    assert behaviour._request_handling_deadline == 9999.0


def test_execute_task_inflight_with_executing_calls_ensure_deadline(
    behaviour: Any, params_stub: Any, monkeypatch: Any
) -> None:
    """_execute_task calls _ensure_deadline when inflight and executing_task is set."""
    params_stub.in_flight_req = True
    behaviour._executing_task = {"requestId": 1}
    behaviour._request_handling_deadline = None
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    params_stub.is_cold_start = True
    behaviour._execute_task()
    assert behaviour._request_handling_deadline is not None


def test_execute_task_deadline_reached_calls_handle_timeout(
    behaviour: Any, params_stub: Any, monkeypatch: Any
) -> None:
    """_execute_task calls _handle_timeout_task when request deadline is reached."""
    params_stub.in_flight_req = True
    behaviour._executing_task = {"requestId": 1}
    behaviour._request_handling_deadline = 500.0
    monkeypatch.setattr(time, "time", lambda: 600.0)
    timeout_called: list = []
    monkeypatch.setattr(
        behaviour, "_handle_timeout_task", lambda: timeout_called.append(True)
    )
    behaviour._execute_task()
    assert timeout_called


def test_execute_task_not_inflight_timeout_calls_handle_timeout(
    behaviour: Any, params_stub: Any, monkeypatch: Any
) -> None:
    """_execute_task calls _handle_timeout_task on per-task execution timeout."""
    params_stub.in_flight_req = False
    behaviour._executing_task = {"requestId": 1, "timeout_deadline": 500.0}
    behaviour._async_result = None
    behaviour._invalid_request = False
    monkeypatch.setattr(time, "time", lambda: 600.0)
    timeout_called: list = []
    monkeypatch.setattr(
        behaviour, "_handle_timeout_task", lambda: timeout_called.append(True)
    )
    behaviour._execute_task()
    assert timeout_called


def test_execute_task_pops_from_timed_out_when_pending_empty(
    behaviour: Any,
    params_stub: Any,
    shared_state: Dict[str, Any],
    fake_dialogue: Any,
    monkeypatch: Any,
    patch_ipfs_multihash: Callable,
) -> None:
    """_execute_task picks from timed_out_tasks when pending_tasks is empty."""
    patch_ipfs_multihash()
    params_stub.in_flight_req = False
    timed_task: Dict[str, Any] = {
        "requestId": 200,
        "request_delivery_rate": 100,
        "data": b"ptr",
        "contract_address": "0xmech",
    }
    shared_state[beh_mod.TIMED_OUT_TASKS].append(timed_task)
    assert shared_state[beh_mod.PENDING_TASKS] == []
    monkeypatch.setattr(
        behaviour,
        "_build_ipfs_get_file_req",
        lambda h, timeout=None: (object(), fake_dialogue),
    )
    monkeypatch.setattr(behaviour, "send_message", lambda *a, **k: None)
    behaviour._execute_task()
    assert behaviour._executing_task is not None
    assert behaviour._executing_task["requestId"] == 200
    assert shared_state[beh_mod.TIMED_OUT_TASKS] == []


def test_execute_task_bytes_request_id_converted(
    behaviour: Any,
    params_stub: Any,
    shared_state: Dict[str, Any],
    fake_dialogue: Any,
    monkeypatch: Any,
    patch_ipfs_multihash: Callable,
) -> None:
    """_execute_task converts bytes requestId to int before processing."""
    patch_ipfs_multihash()
    params_stub.in_flight_req = False
    req_id_int = 42
    req_id_bytes = req_id_int.to_bytes(32, byteorder="big")
    task: Dict[str, Any] = {
        "requestId": req_id_bytes,
        "request_delivery_rate": 100,
        "data": b"ptr",
        "contract_address": "0xmech",
    }
    shared_state[beh_mod.PENDING_TASKS].append(task)
    monkeypatch.setattr(
        behaviour,
        "_build_ipfs_get_file_req",
        lambda h, timeout=None: (object(), fake_dialogue),
    )
    monkeypatch.setattr(behaviour, "send_message", lambda *a, **k: None)
    behaviour._execute_task()
    assert behaviour._executing_task["requestId"] == req_id_int


# ---------------------------------------------------------------------------
# _update_pending_tasks
# ---------------------------------------------------------------------------


def test_update_pending_tasks_non_marketplace(behaviour: Any, params_stub: Any) -> None:
    """_update_pending_tasks returns early when not marketplace."""
    params_stub.use_mech_marketplace = False
    behaviour._update_pending_tasks()  # should not raise


def test_update_pending_tasks_no_pending_tasks(
    behaviour: Any, params_stub: Any, monkeypatch: Any
) -> None:
    """_update_pending_tasks returns early when pending_tasks is empty."""
    params_stub.use_mech_marketplace = True
    params_stub.in_flight_req = False
    behaviour.last_status_check_time = 0.0
    monkeypatch.setattr(time, "time", lambda: beh_mod.STATUS_CHECK_INTERVAL + 1.0)
    behaviour._update_pending_tasks()  # should not raise


# ---------------------------------------------------------------------------
# send_message
# ---------------------------------------------------------------------------


def test_send_message_registers_callback_and_deadline(
    behaviour: Any,
    params_stub: Any,
    fake_dialogue: Any,
    monkeypatch: Any,
) -> None:
    """send_message registers callback, error_callback, deadline, and sets in_flight_req."""
    behaviour._executing_task = {"requestId": 1}
    behaviour._request_handling_deadline = None
    params_stub.is_cold_start = True
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    mock_msg = MagicMock()

    def cb(msg: object, dlg: object) -> None:
        """No-op callback."""

    def err_cb(reason: object) -> None:
        """No-op error callback."""

    behaviour.send_message(mock_msg, fake_dialogue, cb, err_cb)
    nonce = "nonce-1"
    assert params_stub.req_to_callback[nonce] is cb
    assert params_stub.req_to_error_callback[nonce] is err_cb
    assert params_stub.in_flight_req is True
    assert nonce in params_stub.req_to_deadline


# ---------------------------------------------------------------------------
# _handle_ipfs_error
# ---------------------------------------------------------------------------


def test_handle_ipfs_error_sets_state(behaviour: Any) -> None:
    """_handle_ipfs_error sets _ipfs_error_reason and _invalid_request."""
    behaviour._handle_ipfs_error("some error detail")
    assert "some error detail" in (behaviour._ipfs_error_reason or "")
    assert behaviour._invalid_request is True


# ---------------------------------------------------------------------------
# _get_designated_marketplace_mech_address
# ---------------------------------------------------------------------------


def test_get_designated_mech_no_marketplace_raises(
    behaviour: Any, params_stub: Any
) -> None:
    """_get_designated_marketplace_mech_address raises ValueError when none found."""
    from types import SimpleNamespace as NS

    params_stub.mech_to_config = {
        "0xmech1": NS(is_marketplace_mech=False),
        "0xmech2": NS(is_marketplace_mech=False),
    }
    with pytest.raises(ValueError, match="No marketplace mech address found"):
        behaviour._get_designated_marketplace_mech_address()


# ---------------------------------------------------------------------------
# _handle_get_task — stepping-in with unknown tool
# ---------------------------------------------------------------------------


def test_handle_get_task_stepping_in_unknown_tool_ignored(
    behaviour: Any, params_stub: Any, monkeypatch: Any
) -> None:
    """_handle_get_task ignores task when stepping in with an unknown tool."""
    from types import SimpleNamespace as NS

    my_mech = "0xmymech"
    params_stub.mech_to_config = {
        my_mech: NS(is_marketplace_mech=True, use_dynamic_pricing=False)
    }
    other_mech = "0xothermech"
    behaviour._executing_task = {
        "requestId": 55,
        "priorityMech": other_mech,
        "request_delivery_rate": 100,
        "contract_address": "0xmech",
    }
    monkeypatch.setattr(behaviour.mech_metrics, "inc_counter", MagicMock())
    monkeypatch.setattr(behaviour.mech_metrics, "set_gauge", MagicMock())
    msg = SimpleNamespace(
        files={"task.json": json.dumps({"prompt": "hi", "tool": "unknown_tool"})}
    )
    behaviour._handle_get_task(msg, MagicMock())
    assert 55 in behaviour._ignored_request_ids
    assert behaviour._executing_task is None


# ---------------------------------------------------------------------------
# _get_executing_task_result branches
# ---------------------------------------------------------------------------


def test_get_executing_task_result_no_task(behaviour: Any) -> None:
    """_get_executing_task_result returns None when no executing task."""
    behaviour._executing_task = None
    assert behaviour._get_executing_task_result() is None


def test_get_executing_task_result_no_async_result(behaviour: Any) -> None:
    """_get_executing_task_result returns None when async_result is None."""
    behaviour._executing_task = {"requestId": 1}
    behaviour._invalid_request = False
    behaviour._async_result = None
    assert behaviour._get_executing_task_result() is None


def test_get_executing_task_result_timeout_error(behaviour: Any) -> None:
    """_get_executing_task_result returns None on TimeoutError."""
    behaviour._executing_task = {"requestId": 1}
    behaviour._invalid_request = False
    mock_fut = MagicMock()
    mock_fut.result.side_effect = TimeoutError("expired")
    behaviour._async_result = mock_fut
    assert behaviour._get_executing_task_result() is None


def test_get_executing_task_result_exception(behaviour: Any) -> None:
    """_get_executing_task_result returns None on generic Exception."""
    behaviour._executing_task = {"requestId": 1}
    behaviour._invalid_request = False
    mock_fut = MagicMock()
    mock_fut.result.side_effect = Exception("boom")
    behaviour._async_result = mock_fut
    assert behaviour._get_executing_task_result() is None


# ---------------------------------------------------------------------------
# _handle_store_response branches
# ---------------------------------------------------------------------------


def test_handle_store_response_no_executing_task(
    behaviour: Any, monkeypatch: Any
) -> None:
    """_handle_store_response logs error and returns early when no executing task."""
    behaviour._executing_task = None
    monkeypatch.setattr(beh_mod, "to_v1", lambda x: x)
    msg = SimpleNamespace(ipfs_hash="bafybeiabc")
    behaviour._handle_store_response(msg, MagicMock())  # should not raise


def test_handle_store_response_invalid_done_task_resets_state(
    behaviour: Any, monkeypatch: Any
) -> None:
    """_handle_store_response resets all state when done_task is None."""
    behaviour._executing_task = {"requestId": 1}
    behaviour._done_task = None
    monkeypatch.setattr(beh_mod, "to_v1", lambda x: x)
    monkeypatch.setattr(behaviour.mech_metrics, "set_gauge", MagicMock())
    msg = SimpleNamespace(ipfs_hash="bafybeiabc")
    behaviour._handle_store_response(msg, MagicMock())
    assert behaviour._executing_task is None
    assert behaviour._done_task is None


def test_handle_store_response_dynamic_pricing_recorded(
    behaviour: Any, params_stub: Any, shared_state: Dict[str, Any], monkeypatch: Any
) -> None:
    """_handle_store_response adds dynamic_tool_cost when tools_to_pricing has the tool."""
    from types import SimpleNamespace as NS

    my_mech = params_stub.agent_mech_contract_address.lower()
    params_stub.mech_to_config = {
        my_mech: NS(is_marketplace_mech=True, use_dynamic_pricing=False)
    }
    behaviour._executing_task = {
        "requestId": 1,
        "tool": "mytool",
        "contract_address": "0xmech",
    }
    behaviour._done_task = {"request_id": 1, "tool": "mytool", "mech_address": "0xmech"}
    behaviour._tools_to_pricing = {"mytool": 100}
    monkeypatch.setattr(beh_mod, "to_v1", lambda x: x)
    monkeypatch.setattr(beh_mod, "to_multihash", lambda x: f"mh:{x}")
    monkeypatch.setattr(behaviour.mech_metrics, "set_gauge", MagicMock())
    msg = SimpleNamespace(ipfs_hash="bafybeiabc")
    behaviour._handle_store_response(msg, MagicMock())
    done_tasks = shared_state[beh_mod.DONE_TASKS]
    assert len(done_tasks) == 1
    assert done_tasks[0]["dynamic_tool_cost"] == 100


# ---------------------------------------------------------------------------
# _build_ipfs_message / _build_ipfs_store_file_req / _build_ipfs_get_file_req
# ---------------------------------------------------------------------------


def test_build_ipfs_message_returns_message_and_dialogue(behaviour: Any) -> None:
    """_build_ipfs_message calls ipfs_dialogues.create and returns its results."""
    msg, dlg = behaviour._build_ipfs_message(
        performative=beh_mod.IpfsMessage.Performative.GET_FILES,  # type: ignore
        ipfs_hash="bafybeiabc",
    )
    assert msg is not None
    assert dlg is not None


def test_build_ipfs_store_file_req_returns_pair(behaviour: Any) -> None:
    """_build_ipfs_store_file_req delegates to _build_ipfs_message correctly."""
    msg, dlg = behaviour._build_ipfs_store_file_req({"metadata.json": '{"x":1}'})
    assert msg is not None
    assert dlg is not None


def test_build_ipfs_get_file_req_returns_pair(behaviour: Any) -> None:
    """_build_ipfs_get_file_req delegates to _build_ipfs_message correctly."""
    msg, dlg = behaviour._build_ipfs_get_file_req("bafybeiabc")
    assert msg is not None
    assert dlg is not None


# ---------------------------------------------------------------------------
# _request_payment_model
# ---------------------------------------------------------------------------


def test_request_payment_model_sets_inflight(behaviour: Any, params_stub: Any) -> None:
    """_request_payment_model puts a message and sets in_flight_req."""
    params_stub.in_flight_req = False
    behaviour._request_payment_model()
    assert params_stub.in_flight_req is True


# ---------------------------------------------------------------------------
# _handle_timeout_task
# ---------------------------------------------------------------------------


def test_handle_timeout_task_no_req_id_resets_state(
    behaviour: Any, params_stub: Any
) -> None:
    """_handle_timeout_task resets state immediately when req_id is missing."""
    behaviour._executing_task = {}  # no requestId
    behaviour._async_result = MagicMock()
    behaviour._handle_timeout_task()
    assert behaviour._executing_task is None
    assert behaviour._async_result is None


def test_handle_timeout_task_adds_back_to_queue(
    behaviour: Any, params_stub: Any, monkeypatch: Any
) -> None:
    """_handle_timeout_task re-queues the task when timeout limit is not reached."""
    behaviour._executing_task = {"requestId": 7, "request_delivery_rate": 100}
    behaviour._async_result = None
    params_stub.request_id_to_num_timeouts[7] = 0
    params_stub.timeout_limit = 3
    monkeypatch.setattr(beh_mod, "ProcessPool", lambda max_workers: MagicMock())
    behaviour._handle_timeout_task()
    assert behaviour._executing_task is None
    assert any(t.get("requestId") == 7 for t in behaviour.pending_tasks)


def test_handle_timeout_task_limit_reached_calls_handle_done(
    behaviour: Any, params_stub: Any, monkeypatch: Any
) -> None:
    """_handle_timeout_task calls _handle_done_task when timeout limit is reached."""
    behaviour._executing_task = {"requestId": 9, "request_delivery_rate": 100}
    behaviour._async_result = None
    params_stub.request_id_to_num_timeouts[9] = 0
    params_stub.timeout_limit = 1
    monkeypatch.setattr(beh_mod, "ProcessPool", lambda max_workers: MagicMock())
    handle_done_calls: list = []
    monkeypatch.setattr(
        behaviour, "_handle_done_task", lambda r: handle_done_calls.append(r)
    )
    behaviour._handle_timeout_task()
    assert len(handle_done_calls) == 1


# ---------------------------------------------------------------------------
# _check_for_new_marketplace_reqs / _check_undelivered_reqs_marketplace
# ---------------------------------------------------------------------------


def test_check_for_new_marketplace_reqs_inflight_sets_ts(
    behaviour: Any,
    params_stub: Any,
    shared_state: Dict[str, Any],
    monkeypatch: Any,
) -> None:
    """When inflight, _check_for_new_marketplace_reqs stamps INFLIGHT_READ_TS."""
    params_stub.in_flight_req = True
    monkeypatch.setattr(time, "time", lambda: 1234.0)
    behaviour._check_for_new_marketplace_reqs()
    assert shared_state.get(beh_mod.INFLIGHT_READ_TS) == 1234.0


def test_check_for_new_marketplace_reqs_not_poll_time(
    behaviour: Any,
    params_stub: Any,
    shared_state: Dict[str, Any],
    monkeypatch: Any,
) -> None:
    """When polling interval not elapsed, stamps LAST_READ_ATTEMPT_TS and returns."""
    params_stub.in_flight_req = False
    marketplace_key = beh_mod.RequestType.MARKETPLACE.value
    params_stub.req_params.last_polling = {marketplace_key: 1000.0}
    params_stub.polling_interval = 300.0
    monkeypatch.setattr(time, "time", lambda: 1100.0)
    behaviour._check_for_new_marketplace_reqs()
    assert shared_state.get(beh_mod.LAST_READ_ATTEMPT_TS) == 1100.0


def test_check_for_new_marketplace_reqs_no_from_block_populates(
    behaviour: Any,
    params_stub: Any,
    monkeypatch: Any,
) -> None:
    """When from_block is None, calls _populate_from_block and returns."""
    params_stub.in_flight_req = False
    marketplace_key = beh_mod.RequestType.MARKETPLACE.value
    params_stub.req_params.last_polling = {marketplace_key: 0.0}
    params_stub.polling_interval = 0.0
    params_stub.req_params.from_block = {marketplace_key: None}
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    populate_called: list = []
    monkeypatch.setattr(
        behaviour, "_populate_from_block", lambda: populate_called.append(True)
    )
    behaviour._check_for_new_marketplace_reqs()
    assert populate_called


def test_check_for_new_marketplace_reqs_full_poll(
    behaviour: Any,
    params_stub: Any,
    shared_state: Dict[str, Any],
    monkeypatch: Any,
) -> None:
    """Full poll path sets in_flight_req, LAST_READ_ATTEMPT_TS, and INFLIGHT_READ_TS."""
    params_stub.in_flight_req = False
    params_stub.use_mech_marketplace = True
    marketplace_key = beh_mod.RequestType.MARKETPLACE.value
    params_stub.req_params.from_block = {marketplace_key: 100}
    params_stub.req_params.last_polling = {marketplace_key: 0.0}
    params_stub.polling_interval = 0.0
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    behaviour._check_for_new_marketplace_reqs()
    assert params_stub.in_flight_req is True
    assert shared_state.get(beh_mod.LAST_READ_ATTEMPT_TS) == 1000.0
    assert shared_state.get(beh_mod.INFLIGHT_READ_TS) == 1000.0


def test_check_undelivered_reqs_marketplace_non_marketplace_returns(
    behaviour: Any, params_stub: Any
) -> None:
    """_check_undelivered_reqs_marketplace returns early when not marketplace."""
    params_stub.use_mech_marketplace = False
    behaviour._check_undelivered_reqs_marketplace()  # should not raise


def test_check_undelivered_reqs_marketplace_puts_message(
    behaviour: Any, params_stub: Any
) -> None:
    """_check_undelivered_reqs_marketplace sends a contract request."""
    params_stub.use_mech_marketplace = True
    marketplace_key = beh_mod.RequestType.MARKETPLACE.value
    params_stub.req_params.from_block = {marketplace_key: 100}
    behaviour._check_undelivered_reqs_marketplace()
    assert params_stub.req_type == marketplace_key


# ---------------------------------------------------------------------------
# send_data_via_acn
# ---------------------------------------------------------------------------


def test_send_data_via_acn_puts_message(behaviour: Any) -> None:
    """send_data_via_acn creates an ACN message and puts it on the outbox."""
    # Should not raise — stub outbox and dialogues accept any message
    behaviour.send_data_via_acn("0xsender", "req-1", {"foo": "bar"})


# ---------------------------------------------------------------------------
# Remaining coverage gaps
# ---------------------------------------------------------------------------


def test_has_executing_task_timed_out_no_executing_task(behaviour: Any) -> None:
    """_has_executing_task_timed_out returns False when executing_task is None."""
    behaviour._executing_task = None
    assert behaviour._has_executing_task_timed_out() is False


def test_download_tools_skips_already_loaded_tool(
    behaviour: Any, fake_dialogue: Any, monkeypatch: Any
) -> None:
    """_download_tools skips tools already in _all_tools and fetches the next one."""
    behaviour._tools_to_package_hash = {"tool1": "h1", "tool2": "h2"}
    behaviour._all_tools = {"tool1": ("py", "run", {})}  # tool1 already loaded
    monkeypatch.setattr(
        behaviour,
        "_build_ipfs_get_file_req",
        lambda h, timeout=None: (object(), fake_dialogue),
    )
    sent: list = []
    monkeypatch.setattr(
        behaviour,
        "send_message",
        lambda msg, dlg, cb: sent.append(True),
    )
    behaviour._download_tools()
    # should have fetched tool2 (the missing one), not tool1
    assert behaviour._inflight_tool_req == "tool2"
    assert len(sent) == 1


def test_handle_get_tool_loads_and_stores(behaviour: Any, monkeypatch: Any) -> None:
    """_handle_get_tool calls ComponentPackageLoader.load and stores result."""
    behaviour._inflight_tool_req = "mytool"
    fake_yaml = {"params": {"default_model": "gpt-4o"}}
    monkeypatch.setattr(
        beh_mod.ComponentPackageLoader,
        "load",
        staticmethod(lambda files: (fake_yaml, "tool_py_src", "run")),
    )
    msg = SimpleNamespace(files={"component.yaml": "...", "tool.py": "..."})
    behaviour._handle_get_tool(msg, MagicMock())
    assert "mytool" in behaviour._all_tools
    tool_py, callable_method, component_yaml = behaviour._all_tools["mytool"]
    assert tool_py == "tool_py_src"
    assert callable_method == "run"
    assert behaviour._inflight_tool_req is None


def test_populate_from_block_sets_inflight(behaviour: Any, params_stub: Any) -> None:
    """_populate_from_block sends a ledger request and sets in_flight_req."""
    params_stub.in_flight_req = False
    behaviour._populate_from_block()
    assert params_stub.in_flight_req is True


def test_get_payment_types_requests_unknown_mech(
    behaviour: Any, params_stub: Any, shared_state: Dict[str, Any]
) -> None:
    """_get_payment_types makes a contract call when unknown mechs are present."""
    params_stub.in_flight_req = False
    params_stub.agent_mech_contract_address = "0xmymech"
    # payment_info is empty and the task's mech is not our own
    shared_state[beh_mod.UNPROCESSED_TIMED_OUT_TASKS] = [
        {"priorityMech": "0xothermech", "requestId": 1}
    ]
    result = behaviour._get_payment_types()
    assert result is False
    assert params_stub.in_flight_req is True


def test_filter_out_incompatible_reqs_skips_own_mech_not_in_info(
    behaviour: Any, params_stub: Any, shared_state: Dict[str, Any]
) -> None:
    """_filter_out_incompatible_reqs skips task for own mech not in payment_info."""
    params_stub.use_mech_marketplace = True
    params_stub.in_flight_req = False
    params_stub.agent_mech_contract_address = "0xmymech"
    params_stub.step_in_list_size = 20
    # priorityMech is our own mech but not in payment_info
    # → _get_payment_types() returns True (skipped from to_request)
    # → loop runs, req_mech not in payment_info → continue
    shared_state[beh_mod.UNPROCESSED_TIMED_OUT_TASKS] = [
        {"priorityMech": "0xmymech", "requestId": 2}
    ]
    behaviour._filter_out_incompatible_reqs()
    # The task should have been dropped (not added to timed_out_tasks)
    assert shared_state[beh_mod.TIMED_OUT_TASKS] == []


def test_filter_out_incompatible_reqs_same_payment_type(
    behaviour: Any, params_stub: Any, shared_state: Dict[str, Any]
) -> None:
    """_filter_out_incompatible_reqs keeps tasks with matching payment type."""
    params_stub.use_mech_marketplace = True
    params_stub.in_flight_req = False
    params_stub.agent_mech_contract_address = "0xother"
    params_stub.step_in_list_size = 20
    shared_state[beh_mod.PAYMENT_MODEL] = "fixed"
    shared_state[beh_mod.PAYMENT_INFO] = {"0xmech1": "fixed"}
    shared_state[beh_mod.UNPROCESSED_TIMED_OUT_TASKS] = [
        {"priorityMech": "0xmech1", "requestId": 3}
    ]
    behaviour._filter_out_incompatible_reqs()
    # Task should be moved to timed_out_tasks
    assert any(t.get("requestId") == 3 for t in shared_state[beh_mod.TIMED_OUT_TASKS])


def test_execute_task_both_queues_empty_returns_cleanly(
    behaviour: Any, params_stub: Any, shared_state: Dict[str, Any]
) -> None:
    """_execute_task returns without error when both pending and timed_out are empty."""
    params_stub.in_flight_req = False
    behaviour._executing_task = None
    assert shared_state[beh_mod.PENDING_TASKS] == []
    assert shared_state[beh_mod.TIMED_OUT_TASKS] == []
    behaviour._execute_task()  # should not raise


def test_handle_timeout_task_cancels_async_result(
    behaviour: Any, params_stub: Any, monkeypatch: Any
) -> None:
    """_handle_timeout_task cancels the in-flight async result."""
    behaviour._executing_task = {"requestId": 11, "request_delivery_rate": 100}
    mock_fut = MagicMock()
    behaviour._async_result = mock_fut
    params_stub.request_id_to_num_timeouts[11] = 0
    params_stub.timeout_limit = 3
    monkeypatch.setattr(beh_mod, "ProcessPool", lambda max_workers: MagicMock())
    behaviour._handle_timeout_task()
    mock_fut.cancel.assert_called_once()


def test_filter_out_incompatible_reqs_warning_for_unknown_mech(
    behaviour: Any, params_stub: Any, shared_state: Dict[str, Any], monkeypatch: Any
) -> None:
    """_filter_out_incompatible_reqs logs a warning when a mech has no payment info."""
    params_stub.use_mech_marketplace = True
    params_stub.agent_mech_contract_address = "0xmymech"
    params_stub.step_in_list_size = 20
    shared_state[beh_mod.UNPROCESSED_TIMED_OUT_TASKS] = [
        {"priorityMech": "0xothermech", "requestId": 99}
    ]
    # Force _get_payment_types to return True to reach the inner loop
    monkeypatch.setattr(behaviour, "_get_payment_types", lambda: True)
    # payment_info does not contain "0xothermech" → triggers warning branch
    behaviour._filter_out_incompatible_reqs()
    # task should be dropped (not added to timed_out_tasks)
    assert shared_state[beh_mod.TIMED_OUT_TASKS] == []


def test_filter_out_incompatible_reqs_different_payment_type_drops_task(
    behaviour: Any, params_stub: Any, shared_state: Dict[str, Any]
) -> None:
    """_filter_out_incompatible_reqs drops tasks with mismatched payment type."""
    params_stub.use_mech_marketplace = True
    params_stub.in_flight_req = False
    params_stub.agent_mech_contract_address = "0xother"
    params_stub.step_in_list_size = 20
    shared_state[beh_mod.PAYMENT_MODEL] = "fixed"
    shared_state[beh_mod.PAYMENT_INFO] = {
        "0xmech1": "dynamic"
    }  # different from "fixed"
    shared_state[beh_mod.UNPROCESSED_TIMED_OUT_TASKS] = [
        {"priorityMech": "0xmech1", "requestId": 4}
    ]
    behaviour._filter_out_incompatible_reqs()
    # task with mismatched payment type should be filtered out
    assert shared_state[beh_mod.TIMED_OUT_TASKS] == []
