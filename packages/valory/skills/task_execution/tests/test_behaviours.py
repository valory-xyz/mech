import json
from types import SimpleNamespace

import packages.valory.skills.task_execution.behaviours as beh_mod


def test_happy_path_executes_and_stores(
    behaviour,
    shared_state,
    params_stub,
    fake_dialogue,
    done_future,
    monkeypatch,
    patch_ipfs_multihash,
    disable_polling,
):
    patch_ipfs_multihash()
    disable_polling()

    valid_cid = "bafybeigdyrzt5u36sq3x7xvaf2h2k6g2r5fpmy7bcxfbcdx7djzn2k2f3u"

    behaviour._all_tools["sum"] = (
        "tool_py_src",
        "run",
        {"params": {"default_model": "gpt-4o-mini"}},
    )
    params_stub.tools_to_pricing = {"sum": 0}

    req_id = 42
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

    call_no = {"n": 0}

    def send_message_stub(msg, dlg, callback):
        call_no["n"] += 1
        if call_no["n"] == 1:
            task_body = {"prompt": "add 2+2", "tool": "sum"}
            fake_get_response = type(
                "Msg", (), {"files": {"task.json": json.dumps(task_body)}}
            )()
            callback(fake_get_response, dlg)
        else:
            fake_store_response = type("Msg", (), {"ipfs_hash": valid_cid})()
            callback(fake_store_response, dlg)
        params_stub.in_flight_req = True

    monkeypatch.setattr(behaviour, "send_message", send_message_stub)

    token_cb = type("CB", (), {"cost_dict": {"input": 10, "output": 5}})()
    keychain = object()
    result_tuple = ("4", "add 2+2", {"tx": "0xabc"}, token_cb, keychain)
    monkeypatch.setattr(
        behaviour, "_submit_task", lambda *a, **k: done_future(result_tuple)
    )

    params_stub.in_flight_req = False
    behaviour.act()
    params_stub.in_flight_req = False
    behaviour.act()

    assert len(shared_state[beh_mod.DONE_TASKS]) == 1
    done = shared_state[beh_mod.DONE_TASKS][0]
    assert done["request_id"] == req_id
    assert done["tool"] == "sum"
    assert done["mech_address"] == "0xmech"
    assert done["task_result"] == f"mh:{valid_cid}"

    assert behaviour._executing_task is None
    assert behaviour._async_result is None
    assert behaviour._last_deadline is None


def test_pricing_too_low_marks_invalid_and_stores_stub(
    behaviour,
    shared_state,
    params_stub,
    fake_dialogue,
    monkeypatch,
    done_future,
    patch_ipfs_multihash,
    disable_polling,
):
    patch_ipfs_multihash()
    disable_polling()
    behaviour._tools_to_package_hash = {"sum": "fakehash"}
    behaviour._tools_to_pricing["sum"] = 200
    behaviour._all_tools["sum"] = ("tool_py_src", "run", {"params": {}})

    req_id = 99
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

    # Stub builders
    monkeypatch.setattr(
        behaviour, "_build_ipfs_get_file_req", lambda *a, **k: (object(), fake_dialogue)
    )
    monkeypatch.setattr(
        behaviour,
        "_build_ipfs_store_file_req",
        lambda files, **k: (object(), fake_dialogue),
    )

    def _fail_submit(*a, **k):
        raise AssertionError("_submit_task must not be called for invalid pricing")

    monkeypatch.setattr(behaviour, "_submit_task", _fail_submit)

    valid_cid = "bafybeigdyrzt5u36sq3x7xvaf2h2k6g2r5fpmy7bcxfbcdx7djzn2k2f3u"
    calls = {"n": 0}

    def send_message_stub(msg, dlg, callback):
        calls["n"] += 1
        if calls["n"] == 1:
            body = {"prompt": "add 2+2", "tool": "sum"}
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
    done = shared_state[beh_mod.DONE_TASKS][0]
    assert done["request_id"] == req_id
    assert done.get("tool") is None
    assert "dynamic_tool_cost" not in done
    assert done["task_result"] == f"mh:{valid_cid}"
    assert behaviour._executing_task is None


def test_broken_process_pool_restart(
    behaviour,
    shared_state,
    params_stub,
    fake_dialogue,
    done_future,
    monkeypatch,
    patch_ipfs_multihash,
    disable_polling,
):
    patch_ipfs_multihash()
    disable_polling()

    behaviour._all_tools["sum"] = ("py", "run", {"params": {}})
    behaviour._tools_to_package_hash["sum"] = "fake-package-hash"

    req_id = 1
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

    monkeypatch.setattr(
        behaviour, "_build_ipfs_get_file_req", lambda *a, **k: (object(), fake_dialogue)
    )
    monkeypatch.setattr(
        behaviour,
        "_build_ipfs_store_file_req",
        lambda files, **k: (object(), fake_dialogue),
    )

    calls = {"n": 0}

    class BrokenOnceExec:
        def submit(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                from concurrent.futures.process import BrokenProcessPool

                raise BrokenProcessPool("boom")
            return done_future(
                ("ok", "p", {"tx": 1}, type("CB", (), {"cost_dict": {}})(), object())
            )

    monkeypatch.setattr(behaviour, "_executor", BrokenOnceExec())

    restarted = {"flag": False}
    monkeypatch.setattr(
        behaviour, "_restart_executor", lambda: restarted.__setitem__("flag", True)
    )

    # Shape the fake messages based on WHICH callback is used
    def send_message_stub(msg, dlg, cb):
        func = getattr(cb, "__func__", cb)
        if func is beh_mod.TaskExecutionBehaviour._handle_get_task:
            body = {"prompt": "p", "tool": "sum"}
            cb(SimpleNamespace(files={"task.json": json.dumps(body)}), dlg)
        elif func is beh_mod.TaskExecutionBehaviour._handle_store_response:
            cb(SimpleNamespace(ipfs_hash="bafyok"), dlg)
        else:
            raise AssertionError(f"Unexpected callback: {cb}")
        # keep in-flight True during this tick so polling doesn't run
        params_stub.in_flight_req = True

    monkeypatch.setattr(behaviour, "send_message", send_message_stub)

    params_stub.in_flight_req = False
    behaviour.act()

    params_stub.in_flight_req = False
    behaviour.act()

    assert restarted[
        "flag"
    ], "executor should have been restarted after BrokenProcessPool"
    assert len(shared_state[beh_mod.DONE_TASKS]) == 1
    done = shared_state[beh_mod.DONE_TASKS][0]
    assert done["request_id"] == req_id
    assert done["task_result"] == "mh:bafyok"


def test_invalid_tool_is_recorded_and_no_execution(
    behaviour,
    shared_state,
    params_stub,
    fake_dialogue,
    monkeypatch,
    patch_ipfs_multihash,
    disable_polling,
):
    patch_ipfs_multihash()
    disable_polling()

    req_id = 5
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

    def send_message_stub(msg, dlg, cb):
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
        # keep in-flight True within this tick to suppress polling
        params_stub.in_flight_req = True

    monkeypatch.setattr(behaviour, "send_message", send_message_stub)

    params_stub.in_flight_req = False
    behaviour.act()
    params_stub.in_flight_req = False
    behaviour.act()

    # Now DONE_TASKS has one item
    assert len(shared_state[beh_mod.DONE_TASKS]) == 1
    done = shared_state[beh_mod.DONE_TASKS][0]
    assert done["request_id"] == req_id
    assert done["tool"] == "unknown"
    assert done["task_result"] == "mh:bafyinval"


def test_ipfs_aux_task_removed_from_queue(
    behaviour,
    shared_state,
    params_stub,
    fake_dialogue,
    monkeypatch,
    disable_polling,
    patch_ipfs_multihash,
):
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
    monkeypatch.setattr(
        behaviour,
        "send_message",
        lambda msg, dlg, cb: (
            cb(SimpleNamespace(ipfs_hash="bafyaux"), dlg),
            setattr(params_stub, "in_flight_req", True),
        ),
    )
    params_stub.in_flight_req = False
    behaviour.act()
    assert shared_state[beh_mod.IPFS_TASKS] == []
