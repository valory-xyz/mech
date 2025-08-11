import json

import packages.valory.skills.task_execution.behaviours as beh_mod


def test_happy_path_executes_and_stores(
    behaviour, shared_state, params_stub, fake_dialogue, done_future, monkeypatch
):
    monkeypatch.setattr(beh_mod, "get_ipfs_file_hash", lambda data: "cid-for-task")
    monkeypatch.setattr(beh_mod, "to_v1", lambda cid: cid)
    monkeypatch.setattr(beh_mod, "to_multihash", lambda cid: f"mh:{cid}")
    monkeypatch.setattr(type(behaviour), "_check_for_new_reqs", lambda self: None)
    monkeypatch.setattr(
        type(behaviour), "_check_for_new_marketplace_reqs", lambda self: None
    )

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
        # Set to true so polling doesn't run.
        params_stub.in_flight_req = True

    monkeypatch.setattr(behaviour, "send_message", send_message_stub)

    token_cb = type("CB", (), {"cost_dict": {"input": 10, "output": 5}})()
    keychain = object()
    result_tuple = ("4", "add 2+2", {"tx": "0xabc"}, token_cb, keychain)
    monkeypatch.setattr(
        behaviour, "_submit_task", lambda *a, **k: done_future(result_tuple)
    )

    params_stub.in_flight_req = False  # allow _execute_task to start on first tick
    behaviour.act()
    # next tick: allow it to proceed again
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
    behaviour, shared_state, params_stub, fake_dialogue, monkeypatch, done_future
):
    monkeypatch.setattr(beh_mod, "get_ipfs_file_hash", lambda data: "cid-for-task")
    monkeypatch.setattr(beh_mod, "to_v1", lambda cid: cid)
    monkeypatch.setattr(beh_mod, "to_multihash", lambda cid: f"mh:{cid}")
    monkeypatch.setattr(type(behaviour), "_check_for_new_reqs", lambda self: None)
    monkeypatch.setattr(
        type(behaviour), "_check_for_new_marketplace_reqs", lambda self: None
    )
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
        params_stub.in_flight_req = True  # keep polling suppressed within the tick

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
