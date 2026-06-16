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

"""Behavioural tests for the kv_store connection."""

import itertools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Generator, Optional, Tuple, cast
from unittest.mock import MagicMock

import pytest
from aea.mail.base import Envelope
from aea.protocols.dialogue.base import InvalidDialogueMessage

from packages.valory.connections.kv_store import connection as conn_mod
from packages.valory.connections.kv_store.connection import (
    KvStoreConnection,
    KvStoreDialogue,
    KvStoreDialogues,
    PUBLIC_ID,
    Store,
    _GENERIC_ERROR_MESSAGE,
)
from packages.valory.protocols.kv_store.message import KvStoreMessage
from packages.valory.protocols.kv_store.serialization import KvStoreSerializer

SKILL_ADDRESS = "test_author/test_skill:0.1.0"
CONNECTION_ADDRESS = str(PUBLIC_ID)

_REF_COUNTER = itertools.count()
_REF_LOCK = threading.Lock()


def _next_ref() -> str:
    """Return a unique dialogue reference safe for concurrent callers."""
    with _REF_LOCK:
        return f"ref-{next(_REF_COUNTER)}"


def _make_connection() -> KvStoreConnection:
    """Construct a KvStoreConnection without invoking AEA framework init."""
    instance = KvStoreConnection.__new__(KvStoreConnection)
    instance.logger = logging.getLogger("test.kv_store")  # type: ignore[assignment]
    instance.dialogues = KvStoreDialogues(connection_id=PUBLIC_ID)
    instance.put_envelope = MagicMock()  # type: ignore[method-assign]
    return instance


def _teardown_db() -> None:
    """Drop tables, close, and unbind the module-level peewee DB."""
    if not conn_mod.db.is_closed():
        conn_mod.db.drop_tables([Store])
        conn_mod.db.close()
    conn_mod.db.init(None)


@pytest.fixture()
def fresh_db() -> Generator[None, None, None]:
    """Bind the module-level peewee DB to a fresh in-memory SQLite per test."""
    conn_mod.db.init(":memory:")
    conn_mod.db.connect(reuse_if_open=True)
    conn_mod.db.create_tables([Store])
    try:
        yield
    finally:
        _teardown_db()


@pytest.fixture()
def disk_db(tmp_path: Any) -> Generator[None, None, None]:
    """Bind the module-level peewee DB to a file-backed SQLite per test."""
    conn_mod.db.init(str(tmp_path / "store.db"))
    conn_mod.db.connect(reuse_if_open=True)
    conn_mod.db.create_tables([Store])
    try:
        yield
    finally:
        _teardown_db()


@pytest.fixture()
def kv_connection(fresh_db: None) -> KvStoreConnection:  # noqa: ARG001
    """A KvStoreConnection wired to a fresh in-memory DB."""
    return _make_connection()


@pytest.fixture()
def disk_kv_connection(disk_db: None) -> KvStoreConnection:  # noqa: ARG001
    """A KvStoreConnection wired to a file-backed DB (exercises WAL + locks)."""
    return _make_connection()


def _build_read_request(keys: Tuple[str, ...]) -> KvStoreMessage:
    return KvStoreMessage(
        performative=KvStoreMessage.Performative.READ_REQUEST,  # type: ignore[arg-type]
        dialogue_reference=(_next_ref(), ""),
        message_id=1,
        target=0,
        keys=keys,
    )


def _build_write_request(data: Dict[str, str]) -> KvStoreMessage:
    return KvStoreMessage(
        performative=KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST,  # type: ignore[arg-type]
        dialogue_reference=(_next_ref(), ""),
        message_id=1,
        target=0,
        data=data,
    )


def _build_delete_request(keys: Tuple[str, ...]) -> KvStoreMessage:
    return KvStoreMessage(
        performative=KvStoreMessage.Performative.DELETE_REQUEST,  # type: ignore[arg-type]
        dialogue_reference=(_next_ref(), ""),
        message_id=1,
        target=0,
        keys=keys,
    )


def _build_list_request(key_prefix: str) -> KvStoreMessage:
    return KvStoreMessage(
        performative=KvStoreMessage.Performative.LIST_REQUEST,  # type: ignore[arg-type]
        dialogue_reference=(_next_ref(), ""),
        message_id=1,
        target=0,
        key_prefix=key_prefix,
    )


def _build_list_response(data: Dict[str, str]) -> KvStoreMessage:
    return KvStoreMessage(
        performative=KvStoreMessage.Performative.LIST_RESPONSE,  # type: ignore[arg-type]
        dialogue_reference=(_next_ref(), ""),
        message_id=1,
        target=0,
        data=data,
    )


def _open_dialogue(
    dialogues: KvStoreDialogues, message: KvStoreMessage
) -> KvStoreDialogue:
    """Register a peer-initiated message and return the resulting dialogue."""
    message.sender = SKILL_ADDRESS
    message.to = CONNECTION_ADDRESS
    dialogue = dialogues.update(message)
    assert dialogue is not None, "dialogue should pair for INITIAL_PERFORMATIVES"
    return cast(KvStoreDialogue, dialogue)


def test_read_request_filters_by_keys(kv_connection: KvStoreConnection) -> None:
    """A read for a subset of keys returns only that subset."""
    Store.create(key="a", value="1")
    Store.create(key="b", value="2")
    Store.create(key="c", value="3")

    message = _build_read_request(("a", "b"))
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    response = kv_connection.read_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.READ_RESPONSE
    assert response.data == {"a": "1", "b": "2"}


def test_read_request_unknown_keys_return_empty_dict(
    kv_connection: KvStoreConnection,
) -> None:
    """Asking for keys that do not exist yields an empty data dict."""
    Store.create(key="a", value="1")
    message = _build_read_request(("missing-1", "missing-2"))
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    response = kv_connection.read_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.READ_RESPONSE
    assert response.data == {}


def test_read_request_empty_keys_returns_empty_dict(
    kv_connection: KvStoreConnection,
) -> None:
    """An empty key list returns an empty data dict, not the whole table."""
    Store.create(key="a", value="1")
    message = _build_read_request(())
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    response = kv_connection.read_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.READ_RESPONSE
    assert response.data == {}


def test_create_or_update_inserts_and_updates(
    kv_connection: KvStoreConnection,
) -> None:
    """A mixed batch of inserts and updates lands as one consistent write."""
    Store.create(key="existing", value="old")

    message = _build_write_request({"existing": "new", "fresh": "value"})
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    response = kv_connection.create_or_update_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.SUCCESS
    assert Store.get(Store.key == "existing").value == "new"
    assert Store.get(Store.key == "fresh").value == "value"


def test_create_or_update_atomic_on_handler_error(
    disk_kv_connection: KvStoreConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure mid-batch rolls back earlier writes from the same batch."""
    real_create = Store.create
    call_count = {"n": 0}

    def flaky_create(**kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("synthetic mid-batch failure")
        return real_create(**kwargs)

    monkeypatch.setattr(Store, "create", flaky_create)

    message = _build_write_request({"k1": "v1", "k2": "v2", "k3": "v3"})
    dialogue = _open_dialogue(disk_kv_connection.dialogues, message)

    response = disk_kv_connection.create_or_update_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.ERROR
    assert response.message == _GENERIC_ERROR_MESSAGE
    assert Store.select().count() == 0


def test_create_or_update_returns_error_when_db_raises(
    kv_connection: KvStoreConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DB-layer exception is converted to an ERROR reply, not propagated."""

    def boom(**_kwargs: Any) -> Any:
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(Store, "create", boom)

    message = _build_write_request({"only": "key"})
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    response = kv_connection.create_or_update_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.ERROR
    assert response.message == _GENERIC_ERROR_MESSAGE


def test_delete_request_removes_matching_keys(
    kv_connection: KvStoreConnection,
) -> None:
    """Delete drops matching rows and leaves the rest untouched."""
    Store.create(key="a", value="1")
    Store.create(key="b", value="2")
    Store.create(key="c", value="3")

    message = _build_delete_request(("a", "c"))
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    response = kv_connection.delete_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.SUCCESS
    remaining = {row.key: row.value for row in Store.select()}
    assert remaining == {"b": "2"}


def test_delete_request_missing_keys_is_idempotent(
    kv_connection: KvStoreConnection,
) -> None:
    """Deleting keys that do not exist is a no-op, not an error."""
    Store.create(key="a", value="1")

    message = _build_delete_request(("missing-1", "missing-2"))
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    response = kv_connection.delete_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.SUCCESS
    assert {row.key for row in Store.select()} == {"a"}


def test_delete_request_empty_keys_is_noop(
    kv_connection: KvStoreConnection,
) -> None:
    """An empty key list returns SUCCESS without touching the table."""
    Store.create(key="a", value="1")

    message = _build_delete_request(())
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    response = kv_connection.delete_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.SUCCESS
    assert {row.key for row in Store.select()} == {"a"}


def test_delete_request_returns_error_when_db_raises(
    kv_connection: KvStoreConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DB exception during delete is converted to an ERROR reply."""

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("delete failure")

    monkeypatch.setattr(Store, "delete", boom)

    message = _build_delete_request(("a",))
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    response = kv_connection.delete_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.ERROR
    assert response.message == _GENERIC_ERROR_MESSAGE


def test_list_request_filters_by_prefix(
    kv_connection: KvStoreConnection,
) -> None:
    """List returns only rows whose key starts with the given prefix."""
    Store.create(key="preimage:1", value="a")
    Store.create(key="preimage:2", value="b")
    Store.create(key="other:1", value="c")

    message = _build_list_request("preimage:")
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    response = kv_connection.list_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.LIST_RESPONSE
    assert response.data == {"preimage:1": "a", "preimage:2": "b"}


def test_list_request_no_matches_returns_empty(
    kv_connection: KvStoreConnection,
) -> None:
    """A prefix that matches no rows yields an empty response, not an error."""
    Store.create(key="other:1", value="a")

    message = _build_list_request("preimage:")
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    response = kv_connection.list_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.LIST_RESPONSE
    assert response.data == {}


def test_list_request_empty_prefix_returns_all_rows(
    kv_connection: KvStoreConnection,
) -> None:
    """An empty prefix matches every row in the table."""
    # Sweeper use-case: caller wants to enumerate every entry for retention
    # filtering when they don't know all the prefixes in use.
    Store.create(key="preimage:1", value="a")
    Store.create(key="metric:cpu", value="b")
    Store.create(key="lock", value="c")

    message = _build_list_request("")
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    response = kv_connection.list_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.LIST_RESPONSE
    assert response.data == {"preimage:1": "a", "metric:cpu": "b", "lock": "c"}


@pytest.mark.parametrize("prefix", ["preimage:", ""])
def test_list_request_returns_error_when_db_raises(
    kv_connection: KvStoreConnection, monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    """A DB exception during list converts to an ERROR reply (both branches)."""

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("list failure")

    monkeypatch.setattr(Store, "select", boom)

    message = _build_list_request(prefix)
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    response = kv_connection.list_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.ERROR
    assert response.message == _GENERIC_ERROR_MESSAGE


def test_delete_then_list_roundtrip(
    kv_connection: KvStoreConnection,
) -> None:
    """A delete is reflected on the very next list call (sweeper happy path)."""
    Store.create(key="preimage:1", value="a")
    Store.create(key="preimage:2", value="b")

    delete_msg = _build_delete_request(("preimage:1",))
    delete_dlg = _open_dialogue(kv_connection.dialogues, delete_msg)
    delete_resp = kv_connection.delete_request(delete_msg, delete_dlg)
    assert delete_resp is not None
    assert delete_resp.performative == KvStoreMessage.Performative.SUCCESS

    list_msg = _build_list_request("preimage:")
    list_dlg = _open_dialogue(kv_connection.dialogues, list_msg)
    list_resp = kv_connection.list_request(list_msg, list_dlg)
    assert list_resp is not None
    assert list_resp.data == {"preimage:2": "b"}


def test_read_request_returns_error_when_db_raises(
    kv_connection: KvStoreConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DB exception during read is converted to an ERROR reply."""

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("read failure")

    monkeypatch.setattr(Store, "select", boom)

    message = _build_read_request(("a",))
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    response = kv_connection.read_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.ERROR
    assert response.message == _GENERIC_ERROR_MESSAGE


def test_on_send_drops_message_when_dialogue_cannot_be_built(
    kv_connection: KvStoreConnection, caplog: pytest.LogCaptureFixture
) -> None:
    """An unpairable message logs and does not crash on_send."""
    # READ_RESPONSE is not in KvStoreDialogue.INITIAL_PERFORMATIVES, so
    # dialogues.update() returns None when it arrives as the first message.
    orphan = KvStoreMessage(
        performative=KvStoreMessage.Performative.READ_RESPONSE,  # type: ignore[arg-type]
        dialogue_reference=(_next_ref(), ""),
        message_id=1,
        target=0,
        data={},
    )
    orphan.sender = SKILL_ADDRESS
    orphan.to = CONNECTION_ADDRESS
    envelope = Envelope(to=CONNECTION_ADDRESS, sender=SKILL_ADDRESS, message=orphan)

    with caplog.at_level(logging.ERROR, logger="test.kv_store"):
        kv_connection.on_send(envelope)

    assert kv_connection.put_envelope.call_count == 0
    assert any("Could not associate dialogue" in rec.message for rec in caplog.records)


def test_on_send_replies_error_when_handler_raises(
    kv_connection: KvStoreConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the dispatched handler raises, on_send replies with ERROR."""

    def exploding_handler(_message: Any, _dialogue: Any) -> Any:
        raise RuntimeError("handler boom")

    monkeypatch.setattr(kv_connection, "read_request", exploding_handler)

    request = _build_read_request(("a",))
    request.sender = SKILL_ADDRESS
    request.to = CONNECTION_ADDRESS
    envelope = Envelope(to=CONNECTION_ADDRESS, sender=SKILL_ADDRESS, message=request)

    kv_connection.on_send(envelope)

    assert kv_connection.put_envelope.call_count == 1
    response = kv_connection.put_envelope.call_args[0][0].message
    assert response.performative == KvStoreMessage.Performative.ERROR
    assert response.message == _GENERIC_ERROR_MESSAGE


def test_on_send_replies_error_when_handler_method_missing(
    kv_connection: KvStoreConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paired dialogue with no matching handler method replies ERROR."""
    request = _build_read_request(("a",))
    request.sender = SKILL_ADDRESS
    request.to = CONNECTION_ADDRESS
    monkeypatch.delattr(KvStoreConnection, "read_request")
    envelope = Envelope(to=CONNECTION_ADDRESS, sender=SKILL_ADDRESS, message=request)

    kv_connection.on_send(envelope)

    assert kv_connection.put_envelope.call_count == 1
    response = kv_connection.put_envelope.call_args[0][0].message
    assert response.performative == KvStoreMessage.Performative.ERROR
    assert response.message == _GENERIC_ERROR_MESSAGE


def test_on_send_happy_path_round_trips_a_read(
    kv_connection: KvStoreConnection,
) -> None:
    """A normal READ_REQUEST round-trips through on_send to put_envelope."""
    Store.create(key="hello", value="world")

    request = _build_read_request(("hello",))
    request.sender = SKILL_ADDRESS
    request.to = CONNECTION_ADDRESS
    envelope = Envelope(to=CONNECTION_ADDRESS, sender=SKILL_ADDRESS, message=request)

    kv_connection.on_send(envelope)

    assert kv_connection.put_envelope.call_count == 1
    response = kv_connection.put_envelope.call_args[0][0].message
    assert response.performative == KvStoreMessage.Performative.READ_RESPONSE
    assert response.data == {"hello": "world"}


def test_value_column_uses_text_not_varchar(
    kv_connection: KvStoreConnection,  # noqa: ARG001
) -> None:
    """The Store.value column is rendered as TEXT in the SQL schema."""
    info = conn_mod.db.execute_sql("PRAGMA table_info(store)").fetchall()
    value_col = next(col for col in info if col[1] == "value")
    col_type = value_col[2]
    assert col_type == "TEXT", f"expected TEXT, got {col_type!r}"


def test_value_field_accepts_long_payloads(
    kv_connection: KvStoreConnection,
) -> None:
    """A value larger than 255 chars survives a write/read round-trip."""
    long_value = "x" * 4096

    write_msg = _build_write_request({"big": long_value})
    write_dialogue = _open_dialogue(kv_connection.dialogues, write_msg)
    write_response = kv_connection.create_or_update_request(write_msg, write_dialogue)
    assert write_response is not None
    assert write_response.performative == KvStoreMessage.Performative.SUCCESS

    read_msg = _build_read_request(("big",))
    read_dialogue = _open_dialogue(kv_connection.dialogues, read_msg)
    read_response = kv_connection.read_request(read_msg, read_dialogue)

    assert read_response is not None
    assert read_response.data == {"big": long_value}


def test_concurrent_writes_do_not_deadlock(
    disk_kv_connection: KvStoreConnection,
) -> None:
    """5 threads writing in parallel must all commit without SQLITE_BUSY."""
    instance = disk_kv_connection
    payloads = [
        {f"t{t}-b{b}-k{i}": f"v{t}-{b}-{i}" for i in range(3)}
        for t in range(5)
        for b in range(5)
    ]
    expected = {k: v for p in payloads for k, v in p.items()}

    def submit_write(payload: Dict[str, str]) -> Optional[KvStoreMessage]:
        message = _build_write_request(payload)
        dialogue = _open_dialogue(instance.dialogues, message)
        return instance.create_or_update_request(message, dialogue)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(submit_write, p) for p in payloads]
        responses = [f.result() for f in as_completed(futures)]

    assert all(r is not None for r in responses)
    success = [
        r
        for r in responses
        if r is not None and r.performative == KvStoreMessage.Performative.SUCCESS
    ]
    errors = [
        r
        for r in responses
        if r is not None and r.performative == KvStoreMessage.Performative.ERROR
    ]
    assert not errors, [r.message for r in errors[:3]]
    assert len(success) == len(payloads)

    stored = {row.key: row.value for row in Store.select()}
    assert stored == expected


# --- serialization round-trips for the new performatives --------------------


@pytest.mark.parametrize(
    "message",
    [
        _build_delete_request(("a", "b")),
        _build_delete_request(()),
        _build_list_request("preimage:"),
        _build_list_request(""),
        _build_list_response({"preimage:1": "v1", "preimage:2": "v2"}),
        _build_list_response({}),
    ],
    ids=[
        "delete",
        "delete_empty",
        "list_req",
        "list_req_empty",
        "list_resp",
        "list_resp_empty",
    ],
)
def test_serialization_round_trip(message: KvStoreMessage) -> None:
    """Encode then decode reproduces each new performative's message."""
    decoded = KvStoreSerializer.decode(KvStoreSerializer.encode(message))
    assert decoded == message


# --- on_send dispatch for the new request performatives ----------------------


def test_on_send_dispatches_delete_request(kv_connection: KvStoreConnection) -> None:
    """DELETE_REQUEST dispatches through on_send to its handler and replies."""
    Store.create(key="preimage:1", value="v1")
    request = _build_delete_request(("preimage:1",))
    request.sender = SKILL_ADDRESS
    request.to = CONNECTION_ADDRESS
    envelope = Envelope(to=CONNECTION_ADDRESS, sender=SKILL_ADDRESS, message=request)

    kv_connection.on_send(envelope)

    assert kv_connection.put_envelope.call_count == 1
    response = kv_connection.put_envelope.call_args[0][0].message
    assert response.performative == KvStoreMessage.Performative.SUCCESS


def test_on_send_dispatches_list_request(kv_connection: KvStoreConnection) -> None:
    """LIST_REQUEST round-trips through on_send -> handler -> put_envelope."""
    Store.create(key="preimage:1", value="v1")
    request = _build_list_request("preimage:")
    request.sender = SKILL_ADDRESS
    request.to = CONNECTION_ADDRESS
    envelope = Envelope(to=CONNECTION_ADDRESS, sender=SKILL_ADDRESS, message=request)

    kv_connection.on_send(envelope)

    assert kv_connection.put_envelope.call_count == 1
    response = kv_connection.put_envelope.call_args[0][0].message
    assert response.performative == KvStoreMessage.Performative.LIST_RESPONSE
    assert response.data == {"preimage:1": "v1"}


# --- delete empty-keys is a true no-op (no Store.delete issued) ---------------


def test_delete_request_empty_keys_does_not_call_delete(
    kv_connection: KvStoreConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty key list skips the DELETE entirely (the `if keys:` guard)."""
    delete_spy = MagicMock(
        side_effect=AssertionError("Store.delete must not be called for empty keys")
    )
    monkeypatch.setattr(Store, "delete", delete_spy)

    message = _build_delete_request(())
    dialogue = _open_dialogue(kv_connection.dialogues, message)
    response = kv_connection.delete_request(message, dialogue)

    assert response is not None
    assert response.performative == KvStoreMessage.Performative.SUCCESS
    delete_spy.assert_not_called()


# --- case-sensitive prefix matching ------------------------------------------


def test_list_request_prefix_is_case_sensitive(
    kv_connection: KvStoreConnection,
) -> None:
    """Prefix matching is case-sensitive, so a sweeper can't over-match."""
    Store.create(key="preimage:1", value="v1")
    Store.create(key="PREIMAGE:2", value="v2")

    message = _build_list_request("preimage:")
    dialogue = _open_dialogue(kv_connection.dialogues, message)
    response = kv_connection.list_request(message, dialogue)

    assert response is not None
    assert response.data == {"preimage:1": "v1"}


# --- dialogue rejects an invalid reply (VALID_REPLIES enforcement) -----------


def test_dialogue_rejects_invalid_reply_to_delete_request(
    kv_connection: KvStoreConnection,
) -> None:
    """A READ_RESPONSE reply to a DELETE_REQUEST is rejected by the dialogue."""
    message = _build_delete_request(("a",))
    dialogue = _open_dialogue(kv_connection.dialogues, message)

    with pytest.raises(InvalidDialogueMessage):
        dialogue.reply(
            performative=KvStoreMessage.Performative.READ_RESPONSE,
            target_message=message,
            data={},
        )
