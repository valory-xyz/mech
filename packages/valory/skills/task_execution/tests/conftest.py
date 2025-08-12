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

"""This package contains the fixtures for the rest of the tests."""

import threading
from collections import defaultdict
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, Callable, Dict

import pytest

import packages.valory.skills.task_execution.behaviours as beh_mod
from packages.valory.skills.task_execution import models
from packages.valory.skills.task_execution.behaviours import (
    DONE_TASKS,
    DONE_TASKS_LOCK,
    IPFS_TASKS,
    PENDING_TASKS,
    REQUEST_ID_TO_DELIVERY_RATE_INFO,
    TaskExecutionBehaviour,
)


@pytest.fixture
def shared_state() -> Dict[str, Any]:
    """
    Return initial shared_state mapping used by behaviours/handlers.

    :returns: Mapping with lists/lock for pending/done/ipfs tasks and pricing info.
    :rtype: Dict[str, Any]
    """
    return {
        PENDING_TASKS: [],
        DONE_TASKS: [],
        IPFS_TASKS: [],
        DONE_TASKS_LOCK: threading.Lock(),
        REQUEST_ID_TO_DELIVERY_RATE_INFO: {},
    }


@pytest.fixture
def params_stub() -> SimpleNamespace:
    """
    A minimal Params-like namespace with attributes the code touches.

    :returns: Parameters stub emulating the real Params model for tests.
    :rtype: SimpleNamespace
    """
    return SimpleNamespace(
        tools_to_package_hash={},
        tools_to_pricing={},
        api_keys={},
        req_params=SimpleNamespace(from_block={}, last_polling={}),
        polling_interval=10.0,
        in_flight_req=False,
        req_to_callback={},
        req_to_deadline={},
        request_count=0,
        cleanup_freq=1000,
        req_type=None,
        default_chain_id=100,
        agent_mech_contract_addresses=["0x0000000000000000000000000000000000000000"],
        mech_to_config={"0xmech": SimpleNamespace(is_marketplace_mech=False)},
        use_mech_marketplace=False,
        mech_marketplace_address=None,
        max_block_window=10_000,
        task_deadline=10.0,
        timeout_limit=2,
        request_id_to_num_timeouts=defaultdict(int),
        is_cold_start=True,
        num_agents=1,
        agent_index=0,
        from_block_range=0,
    )


@pytest.fixture
def context_stub(
    shared_state: Dict[str, Any], params_stub: SimpleNamespace
) -> SimpleNamespace:
    """
    A bare-bones AEA-like context with logger/outbox stubs.

    :param shared_state: The shared state mapping used by the skill.
    :type shared_state: Dict[str, Any]
    :param params_stub: The parameters stub object.
    :type params_stub: SimpleNamespace
    :returns: A minimal context exposing attributes accessed by the code.
    :rtype: SimpleNamespace
    """

    class Outbox:
        """Outbox stub which discards messages."""

        def put_message(self, *args: Any, **kwargs: Any) -> None:
            """
            Drop an outgoing message.

            :param args: Ignored positional arguments.
            :type args: tuple[Any, ...]
            :param kwargs: Ignored keyword arguments.
            :type kwargs: dict[str, Any]
            :returns
            """
            pass

    logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )

    return SimpleNamespace(
        logger=logger,
        shared_state=shared_state,
        params=params_stub,
        agent_address="0xagent",
        default_ledger_id="ethereum",
        outbox=Outbox(),
        ipfs_dialogues=None,
        contract_dialogues=None,
        ledger_dialogues=None,
        acn_data_share_dialogues=None,
    )


@pytest.fixture
def behaviour(context_stub: SimpleNamespace) -> TaskExecutionBehaviour:
    """
    TaskExecutionBehaviour instance wired to the stub context.

    :param context_stub: The fake AEA context.
    :type context_stub: SimpleNamespace
    :returns: The behaviour instance after setup().
    :rtype: TaskExecutionBehaviour
    """
    b = TaskExecutionBehaviour(name="task_execution", skill_context=context_stub)
    b.setup()
    return b


class FakeDialogue:
    """Mimic a dialogue with a stable nonce used for callback mapping."""

    class Label:
        """Mock label with a fixed dialogue reference."""

        dialogue_reference = ("nonce-1", "x")

    dialogue_label = Label()


class FakeIpfsMsg:
    """Minimal IpfsMessage-like shape for callback paths."""

    def __init__(
        self, files: Dict[str, Any] | None = None, ipfs_hash: str | None = None
    ) -> None:
        """
        Initialize with optional files/ipfs_hash fields.

        :param files: Optional mapping of filename to content bytes/str.
        :type files: Dict[str, Any]
        :param ipfs_hash: IPFS hash string to emulate a store response.
        :type ipfs_hash: str
        """
        self.files = files or {}
        self.ipfs_hash = ipfs_hash


@pytest.fixture
def fake_dialogue() -> FakeDialogue:
    """
    Fake dialogue object with a fixed nonce.

    :returns: Dialogue stub with a static nonce.
    :rtype: FakeDialogue
    """
    return FakeDialogue()


@pytest.fixture
def done_future() -> Callable[[Any], Future]:
    """
    Factory that produces an already-completed Future.

    :returns: A maker function which sets the result immediately.
    :rtype: Callable[[Any], concurrent.futures.Future]
    """

    def _make(value: Any) -> Future:
        """
        Create a Future completed with `value`.

        :param value: The value to set on the Future.
        :type value: Any
        :returns: A Future already completed with the given value.
        :rtype: Future
        """
        f: Future = Future()
        f.set_result(value)
        return f

    return _make


@pytest.fixture
def patch_ipfs_multihash(monkeypatch: Any) -> Callable[[str], None]:
    """
    A helper that stubs CID/multihash helpers for tests.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: Any
    :returns: Function to apply the stubs with an optional fake file hash.
    :rtype: Callable[[str], None]
    """

    def _apply(file_hash: str = "cid-for-task") -> None:
        """
        Apply monkeypatches for IPFS helpers.

        :param file_hash: Fake CID returned by get_ipfs_file_hash.
        :type file_hash: str
        """
        monkeypatch.setattr(beh_mod, "get_ipfs_file_hash", lambda data: file_hash)
        monkeypatch.setattr(beh_mod, "to_v1", lambda cid: cid)
        monkeypatch.setattr(beh_mod, "to_multihash", lambda cid: f"mh:{cid}")

    return _apply


@pytest.fixture
def disable_polling(monkeypatch: Any) -> Callable[[], None]:
    """
    A helper that disables polling paths on the behaviour.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: Any
    :returns: Function which, when called, disables polling methods.
    :rtype: Callable[[], None]
    """

    def _apply() -> None:
        """Disable polling-related methods on the behaviour."""
        monkeypatch.setattr(
            TaskExecutionBehaviour, "_check_for_new_reqs", lambda self: None
        )
        monkeypatch.setattr(
            TaskExecutionBehaviour, "_check_for_new_marketplace_reqs", lambda self: None
        )

    return _apply


class SentOutbox:
    """Collect messages sent to outbox for later assertions."""

    def __init__(self) -> None:
        """Initialize the sent list."""
        self.sent: list[Any] = []

    def put_message(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """
        Append a message to the sent list.

        :param message: The message to store.
        :type message: Any
        :param *args: Ignored positional arguments.
        :type *args: Any
        :param **kwargs: Ignored keyword arguments.
        :type **kwargs: Any
        """
        self.sent.append(message)


@pytest.fixture
def handler_context(
    shared_state: Dict[str, Any], params_stub: SimpleNamespace
) -> SimpleNamespace:
    """
    A minimal handler context with stubbed dialogues and outbox.

    :param shared_state: The shared state mapping used by the skill.
    :type shared_state: Dict[str, Any]
    :param params_stub: The parameters stub object.
    :type params_stub: SimpleNamespace
    :returns: Context exposing handler dependencies and fake dialogues.
    :rtype: SimpleNamespace
    """
    ctx = SimpleNamespace(
        logger=SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
        shared_state=shared_state,
        params=params_stub,
        default_ledger_id="ethereum",
        outbox=SentOutbox(),
    )

    class HandlersBag:
        """Container for handler attributes to drive cleanup logic."""

        pass

    ctx.handlers = HandlersBag()
    ctx.ipfs_dialogues = SimpleNamespace(
        update=lambda msg: SimpleNamespace(
            dialogue_label=SimpleNamespace(dialogue_reference=("nonce-1", "x"))
        ),
        cleanup=lambda: None,
    )
    ctx.contract_dialogues = SimpleNamespace(cleanup=lambda: None)
    ctx.ledger_dialogues = SimpleNamespace(cleanup=lambda: None)
    return ctx


@pytest.fixture
def http_dialogue() -> Any:
    """
    A fake HttpDialogue whose reply() returns a SimpleNamespace response.

    :returns: Fake dialogue object with a reply() method.
    :rtype: Any
    """

    class FakeHttpDialogue:
        """Minimal HTTP dialogue stub."""

        def reply(
            self,
            performative: Any,
            target_message: Any,
            version: str,
            status_code: int,
            status_text: str,
            headers: str,
            body: bytes,
        ) -> SimpleNamespace:
            """
            Build a fake HTTP response object.

            :param performative: Performative enum/value to set on the response.
            :type performative: Any
            :param target_message: The originating HTTP request message.
            :type target_message: Any
            :param version: HTTP version string (e.g., "1.1").
            :type version: str
            :param status_code: Numeric HTTP status code.
            :type status_code: int
            :param status_text: Human-readable status string.
            :type status_text: str
            :param headers: Raw headers string to include.
            :type headers: str
            :param body: Raw response body bytes.
            :type body: bytes
            :returns: A response-like object with the given fields.
            """
            return SimpleNamespace(
                performative=performative,
                version=version,
                status_code=status_code,
                status_text=status_text,
                headers=headers,
                body=body,
                target=target_message,
            )

    return FakeHttpDialogue()


@pytest.fixture
def dialogue_skill_context(shared_state: Dict[str, Any]) -> SimpleNamespace:
    """
    A minimal skill_context expected by Model.

    :param shared_state: The shared state mapping passed through to the context.
    :type shared_state: Dict[str, Any]
    :returns: Context with skill_id, agent_address, logger, and shared_state.
    :rtype: SimpleNamespace
    """
    return SimpleNamespace(
        skill_id="valory/task_execution:0.1.0",
        agent_address="0xagent",
        logger=SimpleNamespace(
            debug=lambda *a, **k: None,
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
        shared_state=shared_state,
    )


def _get_self_addr(dialogues_obj: Any) -> str | None:
    """
    Self address from dialogues object handling different attribute names.

    :param dialogues_obj: The dialogues instance (IpfsDialogues, etc.).
    :type dialogues_obj: Any
    :returns: The configured self address string, or None if not found.
    :rtype: Optional[str]
    """
    return getattr(
        dialogues_obj, "self_address", getattr(dialogues_obj, "_self_address", None)
    )


@pytest.fixture
def params_kwargs(dialogue_skill_context: SimpleNamespace) -> Dict[str, Any]:
    """
    Minimal good kwargs for Params; individual tests mutate per-case.

    :param dialogue_skill_context: The fake Model skill_context.
    :type dialogue_skill_context: SimpleNamespace
    :returns: Keyword arguments suitable for constructing Params().
    :rtype: Dict[str, Any]
    """
    return dict(
        skill_context=dialogue_skill_context,
        api_keys={},
        tools_to_package_hash={"sum": "hashsum"},
        num_agents=2,
        agent_index=0,
        from_block_range=1000,
        timeout_limit=3,
        max_block_window=10_000,
        mech_to_config={
            "0xMeCh": {"use_dynamic_pricing": True, "is_marketplace_mech": False}
        },
        mech_marketplace_address=models.ZERO_ADDRESS,
        default_chain_id="1",
        tools_to_pricing={},
        polling_interval=12.5,
        task_deadline=111.0,
        cleanup_freq=77,
    )
