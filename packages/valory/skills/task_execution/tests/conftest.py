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

"""Shared test fixtures for the task_execution skill."""

from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Tuple

import pytest

import packages.valory.skills.task_execution.behaviours as beh_mod
from packages.valory.skills.task_execution import models
from packages.valory.skills.task_execution.behaviours import (
    DONE_TASKS,
    DONE_TASKS_LOCK,
    IPFS_TASKS,
    PENDING_TASKS,
    REQUEST_ID_TO_DELIVERY_RATE_INFO,
    TIMED_OUT_TASKS,
    TaskExecutionBehaviour,
    WAIT_FOR_TIMEOUT,
)


# ----------------------------- Shared state ----------------------------------


@pytest.fixture
def shared_state() -> Dict[str, Any]:
    """Return the initial shared_state mapping used by behaviours/handlers."""
    return {
        PENDING_TASKS: [],
        WAIT_FOR_TIMEOUT: [],
        TIMED_OUT_TASKS: [],
        DONE_TASKS: [],
        IPFS_TASKS: [],
        DONE_TASKS_LOCK: threading.Lock(),
        REQUEST_ID_TO_DELIVERY_RATE_INFO: {},
    }


# ----------------------------- Params stub -----------------------------------


@pytest.fixture
def params_stub() -> SimpleNamespace:
    """Return a minimal Params-like namespace with the attributes the code uses."""
    mech_addr = "0xMechAddr"
    mech_addr_lower = mech_addr.lower()

    ns = SimpleNamespace(
        tools_to_package_hash={},  # {tool_name: ipfs_cid}
        tools_to_pricing={},  # {tool_name: price}
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
        agent_mech_contract_addresses=[mech_addr],
        mech_to_config={
            mech_addr_lower: SimpleNamespace(
                is_marketplace_mech=True,
                use_dynamic_pricing=False,
            )
        },
        use_mech_marketplace=True,
        mech_marketplace_address="0xMarketplace",
        max_block_window=10_000,
        task_deadline=15.0,
        timeout_limit=2,
        request_id_to_num_timeouts=defaultdict(int),
        is_cold_start=True,
        num_agents=1,
        agent_index=0,
        from_block_range=0,
        mech_to_max_delivery_rate={},
        step_in_list_size=20,
    )

    # Provide a logger object sometimes accessed via params.
    ns.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    return ns


# ----------------------------- Context stub ----------------------------------


class _Outbox:
    """Outbox stub which discards messages."""

    def put_message(self, *args: Any, **kwargs: Any) -> None:
        """Drop an outgoing message."""
        return None


class _SentOutbox:
    """Collect messages if you need to assert what was sent in handler tests."""

    def __init__(self) -> None:
        """Initialize the container."""
        self.sent: List[Any] = []

    def put_message(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """Append a message to the sent list."""
        self.sent.append(message)


@pytest.fixture
def context_stub(
    shared_state: Dict[str, Any], params_stub: SimpleNamespace
) -> SimpleNamespace:
    """Return a minimal AEA-like context with logger/outbox/dialogue stubs."""
    logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )

    class _DLG:
        """Generic dialogue stub."""

        def cleanup(self) -> None:
            """No-op cleanup."""
            return None

    class _IpfsDLG(_DLG):
        """IPFS dialogue stub with update/create helpers."""

        def update(self, _msg: Any) -> Any:
            """Return an object with a stable dialogue reference."""
            return SimpleNamespace(
                dialogue_label=SimpleNamespace(dialogue_reference=("nonce-1", "x"))
            )

        def create(self, *a: Any, **k: Any) -> Tuple[SimpleNamespace, SimpleNamespace]:
            """Return a (message, dialogue) pair with a stable reference."""
            msg = SimpleNamespace()
            dlg = SimpleNamespace(
                dialogue_label=SimpleNamespace(dialogue_reference=("nonce-1", "x"))
            )
            return msg, dlg

    ctx = SimpleNamespace(
        logger=logger,
        shared_state=shared_state,
        params=params_stub,
        agent_address="0xagent",
        default_ledger_id="ethereum",
        outbox=_Outbox(),
        ipfs_dialogues=_IpfsDLG(),
        contract_dialogues=_DLG(),
        ledger_dialogues=_DLG(),
        acn_data_share_dialogues=_DLG(),
    )

    class _HandlersBag:
        """Container listing handler attributes (for dialogue cleanup)."""

        ipfs_handler: object = object()
        contract_handler: object = object()
        ledger_handler: object = object()

    ctx.handlers = _HandlersBag()
    ctx.params.logger = ctx.logger  # optional: some code uses params.logger
    return ctx


# ----------------------------- Behaviour fixture -----------------------------


@pytest.fixture
def behaviour(context_stub: SimpleNamespace) -> TaskExecutionBehaviour:
    """Create a TaskExecutionBehaviour instance wired to the stub context."""
    b = TaskExecutionBehaviour(name="task_execution", skill_context=context_stub)
    b.setup()
    return b


# ----------------------------- Helpers ---------------------------------------


class FakeDialogue:
    """Dialogue with a stable nonce used by behaviour.send_message mapping."""

    class Label:
        """Label with fixed dialogue reference."""

        dialogue_reference = ("nonce-1", "x")

    dialogue_label = Label()


@pytest.fixture
def fake_dialogue() -> FakeDialogue:
    """Return a dialogue stub with a stable nonce."""
    return FakeDialogue()


@pytest.fixture
def done_future() -> Callable[[Any], Future]:
    """Return a factory that produces an already-completed Future."""

    def _make(value: Any) -> Future:
        """Create a Future that is already completed with the given value."""
        fut: Future = Future()
        fut.set_result(value)
        return fut

    return _make


@pytest.fixture
def patch_ipfs_multihash(monkeypatch: Any) -> Callable[[str], None]:
    """Patch CID/multihash helpers to deterministic stubs for tests."""

    def _apply(file_hash: str = "cid-for-task") -> None:
        """Apply monkeypatches for IPFS helper functions."""
        monkeypatch.setattr(beh_mod, "get_ipfs_file_hash", lambda data: file_hash)
        monkeypatch.setattr(beh_mod, "to_v1", lambda cid: cid)
        monkeypatch.setattr(beh_mod, "to_multihash", lambda cid: f"mh:{cid}")

    return _apply


@pytest.fixture
def disable_polling(monkeypatch: Any) -> Callable[[], None]:
    """Disable marketplace polling to keep behaviour tests deterministic."""

    def _apply() -> None:
        """Apply monkeypatch to skip polling in behaviour.act()."""
        monkeypatch.setattr(
            TaskExecutionBehaviour,
            "_check_for_new_marketplace_reqs",
            lambda self: None,
            raising=False,
        )

    return _apply


# ----------------------- Handler-specific context (optional) ------------------


@pytest.fixture
def handler_context(
    shared_state: Dict[str, Any], params_stub: SimpleNamespace
) -> SimpleNamespace:
    """Return a context variant that records sent messages (for handler tests)."""
    ctx = SimpleNamespace(
        logger=SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
        shared_state=shared_state,
        params=params_stub,
        default_ledger_id="ethereum",
        outbox=_SentOutbox(),
    )

    class _DLG:
        """Dialogue stub with cleanup."""

        def cleanup(self) -> None:
            """No-op cleanup."""
            return None

    ctx.handlers = SimpleNamespace(
        ipfs_handler=object(), contract_handler=object(), ledger_handler=object()
    )
    ctx.ipfs_dialogues = SimpleNamespace(
        update=lambda _msg: SimpleNamespace(
            dialogue_label=SimpleNamespace(dialogue_reference=("nonce-1", "x"))
        ),
        cleanup=lambda: None,
        create=lambda *a, **k: (
            SimpleNamespace(),  # msg
            SimpleNamespace(
                dialogue_label=SimpleNamespace(dialogue_reference=("nonce-1", "x"))
            ),
        ),
    )
    ctx.contract_dialogues = _DLG()
    ctx.ledger_dialogues = _DLG()
    ctx.params.logger = ctx.logger
    return ctx


@pytest.fixture
def http_dialogue() -> Any:
    """Return a fake HttpDialogue whose reply() returns a SimpleNamespace."""

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

            :param performative: Response performative.
            :param target_message: The originating HTTP request message.
            :param version: HTTP version string (e.g., "1.1").
            :param status_code: Numeric HTTP status code.
            :param status_text: Human-readable status string.
            :param headers: Raw headers string to include.
            :param body: Raw response body bytes.
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
    """Return a minimal skill_context used by dialogue classes in tests."""
    logger = SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    return SimpleNamespace(
        skill_id="valory/task_execution:0.1.0",
        agent_address="0xagent",
        logger=logger,
        shared_state=shared_state,
    )


@pytest.fixture
def params_kwargs(dialogue_skill_context: SimpleNamespace) -> Dict[str, Any]:
    """
    Return minimal good kwargs for Params; individual tests can mutate per-case.

    :param dialogue_skill_context: The fake Model skill_context.
    :returns: Keyword arguments suitable for constructing Params().
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
        mech_to_max_delivery_rate={},
    )
