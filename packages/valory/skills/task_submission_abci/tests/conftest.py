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

"""Conftest for the task_submission_abci tests."""

import contextlib
import threading
from types import SimpleNamespace
from typing import Any, Generator
from unittest.mock import MagicMock, patch

from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.task_submission_abci.behaviours import (
    DONE_TASKS,
    DONE_TASKS_LOCK,
    PAYMENT_MODEL,
)


def _make_logger(include_debug: bool = False) -> SimpleNamespace:
    """Create a no-op logger stub. Single source of truth for all test contexts."""
    ns = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    if include_debug:
        ns.debug = lambda *a, **k: None  # type: ignore[attr-defined]
    return ns


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def _make_lock() -> threading.Lock:
    return threading.Lock()


def _make_ctx(
    done_tasks: Any = None,
    payment_model: Any = None,
    lock: Any = None,
    agent_mech_addresses: Any = None,
) -> SimpleNamespace:
    if lock is None:
        lock = _make_lock()
    shared_state: dict = {
        DONE_TASKS_LOCK: lock,
        DONE_TASKS: done_tasks if done_tasks is not None else [],
    }
    if payment_model is not None:
        shared_state[PAYMENT_MODEL] = payment_model
    return SimpleNamespace(
        logger=_make_logger(include_debug=True),
        params=SimpleNamespace(
            profit_split_balance=100,
            on_chain_service_id=1,
            agent_mech_contract_addresses=agent_mech_addresses or ["0xMECH"],
            task_wait_timeout=0.1,
            default_chain_id="100",
        ),
        shared_state=shared_state,
    )


def _make_full_ctx(
    done_tasks: Any = None,
    payment_model: Any = None,
    lock: Any = None,
    agent_mech_addresses: Any = None,
    **overrides: Any,
) -> SimpleNamespace:
    """Extended context with all params needed for complex behaviours.

    Use **overrides to set/override any param attribute, e.g.
    ``_make_full_ctx(multisend_address="0xCUSTOM")``.
    """
    ctx = _make_ctx(
        done_tasks=done_tasks,
        payment_model=payment_model,
        lock=lock,
        agent_mech_addresses=agent_mech_addresses,
    )
    defaults = dict(
        agent_mech_contract_address="0xMECH",
        hash_checkpoint_address="0xHASH",
        mech_marketplace_address="0xMARKET",
        complementary_service_metadata_address="0xMETA",
        metadata_hash="bafytest",
        task_mutable_params=SimpleNamespace(latest_metadata_hash=None),
        service_registry_address="0xSERVICE",
        minimum_agent_balance=100,
        agent_funding_amount=200,
        service_owner_share=1000,
        multisend_address="0xMULTI",
        mech_staking_instance_address="0xSTAKE",
        mech_max_delivery_rate=10,
    )
    defaults.update(overrides)
    for attr, val in defaults.items():
        setattr(ctx.params, attr, val)
    # ctx.state is accessed by BaseBehaviour.shared_state property
    ctx.state = SimpleNamespace(
        mech_delivery_last_block_number=MagicMock(),
        mech_agent_balance=MagicMock(),
        tool_delivery_time=MagicMock(),
        synchronized_data=SimpleNamespace(safe_contract_address="0xSAFE"),
    )
    return ctx


def _make_fs_ctx() -> SimpleNamespace:
    """Minimal context for FundsSplittingBehaviour tests."""
    return SimpleNamespace(
        logger=_make_logger(),
        params=SimpleNamespace(
            profit_split_balance=10,
            on_chain_service_id=100,
            agent_mech_contract_addresses=["0xA"],
        ),
    )


def _make_benchmark_ctx() -> SimpleNamespace:
    """Minimal context with benchmark_tool for async_act tests."""
    return SimpleNamespace(
        logger=_make_logger(include_debug=True),
        params=SimpleNamespace(
            task_wait_timeout=0.01,
            agent_mech_contract_addresses=["0xMECH"],
        ),
        shared_state={
            DONE_TASKS_LOCK: threading.Lock(),
            DONE_TASKS: [],
        },
        # MagicMock supports the context-manager protocol automatically, so
        # benchmark_tool.measure(id).local() and .consensus() work as with-targets.
        benchmark_tool=MagicMock(),
        agent_address="test_agent_address",
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _run_gen(gen: Generator[Any, Any, Any]) -> Any:
    """Run a generator to completion, returning its final value."""
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def _gen_returning(value: Any) -> Any:
    """Create a generator that returns `value` immediately."""

    def _gen(*args: Any, **kwargs: Any) -> Any:
        if False:
            yield
        return value

    return _gen


def _noop_gen() -> Generator:
    """Generator that returns immediately with None."""
    yield from ()


def _noop_gen_with_args(*_args: object, **_kwargs: object) -> Generator:
    """Generator that accepts any arguments and returns immediately."""
    yield from ()


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def _state_contract_msg(body: Any = None) -> MagicMock:
    """Return a MagicMock with STATE performative and given body."""
    msg = MagicMock()
    msg.performative = ContractApiMessage.Performative.STATE
    if body is not None:
        msg.state.body = body
    return msg


def _error_contract_msg() -> MagicMock:
    """Return a MagicMock with ERROR performative (non-STATE)."""
    msg = MagicMock()
    msg.performative = ContractApiMessage.Performative.ERROR
    return msg


def _error_contract_msg_with_detail(
    code: int = 500, message: str = "boom"
) -> MagicMock:
    """Return an ERROR MagicMock with explicit code + message, mimicking the ledger dispatcher."""
    msg = MagicMock()
    msg.performative = ContractApiMessage.Performative.ERROR
    msg.code = code
    msg.message = message
    msg.is_set = MagicMock(side_effect=lambda key: key in {"code", "message"})
    return msg


def _raw_tx_contract_msg(body: Any = None) -> MagicMock:
    """Return a MagicMock with RAW_TRANSACTION performative."""
    msg = MagicMock()
    msg.performative = ContractApiMessage.Performative.RAW_TRANSACTION
    if body is not None:
        msg.raw_transaction.body = body
    return msg


def _state_ledger_msg(body: Any = None) -> MagicMock:
    """Return a MagicMock with STATE performative and given body (ledger API)."""
    msg = MagicMock()
    msg.performative = LedgerApiMessage.Performative.STATE
    if body is not None:
        msg.state.body = body
    return msg


def _error_ledger_msg() -> MagicMock:
    """Return a MagicMock with non-STATE performative (ledger API)."""
    msg = MagicMock()
    msg.performative = LedgerApiMessage.Performative.ERROR
    return msg


def _mock_to_multihash() -> contextlib.AbstractContextManager:
    """Patch to_multihash in behaviours; use as a context manager."""
    return patch("packages.valory.skills.task_submission_abci.behaviours.to_multihash")
