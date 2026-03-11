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

import threading
from types import SimpleNamespace
from typing import Any, Callable, Generator
from unittest.mock import MagicMock

import pytest

from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.task_submission_abci.behaviours import (
    DONE_TASKS,
    DONE_TASKS_LOCK,
    PAYMENT_MODEL,
)

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
        logger=SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        ),
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
) -> SimpleNamespace:
    """Extended context with all params needed for complex behaviours."""
    ctx = _make_ctx(
        done_tasks=done_tasks,
        payment_model=payment_model,
        lock=lock,
        agent_mech_addresses=agent_mech_addresses,
    )
    ctx.params.agent_mech_contract_address = "0xMECH"
    ctx.params.hash_checkpoint_address = "0xHASH"
    ctx.params.mech_marketplace_address = "0xMARKET"
    ctx.params.complementary_service_metadata_address = "0xMETA"
    ctx.params.metadata_hash = "bafytest"
    ctx.params.task_mutable_params = SimpleNamespace(latest_metadata_hash=None)
    ctx.params.service_registry_address = "0xSERVICE"
    ctx.params.minimum_agent_balance = 100
    ctx.params.agent_funding_amount = 200
    ctx.params.service_owner_share = 1000
    ctx.params.multisend_address = "0xMULTI"
    ctx.params.mech_staking_instance_address = "0xSTAKE"
    ctx.params.mech_max_delivery_rate = 10
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
        logger=SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
        params=SimpleNamespace(
            profit_split_balance=10,
            on_chain_service_id=100,
            agent_mech_contract_addresses=["0xA"],
        ),
    )


def _make_benchmark_ctx() -> SimpleNamespace:
    """Minimal context with benchmark_tool for async_act tests."""
    return SimpleNamespace(
        logger=SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        ),
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


# ---------------------------------------------------------------------------
# Pytest fixtures (factory versions of the helpers above)
# ---------------------------------------------------------------------------


@pytest.fixture
def make_lock() -> Callable[[], threading.Lock]:
    """Fixture factory for threading.Lock."""
    return _make_lock


@pytest.fixture
def make_ctx() -> Callable[..., SimpleNamespace]:
    """Fixture factory for minimal skill context."""
    return _make_ctx


@pytest.fixture
def make_full_ctx() -> Callable[..., SimpleNamespace]:
    """Fixture factory for extended skill context."""
    return _make_full_ctx


@pytest.fixture
def make_fs_ctx() -> Callable[[], SimpleNamespace]:
    """Fixture factory for FundsSplitting context."""
    return _make_fs_ctx


@pytest.fixture
def make_benchmark_ctx() -> Callable[[], SimpleNamespace]:
    """Fixture factory for benchmark context."""
    return _make_benchmark_ctx


@pytest.fixture
def run_gen() -> Callable[[Generator[Any, Any, Any]], Any]:
    """Fixture for running a generator to completion."""
    return _run_gen
