# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025-2026 Valory AG
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

"""Shared test helpers and fixtures for the mech_abci skill tests."""

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Generator, Optional
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.connections.http_server.connection import (
    PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
)
from packages.valory.protocols.http.message import HttpMessage
from packages.valory.skills.mech_abci.composition import MechAbciApp
from packages.valory.skills.mech_abci.handlers import HttpHandler
from packages.valory.skills.mech_abci.models import SharedState
from packages.valory.skills.task_execution.handlers import MechHttpHandler
from packages.valory.skills.task_submission_abci.models import (
    SharedState as TaskExecSharedState,
)

# ---------------------------------------------------------------------------
# Helpers from test_handlers.py
# ---------------------------------------------------------------------------


def _make_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.logger = MagicMock()
    ctx.params.service_endpoint_base = "http://localhost:8080/"
    ctx.params.reset_pause_duration = 30.0
    ctx.shared_state = {}
    ctx.outbox = MagicMock()
    return ctx


def _make_handler(ctx: Optional[MagicMock] = None) -> HttpHandler:
    if ctx is None:
        ctx = _make_ctx()
    # MechHttpHandler.setup() populates shared_state["routes_info"] which HttpHandler.setup() needs
    mech_h = MechHttpHandler(name="mech_http", skill_context=ctx)
    with patch.object(mech_h, "start_prometheus_server"):
        mech_h.setup()
    h = HttpHandler(name="http", skill_context=ctx)
    h.setup()
    return h


def _make_http_msg(
    performative: Any = None,
    url: str = "http://localhost:8080/healthcheck",
    method: str = "get",
    body: bytes = b"",
    sender: Optional[str] = None,
) -> MagicMock:
    if performative is None:
        performative = HttpMessage.Performative.REQUEST
    if sender is None:
        sender = str(HTTP_SERVER_PUBLIC_ID.without_hash())
    msg = MagicMock()
    msg.performative = performative
    msg.url = url
    msg.method = method
    msg.body = body
    msg.sender = sender
    msg.version = "1.1"
    msg.headers = ""
    return msg


def _make_dialogue() -> MagicMock:
    dlg = MagicMock()
    dlg.reply.return_value = SimpleNamespace(
        status_code=200, body=b"ok", version="1.1", headers=""
    )
    return dlg


def _make_round_sequence(
    last_transition: Optional[datetime] = None,
    stall_expired: bool = False,
    has_abci_app: bool = False,
) -> MagicMock:
    rs = MagicMock()
    rs._last_round_transition_timestamp = last_transition
    rs.block_stall_deadline_expired = stall_expired
    if has_abci_app:
        rs._abci_app.current_round.round_id = "some_round"
        rs._abci_app._previous_rounds = []
    else:
        rs._abci_app = None
    return rs


# ---------------------------------------------------------------------------
# Helpers from test_models.py
# ---------------------------------------------------------------------------


def _make_context(round_timeout=10.0, validate=20.0, finalize=30.0, reset_pause=60.0):  # type: ignore
    return SimpleNamespace(
        params=SimpleNamespace(
            round_timeout_seconds=round_timeout,
            validate_timeout=validate,
            finalize_timeout=finalize,
            reset_pause_duration=reset_pause,
        )
    )


def _make_shared_state(ctx=None) -> SharedState:  # type: ignore
    """Create a SharedState instance bypassing the AEA framework init."""
    with patch.object(TaskExecSharedState, "__init__", return_value=None):
        state = SharedState.__new__(SharedState)
        SharedState.__init__(state)
    # context is a read-only property backed by _context
    object.__setattr__(state, "_context", ctx or _make_context())
    return state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def preserve_event_to_timeout() -> Generator[None, None, None]:
    """Save and restore MechAbciApp.event_to_timeout around a test."""
    original = dict(MechAbciApp.event_to_timeout)
    yield
    MechAbciApp.event_to_timeout.clear()
    MechAbciApp.event_to_timeout.update(original)
