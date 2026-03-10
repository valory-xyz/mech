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

"""Test the handlers.py module of the mech_abci skill."""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union
from unittest.mock import MagicMock, patch

import pytest

import packages.valory.skills.mech_abci.handlers as hmod
from packages.valory.connections.http_server.connection import (
    PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
)
from packages.valory.protocols.http.message import HttpMessage
from packages.valory.skills.mech_abci.handlers import HttpHandler, HttpMethod
from packages.valory.skills.task_execution.handlers import MechHttpHandler

PACKAGE_DIR = Path(__file__).parents[1]


@dataclass
class GetHandlerTestCase:
    """Get Handler test case."""

    name: str
    url: str
    method: str
    expected_handler: Union[str, None]
    is_mech_handler: Optional[bool] = False


class TestHttpHandler:
    """Test HttpHandler of mech_abci."""

    path_to_skill = PACKAGE_DIR

    def setup_class(self) -> None:
        """Setup the test class."""
        self.context = MagicMock()
        self.context.logger = MagicMock()
        self.handler = HttpHandler(name="", skill_context=self.context)
        self.mech_handler = MechHttpHandler(name="", skill_context=self.context)
        self.mech_handler.context.shared_state = {}
        self.handler.context.params.service_endpoint_base = "http://localhost:8080/"
        self.mech_handler.setup()
        self.handler.setup()

    def test_setup(self) -> None:
        """Test the setup method of the handler."""
        service_endpoint_base = "localhost"
        propel_uri_base_hostname = (
            r"https?:\/\/[a-zA-Z0-9]{16}.agent\.propel\.(staging\.)?autonolas\.tech"
        )
        hostname_regex = rf".*({service_endpoint_base}|{propel_uri_base_hostname}|localhost|127.0.0.1|0.0.0.0)(:\d+)?"
        health_url_regex = rf"{hostname_regex}\/healthcheck"
        send_signed_url = rf"{hostname_regex}\/send_signed_requests"
        fetch_offchain_info_url = rf"{hostname_regex}\/fetch_offchain_info"

        assert self.handler.handler_url_regex == rf"{hostname_regex}\/.*"
        assert self.handler.routes == {
            (HttpMethod.GET.value, HttpMethod.HEAD.value): [
                (health_url_regex, self.handler._handle_get_health),
                (
                    fetch_offchain_info_url,
                    self.mech_handler._handle_offchain_request_info,
                ),
            ],
            (HttpMethod.POST.value,): [
                (send_signed_url, self.mech_handler._handle_signed_requests)
            ],
        }
        assert self.handler.json_content_header == "Content-Type: application/json\n"

    @pytest.mark.parametrize(
        "test_case",
        [
            GetHandlerTestCase(
                name="Happy Path",
                url="http://localhost:8080/healthcheck",
                method=HttpMethod.GET.value,
                expected_handler="_handle_get_health",
            ),
            GetHandlerTestCase(
                name="Happy Path",
                url="http://localhost:8080/send_signed_requests",
                method=HttpMethod.POST.value,
                expected_handler="_handle_signed_requests",
                is_mech_handler=True,
            ),
            GetHandlerTestCase(
                name="Happy Path",
                url="http://localhost:8080/fetch_offchain_info",
                method=HttpMethod.GET.value,
                expected_handler="_handle_offchain_request_info",
                is_mech_handler=True,
            ),
            GetHandlerTestCase(
                name="Happy Path",
                url="http://localhost:8080/fetch_offchain_info/1",
                method=HttpMethod.GET.value,
                expected_handler="_handle_offchain_request_info",
                is_mech_handler=True,
            ),
            GetHandlerTestCase(
                name="No url match",
                url="http://invalid.url/not/matching",
                method=HttpMethod.GET.value,
                expected_handler=None,
            ),
            GetHandlerTestCase(
                name="No method match",
                url="http://localhost:8080/some/path",
                method=HttpMethod.POST.value,
                expected_handler="_handle_bad_request",
            ),
        ],
    )
    def test_mech_get_handler(self, test_case: GetHandlerTestCase) -> None:
        """Test _get_handler."""
        url = test_case.url
        method = test_case.method
        is_mech_handler = test_case.is_mech_handler

        if test_case.expected_handler is not None:
            if is_mech_handler:
                expected_handler = getattr(
                    self.mech_handler, test_case.expected_handler
                )
            else:
                expected_handler = getattr(self.handler, test_case.expected_handler)
        else:
            expected_handler = test_case.expected_handler
        expected_captures: Dict[Any, Any] = {}

        handler, captures = self.handler._get_handler(url, method)

        assert handler == expected_handler
        assert captures == expected_captures


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestHttpHandlerProperties:
    """Tests for the shared_state-backed properties of HttpHandler."""

    def test_last_successful_read_none(self) -> None:
        ctx = _make_ctx()
        h = _make_handler(ctx)
        assert h.last_successful_read is None  # line 152

    def test_last_successful_read_set(self) -> None:
        ctx = _make_ctx()
        h = _make_handler(ctx)
        ctx.shared_state[hmod.LAST_SUCCESSFUL_READ] = (100, 1.0)
        assert h.last_successful_read == (100, 1.0)

    def test_last_attempt_ts(self) -> None:
        ctx = _make_ctx()
        h = _make_handler(ctx)
        assert h.last_attempt_ts is None  # line 160
        ctx.shared_state[hmod.LAST_READ_ATTEMPT_TS] = 9.5
        assert h.last_attempt_ts == 9.5

    def test_inflight_ts(self) -> None:
        ctx = _make_ctx()
        h = _make_handler(ctx)
        assert h.inflight_ts is None  # line 165
        ctx.shared_state[hmod.INFLIGHT_READ_TS] = 7.7
        assert h.inflight_ts == 7.7

    def test_last_successful_executed_task(self) -> None:
        ctx = _make_ctx()
        h = _make_handler(ctx)
        assert h.last_successful_executed_task is None  # line 170
        ctx.shared_state[hmod.LAST_SUCCESSFUL_EXECUTED_TASK] = ("req-1", 2.0)
        assert h.last_successful_executed_task == ("req-1", 2.0)

    def test_was_last_read_successful_defaults_true(self) -> None:
        ctx = _make_ctx()
        h = _make_handler(ctx)
        assert h.was_last_read_successful is True  # line 178 — absent → not False → True

    def test_was_last_read_successful_false(self) -> None:
        ctx = _make_ctx()
        h = _make_handler(ctx)
        ctx.shared_state[hmod.WAS_LAST_READ_SUCCESSFUL] = False
        assert h.was_last_read_successful is False

    def test_last_tx(self) -> None:
        ctx = _make_ctx()
        h = _make_handler(ctx)
        assert h.last_tx is None  # line 183
        ctx.shared_state[hmod.LAST_TX] = ("0xabc", 3.0)
        assert h.last_tx == ("0xabc", 3.0)

    def test_synchronized_data(self) -> None:
        ctx = _make_ctx()
        h = _make_handler(ctx)
        # SynchronizedData wraps the db from context.state; just verify no AttributeError
        with patch("packages.valory.skills.mech_abci.handlers.SynchronizedData") as MockSD:
            MockSD.return_value = MagicMock(period_count=5)
            sd = h.synchronized_data  # line 188
        assert MockSD.called


# ---------------------------------------------------------------------------
# handle() branches
# ---------------------------------------------------------------------------


class TestHttpHandlerHandle:
    """Tests for HttpHandler.handle() dispatch."""

    def test_handle_non_request_performative_calls_super(self) -> None:
        ctx = _make_ctx()
        h = _make_handler(ctx)
        msg = _make_http_msg(performative=HttpMessage.Performative.RESPONSE)

        with patch.object(HttpHandler.__bases__[0], "handle") as mock_super:
            h.handle(msg)  # lines 241-245
            mock_super.assert_called_once_with(msg)

    def test_handle_wrong_sender_calls_super(self) -> None:
        ctx = _make_ctx()
        h = _make_handler(ctx)
        msg = _make_http_msg(sender="some_other_skill")

        with patch.object(HttpHandler.__bases__[0], "handle") as mock_super:
            h.handle(msg)
            mock_super.assert_called_once_with(msg)

    def test_handle_no_matching_route_calls_super(self) -> None:
        ctx = _make_ctx()
        h = _make_handler(ctx)
        msg = _make_http_msg(url="http://notmyhost.com/healthcheck")

        with patch.object(HttpHandler.__bases__[0], "handle") as mock_super:
            h.handle(msg)  # line 250
            mock_super.assert_called_once_with(msg)

    def test_handle_none_dialogue_returns_early(self) -> None:
        ctx = _make_ctx()
        h = _make_handler(ctx)
        msg = _make_http_msg()
        ctx.http_dialogues.update.return_value = None  # None dialogue → line 264

        h.handle(msg)  # should not raise, no outbox call

        ctx.outbox.put_message.assert_not_called()

    def test_handle_valid_request_calls_handler(self) -> None:
        ctx = _make_ctx()
        h = _make_handler(ctx)
        msg = _make_http_msg(url="http://localhost:8080/healthcheck")
        ctx.http_dialogues.update.return_value = _make_dialogue()

        mock_route_fn = MagicMock()
        with patch.object(h, "_get_handler", return_value=(mock_route_fn, {})):
            h.handle(msg)  # lines 266-274
            mock_route_fn.assert_called_once()


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


class TestHttpHandlerResponseHelpers:
    """Tests for _handle_bad_request, _send_ok_response, _send_not_found_response."""

    def setup_method(self) -> None:
        self.ctx = _make_ctx()
        self.h = _make_handler(self.ctx)
        self.http_msg = _make_http_msg()
        self.dlg = _make_dialogue()

    def test_handle_bad_request_sends_400(self) -> None:
        self.h._handle_bad_request(self.http_msg, self.dlg)  # lines 285-297
        self.ctx.outbox.put_message.assert_called_once()
        resp = self.dlg.reply.call_args
        assert resp.kwargs["status_code"] == 400

    def test_send_ok_response_sends_200(self) -> None:
        self.h._send_ok_response(self.http_msg, self.dlg, {"key": "val"})  # lines 306-318
        self.ctx.outbox.put_message.assert_called_once()
        resp = self.dlg.reply.call_args
        assert resp.kwargs["status_code"] == 200

    def test_send_not_found_response_sends_404(self) -> None:
        self.h._send_not_found_response(self.http_msg, self.dlg)  # lines 324-335
        self.ctx.outbox.put_message.assert_called_once()
        resp = self.dlg.reply.call_args
        assert resp.kwargs["status_code"] == 404


# ---------------------------------------------------------------------------
# _handle_get_health branches
# ---------------------------------------------------------------------------


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


class TestHandleGetHealth:
    """Tests for _handle_get_health covering all branches."""

    def _run(
        self,
        round_sequence: Any = None,
        shared_state: Optional[Dict] = None,
        reset_pause: float = 30.0,
    ) -> Dict:
        ctx = _make_ctx()
        ctx.params.reset_pause_duration = reset_pause
        ctx.shared_state = shared_state or {}
        if round_sequence is None:
            round_sequence = _make_round_sequence()
        ctx.state.round_sequence = round_sequence
        h = _make_handler(ctx)

        sent: Dict = {}

        def capture_ok(http_msg: Any, http_dlg: Any, data: Any) -> None:
            sent["data"] = data

        with (
            patch.object(h, "_send_ok_response", side_effect=capture_ok),
            patch("packages.valory.skills.mech_abci.handlers.SynchronizedData") as MockSD,
        ):
            MockSD.return_value = MagicMock(period_count=1)
            h._handle_get_health(MagicMock(), _make_dialogue())

        return sent.get("data", {})

    def test_cold_start_no_fsm_no_backlog(self) -> None:
        """No FSM data, no backlog → no-fsm-data liveness, idle readiness/progress."""
        data = self._run(round_sequence=_make_round_sequence(last_transition=None))
        assert data["liveness"]["reason"] == "no-fsm-data"
        assert data["readiness"]["reason"] == "idle-ok"
        assert data["progress"]["reason"] == "idle-ok"
        assert data["seconds_since_last_transition"] is None
        assert data["last_successful_read"] is None
        assert data["last_successful_executed_task"] is None
        assert data["last_tx"] is None

    def test_warm_idle_healthy_fsm(self) -> None:
        """Recent FSM transition, no backlog → liveness ok, idle readiness/progress."""
        rs = _make_round_sequence(
            last_transition=datetime.now(),
            stall_expired=False,
            has_abci_app=True,
        )
        ss = {
            hmod.LAST_SUCCESSFUL_READ: (50, time.time()),
            hmod.LAST_TX: ("0xabc", time.time()),
        }
        data = self._run(round_sequence=rs, shared_state=ss)
        assert data["liveness"]["ok"] is True
        assert data["readiness"]["reason"] == "idle-ok"
        assert data["progress"]["reason"] == "idle-ok"
        assert data["last_successful_read"] is not None
        assert data["last_tx"] is not None
        assert data["current_round"] == "some_round"

    def test_with_backlog_fresh_deps_and_executed_task(self) -> None:
        """Backlog exists, heartbeat recent, task executed recently → all healthy."""
        rs = _make_round_sequence(
            last_transition=datetime.now(), stall_expired=False
        )
        now = time.time()
        ss = {
            hmod.PENDING_TASKS: ["task1"],
            hmod.LAST_READ_ATTEMPT_TS: now,
            hmod.LAST_SUCCESSFUL_EXECUTED_TASK: ("req-1", now),
        }
        data = self._run(round_sequence=rs, shared_state=ss)
        assert data["progress"]["expected_work"] is True
        assert data["readiness"]["ok"] is True
        assert data["progress"]["ok"] is True
        assert data["last_successful_executed_task"] is not None

    def test_with_backlog_stale_deps_no_fsm_transition(self) -> None:
        """Backlog exists, stale heartbeat, no recent transition → readiness/progress fail."""
        rs = _make_round_sequence(last_transition=None)
        old_ts = time.time() - 9999
        ss = {
            hmod.PENDING_TASKS: ["task1"],
            hmod.LAST_READ_ATTEMPT_TS: old_ts,
        }
        data = self._run(round_sequence=rs, shared_state=ss, reset_pause=1.0)
        assert data["readiness"]["reason"] == "stale-read"
        assert data["progress"]["reason"] == "no-progress-with-backlog"

    def test_tm_unhealthy_liveness_false(self) -> None:
        """Tendermint stall detected → liveness not ok."""
        rs = _make_round_sequence(
            last_transition=datetime.now(), stall_expired=True
        )
        data = self._run(round_sequence=rs)
        assert data["liveness"]["ok"] is False
        assert data["liveness"]["reason"] == "tm-unhealthy"

    def test_inflight_recent_covers_readiness(self) -> None:
        """inflight_ts recent enough → readiness deps-inflight even without heartbeat."""
        rs = _make_round_sequence(
            last_transition=datetime.now(), stall_expired=False
        )
        now = time.time()
        ss = {
            hmod.PENDING_TASKS: ["task1"],
            hmod.INFLIGHT_READ_TS: now,  # recent inflight
        }
        data = self._run(round_sequence=rs, shared_state=ss)
        assert data["readiness"]["ok"] is True
        assert data["readiness"]["reason"] == "deps-inflight"

    def test_len_or_zero_int_and_exception_branches(self) -> None:
        """_len_or_zero: int value hits line 409; object() with no __len__ hits 412-413."""
        rs = _make_round_sequence(last_transition=None)
        ss = {
            hmod.PENDING_TASKS: 5,    # int → line 409
            hmod.IPFS_TASKS: object(),  # len() raises TypeError → lines 412-413
        }
        data = self._run(round_sequence=rs, shared_state=ss)
        # pending=5, ipfsq=0 (exception→0), waiting=0 → backlog_size=5
        assert data["progress"]["backlog_size"] == 5
