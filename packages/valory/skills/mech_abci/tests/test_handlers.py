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
from typing import Any, Dict, Optional, Union
from unittest.mock import MagicMock, patch

import pytest

import packages.valory.skills.mech_abci.handlers as hmod
from packages.valory.protocols.http.message import HttpMessage
from packages.valory.skills.mech_abci.handlers import HttpHandler, HttpMethod
from packages.valory.skills.mech_abci.tests.conftest import (
    _make_ctx,
    _make_dialogue,
    _make_handler,
    _make_http_msg,
    _make_round_sequence,
)
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

    @classmethod
    def setup_class(cls) -> None:
        """Create handler instances for the test class."""
        cls.context = MagicMock()
        cls.context.logger = MagicMock()
        cls.handler = HttpHandler(name="", skill_context=cls.context)
        cls.mech_handler = MechHttpHandler(name="", skill_context=cls.context)
        cls.mech_handler.context.shared_state = {}
        cls.handler.context.params.service_endpoint_base = "http://localhost:8080/"
        with patch.object(cls.mech_handler, "start_prometheus_server"):
            cls.mech_handler.setup()
        cls.handler.setup()

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
                name="Happy Path: healthcheck",
                url="http://localhost:8080/healthcheck",
                method=HttpMethod.GET.value,
                expected_handler="_handle_get_health",
            ),
            GetHandlerTestCase(
                name="Happy Path: signed requests",
                url="http://localhost:8080/send_signed_requests",
                method=HttpMethod.POST.value,
                expected_handler="_handle_signed_requests",
                is_mech_handler=True,
            ),
            GetHandlerTestCase(
                name="Happy Path: fetch offchain info",
                url="http://localhost:8080/fetch_offchain_info",
                method=HttpMethod.GET.value,
                expected_handler="_handle_offchain_request_info",
                is_mech_handler=True,
            ),
            GetHandlerTestCase(
                name="Happy Path: fetch offchain info with ID",
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
        if test_case.expected_handler is not None:
            owner = self.mech_handler if test_case.is_mech_handler else self.handler
            expected_handler = getattr(owner, test_case.expected_handler)
        else:
            expected_handler = None

        handler, captures = self.handler._get_handler(test_case.url, test_case.method)

        assert handler == expected_handler
        assert captures == {}


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestHttpHandlerProperties:
    """Tests for the shared_state-backed properties of HttpHandler."""

    def setup_method(self) -> None:
        """Create fresh ctx and handler for each test."""
        self.ctx = _make_ctx()
        self.h = _make_handler(self.ctx)

    def test_last_successful_read_none(self) -> None:
        """Test last successful read none."""
        assert self.h.last_successful_read is None

    def test_last_successful_read_set(self) -> None:
        """Test last successful read set."""
        self.ctx.shared_state[hmod.LAST_SUCCESSFUL_READ] = (100, 1.0)
        assert self.h.last_successful_read == (100, 1.0)

    def test_last_attempt_ts(self) -> None:
        """Test last attempt ts."""
        assert self.h.last_attempt_ts is None
        self.ctx.shared_state[hmod.LAST_READ_ATTEMPT_TS] = 9.5
        assert self.h.last_attempt_ts == 9.5

    def test_inflight_ts(self) -> None:
        """Test inflight ts."""
        assert self.h.inflight_ts is None
        self.ctx.shared_state[hmod.INFLIGHT_READ_TS] = 7.7
        assert self.h.inflight_ts == 7.7

    def test_last_successful_executed_task(self) -> None:
        """Test last successful executed task."""
        assert self.h.last_successful_executed_task is None
        self.ctx.shared_state[hmod.LAST_SUCCESSFUL_EXECUTED_TASK] = ("req-1", 2.0)
        assert self.h.last_successful_executed_task == ("req-1", 2.0)

    def test_was_last_read_successful_defaults_true(self) -> None:
        """Test was last read successful defaults true."""
        assert self.h.was_last_read_successful is True

    def test_was_last_read_successful_false(self) -> None:
        """Test was last read successful false."""
        self.ctx.shared_state[hmod.WAS_LAST_READ_SUCCESSFUL] = False
        assert self.h.was_last_read_successful is False

    def test_last_tx(self) -> None:
        """Test last tx."""
        assert self.h.last_tx is None
        self.ctx.shared_state[hmod.LAST_TX] = ("0xabc", 3.0)
        assert self.h.last_tx == ("0xabc", 3.0)

    def test_synchronized_data(self) -> None:
        """Test synchronized data."""
        with patch(
            "packages.valory.skills.mech_abci.handlers.SynchronizedData"
        ) as MockSD:
            MockSD.return_value = MagicMock(period_count=5)
            self.h.synchronized_data
        assert MockSD.called


# ---------------------------------------------------------------------------
# handle() branches
# ---------------------------------------------------------------------------


class TestHttpHandlerHandle:
    """Tests for HttpHandler.handle() dispatch."""

    def test_handle_non_request_performative_calls_super(self) -> None:
        """Test handle non request performative calls super."""
        ctx = _make_ctx()
        h = _make_handler(ctx)
        msg = _make_http_msg(performative=HttpMessage.Performative.RESPONSE)

        with patch.object(HttpHandler.__bases__[0], "handle") as mock_super:
            h.handle(msg)  # lines 241-245
            mock_super.assert_called_once_with(msg)

    def test_handle_wrong_sender_calls_super(self) -> None:
        """Test handle wrong sender calls super."""
        ctx = _make_ctx()
        h = _make_handler(ctx)
        msg = _make_http_msg(sender="some_other_skill")

        with patch.object(HttpHandler.__bases__[0], "handle") as mock_super:
            h.handle(msg)
            mock_super.assert_called_once_with(msg)

    def test_handle_no_matching_route_calls_super(self) -> None:
        """Test handle no matching route calls super."""
        ctx = _make_ctx()
        h = _make_handler(ctx)
        msg = _make_http_msg(url="http://notmyhost.com/healthcheck")

        with patch.object(HttpHandler.__bases__[0], "handle") as mock_super:
            h.handle(msg)  # line 250
            mock_super.assert_called_once_with(msg)

    def test_handle_none_dialogue_returns_early(self) -> None:
        """Test handle none dialogue returns early."""
        ctx = _make_ctx()
        h = _make_handler(ctx)
        msg = _make_http_msg()
        ctx.http_dialogues.update.return_value = None  # None dialogue → line 264

        h.handle(msg)  # should not raise, no outbox call

        ctx.outbox.put_message.assert_not_called()

    def test_handle_valid_request_calls_handler(self) -> None:
        """Test handle valid request calls handler."""
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
        """Test setup method."""
        self.ctx = _make_ctx()
        self.h = _make_handler(self.ctx)
        self.http_msg = _make_http_msg()
        self.dlg = _make_dialogue()

    def test_handle_bad_request_sends_400(self) -> None:
        """Test handle bad request sends 400."""
        self.h._handle_bad_request(self.http_msg, self.dlg)  # lines 285-297
        self.ctx.outbox.put_message.assert_called_once()
        resp = self.dlg.reply.call_args
        assert resp.kwargs["status_code"] == 400

    def test_send_ok_response_sends_200(self) -> None:
        """Test send ok response sends 200."""
        self.h._send_ok_response(
            self.http_msg, self.dlg, {"key": "val"}
        )  # lines 306-318
        self.ctx.outbox.put_message.assert_called_once()
        resp = self.dlg.reply.call_args
        assert resp.kwargs["status_code"] == 200

    def test_send_not_found_response_sends_404(self) -> None:
        """Test send not found response sends 404."""
        self.h._send_not_found_response(self.http_msg, self.dlg)  # lines 324-335
        self.ctx.outbox.put_message.assert_called_once()
        resp = self.dlg.reply.call_args
        assert resp.kwargs["status_code"] == 404


# ---------------------------------------------------------------------------
# _handle_get_health branches
# ---------------------------------------------------------------------------


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
        if round_sequence is None:
            round_sequence = _make_round_sequence()
        ctx.state.round_sequence = round_sequence
        h = _make_handler(ctx)
        # Merge AFTER handler.setup() so test values aren't overwritten by init
        if shared_state:
            ctx.shared_state.update(shared_state)

        sent: Dict = {}

        def capture_ok(http_msg: Any, http_dlg: Any, data: Any) -> None:
            sent["data"] = data

        with (
            patch.object(h, "_send_ok_response", side_effect=capture_ok),
            patch(
                "packages.valory.skills.mech_abci.handlers.SynchronizedData"
            ) as MockSD,
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
        rs = _make_round_sequence(last_transition=datetime.now(), stall_expired=False)
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
        rs = _make_round_sequence(last_transition=datetime.now(), stall_expired=True)
        data = self._run(round_sequence=rs)
        assert data["liveness"]["ok"] is False
        assert data["liveness"]["reason"] == "tm-unhealthy"

    def test_inflight_recent_covers_readiness(self) -> None:
        """inflight_ts recent enough → readiness deps-inflight even without heartbeat."""
        rs = _make_round_sequence(last_transition=datetime.now(), stall_expired=False)
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
            hmod.PENDING_TASKS: 5,  # int → line 409
            hmod.IPFS_TASKS: object(),  # len() raises TypeError → lines 412-413
        }
        data = self._run(round_sequence=rs, shared_state=ss)
        # pending=5, ipfsq=0 (exception→0), waiting=0 → backlog_size=5
        assert data["progress"]["backlog_size"] == 5

    def test_stuck_no_transition_liveness_fails(self) -> None:
        """FSM data present, TM healthy, but transition too old → stuck-no-transition."""
        from datetime import timedelta

        old = datetime.now() - timedelta(seconds=300)  # 300s ago, factor=3 * pause=30 = 90
        rs = _make_round_sequence(last_transition=old, stall_expired=False)
        data = self._run(round_sequence=rs, reset_pause=30.0)
        assert data["liveness"]["ok"] is False
        assert data["liveness"]["reason"] == "stuck-no-transition"

    def test_backlog_with_executed_task_covers_progress(self) -> None:
        """Backlog + no recent FSM transition, but recent task execution → progress ok."""
        from datetime import timedelta

        old = datetime.now() - timedelta(seconds=300)
        rs = _make_round_sequence(last_transition=old, stall_expired=False)
        now = time.time()
        ss = {
            hmod.PENDING_TASKS: ["task1"],
            hmod.LAST_READ_ATTEMPT_TS: now,
            hmod.LAST_SUCCESSFUL_EXECUTED_TASK: ("req-1", now),
        }
        data = self._run(round_sequence=rs, shared_state=ss, reset_pause=30.0)
        assert data["progress"]["ok"] is True
        assert data["readiness"]["ok"] is True

    def test_backlog_no_executed_task_no_transition_progress_fails(self) -> None:
        """Backlog + no executed task + no recent transition → progress fails."""
        rs = _make_round_sequence(last_transition=None)
        now = time.time()
        ss = {
            hmod.PENDING_TASKS: ["task1"],
            hmod.LAST_READ_ATTEMPT_TS: now,
        }
        data = self._run(round_sequence=rs, shared_state=ss)
        assert data["progress"]["ok"] is False
        assert data["progress"]["reason"] == "no-progress-with-backlog"

    def test_backlog_no_executed_task_but_fast_transition_progress_ok(self) -> None:
        """Backlog + no executed task but fast FSM transition → progress ok."""
        rs = _make_round_sequence(
            last_transition=datetime.now(), stall_expired=False
        )
        now = time.time()
        ss = {
            hmod.PENDING_TASKS: ["task1"],
            hmod.LAST_READ_ATTEMPT_TS: now,
        }
        data = self._run(round_sequence=rs, shared_state=ss)
        assert data["progress"]["ok"] is True

    def test_all_three_dimensions_independent(self) -> None:
        """Healthy only when ALL three dimensions are ok."""
        # Liveness ok, readiness ok (idle), progress ok (idle)
        rs = _make_round_sequence(last_transition=datetime.now(), stall_expired=False)
        data = self._run(round_sequence=rs)
        assert data["is_healthy"] is True

        # Liveness fails → is_healthy False even if readiness/progress ok
        rs_bad = _make_round_sequence(last_transition=None)
        data2 = self._run(round_sequence=rs_bad)
        assert data2["is_healthy"] is False

    def test_wait_for_timeout_contributes_to_backlog(self) -> None:
        """WAIT_FOR_TIMEOUT items add to backlog_size."""
        rs = _make_round_sequence(last_transition=datetime.now(), stall_expired=False)
        now = time.time()
        ss = {
            hmod.WAIT_FOR_TIMEOUT: ["w1", "w2"],
            hmod.LAST_READ_ATTEMPT_TS: now,
        }
        data = self._run(round_sequence=rs, shared_state=ss)
        assert data["progress"]["backlog_size"] == 2
        assert data["progress"]["expected_work"] is True

    def test_ipfs_tasks_contribute_to_backlog(self) -> None:
        """IPFS_TASKS items add to backlog_size."""
        rs = _make_round_sequence(last_transition=datetime.now(), stall_expired=False)
        now = time.time()
        ss = {
            hmod.IPFS_TASKS: ["i1"],
            hmod.LAST_READ_ATTEMPT_TS: now,
        }
        data = self._run(round_sequence=rs, shared_state=ss)
        assert data["progress"]["backlog_size"] == 1

    def test_deps_ok_takes_priority_over_inflight(self) -> None:
        """When both deps and inflight are recent, reason should be 'deps-ok' not 'deps-inflight'."""
        rs = _make_round_sequence(last_transition=datetime.now(), stall_expired=False)
        now = time.time()
        ss = {
            hmod.PENDING_TASKS: ["task1"],
            hmod.LAST_READ_ATTEMPT_TS: now,
            hmod.INFLIGHT_READ_TS: now,
        }
        data = self._run(round_sequence=rs, shared_state=ss)
        assert data["readiness"]["reason"] == "deps-ok"

    def test_is_transitioning_fast_when_fsm_healthy(self) -> None:
        """is_transitioning_fast is True when TM healthy and recent transition."""
        rs = _make_round_sequence(
            last_transition=datetime.now(), stall_expired=False, has_abci_app=True
        )
        data = self._run(round_sequence=rs, reset_pause=30.0)
        assert data["is_transitioning_fast"] is True

    def test_is_transitioning_fast_none_when_no_fsm(self) -> None:
        """is_transitioning_fast is None when no FSM data."""
        data = self._run(round_sequence=_make_round_sequence(last_transition=None))
        assert data["is_transitioning_fast"] is None

    def test_health_version_always_2(self) -> None:
        """Response always includes health_version=2."""
        data = self._run()
        assert data["health_version"] == 2
