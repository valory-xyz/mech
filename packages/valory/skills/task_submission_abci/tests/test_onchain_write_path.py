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

"""Tests for the on-chain write path additions in :py:class:`PostTxSettlementBehaviour`.

Three pieces:

  - ``_sweep_pending_undelivered`` walks ``shared_state[PENDING_TASKS]``
    and RETURNS ``(events, swept_request_ids)`` for tasks past the local
    sweep age, without mutating the queue. All gated on
    ``use_offchain`` AND ``mech_events_sweep_pending_enabled``.
  - ``_drop_swept_from_pending`` mutates ``PENDING_TASKS`` in place to
    drop the swept tasks. The caller only fires this after a confirmed
    predict-api POST 2xx, so any of the six early-return paths in
    ``_do_predict_api_write_best_effort`` (missing mech address, missing
    marketplace, mixed chains, unconfigured chain id, digest failure,
    POST failure) leaves the tasks on the queue for a retry.
  - ``_build_request_only_event`` shapes one entry into the predict-api
    event payload (``request`` half, ``response: None``,
    ``source: 'mech_onchain'``).

The behaviour class is instantiated indirectly by reaching into the
module's class definition; the methods under test are pure functions of
``self.context.shared_state`` + ``self.params``, so a SimpleNamespace
context plus the unbound method invoked with that context is enough —
no FSM scaffolding required. Same pattern matches what the existing
tests in ``test_behaviours.py`` use for narrow-scope unit checks.

"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Dict, List, cast

from packages.valory.skills.task_execution.behaviours import PREDICT_API_EVENTS
from packages.valory.skills.task_submission_abci.behaviours import (
    PENDING_TASKS,
    PostTxSettlementBehaviour,
)


def _make_logger() -> SimpleNamespace:
    """No-op logger that absorbs every level."""
    return SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )


def _make_self(
    *,
    pending: List[Any] | None = None,
    enabled: bool = True,
    sweep_enabled: bool = True,
    max_age: float = 60.0,
    chain_id: int = 100,
    marketplace_address: str = "0xMARKET",
) -> PostTxSettlementBehaviour:
    """Build the minimum 'self' a sweep / build call expects.

    The behaviour's :py:meth:`_sweep_pending_undelivered` and
    :py:meth:`_build_request_only_event` only touch ``self.context``,
    ``self.params``, and (indirectly) ``self.context.logger``, so a flat
    :class:`SimpleNamespace` context is enough at runtime.
    ``_sweep_pending_undelivered`` calls
    ``self._build_request_only_event`` on the same self_, so we wire
    that method onto the namespace as well so the unbound-method
    invocation pattern works (calling
    ``PostTxSettlementBehaviour._sweep_pending_undelivered(self_)``).

    The return is annotated as :class:`PostTxSettlementBehaviour` via
    :func:`typing.cast` so the many unbound-method callsites in the
    tests below type-check under the stricter mypy config the CI runs
    (``--disallow-untyped-defs``). At runtime this is still a
    :class:`SimpleNamespace`; the cast is a mypy annotation, not a
    conversion.

    ``pending`` accepts ``List[Any]`` (not ``List[Dict[str, Any]]``) so
    the "non-dict entry" defensive test can pass a mixed
    ``["garbage", {...}]`` without a ``[list-item]`` mypy error; the
    sweep code path skips non-dict entries at runtime.

    :param pending: value to stash under ``shared_state[PENDING_TASKS]``,
        or ``None`` to leave the key unset (simulates a fresh mech boot).
    :param enabled: value for ``params.use_offchain``.
    :param sweep_enabled: value for ``params.mech_events_sweep_pending_enabled``.
    :param max_age: value for ``params.mech_events_sweep_max_age_seconds``.
    :param chain_id: value for ``params.mech_events_chain_id``.
    :param marketplace_address: value for ``params.mech_marketplace_address``.
    :return: a fixture typed as :class:`PostTxSettlementBehaviour` (a
        :class:`SimpleNamespace` at runtime) suitable for the unbound-method
        call pattern used by every test in this module.
    """
    shared_state: Dict[str, Any] = {}
    if pending is not None:
        shared_state[PENDING_TASKS] = pending
    self_ = SimpleNamespace(
        context=SimpleNamespace(
            shared_state=shared_state,
            logger=_make_logger(),
        ),
        params=SimpleNamespace(
            use_offchain=enabled,
            mech_events_sweep_pending_enabled=sweep_enabled,
            mech_events_sweep_max_age_seconds=max_age,
            mech_events_chain_id=chain_id,
            mech_marketplace_address=marketplace_address,
        ),
    )
    # Bind ``_build_request_only_event`` to the namespace so the sweep's
    # internal ``self._build_request_only_event(...)`` call resolves.
    # Cast the inner ``self_`` for the same mypy reason as the outer cast
    # on this function's return.
    self_._build_request_only_event = (
        lambda task, now_ts: PostTxSettlementBehaviour._build_request_only_event(
            cast(PostTxSettlementBehaviour, self_), task, now_ts
        )
    )
    return cast(PostTxSettlementBehaviour, self_)


def _make_pending_task(
    request_id: str = "req-1",
    *,
    age_seconds: float = 0.0,
    priority_mech: str = "0xPRIO",
    requester: str = "0xREQUESTER",
    delivery_rate: int | None = 10**16,
    data: Any = "bafy-cid",
    include_enqueued_at: bool = True,
) -> Dict[str, Any]:
    """Build one PENDING_TASKS entry. ``age_seconds`` ago = now - age."""
    task: Dict[str, Any] = {
        "requestId": request_id,
        "priorityMech": priority_mech,
        "requester": requester,
        "request_delivery_rate": delivery_rate,
        "data": data,
    }
    if include_enqueued_at:
        task["enqueued_at_local"] = time.time() - age_seconds
    return task


# Gating -------------------------------------------------------------------


def test_sweep_returns_empty_when_mech_events_disabled() -> None:
    """``use_offchain=False`` short-circuits before any sweep."""
    self_ = _make_self(
        pending=[_make_pending_task("r-disabled", age_seconds=9999)],
        enabled=False,
    )
    events, swept = PostTxSettlementBehaviour._sweep_pending_undelivered(self_)
    assert events == []
    assert swept == []
    # Pending list untouched.
    assert len(self_.context.shared_state[PENDING_TASKS]) == 1


def test_sweep_returns_empty_when_sweep_flag_disabled() -> None:
    """``mech_events_sweep_pending_enabled=False`` short-circuits the sweep on its own.

    The sweep can be flipped off independently from the delivered-event
    write, so operators can roll each out separately.
    """
    self_ = _make_self(
        pending=[_make_pending_task("r-sweep-off", age_seconds=9999)],
        enabled=True,
        sweep_enabled=False,
    )
    events, swept = PostTxSettlementBehaviour._sweep_pending_undelivered(self_)
    assert events == []
    assert swept == []
    assert len(self_.context.shared_state[PENDING_TASKS]) == 1


def test_sweep_returns_empty_when_max_age_zero() -> None:
    """A zero (or negative) max-age is treated as 'sweep off' to avoid accidentally sweeping every fresh task that the agent just enqueued."""
    self_ = _make_self(
        pending=[_make_pending_task("r-zero", age_seconds=9999)],
        max_age=0.0,
    )
    events, swept = PostTxSettlementBehaviour._sweep_pending_undelivered(self_)
    assert events == []
    assert swept == []


def test_sweep_returns_empty_when_pending_missing() -> None:
    """No ``PENDING_TASKS`` in shared_state → no-op, no crash. The mech boots before the handler stamps the key on some agent configs."""
    self_ = _make_self(pending=None)
    events, swept = PostTxSettlementBehaviour._sweep_pending_undelivered(self_)
    assert events == []
    assert swept == []


def test_sweep_returns_empty_when_pending_empty() -> None:
    """An empty ``PENDING_TASKS`` list produces no events and no swept ids."""
    self_ = _make_self(pending=[])
    events, swept = PostTxSettlementBehaviour._sweep_pending_undelivered(self_)
    assert events == []
    assert swept == []


# Age-driven selection -----------------------------------------------------


def test_sweep_skips_recent_tasks() -> None:
    """Tasks within the max-age window stay in the queue and emit nothing."""
    self_ = _make_self(
        pending=[
            _make_pending_task("r-fresh-a", age_seconds=5),
            _make_pending_task("r-fresh-b", age_seconds=30),
        ],
        max_age=60.0,
    )
    events, swept = PostTxSettlementBehaviour._sweep_pending_undelivered(self_)
    assert events == []
    assert swept == []
    assert [t["requestId"] for t in self_.context.shared_state[PENDING_TASKS]] == [
        "r-fresh-a",
        "r-fresh-b",
    ]


def test_sweep_emits_for_stale_task_and_leaves_queue_untouched() -> None:
    """A stale task becomes a request-only event and its ``request_id`` is returned.

    The sweep itself does NOT mutate the queue — the caller drops the
    swept tasks only after a confirmed predict-api POST 2xx (via
    ``_drop_swept_from_pending``). The DB's ON CONFLICT (predict-api
    side) handles concurrent step-in, so the mech doesn't need to
    consult the contract to decide.
    """
    self_ = _make_self(
        pending=[_make_pending_task("r-stale", age_seconds=120)],
        max_age=60.0,
    )
    events, swept = PostTxSettlementBehaviour._sweep_pending_undelivered(self_)
    assert len(events) == 1
    event = events[0]
    assert event["request"]["request_id"] == "r-stale"
    assert event["response"] is None
    assert event["source"] == "mech_onchain"
    assert swept == ["r-stale"]
    # Sweep leaves the queue as-is until _drop_swept_from_pending fires.
    assert [t["requestId"] for t in self_.context.shared_state[PENDING_TASKS]] == [
        "r-stale"
    ]


def test_sweep_mixed_queue_returns_stale_and_leaves_all_pending() -> None:
    """Mixed queue: only stale entries become events, and ``swept_request_ids`` mirrors that.

    The sweep does not drop anything from the queue — that happens
    post-write in ``_drop_swept_from_pending``.
    """
    pending = [
        _make_pending_task("r-stale-1", age_seconds=120),
        _make_pending_task("r-fresh-1", age_seconds=10),
        _make_pending_task("r-stale-2", age_seconds=200),
        _make_pending_task("r-fresh-2", age_seconds=30),
    ]
    pending_ref = pending  # capture identity to confirm no rebinding
    self_ = _make_self(pending=pending, max_age=60.0)
    events, swept = PostTxSettlementBehaviour._sweep_pending_undelivered(self_)
    emitted = {e["request"]["request_id"] for e in events}
    assert emitted == {"r-stale-1", "r-stale-2"}
    assert set(swept) == {"r-stale-1", "r-stale-2"}
    # The queue is still all four entries — deferred mutation.
    assert [t["requestId"] for t in self_.context.shared_state[PENDING_TASKS]] == [
        "r-stale-1",
        "r-fresh-1",
        "r-stale-2",
        "r-fresh-2",
    ]
    # Same list object (no rebinding), so other consumers see the queue.
    assert self_.context.shared_state[PENDING_TASKS] is pending_ref


def test_sweep_leaves_tasks_without_enqueued_stamp() -> None:
    """A pending task without ``enqueued_at_local`` is left in the queue rather than swept blindly.

    Pre-sweep-deployment entries won't carry the stamp. The next enqueue
    path stamps fresh; the gap is bounded by mech restart cadence.
    """
    self_ = _make_self(
        pending=[
            _make_pending_task("r-pre-sweep", include_enqueued_at=False),
            _make_pending_task("r-stale", age_seconds=999),
        ],
        max_age=60.0,
    )
    events, swept = PostTxSettlementBehaviour._sweep_pending_undelivered(self_)
    assert [e["request"]["request_id"] for e in events] == ["r-stale"]
    assert swept == ["r-stale"]
    # Queue still holds both entries until the post-write drop.
    assert [t["requestId"] for t in self_.context.shared_state[PENDING_TASKS]] == [
        "r-pre-sweep",
        "r-stale",
    ]


def test_sweep_leaves_non_dict_entries_alone() -> None:
    """Whatever bug puts non-dict entries in PENDING_TASKS is not the sweep's problem; leave them as-is rather than dropping silently."""
    self_ = _make_self(
        pending=[
            "garbage",  # malformed entry
            _make_pending_task("r-stale", age_seconds=120),
        ],
        max_age=60.0,
    )
    events, swept = PostTxSettlementBehaviour._sweep_pending_undelivered(self_)
    assert [e["request"]["request_id"] for e in events] == ["r-stale"]
    assert swept == ["r-stale"]
    # The garbage entry survives, and so does the stale task pre-drop.
    assert "garbage" in self_.context.shared_state[PENDING_TASKS]


# Deferred queue mutation --------------------------------------------------


def test_drop_swept_from_pending_removes_only_swept_ids() -> None:
    """``_drop_swept_from_pending`` mutates the queue only for the ids in ``swept_request_ids``.

    Everything else in the queue is left in place, in the same list
    object (in-place mutation, not a rebind).
    """
    pending = [
        _make_pending_task("r-a"),
        _make_pending_task("r-b"),
        _make_pending_task("r-c"),
    ]
    self_ = _make_self(pending=pending)
    PostTxSettlementBehaviour._drop_swept_from_pending(self_, ["r-a", "r-c"])
    assert [t["requestId"] for t in self_.context.shared_state[PENDING_TASKS]] == [
        "r-b"
    ]
    # In-place mutation, not a rebind.
    assert self_.context.shared_state[PENDING_TASKS] is pending


def test_drop_swept_from_pending_noop_on_empty_swept() -> None:
    """An empty ``swept_request_ids`` (no request-only events staged this round) leaves the queue untouched."""
    pending = [_make_pending_task("r-only")]
    self_ = _make_self(pending=pending)
    PostTxSettlementBehaviour._drop_swept_from_pending(self_, [])
    assert [t["requestId"] for t in self_.context.shared_state[PENDING_TASKS]] == [
        "r-only"
    ]


def test_drop_swept_from_pending_noop_when_pending_missing() -> None:
    """``_drop_swept_from_pending`` is safe to call when ``shared_state[PENDING_TASKS]`` was never populated."""
    self_ = _make_self(pending=None)
    # Should not raise, should not add the key.
    PostTxSettlementBehaviour._drop_swept_from_pending(self_, ["r-x"])
    assert PENDING_TASKS not in self_.context.shared_state


# Request-only event shape --------------------------------------------------


def test_request_only_event_carries_expected_fields() -> None:
    """Event carries ``request.delivery_mech=None``, ``source='mech_onchain'``, ``response=None``.

    Mirrors the shape the predict-api server's MechEvent model expects.
    Mirrors the server-side validator in ``server/src/models/mech.py``:
    a request-only event with ``source='mech_offchain'`` would be
    rejected (an off-chain HTTP path doesn't produce undelivered events
    by construction).

    """
    self_ = _make_self()
    task = _make_pending_task(
        "r-shape", age_seconds=60, priority_mech="0xPRIORITY", requester="0xREQ"
    )
    event = PostTxSettlementBehaviour._build_request_only_event(
        self_, task, time.time()
    )
    assert event is not None
    assert event["source"] == "mech_onchain"
    assert event["response"] is None
    req = event["request"]
    assert req["request_id"] == "r-shape"
    assert req["priority_mech"] == "0xPRIORITY"
    assert req["requester"] == "0xREQ"
    assert req["delivery_mech"] is None
    # Chain bits flow from params.
    assert req["chain_id"] == 100
    assert req["marketplace_address"] == "0xMARKET"
    # requested_at is ISO 8601 with Z (matches the off-chain build pattern).
    assert req["requested_at"].endswith("Z")
    # delivery_rate stays a string (matches off-chain encoder).
    assert req["delivery_rate"] == str(10**16)
    # Placeholder prompt/tool — predict-api requires them non-empty; the lake's
    # undelivered signal is the absent mech_responses row, not these fields.
    assert req["prompt"] == "[onchain undelivered]"
    assert req["tool"] == "unknown"


def test_request_only_event_handles_bytes_cid() -> None:
    """``data`` on the pending task can be raw bytes from the contract event; surface it as a 0x-hex content_cid for the lake."""
    self_ = _make_self()
    task = _make_pending_task("r-bytes", data=b"\xab" * 32)
    event = PostTxSettlementBehaviour._build_request_only_event(
        self_, task, time.time()
    )
    assert event is not None
    assert event["request"]["content_cid"] == "0x" + "ab" * 32


def test_request_only_event_coerces_bytes_request_id_to_decimal_string() -> None:
    r"""Undelivered marketplace tasks hold ``requestId`` as raw bytes.

    ``get_marketplace_undelivered_reqs`` populates ``requestId`` with the
    contract-event byte payload; the int conversion only happens later,
    once the task actually runs in task_execution. If the sweep did a
    plain ``str(bytes)`` it would emit ``"b'\x12\x34...'"`` — completely
    different from the decimal string the delivered event carries via
    the same request_id, so predict-api's ON CONFLICT reconciliation
    would never fire and a request that later gets delivered would stay
    marked undelivered forever.

    Regression guard: assert the built event's ``request.request_id`` is
    the decimal string form of the bytes, matching the delivered shape.
    """
    self_ = _make_self()
    # 42 as big-endian bytes → same integer the delivered path would emit.
    task = _make_pending_task("placeholder")
    task["requestId"] = (42).to_bytes(4, "big")
    event = PostTxSettlementBehaviour._build_request_only_event(
        self_, task, time.time()
    )
    assert event is not None
    assert event["request"]["request_id"] == "42"


def test_request_only_event_skipped_for_missing_request_id() -> None:
    """Malformed pending entries (missing requestId / priorityMech /

    requester) return ``None`` so the sweep keeps the task on the queue
    rather than emitting a predict-api 422-bait row.
    """
    self_ = _make_self()
    bad = _make_pending_task("r-missing")
    del bad["requestId"]
    assert (
        PostTxSettlementBehaviour._build_request_only_event(self_, bad, time.time())
        is None
    )


def test_request_only_event_skipped_for_missing_priority_mech() -> None:
    """A pending entry with an empty ``priorityMech`` returns ``None`` from the event builder.

    Same defensive skip as the missing-request_id case.
    """
    self_ = _make_self()
    bad = _make_pending_task("r-no-pri")
    bad["priorityMech"] = ""
    assert (
        PostTxSettlementBehaviour._build_request_only_event(self_, bad, time.time())
        is None
    )


def test_request_only_event_falls_back_to_now_on_bad_timestamp() -> None:
    """A non-numeric ``enqueued_at_local`` falls back to ``now`` rather than crashing the sweep.

    Defends against a future code path that stores a string there.
    """
    self_ = _make_self()
    bad = _make_pending_task("r-bad-ts")
    bad["enqueued_at_local"] = "not-a-number"
    event = PostTxSettlementBehaviour._build_request_only_event(self_, bad, time.time())
    assert event is not None
    assert event["request"]["requested_at"].endswith("Z")


# Extract -----------------------------------------------------------------


def test_extract_offchain_events_reads_from_shared_state_by_request_id() -> None:
    """Events are looked up from agent-local ``shared_state[PREDICT_API_EVENTS]`` by request_id."""
    now = time.time()
    self_ = cast(
        PostTxSettlementBehaviour,
        SimpleNamespace(
            synchronized_data=SimpleNamespace(
                done_tasks=[
                    {"request_id": "r-off"},
                    {"request_id": "r-on"},
                    {"request_id": "r-not-local"},
                    {"request_id": None},
                ]
            ),
            context=SimpleNamespace(
                shared_state={
                    PREDICT_API_EVENTS: {
                        "r-off": {"event": {"src": "off"}, "written_at": now},
                        "r-on": {"event": {"src": "on"}, "written_at": now},
                    }
                }
            ),
        ),
    )
    events = PostTxSettlementBehaviour._extract_offchain_events(self_)
    assert events == [{"src": "off"}, {"src": "on"}]


def test_extract_offchain_events_returns_empty_when_shared_state_unset() -> None:
    """No local events → empty batch (other agents post their own)."""
    self_ = cast(
        PostTxSettlementBehaviour,
        SimpleNamespace(
            synchronized_data=SimpleNamespace(
                done_tasks=[{"request_id": "r-off"}, {"request_id": "r-on"}]
            ),
            context=SimpleNamespace(shared_state={}),
        ),
    )
    events = PostTxSettlementBehaviour._extract_offchain_events(self_)
    assert events == []


# ---------------------------------------------------------------------------
# _do_predict_api_write_best_effort egress gate: use_offchain drives whether
# the external HTTP POST is even attempted, plus a divergence-warning
# regression when the two flag copies drift out of sync.
# ---------------------------------------------------------------------------


def _make_egress_self(
    *,
    use_offchain: bool,
    predict_api_events_url: str = "https://mpp.autonolas.tech/mech/events",
    done_tasks: List[Dict[str, Any]] | None = None,
    warnings_sink: List[str] | None = None,
    debug_sink: List[str] | None = None,
    predict_api_events: Dict[str, Any] | None = None,
) -> PostTxSettlementBehaviour:
    """Build a minimum ``self`` for a ``_do_predict_api_write_best_effort`` call.

    The behaviour's egress path only touches ``self.params``,
    ``self.synchronized_data.done_tasks``, ``self.context.logger``, and
    the two internal helpers ``_extract_offchain_events`` /
    ``_sweep_pending_undelivered`` / ``_build_http_request_message``.
    We stub the latter three so this test proves the *gate* rather than
    the batch construction (already covered by
    ``_extract_offchain_events`` and ``_build_request_only_event`` tests
    above).

    :param use_offchain: value of ``params.use_offchain``.
    :param predict_api_events_url: value of ``params.predict_api_events_url``.
    :param done_tasks: ``synchronized_data.done_tasks`` list; used only
        by the divergence-warning check.
    :param warnings_sink: list to capture WARNING-level log lines.
    :param debug_sink: list to capture DEBUG-level log lines.
    :param predict_api_events: seed ``shared_state[PREDICT_API_EVENTS]``
        for the divergence-warning check.
    :return: a fixture typed as :class:`PostTxSettlementBehaviour` for
        the unbound-method call pattern used by every test in this
        module.
    """
    warnings_sink = warnings_sink if warnings_sink is not None else []
    debug_sink = debug_sink if debug_sink is not None else []
    build_calls: List[Any] = []

    logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda msg, *a, **k: warnings_sink.append(msg),
        error=lambda *a, **k: None,
        debug=lambda msg, *a, **k: debug_sink.append(msg),
    )
    shared_state: Dict[str, Any] = {}
    if predict_api_events is not None:
        shared_state[PREDICT_API_EVENTS] = predict_api_events
    self_ = SimpleNamespace(
        context=SimpleNamespace(shared_state=shared_state, logger=logger),
        params=SimpleNamespace(
            use_offchain=use_offchain,
            predict_api_events_url=predict_api_events_url,
            mech_events_sweep_pending_enabled=False,
            mech_events_sweep_max_age_seconds=60.0,
            mech_events_chain_id=100,
            mech_marketplace_address="0xMARKET",
            agent_mech_contract_address="0xMECHCONTRACT",
        ),
        synchronized_data=SimpleNamespace(done_tasks=done_tasks or []),
        # If the gate lets us through, ``_do_predict_api_write_best_effort``
        # calls ``_build_http_request_message`` to construct the POST.
        # Track the calls so the "gate off => no POST" invariant is
        # observable without wiring the full HTTP dialogue.
        _build_http_request_message=lambda *a, **k: build_calls.append((a, k)),
        # Called by the egress helper when there are no offchain events
        # in done_tasks; return an empty batch so the "no events" short-
        # circuit fires cleanly.
        _extract_offchain_events=lambda: [],
        _sweep_pending_undelivered=lambda: ([], []),
    )
    # Expose the tracker on the fixture so tests can assert against it.
    self_._egress_build_calls = build_calls  # type: ignore[attr-defined]
    return cast(PostTxSettlementBehaviour, self_)


def _drive_generator_once(gen: Any) -> None:
    """Advance a short-circuited generator to completion.

    ``_do_predict_api_write_best_effort`` is a Python generator. Every
    egress-gate path under test returns before the first ``yield``, so
    ``next(gen)`` either raises ``StopIteration`` cleanly (the ``return``
    completed) or would suspend on a real dialogue — which is exactly
    the surface these gate tests must not reach.

    :param gen: the generator returned by
        ``_do_predict_api_write_best_effort``.
    """
    try:
        next(gen)
    except StopIteration:
        pass


def test_egress_gate_off_short_circuits_without_http_call() -> None:
    """``use_offchain=False`` and no divergence: no HTTP build, no warning.

    The Phase-1 dark ship promise on the egress side. Sibling to the
    ingress-side test in
    ``task_execution/tests/test_behaviours.py::test_finalize_done_task_omits_predict_api_event_when_use_offchain_off``.
    """
    self_ = _make_egress_self(use_offchain=False)
    _drive_generator_once(
        PostTxSettlementBehaviour._do_predict_api_write_best_effort(self_)
    )
    assert self_._egress_build_calls == []  # type: ignore[attr-defined]


def test_egress_gate_off_with_diverged_ingress_emits_divergence_warning() -> None:
    """``use_offchain=False`` but ``done_tasks`` carries predict_api_events: WARN and drop.

    The two ``use_offchain`` copies (on ``task_execution`` and
    ``task_submission_abci``) are independent env-overridable params —
    an operator can flip one but forget the other. If the ingress side
    is on but this side is off, ``done_tasks`` will already carry
    ``predict_api_event`` entries at consensus-replication cost, and
    dropping them silently (as the pre-fix shape did) is invisible to
    alerting. The safety net here fires a WARNING so the misconfig
    surfaces in ops logs.
    """
    warnings_sink: List[str] = []
    self_ = _make_egress_self(
        use_offchain=False,
        predict_api_events={"r1": {"src": "off"}},
        warnings_sink=warnings_sink,
    )
    _drive_generator_once(
        PostTxSettlementBehaviour._do_predict_api_write_best_effort(self_)
    )
    assert self_._egress_build_calls == []  # type: ignore[attr-defined]
    assert any(
        "diverging" in w and "task_execution" in w and "task_submission_abci" in w
        for w in warnings_sink
    ), warnings_sink


def test_egress_gate_off_with_no_events_stays_silent() -> None:
    """``use_offchain=False`` and empty ``done_tasks``: no warning fires.

    Guards against the divergence warning becoming its own boot-noise
    source on production mechs that ship dark by default (both flags
    off, no ingress work happened, nothing to warn about).
    """
    warnings_sink: List[str] = []
    self_ = _make_egress_self(
        use_offchain=False, done_tasks=[], warnings_sink=warnings_sink
    )
    _drive_generator_once(
        PostTxSettlementBehaviour._do_predict_api_write_best_effort(self_)
    )
    assert self_._egress_build_calls == []  # type: ignore[attr-defined]
    assert warnings_sink == []


def _make_egress_self_for_enricher(
    *,
    delivered_events: List[Dict[str, Any]],
    final_tx_hash: Any,
    warnings_sink: List[str] | None = None,
    info_sink: List[str] | None = None,
    shared_state: Dict[str, Any] | None = None,
    raise_on_read: bool = False,
) -> Any:
    """Build a self with a non-empty delivered batch + a stubbed POST helper.

    The `_do_predict_api_write_best_effort` flow reaches the enricher
    only when `_extract_offchain_events` returns a non-empty list AND
    every earlier gate passes. Stub the read side to hand back the
    caller-supplied list, and stub `_post_predict_api_batch` to a
    no-op generator so the test can inspect the (mutated) events
    without needing a real HTTP dialogue.

    :param delivered_events: the list ``_extract_offchain_events`` is
        stubbed to return; the enricher mutates entries in place.
    :param final_tx_hash: value the fake ``synchronized_data.final_tx_hash``
        returns. Ignored when ``raise_on_read`` is True.
    :param warnings_sink: mutable list captured by the logger stub;
        defaults to a fresh list.
    :param info_sink: mutable list captured by the logger stub;
        defaults to a fresh list.
    :param shared_state: mapping the fake ``context.shared_state`` holds;
        defaults to a fresh dict.
    :param raise_on_read: when True the fake ``final_tx_hash`` property
        raises ``ValueError`` — mirrors the cold-entry FSM path.
    :return: a ``SimpleNamespace`` shaped like the pieces of
        ``self`` the enricher touches, with ``_post_calls`` attached
        as a captured list of POST invocations.
    """
    warnings_sink = warnings_sink if warnings_sink is not None else []
    info_sink = info_sink if info_sink is not None else []
    shared_state = shared_state if shared_state is not None else {}

    logger = SimpleNamespace(
        info=lambda msg, *a, **k: info_sink.append(msg % a if a else str(msg)),
        warning=lambda msg, *a, **k: warnings_sink.append(msg % a if a else str(msg)),
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )

    class _SyncData:
        @property
        def final_tx_hash(self) -> str:  # type: ignore[return]
            if raise_on_read:
                raise ValueError("final_tx_hash not in sync db")
            return final_tx_hash

        done_tasks: List[Dict[str, Any]] = []

    post_calls: List[Dict[str, Any]] = []

    def _fake_post(**kwargs: Any) -> Any:
        post_calls.append(kwargs)
        # ``yield from`` on a generator that returns without yielding
        # completes cleanly. Match that shape.
        if False:
            yield None

    self_ = SimpleNamespace(
        context=SimpleNamespace(shared_state=shared_state, logger=logger),
        params=SimpleNamespace(
            use_offchain=True,
            predict_api_events_url="https://mpp.autonolas.tech/mech/events",
            mech_events_sweep_pending_enabled=False,
            mech_events_sweep_max_age_seconds=60.0,
            mech_events_chain_id=100,
            mech_marketplace_address="0xMARKET",
            agent_mech_contract_address="0xMECHCONTRACT",
        ),
        synchronized_data=_SyncData(),
        _extract_offchain_events=lambda: delivered_events,
        _sweep_pending_undelivered=lambda: ([], []),
        _post_predict_api_batch=_fake_post,
    )
    self_._post_calls = post_calls  # type: ignore[attr-defined]
    return self_


def test_enricher_stamps_delivery_tx_hash_on_every_delivered_event() -> None:
    """Every event in the delivered batch gets ``final_tx_hash`` stamped.

    Regression against three mutations that would otherwise slip:
    (a) delete the enrichment loop entirely (every event ships with
    None); (b) flip ``if final_tx_hash is not None`` to ``is None``
    (nothing gets stamped); (c) truncate the iteration
    (``delivered_events[:1]``) — this asserts every event, so N=3
    catches a slice-off-by-one.
    """
    hash_value = "0x" + "ab" * 32
    delivered_events: List[Dict[str, Any]] = [
        {"request": {"request_id": f"req-{i}"}, "response": {"delivery_tx_hash": None}}
        for i in range(3)
    ]
    self_ = _make_egress_self_for_enricher(
        delivered_events=delivered_events,
        final_tx_hash=hash_value,
    )
    _drive_generator_once(
        PostTxSettlementBehaviour._do_predict_api_write_best_effort(self_)
    )
    # Every event's response was stamped, and the POST was made with
    # the same batch we mutated (arg-alias — behaviour passes the
    # same list).
    assert all(
        e["response"]["delivery_tx_hash"] == hash_value for e in delivered_events
    )
    assert len(self_._post_calls) == 1  # type: ignore[attr-defined]
    assert self_._post_calls[0]["events"] is delivered_events  # type: ignore[attr-defined]


def test_enricher_skips_when_final_tx_hash_raises_and_ships_batch_anyway() -> None:
    """Cold entry: ``final_tx_hash`` raises → skip stamping, still ship.

    ``SynchronizedData.final_tx_hash`` raises ``ValueError`` when the
    underlying key is absent. Without the try/except added in this
    PR, the raise escapes ``_do_predict_api_write_best_effort`` (whose
    docstring promises to never raise) and wedges the FSM. Assert we
    log the skip AND still ship the batch (with delivery_tx_hash left
    at whatever placeholder the ingress set — typically None).
    """
    delivered_events: List[Dict[str, Any]] = [
        {"request": {"request_id": "req-cold"}, "response": {"delivery_tx_hash": None}}
    ]
    info_sink: List[str] = []
    self_ = _make_egress_self_for_enricher(
        delivered_events=delivered_events,
        final_tx_hash=None,
        raise_on_read=True,
        info_sink=info_sink,
    )
    _drive_generator_once(
        PostTxSettlementBehaviour._do_predict_api_write_best_effort(self_)
    )
    assert delivered_events[0]["response"]["delivery_tx_hash"] is None
    assert any("no settlement tx hash resolved" in msg for msg in info_sink), info_sink
    assert len(self_._post_calls) == 1  # type: ignore[attr-defined]


def test_enricher_warns_and_skips_when_final_tx_hash_is_empty_string() -> None:
    """Empty-string ``final_tx_hash`` surfaces as a warning, not silently.

    A future FSM regression that lands ``final_tx_hash=""`` would take
    the truthiness-false branch and skip stamping, but the guard on
    ``elif not final_tx_hash:`` is what makes that legible as an
    "investigate upstream" warning rather than being indistinguishable
    from the routine ``None`` "no settlement resolved" path. Without
    this branch, empty-string would either:

    * ride through as-is (invalid tx hash on the wire), or
    * hit the else branch and stamp ``response["delivery_tx_hash"] = ""``
      which the predict-api server would then have to reject with
      ``BATCH_HASH_MISMATCH`` — recovering only via backfill.
    """
    delivered_events: List[Dict[str, Any]] = [
        {"request": {"request_id": "req-empty"}, "response": {"delivery_tx_hash": None}}
    ]
    warnings_sink: List[str] = []
    self_ = _make_egress_self_for_enricher(
        delivered_events=delivered_events,
        final_tx_hash="",
        warnings_sink=warnings_sink,
    )
    _drive_generator_once(
        PostTxSettlementBehaviour._do_predict_api_write_best_effort(self_)
    )
    assert delivered_events[0]["response"]["delivery_tx_hash"] is None
    assert any(
        "final_tx_hash resolved to" in msg and "falsy" in msg for msg in warnings_sink
    ), warnings_sink
    # Batch still ships — best-effort contract: the settlement row
    # lands NULL and backfill fills it in.
    assert len(self_._post_calls) == 1  # type: ignore[attr-defined]


def test_enricher_warns_when_response_is_not_a_dict_and_ships_batch() -> None:
    """Non-dict ``response`` on an event: warn, skip that stamp, ship the batch.

    ``_build_predict_api_event`` always emits a dict for ``response``,
    but the batch composes from ``synchronized_data.done_tasks`` which
    was serialised across an FSM boundary. A future upstream mutation
    that hands us a scalar / list / None here would silently be
    dropped by the ``isinstance(response, dict)`` false branch; the
    warning makes the drop visible so ops can chase the shape bug.

    Regression: without the ``else`` branch a shape bug degrades to
    "one row missing delivery_tx_hash forever, no signal in logs".
    """
    hash_value = "0x" + "cd" * 32
    delivered_events: List[Dict[str, Any]] = [
        {"request": {"request_id": "req-dict"}, "response": {"delivery_tx_hash": None}},
        {"request": {"request_id": "req-scalar"}, "response": "not-a-dict"},
    ]
    warnings_sink: List[str] = []
    self_ = _make_egress_self_for_enricher(
        delivered_events=delivered_events,
        final_tx_hash=hash_value,
        warnings_sink=warnings_sink,
    )
    _drive_generator_once(
        PostTxSettlementBehaviour._do_predict_api_write_best_effort(self_)
    )
    # Dict response landed a hash; scalar response was skipped intact.
    assert delivered_events[0]["response"]["delivery_tx_hash"] == hash_value
    assert delivered_events[1]["response"] == "not-a-dict"
    # Warning names the type so ops can chase upstream.
    assert any(
        "response is" in msg and "str" in msg for msg in warnings_sink
    ), warnings_sink
    # Batch still ships — one bad shape doesn't cost us the real deliveries.
    assert len(self_._post_calls) == 1  # type: ignore[attr-defined]


def test_egress_gate_on_with_empty_url_short_circuits_at_debug_level() -> None:
    """``use_offchain=True`` but empty ``predict_api_events_url``: DEBUG log, no HTTP call.

    Same shape as the flag-off short-circuit but on the URL-missing
    branch: an operator opted in to the pipeline but hasn't wired the
    endpoint yet, so we degrade quietly (DEBUG, not WARNING) rather
    than every settlement round shouting into ops logs.
    """
    debug_sink: List[str] = []
    self_ = _make_egress_self(
        use_offchain=True,
        predict_api_events_url="",
        debug_sink=debug_sink,
    )
    _drive_generator_once(
        PostTxSettlementBehaviour._do_predict_api_write_best_effort(self_)
    )
    assert self_._egress_build_calls == []  # type: ignore[attr-defined]
    assert any("predict_api_events_url empty" in msg for msg in debug_sink), debug_sink
