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

"""Tests for the olas_mech contract package."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterator, List
from unittest.mock import patch

import pytest
import yaml

from packages.valory.contracts.olas_mech import contract as contract_module
from packages.valory.contracts.olas_mech.contract import (
    MAX_INFLIGHT_SIMULATIONS,
    OlasMechContract,
    SIMULATION_OK_KEY,
    SIMULATION_RPC_DEADLINE_SECONDS,
    SIMULATION_TIMED_OUT_KEY,
    simulate_with_deadline,
)

_SENDER = "0x3333333333333333333333333333333333333333"
_MECH = "0x2222222222222222222222222222222222222222"
_SHORT_DEADLINE = 0.2
# Generous upper bound on how long a bounded call may take to return.
_PROMPT_RETURN_SECONDS = 1.5
_RELEASE_WAIT_SECONDS = 5.0
_LEDGER_CONNECTION_YAML = (
    Path(__file__).parents[3] / "connections" / "ledger" / "connection.yaml"
)

_OK = {SIMULATION_OK_KEY: True, SIMULATION_TIMED_OUT_KEY: False}
_FAILED = {SIMULATION_OK_KEY: False, SIMULATION_TIMED_OUT_KEY: False}
_INCONCLUSIVE = {SIMULATION_OK_KEY: False, SIMULATION_TIMED_OUT_KEY: True}


class _FakeEth:
    """Minimal ``web3.eth`` stand-in whose ``call`` runs an injected behaviour."""

    def __init__(self, behaviour: Callable[[], object]) -> None:
        """Store the behaviour run on each ``call``."""
        self._behaviour = behaviour
        self.calls = 0

    def call(self, _tx: dict) -> object:
        """Run the injected behaviour as the simulated ``eth_call``."""
        self.calls += 1
        return self._behaviour()


class _FakeApi:
    """Minimal ``EthereumApi.api`` stand-in exposing ``eth`` and address helper."""

    def __init__(self, behaviour: Callable[[], object]) -> None:
        """Wire up the fake ``eth`` object."""
        self.eth = _FakeEth(behaviour)

    @staticmethod
    def to_checksum_address(address: str) -> str:
        """Return the address unchanged; checksumming is irrelevant to the bound."""
        return address


class _FakeLedgerApi:
    """Minimal ``EthereumApi`` stand-in for exercising simulations directly."""

    def __init__(self, behaviour: Callable[[], object]) -> None:
        """Wire up the fake ``api`` object."""
        self.api = _FakeApi(behaviour)


class _Hang:
    """``eth_call`` behaviour that blocks until the test releases it."""

    def __init__(self) -> None:
        """Create the release and start signals."""
        self.release = threading.Event()
        self.started = threading.Event()

    def __call__(self) -> object:
        """Block until released, then return an empty result."""
        self.started.set()
        self.release.wait(_RELEASE_WAIT_SECONDS)
        return b""


def _raising(error: Exception) -> Callable[[], object]:
    """Return an ``eth_call`` behaviour that raises ``error``."""

    def _raise() -> object:
        raise error

    return _raise


def _simulate(ledger: _FakeLedgerApi, deadline: float = _SHORT_DEADLINE) -> dict:
    return simulate_with_deadline(ledger, _MECH, _SENDER, "0x", deadline)  # type: ignore[arg-type]


@pytest.fixture
def hang() -> Iterator[_Hang]:
    """Yield a hanging behaviour and free every simulation slot afterwards."""
    behaviour = _Hang()
    yield behaviour
    behaviour.release.set()
    acquired: List[bool] = [
        contract_module._inflight_simulations.acquire(timeout=_RELEASE_WAIT_SECONDS)
        for _ in range(MAX_INFLIGHT_SIMULATIONS)
    ]
    for got in acquired:
        if got:
            contract_module._inflight_simulations.release()
    assert all(acquired), "a simulation thread did not exit after release"


class TestSimulateWithDeadline:
    """``simulate_with_deadline`` separates node answers from inconclusive outcomes."""

    def test_returning_call_is_ok(self) -> None:
        """A returning ``eth_call`` is a conclusive success."""
        assert _simulate(_FakeLedgerApi(lambda: b"")) == _OK

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("execution reverted: GS026"),
            RuntimeError("unexpected node answer"),
            KeyError("result"),
        ],
    )
    def test_node_error_is_a_conclusive_failure(self, error: Exception) -> None:
        """Errors that are not transport failures are charged like a revert."""
        assert _simulate(_FakeLedgerApi(_raising(error))) == _FAILED

    @pytest.mark.parametrize(
        "error",
        [
            TimeoutError("read timed out"),
            ConnectionResetError("connection reset by peer"),
            OSError("network is unreachable"),
        ],
    )
    def test_transport_error_is_inconclusive(self, error: Exception) -> None:
        """A transport failure, including the call's own timeout, is inconclusive."""
        assert _simulate(_FakeLedgerApi(_raising(error))) == _INCONCLUSIVE

    def test_hanging_call_is_inconclusive_on_the_deadline(self, hang: _Hang) -> None:
        """A call that overruns returns on the deadline, not when the call unblocks."""
        begin = time.monotonic()
        result = _simulate(_FakeLedgerApi(hang))
        elapsed = time.monotonic() - begin

        assert result == _INCONCLUSIVE
        assert hang.started.is_set()
        assert elapsed < _PROMPT_RETURN_SECONDS

    def test_full_pool_is_inconclusive_without_calling(self, hang: _Hang) -> None:
        """With every slot held by a hung call, a new call never starts."""
        for _ in range(MAX_INFLIGHT_SIMULATIONS):
            assert _simulate(_FakeLedgerApi(hang)) == _INCONCLUSIVE

        fresh = _FakeLedgerApi(lambda: b"")
        begin = time.monotonic()
        result = _simulate(fresh, deadline=_RELEASE_WAIT_SECONDS)
        elapsed = time.monotonic() - begin

        assert result == _INCONCLUSIVE
        assert fresh.api.eth.calls == 0
        assert elapsed < _PROMPT_RETURN_SECONDS

    def test_thread_start_failure_is_inconclusive_and_frees_the_slot(
        self, hang: _Hang
    ) -> None:
        """A worker that cannot start leaves no slot held."""
        with patch.object(
            contract_module.threading.Thread,
            "start",
            side_effect=RuntimeError("can't start new thread"),
        ):
            for _ in range(MAX_INFLIGHT_SIMULATIONS + 1):
                assert _simulate(_FakeLedgerApi(hang)) == _INCONCLUSIVE
        assert not hang.started.is_set()
        assert _simulate(_FakeLedgerApi(lambda: b"")) == _OK

    def test_slot_is_reusable_after_a_hung_call_ends(self, hang: _Hang) -> None:
        """A slot held by a hung call is returned once that call finishes."""
        for _ in range(MAX_INFLIGHT_SIMULATIONS):
            _simulate(_FakeLedgerApi(hang))
        hang.release.set()

        deadline = time.monotonic() + _RELEASE_WAIT_SECONDS
        result = _INCONCLUSIVE
        while result != _OK and time.monotonic() < deadline:
            result = _simulate(_FakeLedgerApi(lambda: b""))
        assert result == _OK


def test_default_deadline_fits_a_retried_ledger_request() -> None:
    """The default deadline outlasts two request attempts of the ledger connection.

    A shorter deadline fires together with the first attempt's timeout, so a
    request that stalls once is reported inconclusive before its retry runs.
    """
    with open(_LEDGER_CONNECTION_YAML, encoding="utf-8") as f:
        config = yaml.safe_load(f)["config"]
    request_timeouts = [
        api["timeout"] for api in config["ledger_apis"].values() if "timeout" in api
    ]
    assert request_timeouts
    assert SIMULATION_RPC_DEADLINE_SECONDS > 2 * max(request_timeouts)


class TestDeliverDataCarriesTheVerdict:
    """The deliver-data callables return the simulation verdict with the calldata."""

    def test_simulate_tx_keeps_its_data_key(self) -> None:
        """``simulate_tx`` still reports ``simulation_ok`` under ``data``."""
        ledger = _FakeLedgerApi(lambda: b"")
        result = OlasMechContract.simulate_tx(ledger, _MECH, _SENDER, "0x")  # type: ignore[arg-type]
        assert result == {"data": True, SIMULATION_TIMED_OUT_KEY: False}

    def test_marketplace_deliver_data_honours_the_deadline(self, hang: _Hang) -> None:
        """The deadline argument bounds the marketplace deliver simulation."""
        instance = SimpleNamespace(encode_abi=lambda **_kwargs: "0xabcd")
        begin = time.monotonic()
        with patch.object(OlasMechContract, "get_instance", return_value=instance):
            result = OlasMechContract.get_marketplace_deliver_data(
                _FakeLedgerApi(hang),  # type: ignore[arg-type]
                _MECH,
                _SENDER,
                requestIds=[],
                datas=[],
                deadline_seconds=_SHORT_DEADLINE,
            )
        elapsed = time.monotonic() - begin

        assert result == {"data": b"\xab\xcd", **_INCONCLUSIVE}
        assert elapsed < _PROMPT_RETURN_SECONDS
