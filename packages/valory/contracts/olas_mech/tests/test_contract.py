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
from typing import Callable

from packages.valory.contracts.olas_mech.contract import OlasMechContract

_SENDER = "0x3333333333333333333333333333333333333333"
_MECH = "0x2222222222222222222222222222222222222222"


class _FakeEth:
    """Minimal ``web3.eth`` stand-in whose ``call`` runs an injected behaviour."""

    def __init__(self, behaviour: Callable[[], object]) -> None:
        """Store the behaviour run on each ``call``."""
        self._behaviour = behaviour

    def call(self, _tx: dict) -> object:
        """Run the injected behaviour as the simulated ``eth_call``."""
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
    """Minimal ``EthereumApi`` stand-in for exercising ``simulate_tx`` directly."""

    def __init__(self, behaviour: Callable[[], object]) -> None:
        """Wire up the fake ``api`` object."""
        self.api = _FakeApi(behaviour)


class TestSimulateTxWallClockBound:
    """`simulate_tx` reports the outcome and never blocks past its deadline."""

    def test_returns_true_when_call_succeeds(self) -> None:
        """A returning ``eth_call`` yields ``simulation_ok=True``."""
        ledger = _FakeLedgerApi(lambda: b"")
        result = OlasMechContract.simulate_tx(ledger, _MECH, _SENDER, "0x")
        assert result == {"data": True}

    def test_returns_false_when_call_reverts(self) -> None:
        """A reverting ``eth_call`` is flattened to ``simulation_ok=False``."""

        def _revert() -> object:
            raise ValueError("execution reverted: GS026")

        ledger = _FakeLedgerApi(_revert)
        result = OlasMechContract.simulate_tx(ledger, _MECH, _SENDER, "0x")
        assert result == {"data": False}

    def test_returns_false_without_waiting_for_a_hanging_call(self) -> None:
        """A call that overruns the deadline returns False promptly.

        This is the property that keeps a single unresponsive simulation from
        consuming the whole preparation round: ``simulate_tx`` must return on
        the deadline, not after the underlying call finally unblocks.
        """
        started = threading.Event()

        def _hang() -> object:
            started.set()
            # Blocks far longer than the deadline; the bound must return first.
            time.sleep(3.0)
            return b""

        ledger = _FakeLedgerApi(_hang)
        begin = time.monotonic()
        result = OlasMechContract.simulate_tx(
            ledger, _MECH, _SENDER, "0x", deadline_seconds=0.2
        )
        elapsed = time.monotonic() - begin

        assert result == {"data": False}
        assert started.wait(1.0)  # the simulated call really started
        assert elapsed < 1.5  # returned on the deadline, not after the 3s block
