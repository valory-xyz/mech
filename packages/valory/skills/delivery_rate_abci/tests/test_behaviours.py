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

"""Tests for delivery_rate_abci behaviours — C-1 fix verification."""

# pylint: skip-file

import ast
import pathlib
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock

import pytest

ZERO_ETHER_VALUE = 0


def _run_generator(gen: Generator, values: Optional[List[Any]] = None) -> Any:
    """Drive a generator to completion, sending values from the list."""
    values = values or []
    idx = 0
    try:
        next(gen)
        while True:
            send_val = values[idx] if idx < len(values) else None
            idx += 1
            gen.send(send_val)
    except StopIteration as e:
        return e.value


class _FakeBehaviour:
    """Minimal stand-in for UpdateDeliveryRateBehaviour with just enough to test."""

    def __init__(self, mech_to_max_delivery_rate: Dict[str, int]) -> None:
        self.params = MagicMock()
        self.params.mech_to_max_delivery_rate = mech_to_max_delivery_rate
        self.context = MagicMock()
        self._should_update_fn = None  # type: ignore
        self._get_tx_fn = None  # type: ignore

    def _should_update_delivery_rate(
        self, mech_address: str, delivery_rate: int
    ) -> Generator[None, None, Optional[bool]]:
        return self._should_update_fn(mech_address, delivery_rate)

    def _get_delivery_rate_update_tx(
        self, mech_address: str, delivery_rate: int
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        return self._get_tx_fn(mech_address, delivery_rate)

    def get_delivery_rate_update_txs(
        self,
    ) -> Generator[None, None, List[Dict[str, Any]]]:
        """Get the mech update hash tx — mirrors the real implementation."""
        txs: List[Dict[str, Any]] = []
        for (
            mech_address,
            delivery_rate,
        ) in self.params.mech_to_max_delivery_rate.items():
            should_update = yield from self._should_update_delivery_rate(
                mech_address, delivery_rate
            )
            if should_update is None:
                self.context.logger.warning(
                    f"Could not check if delivery_rate should be updated for {mech_address}."
                )
                continue
            if not should_update:
                self.context.logger.info(
                    f"No need to update delivery_rate for {mech_address}."
                )
                continue

            tx = yield from self._get_delivery_rate_update_tx(
                mech_address, delivery_rate
            )
            if tx is None:
                self.context.logger.warning(
                    f"Could not get delivery_rate update tx for {mech_address}."
                )
                continue

            txs.append(tx)

        return txs


def _make_gen(
    value: Any,
) -> Any:
    """Create a generator that yields once and returns value."""

    def gen(*args: Any, **kwargs: Any) -> Generator:
        yield
        return value

    return gen


class TestGetDeliveryRateUpdateTxs:
    """Tests for get_delivery_rate_update_txs — C-1 fix verification."""

    def test_none_tx_not_appended(self) -> None:
        """When _get_delivery_rate_update_tx returns None, it must NOT appear in txs."""
        b = _FakeBehaviour({"0xMech1": 100})
        b._should_update_fn = _make_gen(True)
        b._get_tx_fn = _make_gen(None)

        txs = _run_generator(b.get_delivery_rate_update_txs())

        assert txs == [], "None tx should not be appended to the list"
        b.context.logger.warning.assert_called_once()

    def test_valid_tx_appended(self) -> None:
        """When _get_delivery_rate_update_tx returns a valid dict, it must be in txs."""
        expected_tx = {"to": "0xMech1", "value": 0, "data": b"\x00"}
        b = _FakeBehaviour({"0xMech1": 100})
        b._should_update_fn = _make_gen(True)
        b._get_tx_fn = _make_gen(expected_tx)

        txs = _run_generator(b.get_delivery_rate_update_txs())

        assert len(txs) == 1
        assert txs[0] == expected_tx

    def test_should_update_none_skips(self) -> None:
        """When _should_update_delivery_rate returns None, the mech is skipped."""
        b = _FakeBehaviour({"0xMech1": 100})
        b._should_update_fn = _make_gen(None)

        txs = _run_generator(b.get_delivery_rate_update_txs())

        assert txs == []
        b.context.logger.warning.assert_called_once()

    def test_should_update_false_skips(self) -> None:
        """When _should_update_delivery_rate returns False, no tx is fetched."""
        b = _FakeBehaviour({"0xMech1": 100})
        b._should_update_fn = _make_gen(False)

        txs = _run_generator(b.get_delivery_rate_update_txs())

        assert txs == []
        b.context.logger.info.assert_called()

    def test_mixed_valid_and_none_txs(self) -> None:
        """With multiple mechs, only valid txs are collected; None ones are skipped."""
        valid_tx = {"to": "0xMech1", "value": 0, "data": b"\x01"}
        b = _FakeBehaviour({"0xMech1": 100, "0xMech2": 200})
        b._should_update_fn = _make_gen(True)
        call_count = {"n": 0}

        def tx_gen(*args: Any) -> Generator:
            yield
            call_count["n"] += 1
            if call_count["n"] == 1:
                return valid_tx
            return None

        b._get_tx_fn = tx_gen

        txs = _run_generator(b.get_delivery_rate_update_txs())

        assert len(txs) == 1
        assert txs[0] == valid_tx
        assert None not in txs


class TestSourceCodeContinuePresent:
    """Verify the actual source file has the fix applied."""

    def test_continue_after_none_tx_check(self) -> None:
        """The source must have `continue` after the `tx is None` warning block."""
        src = pathlib.Path(
            "packages/valory/skills/delivery_rate_abci/behaviours.py"
        ).read_text()
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "get_delivery_rate_update_txs":
                continue
            # Find the `if tx is None:` block
            for i, stmt in enumerate(node.body):
                if not isinstance(stmt, ast.For):
                    continue
                for_body = stmt.body
                for j, inner in enumerate(for_body):
                    if not isinstance(inner, ast.If):
                        continue
                    # Check if it's the `tx is None` check
                    test = inner.test
                    if not isinstance(test, ast.Compare):
                        continue
                    if not (isinstance(test.left, ast.Name) and test.left.id == "tx"):
                        continue
                    # Verify the body ends with Continue
                    last_stmt = inner.body[-1]
                    assert isinstance(
                        last_stmt, ast.Continue
                    ), "Expected `continue` after `if tx is None:` warning block"
                    return

        pytest.fail("Could not find `if tx is None:` block in source")
