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

"""Conftest for the task_submission_abci."""

from types import SimpleNamespace
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Type, cast

import pytest

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.task_submission_abci.behaviours import (
    FundsSplittingBehaviour,
)


@pytest.fixture
def run_to_completion() -> Callable[[Generator[Any, None, Any]], Any]:
    """
    Return a helper that exhausts a generator and yields its final value.

    :returns: A function that runs a generator until completion and returns the StopIteration value.
    :rtype: Callable[[Generator[Any, None, Any]], Any]
    """

    def _run(gen: Generator[Any, None, Any]) -> Any:
        try:
            while True:
                next(gen)
        except StopIteration as e:  # pragma: no cover - control path end
            return e.value

    return _run


class DummyFundsSplit(FundsSplittingBehaviour):
    """Concrete subclass to allow instantiation for unit testing only."""

    matching_round: Type[AbstractRound] = cast(
        Type[AbstractRound], type("DummyRound", (), {})
    )

    def async_act(self) -> Generator[None, None, None]:
        """Satisfy abstract method for BaseBehaviour."""
        if False:  # pragma: no cover - keep it a generator
            yield
        return None


@pytest.fixture
def fs_ctx() -> SimpleNamespace:
    """
    Return a minimal skill context with logger and params used by _should_split_profits.

    :returns: A context namespace exposing logger and params (profit_split_balance, agent_mech_contract_addresses).
    :rtype: SimpleNamespace
    """
    return SimpleNamespace(
        logger=SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
        params=SimpleNamespace(
            profit_split_balance=10,
            agent_mech_contract_addresses=["0xA"],
        ),
    )


@pytest.fixture
def fs_behaviour(fs_ctx: SimpleNamespace) -> DummyFundsSplit:
    """
    Create a behaviour instance bound to the minimal context.

    :param fs_ctx: The fake skill context.
    :type fs_ctx: SimpleNamespace
    :returns: A DummyFundsSplit behaviour ready for testing.
    :rtype: DummyFundsSplit
    """
    return DummyFundsSplit(name="fs", skill_context=fs_ctx)


@pytest.fixture
def patch_mech_info(
    monkeypatch: pytest.MonkeyPatch, fs_behaviour: DummyFundsSplit
) -> Callable[[Dict[str, int]], None]:
    """
    Return a helper to stub _get_mech_info to return balances per mech address.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :param fs_behaviour: The behaviour instance under test.
    :type fs_behaviour: DummyFundsSplit
    :returns: A function that accepts a mapping {mech_address: balance} and applies the stub.
    :rtype: Callable[[Dict[str, int]], None]
    """

    def _apply(balances_by_addr: Dict[str, int]) -> None:
        """
        Apply the stub for _get_mech_info, returning tuples (mech_type, balance_tracker, balance).

        :param balances_by_addr: Mapping from mech address to desired balance.
        :type balances_by_addr: Dict[str, int]
        """

        def _fake(
            self: DummyFundsSplit, mech_address: str
        ) -> Generator[None, None, Optional[Tuple[bytes, str, int]]]:
            if False:  # pragma: no cover - make this a generator
                yield
            bal = balances_by_addr.get(mech_address)
            if bal is None:
                return None
            return (b"\x00", "0xBalanceTracker", bal)

        monkeypatch.setattr(DummyFundsSplit, "_get_mech_info", _fake)

    return _apply
