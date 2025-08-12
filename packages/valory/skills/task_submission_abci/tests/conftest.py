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
from typing import Any, Callable, Generator, Optional, Type

import pytest

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.task_submission_abci.behaviours import (
    FundsSplittingBehaviour,
)


class DummyFundsSplit(FundsSplittingBehaviour):
    """Concrete subclass to allow instantiation for unit testing only."""

    matching_round: Type[AbstractRound] = AbstractRound

    def async_act(self) -> Generator:
        """No-op async act to satisfy the abstract method requirement."""
        if False:
            yield


@pytest.fixture
def logger_stub() -> SimpleNamespace:
    """
    No-op logger object.

    :returns: SimpleNamespace with info/warning/error callables.
    :rtype: SimpleNamespace
    """
    return SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )


@pytest.fixture
def params_profit5() -> SimpleNamespace:
    """
    Params stub with profit_split_freq = 5.

    :returns: SimpleNamespace exposing profit_split_freq.
    :rtype: SimpleNamespace
    """
    return SimpleNamespace(profit_split_freq=5)


@pytest.fixture
def behaviour_context(
    logger_stub: SimpleNamespace, params_profit5: SimpleNamespace
) -> SimpleNamespace:
    """
    Minimal skill_context for FundsSplittingBehaviour.

    :param logger_stub: No-op logger.
    :type logger_stub: SimpleNamespace
    :param params_profit5: Params with profit_split_freq set to 5.
    :type params_profit5: SimpleNamespace
    :returns: Context exposing logger and params.
    :rtype: SimpleNamespace
    """
    return SimpleNamespace(logger=logger_stub, params=params_profit5)


@pytest.fixture
def fs_behaviour(behaviour_context: SimpleNamespace) -> DummyFundsSplit:
    """
    Instantiated FundsSplittingBehaviour under test.

    :param behaviour_context: Minimal skill_context.
    :type behaviour_context: SimpleNamespace
    :returns: Concrete behaviour instance.
    :rtype: DummyFundsSplit
    """
    return DummyFundsSplit(name="fs", skill_context=behaviour_context)


@pytest.fixture
def run_to_completion() -> Callable[[Generator[Any, None, Any]], Any]:
    """
    Exhaust a behaviour generator and return its final value.

    :returns: Callable that runs a generator to completion and returns StopIteration.value.
    :rtype: Callable[[Generator[Any, None, Any]], Any]
    """

    def _run(gen: Generator[Any, None, Any]) -> Any:
        """Run the generator until completion and return its result."""
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    return _run


@pytest.fixture
def patch_num_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[int | list[int]], None]:
    """
    Stub `_get_num_requests_delivered` to return a value or sequence.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :returns: Function that applies a stub returning a single int or successive ints.
    :rtype: Callable[[int | list[int]], None]
    """

    def _apply(value_or_seq: int | list[int]) -> None:
        """Apply the stub with either a single value or a sequence across calls."""
        if isinstance(value_or_seq, list):
            it = iter(value_or_seq)

            def _fn(self) -> Generator[None, None, Optional[int]]:
                if False:  # ensure generator type
                    yield
                return next(it)

        else:

            def _fn(self) -> Generator[None, None, Optional[int]]:
                if False:
                    yield
                return int(value_or_seq)

        monkeypatch.setattr(DummyFundsSplit, "_get_num_requests_delivered", _fn)

    return _apply
