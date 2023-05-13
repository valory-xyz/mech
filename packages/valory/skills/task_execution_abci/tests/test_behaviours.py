# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 Valory AG
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

"""This package contains round behaviours of ProposalCollector."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Type
from unittest import mock

import pytest

from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.abstract_round_abci.behaviour_utils import \
    BaseBehaviour
from packages.valory.skills.abstract_round_abci.behaviours import \
    make_degenerate_behaviour
from packages.valory.skills.abstract_round_abci.io_.store import \
    SupportedFiletype
from packages.valory.skills.abstract_round_abci.test_tools.base import \
    FSMBehaviourBaseCase
from packages.valory.skills.task_execution_abci.behaviours import (
    TaskExecutionAbciBehaviour, TaskExecutionBaseBehaviour)
from packages.valory.skills.task_execution_abci.rounds import (
    Event, FinishedTaskExecutionRound, SynchronizedData)


def wrap_dummy_get_from_ipfs(return_value: Optional[SupportedFiletype]) -> Callable:
    """Wrap dummy_get_from_ipfs."""

    def dummy_get_from_ipfs(
        *args: Any, **kwargs: Any
    ) -> Generator[None, None, Optional[SupportedFiletype]]:
        """A mock get_from_ipfs."""
        return return_value
        yield

    return dummy_get_from_ipfs


@dataclass
class BehaviourTestCase:
    """BehaviourTestCase"""

    name: str
    initial_data: Dict[str, Any]
    event: Event
    next_behaviour_class: Optional[Type[TaskExecutionBaseBehaviour]] = None


class BaseProposalCollectorTest(FSMBehaviourBaseCase):
    """Base test case."""

    path_to_skill = Path(__file__).parent.parent

    behaviour: TaskExecutionBaseBehaviour  # type: ignore
    behaviour_class: Type[TaskExecutionBaseBehaviour]
    next_behaviour_class: Type[TaskExecutionBaseBehaviour]
    synchronized_data: SynchronizedData
    done_event = Event.DONE

    def setup_class(self, **kwargs: Any) -> None:
        """setup_class"""
        super().setup_class(**kwargs)
        self._skill.skill_context.shared_state["pending_tasks"] = ["bafybeieocezdbktaahaktnmjgttwqlbkvncs274pphvlnpblmrrno2hqnq"]

    def fast_forward(self, data: Optional[Dict[str, Any]] = None) -> None:
        """Fast-forward on initialization"""

        data = data if data is not None else {}
        self.fast_forward_to_behaviour(
            self.behaviour,  # type: ignore
            self.behaviour_class.auto_behaviour_id(),
            SynchronizedData(AbciAppDB(setup_data=AbciAppDB.data_to_lists(data))),
        )
        assert (
            self.behaviour.current_behaviour.auto_behaviour_id()  # type: ignore
            == self.behaviour_class.auto_behaviour_id()
        )

    def complete(self, event: Event) -> None:
        """Complete test"""

        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round(done_event=event)
        assert (
            self.behaviour.current_behaviour.auto_behaviour_id()  # type: ignore
            == self.next_behaviour_class.auto_behaviour_id()
        )


class TestTaskExecutionBehaviour(BaseProposalCollectorTest):
    """Tests TestTaskExecutionBehaviour"""

    behaviour_class = TaskExecutionAbciBehaviour
    next_behaviour_class = make_degenerate_behaviour(FinishedTaskExecutionRound)

    @pytest.mark.parametrize(
        "test_case, kwargs",
        [
            (
                BehaviourTestCase(
                    "Happy path",
                    initial_data=dict(),
                    event=Event.DONE,
                ),
                {},
            ),
        ],
    )
    def test_run(self, test_case: BehaviourTestCase, kwargs: Any) -> None:
        """Run tests."""
        self.fast_forward(test_case.initial_data)

        dummy_object = {
            "prompt": "Write a poem about ETHGlobal Lisbon.",
            "tool": "openai-gpt4"
        }

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_get_from_ipfs(dummy_object),
        ):

            self.behaviour.act_wrapper()
            self.complete(test_case.event)
