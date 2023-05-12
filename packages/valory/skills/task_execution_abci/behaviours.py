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

"""This package contains round behaviours of TaskExecutionAbciApp."""

from abc import ABC
from multiprocessing.pool import AsyncResult
from typing import Any, Generator, Optional, Set, Type, cast

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour, BaseBehaviour)
from packages.valory.skills.abstract_round_abci.io_.store import \
    SupportedFiletype
from packages.valory.skills.task_execution_abci.models import Params
from packages.valory.skills.task_execution_abci.rounds import (
    SynchronizedData, TaskExecutionAbciApp, TaskExecutionAbciPayload,
    TaskExecutionRound)
from packages.valory.skills.task_execution_abci.tasks import LLMTask


class TaskExecutionBaseBehaviour(BaseBehaviour, ABC):
    """Base behaviour for the task_execution_abci skill."""

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return cast(SynchronizedData, super().synchronized_data)

    @property
    def params(self) -> Params:
        """Return the params."""
        return cast(Params, super().params)


class TaskExecutionAbciBehaviour(TaskExecutionBaseBehaviour):
    """TaskExecutionAbciBehaviour"""

    matching_round: Type[AbstractRound] = TaskExecutionRound

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Behaviour."""
        super().__init__(**kwargs)
        self._async_result: Optional[AsyncResult] = None
        self._is_task_prepared = False

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():

            # Check whether the task already exists
            if not self._is_task_prepared:
                task_data = self.context.shared_state.get("pending_tasks").pop(0)
                # Verify the data format

                # For now, data is a hash
                file_hash = task_data

                # Get the file from IPFS
                task_data = yield from self.get_from_ipfs(
                    ipfs_hash=file_hash,
                    filetype=SupportedFiletype.JSON,
                )

                self.prepare_task(task_data)

            # Check whether the task is finished
            self._async_result = cast(AsyncResult, self._async_result)
            if not self._async_result.ready():
                self.context.logger.debug("The task is not finished yet.")
                yield from self.sleep(self.params.sleep_time)
                return

            # The task is finished
            completed_task = self._async_result.get()

            sender = self.context.agent_address
            payload = TaskExecutionAbciPayload(sender=sender, content=completed_task)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()


    def prepare_task(self, task_data):
        """Prepare the task."""
        if task_data["type"] == "llm":
            llm_task = LLMTask()
            task_id = self.context.task_manager.enqueue_task(
                llm_task, args=(task_data)
            )
            self._async_result = self.context.task_manager.get_task_result(task_id)
            self._is_task_prepared = True


class TaskExecutionRoundBehaviour(AbstractRoundBehaviour):
    """TaskExecutionRoundBehaviour"""

    initial_behaviour_cls = TaskExecutionAbciBehaviour
    abci_app_cls = TaskExecutionAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = {
        TaskExecutionAbciBehaviour
    }
