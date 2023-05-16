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
import json
import os
from abc import ABC
from multiprocessing.pool import AsyncResult
from typing import Any, Generator, Optional, Set, Type, cast

import multibase
import multicodec
from aea.helpers.cid import to_v1

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour, BaseBehaviour)
from packages.valory.skills.abstract_round_abci.io_.store import \
    SupportedFiletype
from packages.valory.skills.task_execution_abci.models import Params
from packages.valory.skills.task_execution_abci.rounds import (
    SynchronizedData, TaskExecutionAbciApp, TaskExecutionAbciPayload,
    TaskExecutionRound)
from packages.valory.skills.task_execution_abci.tasks import OpenAITask


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
        self.request_id = None
        self._is_task_prepared = False
        self._invalid_request = False

    def is_json(self, obj: Any) -> bool:
        """Checks whether an object is json"""
        try:
            json.loads(obj)
        except ValueError as _:
            return False
        return True

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():

            # Check whether the task already exists
            if not self._is_task_prepared and not self._invalid_request:
                # Get the first task in the queue
                task_data = self.context.shared_state.get("pending_tasks").pop(0)
                self.context.logger.info(f"Preparing task with data: {task_data}")

                # Request format
                # {
                #     "requestId": <id>
                #     "data": <ipfs_hash>
                # }

                self.request_id = task_data["requestId"]

                # Verify the data hash and handle encoding
                file_hash = task_data["data"].decode("utf-8")
                file_hash = "f01701220" + file_hash[2:]  # CID prefix

                # Get the file from IPFS
                task_data = yield from self.get_from_ipfs(
                    ipfs_hash=file_hash,
                    filetype=SupportedFiletype.JSON,
                )

                # Verify the file data
                is_data_valid = task_data and self.is_json(task_data) and "prompt" in task_data and "tool" in task_data
                if is_data_valid:
                    self.prepare_task(task_data)
                else:
                    self.context.logger.warning("Data is not valid")
                    self._invalid_request = True

            response_obj = None

            # Handle invalid requests
            if self._invalid_request:
                task_result = "no_op"
                response_obj = {"requestId": self.request_id, "result": task_result}

            self._async_result = cast(AsyncResult, self._async_result)

            # Handle unfinished task
            if not self._invalid_request and not self._async_result.ready():
                self.context.logger.debug("The task is not finished yet.")
                yield from self.sleep(self.params.sleep_time)
                return

            # Handle finished task
            if not self._invalid_request and self._async_result.ready():
                task_result = self._async_result.get()
                response_obj = {"requestId": self.request_id, "result": task_result}

            self.context.logger.info(f"Response object: {response_obj}")

            # Write to IPFS
            file_path = os.path.join(self.context.data_dir, str(self.request_id))

            obj_hash = yield from self.send_to_ipfs(
                filename=file_path,
                obj=response_obj,
                filetype=SupportedFiletype.JSON,
            )
            obj_hash= to_v1(obj_hash) # from 2 to 1: base32 encoded CID

            # The original Base32 encoded CID
            base32_cid = obj_hash

            # Decode the Base32 CID to bytes
            cid_bytes = multibase.decode(base32_cid)

            # Remove the multicodec prefix (0x01) from the bytes
            multihash_bytes = multicodec.remove_prefix(cid_bytes)

            # Convert the multihash bytes to a hexadecimal string
            hex_multihash = multihash_bytes.hex()

            hex_multihash = hex_multihash[6:]

            payload_content = json.dumps({"request_id": self.request_id, "task_result": hex_multihash}, sort_keys=True)
            sender = self.context.agent_address
            payload = TaskExecutionAbciPayload(sender=sender, content=payload_content)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def prepare_task(self, task_data):
        """Prepare the task."""
        if task_data["tool"] == "openai-gpt4":
            openai_task = OpenAITask()
            task_data["use_gpt4"] = False
            task_data["openai_api_key"] = self.params.openai_api_key
            task_id = self.context.task_manager.enqueue_task(
                openai_task, kwargs=task_data
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
