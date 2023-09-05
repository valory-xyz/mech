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

"""This package contains the implementation of ."""
import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from aea.helpers.cid import to_v1
from aea.mail.base import EnvelopeContext
from aea.protocols.base import Message
from aea.protocols.dialogue.base import Dialogue
from aea.skills.behaviours import SimpleBehaviour

from packages.valory.connections.ipfs.connection import IpfsDialogues
from packages.valory.connections.ipfs.connection import PUBLIC_ID as IPFS_CONNECTION_ID
from packages.valory.connections.ledger.connection import (
    PUBLIC_ID as LEDGER_CONNECTION_PUBLIC_ID,
)
from packages.valory.connections.p2p_libp2p_client.connection import (
    PUBLIC_ID as P2P_CLIENT_PUBLIC_ID,
)
from packages.valory.contracts.agent_mech.contract import AgentMechContract
from packages.valory.protocols.acn_data_share import AcnDataShareMessage
from packages.valory.protocols.acn_data_share.dialogues import AcnDataShareDialogues
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ipfs import IpfsMessage
from packages.valory.protocols.ipfs.dialogues import IpfsDialogue
from packages.valory.skills.task_execution.models import Params
from packages.valory.skills.task_execution.utils.ipfs import (
    get_ipfs_file_hash,
    to_multihash,
)
from packages.valory.skills.task_execution.utils.task import AnyToolAsTask


PENDING_TASKS = "pending_tasks"
DONE_TASKS = "ready_tasks"
DONE_TASKS_LOCK = "lock"


LEDGER_API_ADDRESS = str(LEDGER_CONNECTION_PUBLIC_ID)


class TaskExecutionBehaviour(SimpleBehaviour):
    """A class to execute tasks."""

    def __init__(self, **kwargs: Any):
        """Initialise the agent."""
        super().__init__(**kwargs)
        self._executing_task: Optional[Dict[str, Any]] = None
        self._tools_to_file_hash: Dict[str, str] = {}
        self._all_tools: Dict[str, str] = {}
        self._inflight_tool_req: Optional[str] = None
        self._done_task: Optional[Dict[str, Any]] = None
        self._last_polling: Optional[float] = None
        self._invalid_request = False

    def setup(self) -> None:
        """Implement the setup."""
        self.context.logger.info("Setting up TaskExecutionBehaviour")
        self._tools_to_file_hash = {
            value: key
            for key, values in self.params.file_hash_to_tools.items()
            for value in values
        }

    def act(self) -> None:
        """Implement the act."""
        self._download_tools()
        self._execute_task()
        self._check_for_new_reqs()

    @property
    def done_tasks_lock(self) -> threading.Lock:
        """Get done_tasks_lock."""
        return self.context.shared_state[DONE_TASKS_LOCK]

    @property
    def params(self) -> Params:
        """Get the parameters."""
        return cast(Params, self.context.params)

    @property
    def pending_tasks(self) -> List[Dict[str, Any]]:
        """Get pending_tasks."""
        return self.context.shared_state[PENDING_TASKS]

    @property
    def done_tasks(self) -> List[Dict[str, Any]]:
        """Get done_tasks."""
        return self.context.shared_state[DONE_TASKS]

    def _should_poll(self) -> bool:
        """If we should poll the contract."""
        if self._last_polling is None:
            return True
        return self._last_polling + self.params.polling_interval <= time.time()

    def _is_executing_task_ready(self) -> bool:
        """Check if the executing task is ready."""
        if self._executing_task is None:
            return False
        task_id = self._executing_task.get("async_task_id", None)
        if task_id is None:
            return False

        return self.context.task_manager.get_task_result(task_id).ready()

    def _has_executing_task_timed_out(self) -> bool:
        """Check if the executing task timed out."""
        if self._executing_task is None:
            return False
        timeout_deadline = self._executing_task.get("timeout_deadline", None)
        if timeout_deadline is None:
            return False
        return timeout_deadline <= time.time()

    def _get_executing_task_result(self) -> Any:
        """Get the executing task result."""
        if self._executing_task is None:
            raise ValueError("Executing task is None")
        if self._invalid_request:
            return None
        task_id = self._executing_task.get("async_task_id", None)
        if task_id is None:
            raise ValueError("Executing task has no async_task_id")
        return self.context.task_manager.get_task_result(task_id).get()

    def _download_tools(self) -> None:
        """Download tools."""
        if self._inflight_tool_req is not None:
            # there already is a req in flight
            return
        if len(self._tools_to_file_hash) == len(self._all_tools):
            # we already have all the tools
            return
        for tool, file_hash in self._tools_to_file_hash.items():
            if tool in self._all_tools:
                continue
            # read one at a time
            ipfs_msg, message = self._build_ipfs_get_file_req(file_hash)
            self._inflight_tool_req = tool
            self.send_message(ipfs_msg, message, self._handle_get_tool)
            return

    def _handle_get_tool(self, message: IpfsMessage, dialogue: Dialogue) -> None:
        """Handle get tool response"""
        tool_py = list(message.files.values())[0]
        tool_req = cast(str, self._inflight_tool_req)
        local_namespace: Dict[str, Any] = globals().copy()
        if "run" in local_namespace:
            del local_namespace["run"]
        exec(tool_py, local_namespace)  # pylint: disable=W0122  # nosec
        self._all_tools[tool_req] = local_namespace["run"]
        self._inflight_tool_req = None

    def _check_for_new_reqs(self) -> None:
        """Check for new reqs."""
        if self.params.in_flight_req or not self._should_poll():
            # do nothing if there is an in flight request
            # or if we should not poll yet
            return

        contract_api_msg, _ = self.context.contract_dialogues.create(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=self.params.agent_mech_contract_address,
            contract_id=str(AgentMechContract.contract_id),
            callable="get_undelivered_reqs",
            kwargs=ContractApiMessage.Kwargs(dict(from_block=self.params.from_block)),
            counterparty=LEDGER_API_ADDRESS,
            ledger_id=self.context.default_ledger_id,
        )
        self.context.outbox.put_message(message=contract_api_msg)
        self.params.in_flight_req = True
        self._last_polling = time.time()

    def _execute_task(self) -> None:
        """Execute tasks."""
        # check if there is a task already executing
        if self.params.in_flight_req:
            # there is an in flight request
            return

        if self._executing_task is not None:
            if self._is_executing_task_ready() or self._invalid_request:
                self._handle_done_task()
            elif self._has_executing_task_timed_out():
                self._handle_timeout_task()
            return

        if len(self.pending_tasks) == 0:
            # not tasks (requests) to execute
            return

        # create new task
        task_data = self.pending_tasks.pop(0)
        self.context.logger.info(f"Preparing task with data: {task_data}")
        self._executing_task = task_data
        task_data_ = task_data["data"]
        ipfs_hash = get_ipfs_file_hash(task_data_)
        self.context.logger.info(f"IPFS hash: {ipfs_hash}")
        ipfs_msg, message = self._build_ipfs_get_file_req(ipfs_hash)
        self.send_message(ipfs_msg, message, self._handle_get_task)

    def send_message(
        self, msg: Message, dialogue: Dialogue, callback: Callable
    ) -> None:
        """Send message."""
        self.context.outbox.put_message(message=msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        self.params.req_to_callback[nonce] = callback
        self.params.in_flight_req = True

    def _handle_done_task(self) -> None:
        """Handle done tasks"""
        executing_task = cast(Dict[str, Any], self._executing_task)
        req_id = executing_task.get("requestId", None)
        task_result = self._get_executing_task_result()
        response = {"requestId": req_id, "result": "Invalid response"}
        self._done_task = {"request_id": req_id}
        if task_result is not None:
            # task succeeded
            deliver_msg, transaction = task_result
            response = {**response, "result": deliver_msg}
            self._done_task["transaction"] = transaction

        self.context.logger.info(f"Task result for request {req_id}: {task_result}")
        msg, dialogue = self._build_ipfs_store_file_req(
            {str(req_id): json.dumps(response)}
        )
        self.send_message(msg, dialogue, self._handle_store_response)

    def _handle_timeout_task(self) -> None:
        """Handle timeout tasks"""
        executing_task = cast(Dict[str, Any], self._executing_task)
        req_id = executing_task.get("requestId", None)
        self.context.logger.info(f"Task timed out for request {req_id}")
        # added to end of queue
        self.pending_tasks.append(executing_task)
        self._executing_task = None

    def _handle_get_task(self, message: IpfsMessage, dialogue: Dialogue) -> None:
        """Handle the response from ipfs for a task request."""
        task_data = [json.loads(content) for content in message.files.values()][0]
        is_data_valid = (
            task_data
            and isinstance(task_data, dict)
            and "prompt" in task_data
            and "tool" in task_data
        )  # pylint: disable=C0301
        if is_data_valid and task_data["tool"] in self._tools_to_file_hash:
            self._prepare_task(task_data)
        elif is_data_valid:
            tool = task_data["tool"]
            self.context.logger.warning(f"Tool {tool} is not valid.")
            self._invalid_request = True
        else:
            self.context.logger.warning("Data for task is not valid.")
            self._invalid_request = True

    def _prepare_task(self, task_data: Dict[str, Any]) -> None:
        """Prepare the task."""
        tool_task = AnyToolAsTask()
        task_data["method"] = self._all_tools[task_data["tool"]]
        task_data["api_keys"] = self.params.api_keys
        task_id = self.context.task_manager.enqueue_task(tool_task, kwargs=task_data)
        executing_task = cast(Dict[str, Any], self._executing_task)
        executing_task["async_task_id"] = task_id
        executing_task["timeout_deadline"] = time.time() + self.params.task_deadline
        self._async_result = self.context.task_manager.get_task_result(task_id)

    def _build_ipfs_message(
        self,
        performative: IpfsMessage.Performative,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[IpfsMessage, IpfsDialogue]:
        """Builds an IPFS message."""
        ipfs_dialogues = cast(IpfsDialogues, self.context.ipfs_dialogues)
        message, dialogue = ipfs_dialogues.create(
            counterparty=str(IPFS_CONNECTION_ID),
            performative=performative,
            timeout=timeout,
            **kwargs,
        )
        return message, dialogue

    def _build_ipfs_store_file_req(  # pylint: disable=too-many-arguments
        self,
        filename_to_obj: Dict[str, str],
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[IpfsMessage, IpfsDialogue]:
        """Builds a STORE_FILES ipfs message."""
        message, dialogue = self._build_ipfs_message(
            performative=IpfsMessage.Performative.STORE_FILES,  # type: ignore
            files=filename_to_obj,
            timeout=timeout,
            **kwargs,
        )
        return message, dialogue

    def _build_ipfs_get_file_req(
        self,
        ipfs_hash: str,
        timeout: Optional[float] = None,
    ) -> Tuple[IpfsMessage, IpfsDialogue]:
        """
        Builds a GET_FILES IPFS request.

        :param ipfs_hash: the ipfs hash of the file/dir to download.
        :param timeout: timeout for the request.
        :returns: the ipfs message, and its corresponding dialogue.
        """
        message, dialogue = self._build_ipfs_message(
            performative=IpfsMessage.Performative.GET_FILES,  # type: ignore
            ipfs_hash=ipfs_hash,
            timeout=timeout,
        )
        return message, dialogue

    def _handle_store_response(self, message: IpfsMessage, dialogue: Dialogue) -> None:
        """Handle the response from ipfs for a store response request."""
        executing_task = cast(Dict[str, Any], self._executing_task)
        req_id, sender = (
            executing_task["requestId"],
            executing_task["sender"],
        )
        self.context.logger.info(f"Response for request {req_id} stored on IPFS.")
        ipfs_hash = to_v1(message.ipfs_hash)
        self.send_data_via_acn(
            sender_address=sender,
            request_id=str(req_id),
            data=ipfs_hash,
        )
        done_task = cast(Dict[str, Any], self._done_task)
        done_task["task_result"] = to_multihash(ipfs_hash)
        # add to done tasks, in thread safe way
        with self.done_tasks_lock:
            self.done_tasks.append(done_task)
        # reset tasks
        self._executing_task = None
        self._done_task = None
        self._invalid_request = False

    def send_data_via_acn(
        self,
        sender_address: str,
        request_id: str,
        data: Any,
    ) -> None:
        """Handle callbacks."""
        self.context.logger.info(
            f"Sending data to {sender_address} via ACN for request ID {request_id}"
        )
        response, _ = cast(
            AcnDataShareDialogues, self.context.acn_data_share_dialogues
        ).create(
            counterparty=sender_address,
            performative=AcnDataShareMessage.Performative.DATA,
            request_id=request_id,
            content=data,
        )
        self.context.outbox.put_message(
            message=response,
            context=EnvelopeContext(connection_id=P2P_CLIENT_PUBLIC_ID),
        )
