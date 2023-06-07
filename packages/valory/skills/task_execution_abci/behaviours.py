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
import os
from abc import ABC
from multiprocessing.pool import AsyncResult
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Type, cast

import multibase
import multicodec
import openai  # noqa
from aea.helpers.cid import CID, to_v1

from packages.valory.contracts.agent_mech.contract import AgentMechContract
from packages.valory.contracts.gnosis_safe.contract import (
    GnosisSafeContract,
    SafeOperation,
)
from packages.valory.contracts.multisend.contract import (
    MultiSendContract,
    MultiSendOperation,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.abstract_round_abci.io_.store import SupportedFiletype
from packages.valory.skills.task_execution_abci.models import Params
from packages.valory.skills.task_execution_abci.rounds import (
    SynchronizedData,
    TaskExecutionAbciApp,
    TaskExecutionAbciPayload,
    TaskExecutionRound,
)
from packages.valory.skills.task_execution_abci.tasks import AnyToolAsTask
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)


CID_PREFIX = "f01701220"
ZERO_ETHER_VALUE = 0
SAFE_GAS = 0


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

    def async_act(self) -> Generator:  # pylint: disable=R0914,R0915
        """Do the act, supporting asynchronous execution."""

        if not self.context.params.all_tools:
            all_tools = {}
            for file_hash, tools in self.context.params.file_hash_to_tools.items():
                tool_py = yield from self.get_from_ipfs(
                    file_hash, custom_loader=lambda plain: plain
                )
                if tool_py is None:
                    self.context.logger.error(
                        f"Failed to get the tools {tools} with file_hash {file_hash} from IPFS!"
                    )
                all_tools.update({tool: tool_py for tool in tools})
            self.context.params.__dict__["_frozen"] = False
            self.context.params.all_tools = all_tools
            self.context.params.__dict__["_frozen"] = True

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            task_result = yield from self.get_task_result()
            if task_result is None:
                # the task is not ready yet, check in the next iteration
                return
            payload_content = yield from self.get_payload_content(task_result)
            sender = self.context.agent_address
            payload = TaskExecutionAbciPayload(sender=sender, content=payload_content)
        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
        self.set_done()

    def get_payload_content(
        self, task_result: Optional[Tuple[str, str, List[Dict]]]
    ) -> Generator[None, None, str]:
        """Get the payload content."""
        if task_result is None:
            # something went wrong, respond with ERROR payload for now
            return TaskExecutionRound.ERROR_PAYLOAD

        request_id, deliver_msg_hash, multisend_txs = task_result
        deliver_tx = yield from self._get_deliver_tx(
            {"request_id": request_id, "task_result": deliver_msg_hash}
        )
        if deliver_tx is None:
            # something went wrong, respond with ERROR payload for now
            return TaskExecutionRound.ERROR_PAYLOAD

        all_txs = multisend_txs + [deliver_tx]
        multisend_tx_str = yield from self._to_multisend(all_txs)
        if multisend_tx_str is None:
            # something went wrong, respond with ERROR payload for now
            return TaskExecutionRound.ERROR_PAYLOAD

        return multisend_tx_str

    def get_task_result(  # pylint: disable=R0914,R1710
        self,
    ) -> Generator[None, None, Optional[Tuple[str, str, List[Dict]]]]:
        """
        Execute a task in the background and wait for the result asynchronously.

        :return: A tuple containing request_id, deliver_msg_hash and multisend transactions.
        :yields: None
        """
        # Check whether the task already exists
        if not self._is_task_prepared and not self._invalid_request:
            # Get the first task in the queue - format: {"requestId": <id>, "data": <ipfs_hash>}
            pending_tasks = self.context.shared_state.get("pending_tasks")
            if len(pending_tasks) == 0:
                # something went wrong, we should not be here, send an error payload
                return None

            task_data = self.context.shared_state.get("pending_tasks").pop(0)
            self.context.logger.info(f"Preparing task with data: {task_data}")
            self.request_id = task_data["requestId"]
            task_data_ = task_data["data"]

            # Verify the data hash and handle encoding
            try:
                file_hash = self._get_ipfs_file_hash(task_data_)
                # Get the file from IPFS
                self.context.logger.info(f"Getting data from IPFS: {file_hash}")
                task_data = yield from self.get_from_ipfs(
                    ipfs_hash=file_hash,
                    filetype=SupportedFiletype.JSON,
                )
                self.context.logger.info(f"Got data from IPFS: {task_data}")

                # Verify the file data
                is_data_valid = (
                    task_data
                    and isinstance(task_data, dict)
                    and "prompt" in task_data
                    and "tool" in task_data
                )  # pylint: disable=C0301
                if (
                    is_data_valid
                    and task_data["tool"] in self.context.params.tools_to_file_hash
                ):
                    self._prepare_task(task_data)
                elif is_data_valid:
                    tool = task_data["tool"]
                    self.context.logger.warning(f"Tool {tool} is not valid.")
                    self._invalid_request = True
                else:
                    self.context.logger.warning("Data is not valid.")
                    self._invalid_request = True
            except Exception:  # pylint: disable=W0718
                self.context.logger.warning("Exception when handling data.")
                self._invalid_request = True

        response_obj = None

        # Handle invalid requests
        if self._invalid_request:
            deliver_msg = "no_op"
            # respond with no_op and no multisend transactions
            request_id = cast(str, self.request_id)
            return request_id, deliver_msg, []

        self._async_result = cast(AsyncResult, self._async_result)

        # Handle unfinished task
        if not self._invalid_request and not self._async_result.ready():
            self.context.logger.debug("The task is not finished yet.")
            yield from self.sleep(self.params.sleep_time)
            return None

        # Handle finished task
        transactions: List[Dict] = []
        if not self._invalid_request and self._async_result.ready():
            # the expected response for the task is: Tuple[str, List[Dict]] = (deliver_msg, transactions)
            # deliver_msg: str = is the string containing the deliver message.
            # transactions: List[Dict] = is the list of transactions to be multisent.
            # Should be an empty list if no transactions are needed.
            # example response: ("task_result", [{"to": "0x123", "value": 0, "data": "0x123"}])
            task_result: Tuple[str, List[Dict]] = self._async_result.get()
            if task_result is None:
                return None
            deliver_msg, transactions = task_result
            response_obj = {"requestId": self.request_id, "result": deliver_msg}

        self.context.logger.info(f"Response object: {response_obj}")

        # Write to IPFS
        file_path = os.path.join(self.context.data_dir, str(self.request_id))

        obj_hash = yield from self.send_to_ipfs(
            filename=file_path,
            obj=response_obj,
            filetype=SupportedFiletype.JSON,
        )
        obj_hash = to_v1(obj_hash)  # from 2 to 1: base32 encoded CID

        # The original Base32 encoded CID
        base32_cid = obj_hash

        # Decode the Base32 CID to bytes
        cid_bytes = multibase.decode(base32_cid)

        # Remove the multicodec prefix (0x01) from the bytes
        multihash_bytes = multicodec.remove_prefix(cid_bytes)

        # Convert the multihash bytes to a hexadecimal string
        hex_multihash = multihash_bytes.hex()

        hex_multihash = hex_multihash[6:]
        request_id = cast(str, self.request_id)
        return request_id, hex_multihash, transactions

    def _get_ipfs_file_hash(self, data: bytes) -> str:
        """Get hash from bytes"""
        try:
            return str(CID.from_string(data.decode()))
        except Exception:  # noqa
            # if something goes wrong, fallback to sha256
            file_hash = data.hex()
            file_hash = CID_PREFIX + file_hash
            file_hash = str(CID.from_string(file_hash))
            return file_hash

    def _prepare_task(self, task_data: Dict[str, Any]) -> None:
        """Prepare the task."""
        tool_task = AnyToolAsTask()
        tool_py = self.context.params.all_tools[task_data["tool"]]
        local_namespace: Dict[str, Any] = globals().copy()
        if "run" in local_namespace:
            del local_namespace["run"]
        exec(tool_py, local_namespace)  # pylint: disable=W0122  # nosec
        task_data["method"] = local_namespace["run"]
        task_data["api_keys"] = self.params.api_keys
        task_id = self.context.task_manager.enqueue_task(tool_task, kwargs=task_data)
        self._async_result = self.context.task_manager.get_task_result(task_id)
        self._is_task_prepared = True

    def _to_multisend(
        self, transactions: List[Dict]
    ) -> Generator[None, None, Optional[str]]:
        """Transform payload to MultiSend."""
        multi_send_txs = []
        for transaction in transactions:
            transaction = {
                "operation": transaction.get("operation", MultiSendOperation.CALL),
                "to": transaction["to"],
                "value": transaction["value"],
                "data": transaction.get("data", b""),
            }
            multi_send_txs.append(transaction)

        response = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=self.params.multisend_address,
            contract_id=str(MultiSendContract.contract_id),
            contract_callable="get_tx_data",
            multi_send_txs=multi_send_txs,
        )
        if response.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.context.logger.error(
                f"Couldn't compile the multisend tx. "
                f"Expected performative {ContractApiMessage.Performative.RAW_TRANSACTION.value}, "  # type: ignore
                f"received {response.performative.value}."
            )
            return None

        # strip "0x" from the response
        multisend_data_str = cast(str, response.raw_transaction.body["data"])[2:]
        tx_data = bytes.fromhex(multisend_data_str)
        tx_hash = yield from self._get_safe_tx_hash(tx_data)
        if tx_hash is None:
            # something went wrong
            return None

        payload_data = hash_payload_to_hex(
            safe_tx_hash=tx_hash,
            ether_value=ZERO_ETHER_VALUE,
            safe_tx_gas=SAFE_GAS,
            operation=SafeOperation.DELEGATE_CALL.value,
            to_address=self.params.multisend_address,
            data=tx_data,
        )
        return payload_data

    def _get_safe_tx_hash(self, data: bytes) -> Generator[None, None, Optional[str]]:
        """
        Prepares and returns the safe tx hash.

        This hash will be signed later by the agents, and submitted to the safe contract.
        Note that this is the transaction that the safe will execute, with the provided data.

        :param data: the safe tx data.
        :return: the tx hash
        """
        response = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=self.synchronized_data.safe_contract_address,
            contract_id=str(GnosisSafeContract.contract_id),
            contract_callable="get_raw_safe_transaction_hash",
            to_address=self.params.multisend_address,  # we send the tx to the multisend address
            value=ZERO_ETHER_VALUE,
            data=data,
            safe_tx_gas=SAFE_GAS,
            operation=SafeOperation.DELEGATE_CALL.value,
        )

        if response.performative != ContractApiMessage.Performative.STATE:
            self.context.logger.error(
                f"Couldn't get safe hash. "
                f"Expected response performative {ContractApiMessage.Performative.STATE.value}, "  # type: ignore
                f"received {response.performative.value}."
            )
            return None

        # strip "0x" from the response hash
        tx_hash = cast(str, response.state.body["tx_hash"])[2:]
        return tx_hash

    def _get_deliver_tx(
        self, task_data: Dict[str, Any]
    ) -> Generator[None, None, Optional[Dict]]:
        """Get the deliver tx."""
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=self.params.agent_mech_contract_address,
            contract_id=str(AgentMechContract.contract_id),
            contract_callable="get_deliver_data",
            request_id=task_data["request_id"],
            data=task_data["task_result"],
        )
        if (
            contract_api_msg.performative != ContractApiMessage.Performative.STATE
        ):  # pragma: nocover
            self.context.logger.warning(
                f"get_deliver_data unsuccessful!: {contract_api_msg}"
            )
            return None

        data = cast(bytes, contract_api_msg.state.body["data"])
        return {
            "to": self.params.agent_mech_contract_address,
            "value": ZERO_ETHER_VALUE,
            "data": data,
        }


class TaskExecutionRoundBehaviour(AbstractRoundBehaviour):
    """TaskExecutionRoundBehaviour"""

    initial_behaviour_cls = TaskExecutionAbciBehaviour
    abci_app_cls = TaskExecutionAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = {TaskExecutionAbciBehaviour}
