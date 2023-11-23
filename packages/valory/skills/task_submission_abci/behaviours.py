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
import abc
import json
import threading
import time
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional, Set, Type, cast

import openai  # noqa
from multibase import multibase
from multicodec import multicodec

from packages.valory.contracts.agent_mech.contract import AgentMechContract
from packages.valory.contracts.agent_registry.contract import AgentRegistryContract
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
from packages.valory.skills.task_submission_abci.models import Params
from packages.valory.skills.task_submission_abci.payloads import TransactionPayload
from packages.valory.skills.task_submission_abci.rounds import (
    SynchronizedData,
    TaskPoolingPayload,
    TaskPoolingRound,
    TaskSubmissionAbciApp,
    TransactionPreparationRound,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)


ZERO_ETHER_VALUE = 0
SAFE_GAS = 0
DONE_TASKS = "ready_tasks"
DONE_TASKS_LOCK = "lock"


class TaskExecutionBaseBehaviour(BaseBehaviour, abc.ABC):
    """Base behaviour for the task_execution_abci skill."""

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return cast(SynchronizedData, super().synchronized_data)

    @property
    def params(self) -> Params:
        """Return the params."""
        return cast(Params, super().params)

    @property
    def done_tasks(self) -> List[Dict[str, Any]]:
        """
        Return the done (ready) tasks from shared state.

        Use with care, the returned data here is NOT synchronized with the rest of the agents.

        :returns: the tasks
        """
        done_tasks = deepcopy(self.context.shared_state.get(DONE_TASKS, []))
        return cast(List[Dict[str, Any]], done_tasks)

    def done_tasks_lock(self) -> threading.Lock:
        """Get done_tasks_lock."""
        return self.context.shared_state[DONE_TASKS_LOCK]

    def remove_tasks(self, submitted_tasks: List[Dict[str, Any]]) -> None:
        """
        Pop the tasks from shared state.

        :param submitted_tasks: the done tasks that have already been submitted
        """
        # run this in a lock
        # the amount of done tasks will always be relatively low (<<20)
        # we can afford to do this in a lock
        with self.done_tasks_lock():
            done_tasks = self.done_tasks
            not_submitted = []
            for done_task in done_tasks:
                is_submitted = False
                for submitted_task in submitted_tasks:
                    if submitted_task["request_id"] == done_task["request_id"]:
                        is_submitted = True
                        break
                if not is_submitted:
                    not_submitted.append(done_task)
            self.context.shared_state[DONE_TASKS] = not_submitted


class TaskPoolingBehaviour(TaskExecutionBaseBehaviour):
    """TaskPoolingBehaviour"""

    matching_round: Type[AbstractRound] = TaskPoolingRound

    def async_act(self) -> Generator:  # pylint: disable=R0914,R0915
        """Do the act, supporting asynchronous execution."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            # clean up the queue based on the outcome of the previous period
            self.handle_submitted_tasks()
            # sync new tasks
            payload_content = yield from self.get_payload_content()
            sender = self.context.agent_address
            payload = TaskPoolingPayload(sender=sender, content=payload_content)
        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
        self.set_done()

    def get_payload_content(self) -> Generator[None, None, str]:
        """Get the payload content."""
        done_tasks = yield from self.get_done_tasks(self.params.task_wait_timeout)
        return json.dumps(done_tasks)

    def get_done_tasks(self, timeout: float) -> Generator[None, None, List[Dict]]:
        """Wait for tasks to get done in the specified timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if len(self.done_tasks) == 0:
                one_second = 1.0
                yield from self.sleep(one_second)
                continue
            # there are done tasks, return all of them
            return self.done_tasks

        # no tasks are ready for this agent
        self.context.logger.info("No tasks were ready within the timeout")
        return []

    def handle_submitted_tasks(self) -> None:
        """Handle tasks that have been already submitted before (in a prev. period)."""
        submitted_tasks = cast(List[Dict[str, Any]], self.synchronized_data.done_tasks)
        self.context.logger.info(
            f"Tasks {submitted_tasks} has already been submitted. "
            f"Removing them from the list of tasks to be processed."
        )
        self.remove_tasks(submitted_tasks)


class TransactionPreparationBehaviour(TaskExecutionBaseBehaviour):
    """TransactionPreparationBehaviour"""

    matching_round: Type[AbstractRound] = TransactionPreparationRound

    def async_act(self) -> Generator:  # pylint: disable=R0914,R0915
        """Do the act, supporting asynchronous execution."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            payload_content = yield from self.get_payload_content()
            sender = self.context.agent_address
            payload = TransactionPayload(sender=sender, content=payload_content)
        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
        self.set_done()

    def get_payload_content(self) -> Generator[None, None, str]:
        """Prepare the transaction"""
        all_txs = []
        should_update_hash = yield from self._should_update_hash()
        if should_update_hash:
            update_hash_tx = yield from self._get_mech_update_hash_tx()
            if update_hash_tx is None:
                # something went wrong, respond with ERROR payload for now
                return TransactionPreparationRound.ERROR_PAYLOAD
            all_txs.append(update_hash_tx)

        for task in self.synchronized_data.done_tasks:
            deliver_tx = yield from self._get_deliver_tx(task)
            if deliver_tx is None:
                # something went wrong, respond with ERROR payload for now
                return TransactionPreparationRound.ERROR_PAYLOAD
            all_txs.append(deliver_tx)
            response_tx = task.get("transaction", None)
            if response_tx is not None:
                all_txs.append(response_tx)

        multisend_tx_str = yield from self._to_multisend(all_txs)
        if multisend_tx_str is None:
            # something went wrong, respond with ERROR payload for now
            return TransactionPreparationRound.ERROR_PAYLOAD

        return multisend_tx_str

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
            gas_limit=self.params.manual_gas_limit,
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
            contract_address=task_data["mech_address"],
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
            "to": task_data["mech_address"],
            "value": ZERO_ETHER_VALUE,
            "data": data,
        }

    def _get_latest_hash(self) -> Generator[None, None, Optional[bytes]]:
        """Get latest update hash."""
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=self.params.agent_registry_address,
            contract_id=str(AgentRegistryContract.contract_id),
            contract_callable="get_token_hash",
            token_id=self.params.agent_id,
        )
        if (
            contract_api_msg.performative != ContractApiMessage.Performative.STATE
        ):  # pragma: nocover
            self.context.logger.warning(
                f"get_token_hash unsuccessful!: {contract_api_msg}"
            )
            return None

        latest_hash = cast(bytes, contract_api_msg.state.body["data"])
        return latest_hash

    def _should_update_hash(self) -> Generator:
        """Check if the agent should update the hash."""
        if self.params.task_mutable_params.latest_metadata_hash is None:
            latest_hash = yield from self._get_latest_hash()
            if latest_hash is None:
                self.context.logger.warning(
                    "Could not get latest hash. Don't update the metadata."
                )
                return False
            self.params.task_mutable_params.latest_metadata_hash = latest_hash

        configured_hash = self.to_multihash(self.params.metadata_hash)
        latest_hash = self.params.task_mutable_params.latest_metadata_hash
        return configured_hash != latest_hash

    @staticmethod
    def to_multihash(hash_string: str) -> str:
        """To multihash string."""
        # Decode the Base32 CID to bytes
        cid_bytes = multibase.decode(hash_string)
        # Remove the multicodec prefix (0x01) from the bytes
        multihash_bytes = multicodec.remove_prefix(cid_bytes)
        # Convert the multihash bytes to a hexadecimal string
        hex_multihash = multihash_bytes.hex()
        return hex_multihash[6:]

    def _get_mech_update_hash_tx(self) -> Generator:
        """Get the mech update hash tx."""
        metadata_str = self.to_multihash(self.params.metadata_hash)
        metadata = bytes.fromhex(metadata_str)
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=self.params.agent_registry_address,
            contract_id=str(AgentRegistryContract.contract_id),
            contract_callable="get_update_hash_tx_data",
            token_id=self.params.agent_id,
            metadata_hash=metadata,
        )
        if (
            contract_api_msg.performative != ContractApiMessage.Performative.STATE
        ):  # pragma: nocover
            self.context.logger.warning(
                f"get_mech_update_hash unsuccessful!: {contract_api_msg}"
            )
            return None

        data = cast(bytes, contract_api_msg.state.body["data"])
        return {
            "to": self.params.agent_registry_address,
            "value": ZERO_ETHER_VALUE,
            "data": data,
        }


class TaskSubmissionRoundBehaviour(AbstractRoundBehaviour):
    """TaskSubmissionRoundBehaviour"""

    initial_behaviour_cls = TaskPoolingBehaviour
    abci_app_cls = TaskSubmissionAbciApp
    behaviours: Set[Type[BaseBehaviour]] = {
        TaskPoolingBehaviour,  # type: ignore
        TransactionPreparationBehaviour,  # type: ignore
    }
