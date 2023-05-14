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

"""This package contains round behaviours of TransactionPreparationAbciApp."""

import json
from abc import ABC
from typing import Generator, Optional, Set, Type, cast

from packages.valory.contracts.agent_mech.contract import AgentMechContract
from packages.valory.contracts.gnosis_safe.contract import GnosisSafeContract
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour, BaseBehaviour)
from packages.valory.skills.transaction_preparation_abci.models import Params
from packages.valory.skills.transaction_preparation_abci.rounds import (
    SynchronizedData, TransactionPreparationAbciApp,
    TransactionPreparationAbciPayload, TransactionPreparationRound)
from packages.valory.skills.transaction_settlement_abci.payload_tools import \
    hash_payload_to_hex

SAFE_TX_GAS = 0
ETHER_VALUE = 0


class TransactionPreparationBaseBehaviour(BaseBehaviour, ABC):
    """Base behaviour for the transaction_preparation_abci skill."""

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return cast(SynchronizedData, super().synchronized_data)

    @property
    def params(self) -> Params:
        """Return the params."""
        return cast(Params, super().params)


class TransactionPreparationAbciBehaviour(TransactionPreparationBaseBehaviour):
    """TransactionPreparationAbciBehaviour"""

    matching_round: Type[AbstractRound] = TransactionPreparationRound


    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():

            finished_task_data = self.synchronized_data.finished_task_data

            self.context.logger.info(f"Finished Task Data: {finished_task_data}")

            tx_hash = yield from self._get_safe_tx_hash(finished_task_data)

            if not tx_hash:
                tx_hash = TransactionPreparationRound.ERROR_PAYLOAD

            payload_content = {
                "tx_hash": tx_hash,
            }

            payload = TransactionPreparationAbciPayload(
                sender=self.context.agent_address,
                content=json.dumps(payload_content, sort_keys=True),
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def _get_safe_tx_hash(
        self,
        task_data,
    ) -> Generator[None, None, Optional[str]]:
        """Get the transaction hash of the Safe tx."""
        # Get the raw transaction from the AgentMech contract
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=self.params.agent_mech_contract_address,
            contract_id=str(AgentMechContract.contract_id),
            contract_callable="get_deliver_data",
            request_id=task_data["request_id"],
            data=task_data["task_result"][0].encode("utf-8"),
        )
        if (
            contract_api_msg.performative != ContractApiMessage.Performative.STATE
        ):  # pragma: nocover
            self.context.logger.warning(
                f"get_deliver_data unsuccessful!: {contract_api_msg}"
            )
            return None

        data = cast(bytes, contract_api_msg.state.body["data"])

        # Get the safe transaction hash
        ether_value = ETHER_VALUE
        safe_tx_gas = SAFE_TX_GAS

        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=self.synchronized_data.safe_contract_address,
            contract_id=str(GnosisSafeContract.contract_id),
            contract_callable="get_raw_safe_transaction_hash",
            to_address=self.params.agent_mech_contract_address,
            value=ether_value,
            data=data,
            safe_tx_gas=safe_tx_gas,
        )
        if (
            contract_api_msg.performative != ContractApiMessage.Performative.STATE
        ):  # pragma: nocover
            self.context.logger.warning(
                f"get_raw_safe_transaction_hash unsuccessful!: {contract_api_msg}"
            )
            return None

        safe_tx_hash = cast(str, contract_api_msg.state.body["tx_hash"])
        safe_tx_hash = safe_tx_hash[2:]
        self.context.logger.info(f"Hash of the Safe transaction: {safe_tx_hash}")

        # temp hack
        payload_string = hash_payload_to_hex(
            safe_tx_hash, ether_value, safe_tx_gas, self.params.agent_mech_contract_address, data
        )

        return payload_string


class TransactionPreparationRoundBehaviour(AbstractRoundBehaviour):
    """TransactionPreparationRoundBehaviour"""

    initial_behaviour_cls = TransactionPreparationAbciBehaviour
    abci_app_cls = TransactionPreparationAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = {
        TransactionPreparationAbciBehaviour
    }
