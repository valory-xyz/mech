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

"""This module contains the dynamic_contribution contract definition."""

from typing import Any, Dict, List, cast

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea.crypto.base import LedgerApi
from aea_ledger_ethereum import EthereumApi
from web3.types import BlockIdentifier


class AgentMechContract(Contract):
    """The scaffold contract class for a smart contract."""

    contract_id = PublicId.from_str("valory/agent_mech:0.1.0")

    @classmethod
    def get_raw_transaction(
        cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any
    ) -> JSONLike:
        """
        Handler method for the 'GET_RAW_TRANSACTION' requests.

        Implement this method in the sub class if you want
        to handle the contract requests manually.

        :param ledger_api: the ledger apis.
        :param contract_address: the contract address.
        :param kwargs: the keyword arguments.
        :return: the tx  # noqa: DAR202
        """
        raise NotImplementedError

    @classmethod
    def get_raw_message(
        cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any
    ) -> bytes:
        """
        Handler method for the 'GET_RAW_MESSAGE' requests.

        Implement this method in the sub class if you want
        to handle the contract requests manually.

        :param ledger_api: the ledger apis.
        :param contract_address: the contract address.
        :param kwargs: the keyword arguments.
        :return: the tx  # noqa: DAR202
        """
        raise NotImplementedError

    @classmethod
    def get_state(
        cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any
    ) -> JSONLike:
        """
        Handler method for the 'GET_STATE' requests.

        Implement this method in the sub class if you want
        to handle the contract requests manually.

        :param ledger_api: the ledger apis.
        :param contract_address: the contract address.
        :param kwargs: the keyword arguments.
        :return: the tx  # noqa: DAR202
        """
        raise NotImplementedError

    @classmethod
    def get_deliver_data(
        cls, ledger_api: LedgerApi, contract_address: str, request_id: int, data: str
    ) -> JSONLike:
        """
        Deliver a response to a request.

        :param ledger_api: LedgerApi object
        :param contract_address: the address of the token to be used
        :param request_id: the id of the target request
        :param data: the response data
        :return: the deliver data
        """
        ledger_api = cast(EthereumApi, ledger_api)

        if not isinstance(ledger_api, EthereumApi):
            raise ValueError(f"Only EthereumApi is supported, got {type(ledger_api)}")

        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI(
            fn_name="deliver", args=[request_id, bytes.fromhex(data)]
        )
        return {"data": bytes.fromhex(data[2:])}  # type: ignore

    @classmethod
    def get_request_events(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        from_block: BlockIdentifier = "earliest",
        to_block: BlockIdentifier = "latest",
    ) -> JSONLike:
        """Get the Request events emitted by the contract."""
        ledger_api = cast(EthereumApi, ledger_api)
        contract_instance = cls.get_instance(ledger_api, contract_address)
        entries = contract_instance.events.Request.create_filter(
            fromBlock=from_block,
            toBlock=to_block,
        ).get_all_entries()
        request_events = list(
            {
                "tx_hash": entry.transactionHash.hex(),
                "block_number": entry.blockNumber,
                **entry["args"],
            }
            for entry in entries
        )
        return {"data": request_events}

    @classmethod
    def get_deliver_events(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        from_block: BlockIdentifier = "earliest",
        to_block: BlockIdentifier = "latest",
    ) -> JSONLike:
        """Get the Deliver events emitted by the contract."""
        ledger_api = cast(EthereumApi, ledger_api)
        contract_instance = cls.get_instance(ledger_api, contract_address)
        entries = contract_instance.events.Deliver.create_filter(
            fromBlock=from_block,
            toBlock=to_block,
        ).get_all_entries()
        deliver_events = list(
            {
                "tx_hash": entry.transactionHash.hex(),
                "block_number": entry.blockNumber,
                **entry["args"],
            }
            for entry in entries
        )
        return {"data": deliver_events}

    @classmethod
    def get_undelivered_reqs(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        from_block: BlockIdentifier = "earliest",
        to_block: BlockIdentifier = "latest",
        **kwargs: Any,
    ) -> JSONLike:
        """Get the requests that are not delivered."""
        requests: List[Dict[str, Any]] = cls.get_request_events(
            ledger_api, contract_address, from_block, to_block
        )["data"]
        delivers: List[Dict[str, Any]] = cls.get_deliver_events(
            ledger_api, contract_address, from_block, to_block
        )["data"]
        pending_tasks: List[Dict[str, Any]] = []
        for request in requests:
            if request["requestId"] not in [
                deliver["requestId"] for deliver in delivers
            ]:
                # store each requests in the pending_tasks list, make sure each req is stored once
                pending_tasks.append(request)
        return {"data": pending_tasks}
