# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, cast

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea.crypto.base import LedgerApi
from aea_ledger_ethereum import EthereumApi
from web3 import Web3
from web3.types import BlockIdentifier, TxReceipt


PUBLIC_ID = PublicId.from_str("valory/agent_mech:0.1.0")

_logger = logging.getLogger(
    f"aea.packages.{PUBLIC_ID.author}.contracts.{PUBLIC_ID.name}.contract"
)


BATCH_PRIORITY_PASSED_DATA = {
    "abi": [
        {
            "inputs": [
                {
                    "internalType": "IMechMarketplace",
                    "name": "_marketplace",
                    "type": "address",
                },
                {
                    "internalType": "uint256[]",
                    "name": "_requestIds",
                    "type": "uint256[]",
                },
            ],
            "stateMutability": "nonpayable",
            "type": "constructor",
        }
    ],
    "bytecode": "0x608060405234801561001057600080fd5b5060405161072b38038061072b833981810160405281019061003291906104af565b60008151905060008167ffffffffffffffff81111561005457610053610340565b5b6040519080825280602002602001820160405280156100825781602001602082028036833780820191505090505b5090506000805b838110156101ab578573ffffffffffffffffffffffffffffffffffffffff16639f4713038683815181106100c0576100bf61050b565b5b60200260200101516040518263ffffffff1660e01b81526004016100e49190610549565b602060405180830381865afa92505050801561011e57506040513d601f19601f8201168201806040525081019061011b919061059c565b60015b156101a057801561019e5785828151811061013c5761013b61050b565b5b60200260200101518484815181106101575761015661050b565b5b602002602001019073ffffffffffffffffffffffffffffffffffffffff16908173ffffffffffffffffffffffffffffffffffffffff16815250508261019b90610602565b92505b505b806001019050610089565b5060008167ffffffffffffffff8111156101c8576101c7610340565b5b6040519080825280602002602001820160405280156101f65781602001602082028036833780820191505090505b50905060005b82811015610277578381815181106102175761021661050b565b5b60200260200101518282815181106102325761023161050b565b5b602002602001019073ffffffffffffffffffffffffffffffffffffffff16908173ffffffffffffffffffffffffffffffffffffffff16815250508060010190506101fc565b5060008160405160200161028b9190610708565b60405160208183030381529060405290506020810180590381f35b6000604051905090565b600080fd5b600080fd5b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b60006102e5826102ba565b9050919050565b60006102f7826102da565b9050919050565b610307816102ec565b811461031257600080fd5b50565b600081519050610324816102fe565b92915050565b600080fd5b6000601f19601f8301169050919050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052604160045260246000fd5b6103788261032f565b810181811067ffffffffffffffff8211171561039757610396610340565b5b80604052505050565b60006103aa6102a6565b90506103b6828261036f565b919050565b600067ffffffffffffffff8211156103d6576103d5610340565b5b602082029050602081019050919050565b600080fd5b6103f5816102da565b811461040057600080fd5b50565b600081519050610412816103ec565b92915050565b600061042b610426846103bb565b6103a0565b9050808382526020820190506020840283018581111561044e5761044d6103e7565b5b835b8181101561047757806104638882610403565b845260208401935050602081019050610450565b5050509392505050565b600082601f8301126104965761049561032a565b5b81516104a6848260208601610418565b91505092915050565b600080604083850312156104c6576104c56102b0565b5b60006104d485828601610315565b925050602083015167ffffffffffffffff8111156104f5576104f46102b5565b5b61050185828601610481565b9150509250929050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052603260045260246000fd5b610543816102da565b82525050565b600060208201905061055e600083018461053a565b92915050565b60008115159050919050565b61057981610564565b811461058457600080fd5b50565b60008151905061059681610570565b92915050565b6000602082840312156105b2576105b16102b0565b5b60006105c084828501610587565b91505092915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052601160045260246000fd5b6000819050919050565b600061060d826105f8565b91507fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff820361063f5761063e6105c9565b5b600182019050919050565b600081519050919050565b600082825260208201905092915050565b6000819050602082019050919050565b61067f816102da565b82525050565b60006106918383610676565b60208301905092915050565b6000602082019050919050565b60006106b58261064a565b6106bf8185610655565b93506106ca83610666565b8060005b838110156106fb5781516106e28882610685565b97506106ed8361069d565b9250506001810190506106ce565b5085935050505092915050565b6000602082019050818103600083015261072281846106aa565b90509291505056fe",
}


class MechOperation(Enum):
    """Operation types."""

    CALL = 0
    DELEGATE_CALL = 1


class MechMarketplaceContract(Contract):
    """The scaffold contract class for a smart contract."""

    contract_id = PublicId.from_str("valory/mech_marketplace:0.1.0")

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
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        sender_address: str,
        request_id: int,
        data: str,
        delivery_mech_staking_instance: str,
        delivery_mech_service_id: int,
    ) -> JSONLike:
        """
        Deliver a response to a request.

        :param ledger_api: LedgerApi object
        :param contract_address: the address of the token to be used
        :param sender_address: the address of the sender
        :param request_id: the id of the target request
        :param data: the response data
        :param delivery_mech_staking_instance: the staking instance
        :param delivery_mech_service_id: the service id
        :return: the deliver data
        """
        ledger_api = cast(EthereumApi, ledger_api)
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI(
            fn_name="deliverMarketplace",
            args=[
                request_id,
                data,
                delivery_mech_staking_instance,
                delivery_mech_service_id,
            ],

        )

        simulation_ok = cls.simulate_tx(
            ledger_api, contract_address, sender_address, data
        ).pop("data")
        return {"data": bytes.fromhex(data[2:]), "simulation_ok": simulation_ok}  # type: ignore

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
        entries = contract_instance.events.MarketplaceRequest.create_filter(
            fromBlock=from_block,
            toBlock=to_block,
        ).get_all_entries()

        request_events = list(
            {
                "tx_hash": entry.transactionHash.hex(),
                "block_number": entry.blockNumber,
                **entry["args"],
                "contract_address": contract_address,
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
        entries = contract_instance.events.MarketplaceDeliver.create_filter(
            fromBlock=from_block,
            toBlock=to_block,
        ).get_all_entries()

        request_events = list(
            {
                "tx_hash": entry.transactionHash.hex(),
                "block_number": entry.blockNumber,
                **entry["args"],
                "contract_address": contract_address,
            }
            for entry in entries
        )
        return {"data": request_events}

    @classmethod
    def process_tx_receipt(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        tx_receipt: TxReceipt,
    ) -> JSONLike:
        """Process transaction receipt to filter contract events."""

        ledger_api = cast(EthereumApi, ledger_api)
        contract_instance = cls.get_instance(ledger_api, contract_address)
        event, *_ = contract_instance.events.Request().processReceipt(tx_receipt)
        return dict(event["args"])


    @classmethod
    def has_priority_passed(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        request_ids: List[int],
    ) -> Dict[str, Any]:
        """Check the priority of the requests."""
        # BatchPriorityData contract is a special contract used specifically for checking if the requests have passed
        # the priority timeout. It is not deployed anywhere, nor it needs to be deployed
        batch_workable_contract = ledger_api.api.eth.contract(
            abi=BATCH_PRIORITY_PASSED_DATA["abi"], bytecode=BATCH_PRIORITY_PASSED_DATA["bytecode"]
        )

        # Encode the input data (constructor params)
        encoded_input_data = ledger_api.api.codec.encode_abi(
            ["address", "address[]"], [contract_address, request_ids]
        )

        # Concatenate the bytecode with the encoded input data to create the contract creation code
        contract_creation_code = batch_workable_contract.bytecode + encoded_input_data

        # Call the function with the contract creation code
        # Note that we are not sending any transaction, we are just calling the function
        # This is a special contract creation code that will return some result
        encoded_strategies = ledger_api.api.eth.call({"data": contract_creation_code})

        # Decode the raw response
        # the decoding returns a Tuple with a single element so we need to access the first element of the tuple,
        request_ids = ledger_api.api.codec.decode_abi(
            ["uint256[]"], encoded_strategies
        )[0]
        return dict(request_ids=request_ids)


    @classmethod
    def get_undelivered_reqs(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        from_block: BlockIdentifier = "earliest",
        to_block: BlockIdentifier = "latest",
        max_block_window: int = 1000,
        **kwargs: Any,
    ) -> JSONLike:
        """Get the requests that are not delivered."""
        if from_block == "earliest":
            from_block = 0

        current_block = ledger_api.api.eth.block_number
        requests, delivers = [], []
        for from_block_batch in range(int(from_block), current_block, max_block_window):
            to_block_batch = (from_block_batch + max_block_window) - 1
            if to_block_batch >= current_block:
                to_block_batch = "latest"
            requests_batch: List[Dict[str, Any]] = cls.get_request_events(
                ledger_api, contract_address, from_block_batch, to_block_batch
            )["data"]
            delivers_batch: List[Dict[str, Any]] = cls.get_deliver_events(
                ledger_api, contract_address, from_block_batch, to_block_batch
            )["data"]
            requests.extend(requests_batch)
            delivers.extend(delivers_batch)
        pending_tasks: List[Dict[str, Any]] = []
        for request in requests:
            if request["requestId"] not in [
                deliver["requestId"] for deliver in delivers
            ]:
                # store each requests in the pending_tasks list, make sure each req is stored once
                pending_tasks.append(request)

        request_ids = [req["requestId"] for req in pending_tasks]
        eligible_request_ids = cls.has_priority_passed(ledger_api, contract_address, request_ids).pop("request_ids")
        pending_tasks = [req for req in pending_tasks if req["requestId"] in eligible_request_ids]
        return {"data": pending_tasks}


    @classmethod
    def simulate_tx(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        sender_address: str,
        data: str,
    ) -> JSONLike:
        """Simulate the transaction."""
        try:
            ledger_api.api.eth.call(
                {
                    "from": ledger_api.api.to_checksum_address(sender_address),
                    "to": ledger_api.api.to_checksum_address(contract_address),
                    "data": data,
                }
            )
            simulation_ok = True
        except Exception as e:
            _logger.info(f"Simulation failed: {str(e)}")
            simulation_ok = False

        return dict(data=simulation_ok)
