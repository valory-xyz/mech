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

"""This module contains the dynamic_contribution contract definition."""

import logging
from enum import Enum

from eth_typing import ChecksumAddress
from eth_utils import event_abi_to_log_topic
from hexbytes import HexBytes
from typing import Any, Dict, List, cast

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea.crypto.base import LedgerApi
from aea_ledger_ethereum import EthereumApi
from web3.types import BlockIdentifier, FilterParams, TxReceipt
from web3._utils.events import get_event_data


PUBLIC_ID = PublicId.from_str("valory/agent_mech:0.1.0")

_logger = logging.getLogger(
    f"aea.packages.{PUBLIC_ID.author}.contracts.{PUBLIC_ID.name}.contract"
)


BATCH_PRIORITY_PASSED_DATA = {
    "abi": [
        {
            "inputs": [
                {
                    "internalType": "contract IMechMarketplace",
                    "name": "_marketplace",
                    "type": "address",
                },
                {"internalType": "address", "name": "me", "type": "address"},
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
    "bytecode": "0x608060405234801561001057600080fd5b5060405161081738038061081783398181016040528101906100329190610511565b60008151905060008167ffffffffffffffff81111561005457610053610398565b5b6040519080825280602002602001820160405280156100825781602001602082028036833780820191505090505b5090506000805b838110156102055760008773ffffffffffffffffffffffffffffffffffffffff1663cb261bec8784815181106100c2576100c1610580565b5b60200260200101516040518263ffffffff1660e01b81526004016100e691906105be565b608060405180830381865afa158015610103573d6000803e3d6000fd5b505050506040513d601f19601f820116820180604052508101906101279190610692565b90508673ffffffffffffffffffffffffffffffffffffffff16816000015173ffffffffffffffffffffffffffffffffffffffff1614806101715750806060015163ffffffff164210155b80156101ad5750600073ffffffffffffffffffffffffffffffffffffffff16816020015173ffffffffffffffffffffffffffffffffffffffff16145b156101f9578582815181106101c5576101c4610580565b5b60200260200101518484815181106101e0576101df610580565b5b602002602001018181525050826101f6906106ee565b92505b81600101915050610089565b5060008167ffffffffffffffff81111561022257610221610398565b5b6040519080825280602002602001820160405280156102505781602001602082028036833780820191505090505b50905060005b828110156102a35783818151811061027157610270610580565b5b602002602001015182828151811061028c5761028b610580565b5b602002602001018181525050806001019050610256565b506000816040516020016102b791906107f4565b60405160208183030381529060405290506020810180590381f35b6000604051905090565b600080fd5b600080fd5b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b6000610311826102e6565b9050919050565b600061032382610306565b9050919050565b61033381610318565b811461033e57600080fd5b50565b6000815190506103508161032a565b92915050565b61035f81610306565b811461036a57600080fd5b50565b60008151905061037c81610356565b92915050565b600080fd5b6000601f19601f8301169050919050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052604160045260246000fd5b6103d082610387565b810181811067ffffffffffffffff821117156103ef576103ee610398565b5b80604052505050565b60006104026102d2565b905061040e82826103c7565b919050565b600067ffffffffffffffff82111561042e5761042d610398565b5b602082029050602081019050919050565b600080fd5b6000819050919050565b61045781610444565b811461046257600080fd5b50565b6000815190506104748161044e565b92915050565b600061048d61048884610413565b6103f8565b905080838252602082019050602084028301858111156104b0576104af61043f565b5b835b818110156104d957806104c58882610465565b8452602084019350506020810190506104b2565b5050509392505050565b600082601f8301126104f8576104f7610382565b5b815161050884826020860161047a565b91505092915050565b60008060006060848603121561052a576105296102dc565b5b600061053886828701610341565b93505060206105498682870161036d565b925050604084015167ffffffffffffffff81111561056a576105696102e1565b5b610576868287016104e3565b9150509250925092565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052603260045260246000fd5b6105b881610444565b82525050565b60006020820190506105d360008301846105af565b92915050565b600080fd5b600063ffffffff82169050919050565b6105f7816105de565b811461060257600080fd5b50565b600081519050610614816105ee565b92915050565b6000608082840312156106305761062f6105d9565b5b61063a60806103f8565b9050600061064a8482850161036d565b600083015250602061065e8482850161036d565b60208301525060406106728482850161036d565b604083015250606061068684828501610605565b60608301525092915050565b6000608082840312156106a8576106a76102dc565b5b60006106b68482850161061a565b91505092915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052601160045260246000fd5b60006106f982610444565b91507fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff820361072b5761072a6106bf565b5b600182019050919050565b600081519050919050565b600082825260208201905092915050565b6000819050602082019050919050565b61076b81610444565b82525050565b600061077d8383610762565b60208301905092915050565b6000602082019050919050565b60006107a182610736565b6107ab8185610741565b93506107b683610752565b8060005b838110156107e75781516107ce8882610771565b97506107d983610789565b9250506001810190506107ba565b5085935050505092915050565b6000602082019050818103600083015261080e8184610796565b90509291505056fe",
}

TOPIC_BYTES = 32
TOPIC_CHARS = TOPIC_BYTES * 2
Ox = "0x"
Ox_CHARS = len(Ox)


class MechOperation(Enum):
    """Operation types."""

    CALL = 0
    DELEGATE_CALL = 1


def pad_address_for_topic(address: str) -> HexBytes:
    """Left-pad an Ethereum address to 32 bytes for use in a topic."""
    return HexBytes(Ox + address[Ox_CHARS:].zfill(TOPIC_CHARS))


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
                bytes.fromhex(data),
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
        event_abi = contract_instance.events.MarketplaceRequest().abi
        entries = cls.get_event_entries(
            ledger_api=ledger_api,
            event_abi=event_abi,
            address=contract_instance.address,
            from_block=from_block,
            to_block=to_block,
        )

        request_events = list(
            {
                "tx_hash": entry.transactionHash.hex(),
                "block_number": entry.blockNumber,
                **entry["args"],
                "sender": entry["args"]["requester"],
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
        event_abi = contract_instance.events.MarketplaceDeliver().abi
        entries = cls.get_event_entries(
            ledger_api=ledger_api,
            event_abi=event_abi,
            address=contract_instance.address,
            from_block=from_block,
            to_block=to_block,
        )

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
        my_mech: str,
        request_ids: List[int],
    ) -> Dict[str, Any]:
        """Check if requests are ready to be delivered."""
        # BatchPriorityData contract is a special contract used specifically for checking if the requests have passed
        # the priority timeout. It is not deployed anywhere, nor it needs to be deployed
        batch_workable_contract = ledger_api.api.eth.contract(
            abi=BATCH_PRIORITY_PASSED_DATA["abi"],
            bytecode=BATCH_PRIORITY_PASSED_DATA["bytecode"],
        )

        # Encode the input data (constructor params)
        encoded_input_data = ledger_api.api.codec.encode(
            ["address", "address", "uint256[]"],
            [contract_address, my_mech, request_ids],
        )

        # Concatenate the bytecode with the encoded input data to create the contract creation code
        contract_creation_code = batch_workable_contract.bytecode + encoded_input_data

        # Call the function with the contract creation code
        # Note that we are not sending any transaction, we are just calling the function
        # This is a special contract creation code that will return some result
        encoded_req_ids = ledger_api.api.eth.call({"data": contract_creation_code})

        # Decode the raw response
        # the decoding returns a Tuple with a single element so we need to access the first element of the tuple,
        request_ids = ledger_api.api.codec.decode(["uint256[]"], encoded_req_ids)[0]
        return dict(request_ids=request_ids)

    @classmethod
    def get_undelivered_reqs(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        my_mech: str,
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
        eligible_request_ids = cls.has_priority_passed(
            ledger_api, contract_address, my_mech, request_ids
        ).pop("request_ids")
        pending_tasks = [
            req for req in pending_tasks if req["requestId"] in eligible_request_ids
        ]
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

    @classmethod
    def get_encoded_data_for_request(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        request_id: bytes,
        data: str,
        delivery_rate: int,
    ) -> JSONLike:
        """Fetch info for a given request id."""
        contract_instance = cls.get_instance(ledger_api, contract_address)

        request_id_info = contract_instance.functions.mapRequestIdInfos(
            request_id
        ).call()

        final_delivery_rate = min(request_id_info[4], delivery_rate)
        encoded_data = ledger_api.api.codec.encode(
            ["uint256", "bytes"], [final_delivery_rate, data]
        )

        return dict(data=encoded_data)

    @classmethod
    def get_balance_tracker_for_mech_type(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        mech_type: str,
    ) -> JSONLike:
        """Fetch balance tracker address for a given mech type."""
        contract_instance = cls.get_instance(ledger_api, contract_address)

        balance_tracker_address = (
            contract_instance.functions.mapPaymentTypeBalanceTrackers(mech_type).call()
        )
        return dict(data=balance_tracker_address)

    @classmethod
    def get_event_entries(
        cls,
        ledger_api: EthereumApi,
        event_abi: Any,
        address: ChecksumAddress,
        from_block: BlockIdentifier = "earliest",
        to_block: BlockIdentifier = "latest",
    ) -> List:
        """Helper method to extract the events."""

        event_topic = event_abi_to_log_topic(event_abi)

        filter_params: FilterParams = {
            "fromBlock": from_block,
            "toBlock": to_block,
            "address": address,
            "topics": [event_topic],
        }

        w3 = ledger_api.api.eth
        logs = w3.get_logs(filter_params)
        entries = [get_event_data(w3.codec, event_abi, log) for log in logs]
        return entries
