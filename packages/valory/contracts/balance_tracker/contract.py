# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025-2026 Valory AG
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

"""This module contains the balance_tracker contract definition."""

import logging

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea.crypto.base import LedgerApi
from aea_ledger_ethereum import EthereumApi

from packages.valory.contracts.olas_mech.contract import (
    SIMULATION_OK_KEY,
    SIMULATION_RPC_DEADLINE_SECONDS,
    SIMULATION_TIMED_OUT_KEY,
    simulate_with_deadline,
)

PUBLIC_ID = PublicId.from_str("valory/balance_tracker:0.1.0")

_logger = logging.getLogger(
    f"aea.packages.{PUBLIC_ID.author}.contracts.{PUBLIC_ID.name}.contract"
)


class BalanceTrackerContract(Contract):
    """The scaffold contract class for a smart contract."""

    contract_id = PublicId.from_str("valory/balance_tracker:0.1.0")

    @classmethod
    def get_mech_balance(
        cls, ledger_api: LedgerApi, contract_address: str, mech_address: str
    ) -> JSONLike:
        """Get mech balance"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        mech_balance = contract_instance.functions.mapMechBalances(mech_address).call()
        return {"mech_balance": mech_balance}

    @classmethod
    def get_requester_balance(
        cls, ledger_api: LedgerApi, contract_address: str, requester: str
    ) -> JSONLike:
        """Get requester balance."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        requester_balance = contract_instance.functions.mapRequesterBalances(
            requester
        ).call()
        return {"requester_balance": requester_balance}

    @classmethod
    def get_max_fee_factor(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
    ) -> JSONLike:
        """Get mech balance"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        max_fee_factor = contract_instance.functions.MAX_FEE_FACTOR().call()
        return {"max_fee_factor": max_fee_factor}

    @classmethod
    def get_token_address(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get tx data"""

        contract_instance = cls.get_instance(ledger_api, contract_address)
        token_address = contract_instance.functions.token().call()
        return {"token_address": token_address}  # type: ignore

    @classmethod
    def get_token_credit_ratio(
        cls, ledger_api: LedgerApi, contract_address: str
    ) -> JSONLike:
        """Get token credit ratio"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        token_credit_ratio = contract_instance.functions.tokenCreditRatio().call()
        return {"token_credit_ratio": token_credit_ratio}  # type: ignore

    @classmethod
    def simulate_tx(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        sender_address: str,
        data: str,
        deadline_seconds: float = SIMULATION_RPC_DEADLINE_SECONDS,
    ) -> JSONLike:
        """Simulate the transaction under a wall-clock deadline.

        :param ledger_api: the ledger api.
        :param contract_address: the balance tracker address.
        :param sender_address: the address the simulated call is sent from.
        :param data: the calldata to simulate, hex-encoded.
        :param deadline_seconds: wall-clock upper bound for the call, in seconds.
        :return: ``{"data": <simulation_ok>, "simulation_timed_out": <bool>}``.
        """
        result = simulate_with_deadline(
            ledger_api, contract_address, sender_address, data, deadline_seconds
        )
        return {
            "data": result[SIMULATION_OK_KEY],
            SIMULATION_TIMED_OUT_KEY: result[SIMULATION_TIMED_OUT_KEY],
        }

    @classmethod
    def get_process_payment_tx(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        sender_address: str,
        mech_address: str,
        deadline_seconds: float = SIMULATION_RPC_DEADLINE_SECONDS,
    ) -> JSONLike:
        """Get tx data"""

        contract_instance = cls.get_instance(ledger_api, contract_address)
        tx_data = contract_instance.encode_abi(
            abi_element_identifier="processPaymentByMultisig",
            args=[mech_address],
        )
        simulation = simulate_with_deadline(
            ledger_api, contract_address, sender_address, tx_data, deadline_seconds
        )
        return {"data": bytes.fromhex(tx_data[2:]), **simulation}  # type: ignore

    @classmethod
    def build_deposit_for_data(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        account: str,
        amount: int,
    ) -> JSONLike:
        """Encode depositFor(account, amount) for a Safe multisend batch."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encode_abi(
            abi_element_identifier="depositFor",
            args=[account, amount],
        )
        return {"data": bytes.fromhex(data[2:])}  # type: ignore
