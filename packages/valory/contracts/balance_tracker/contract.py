# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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
from typing import Any, cast

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea.crypto.base import LedgerApi
from aea_ledger_ethereum import EthereumApi


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
    def get_process_payment_tx(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        sender_address: str,
        mech_address: str,
    ) -> JSONLike:
        """Get tx data"""

        contract_instance = cls.get_instance(ledger_api, contract_address)
        tx_data = contract_instance.encodeABI(
            fn_name="processPaymentByMultisig",
            args=[mech_address],
        )
        simulation_ok = cls.simulate_tx(
            ledger_api, contract_address, sender_address, tx_data
        ).pop("data")
        return {"data": bytes.fromhex(tx_data[2:]), "simulation_ok": simulation_ok}  # type: ignore
