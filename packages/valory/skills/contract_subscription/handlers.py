# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 Valory AG
#   Copyright 2023 eightballer
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

"""This package contains a scaffold of a handler."""

import json
import time

from aea.protocols.base import Message
from aea.skills.base import Handler
from web3 import Web3
from web3.types import TxReceipt

from packages.fetchai.protocols.default.message import DefaultMessage

JOB_QUEUE = "pending_tasks"
DISCONNECTION_POINT = "disconnection_point"


class WebSocketHandler(Handler):
    """This class scaffolds a handler."""

    SUPPORTED_PROTOCOL = DefaultMessage.protocol_id
    w3: Web3 = None
    contract = None

    def __init__(self, **kwargs) -> None:
        self.websocket_provider = kwargs.pop("websocket_provider")
        self.contract_to_monitor = kwargs.pop("contract_to_monitor")
        super().__init__(**kwargs)

    def setup(self) -> None:
        """Implement the setup."""
        self.context.shared_state[JOB_QUEUE] = []
        self.context.shared_state[DISCONNECTION_POINT] = None
        # loads the contracts from the config file
        with open("vendor/valory/contracts/agent_mech/build/AgentMech.json", "r", encoding="utf-8") as file:
            abi = json.load(file)['abi']

        self.w3 = Web3(Web3.HTTPProvider(self.websocket_provider))  # pylint: disable=C0103
        self.contract = self.w3.eth.contract(address=self.contract_to_monitor, abi=abi)

    def handle(self, message: Message) -> None:
        """
        Implement the reaction to an envelope.

        :param message: the message
        """
        self.context.logger.info(f"Received message: {message}")
        data = json.loads(message.content)
        if set(data.keys()) == {"id", "result", "jsonrpc"}:
            self.context.logger.info(f"Received response: {data}")
            return

        self.context.logger.info("Extracting data")
        tx_hash = data['params']['result']['transactionHash']
        no_args = True
        limit = 0
        while no_args and limit < 10:
            event_args, no_request = self._get_tx_args(tx_hash)
            if no_request:
                self.context.logger.info("Event not a Request.")
                break
            if len(event_args) == 0:
                self.context.logger.info(f"Could not get event args. tx_hash={tx_hash}")
                time.sleep(1)
                limit += 1
                return
            no_args = False
        if len(event_args) != 0:
            self.context.shared_state[JOB_QUEUE].append(event_args)
            self.context.logger.info(f"Added job to queue: {event_args}")

    def teardown(self) -> None:
        """Implement the handler teardown."""

    def _get_tx_args(self, tx_hash: str):
        """Get the transaction arguments."""
        try:
            tx_receipt: TxReceipt = self.w3.eth.get_transaction_receipt(tx_hash)
            self.context.shared_state[DISCONNECTION_POINT] = tx_receipt["blockNumber"]
            rich_logs = self.contract.events.Request().processReceipt(tx_receipt)  # type: ignore
            return dict(rich_logs[0]['args']), False

        except Exception as exc:  # pylint: disable=W0718
            self.context.logger.error(
                f"An exception occurred while trying to get the transaction arguments for {tx_hash}: {exc}"
            )
            return {}, True
