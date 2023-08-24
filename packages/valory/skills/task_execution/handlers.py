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

"""This package contains a scaffold of a handler."""

from typing import Any, Dict, List, cast

from aea.protocols.base import Message
from aea.skills.base import Handler


from packages.valory.connections.ledger.connection import (
    PUBLIC_ID as LEDGER_CONNECTION_PUBLIC_ID,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.acn_data_share import AcnDataShareMessage
from packages.valory.protocols.ipfs import IpfsMessage
from packages.valory.skills.task_execution.models import Params

PENDING_TASKS = "pending_tasks"
DONE_TASKS = "ready_tasks"

LEDGER_API_ADDRESS = str(LEDGER_CONNECTION_PUBLIC_ID)


class AcnHandler(Handler):
    """ACN API message handler."""

    SUPPORTED_PROTOCOL = AcnDataShareMessage.protocol_id

    def setup(self) -> None:
        """Set up the handler."""
        self.context.logger.info("AcnHandler: setup method called.")

    def handle(self, message: Message) -> None:
        """Handle the message."""
        # we don't respond to ACN messages at this point
        self.context.logger.info(f"Received message: {message}")


class IpfsHandler(Handler):
    """IPFS API message handler."""

    SUPPORTED_PROTOCOL = IpfsMessage.protocol_id

    def setup(self) -> None:
        """Setup the IPFS handler."""
        self.context.logger.info("IPFSHandler: setup method called.")

    @property
    def params(self) -> Params:
        """Get the parameters."""
        return cast(Params, self.context.params)

    def handle(self, message: Message) -> None:
        """
        Implement the reaction to an IPFS message.

        :param message: the message
        """
        self.context.logger.info(f"Received message: {message}")
        ipfs_msg = cast(IpfsMessage, message)
        if ipfs_msg.performative == IpfsMessage.Performative.ERROR:
            self.context.logger.warning(
                f"IPFS Message performative not recognized: {ipfs_msg.performative}"
            )
            return

        dialogue = self.context.ipfs_dialogues.update(ipfs_msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        callback = self.params.req_to_callback[nonce]
        callback(ipfs_msg, dialogue)
        self.params.in_flight_req = False


class ContractHandler(Handler):
    """Contract API message handler."""

    SUPPORTED_PROTOCOL = ContractApiMessage.protocol_id

    def setup(self) -> None:
        """Setup the contract handler."""
        self.context.shared_state[PENDING_TASKS] = []
        self.context.shared_state[DONE_TASKS] = []

    def teardown(self) -> None:
        """Teardown the handler."""
        self.context.logger.info("ContractHandler: teardown called.")

    @property
    def pending_tasks(self) -> List[Dict[str, Any]]:
        """Get pending_tasks."""
        return self.context.shared_state[PENDING_TASKS]

    @property
    def params(self) -> Params:
        """Get the parameters."""
        return cast(Params, self.context.params)

    def handle(self, message: Message) -> None:
        """
        Implement the reaction to a contract message.

        :param message: the message
        """
        self.context.logger.info(f"Received message: {message}")
        contract_api_msg = cast(ContractApiMessage, message)
        if contract_api_msg.performative != ContractApiMessage.Performative.STATE:
            self.context.logger.warning(
                f"Contract API Message performative not recognized: {contract_api_msg.performative}"
            )
            self.params.in_flight_req = False
            return

        body = contract_api_msg.state.body
        self._handle_get_undelivered_reqs(body)
        self.params.in_flight_req = False

    def _handle_get_undelivered_reqs(
        self, body: Dict[str, Any]
    ) -> None:
        """Handle get undelivered reqs."""
        reqs = body.get("data", [])
        if len(reqs) == 0:
            return

        self.context.logger.info(f"Received {len(reqs)} new requests.")
        self.pending_tasks.extend(reqs)
        self.from_block = max([req["block_number"] for req in reqs]) + 1
        self.context.logger.info(f"Monitoring new reqs from block {self.from_block}")
