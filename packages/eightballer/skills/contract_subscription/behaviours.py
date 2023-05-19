# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
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

"""This package contains a scaffold of a behaviour."""

import json
from typing import List, Optional, cast

from aea.mail.base import Envelope
from aea.skills.behaviours import SimpleBehaviour

from packages.eightballer.connections.websocket_client.connection import (
    CONNECTION_ID,
    WebSocketClient,
)
from packages.eightballer.skills.contract_subscription.handlers import (
    DISCONNECTION_POINT,
)
from packages.fetchai.protocols.default.message import DefaultMessage

DEFAULT_ENCODING = "utf-8"
WEBSOCKET_CLIENT_CONNECTION_NAME = "websocket_client"


class SubscriptionBehaviour(SimpleBehaviour):
    """This class scaffolds a behaviour."""

    def setup(self) -> None:
        """Implement the setup."""
        for connection in self.context.outbox._multiplexer.connections:
            if connection.component_id.name == WEBSOCKET_CLIENT_CONNECTION_NAME:
                self._ws_client_connection = cast(WebSocketClient, connection)

    def act(self) -> None:
        """Implement the act."""
        is_connected = cast(WebSocketClient, self._ws_client_connection).is_connected
        disconnection_point = self.context.shared_state.get(DISCONNECTION_POINT, None)

        if is_connected and self._subscription_required:
            # we only subscribe once, because the envelope will remain in the multiplexer until handled
            for contract in self._contracts:
                subscription_msg_template = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_subscribe",
                    "params": ["logs", {"address": contract}],
                }
                self.context.logger.info(f"Sending subscription to: {contract}")
                self._create_call(
                    bytes(json.dumps(subscription_msg_template), DEFAULT_ENCODING)
                )
            self._subscription_required = False
            return

        if is_connected and disconnection_point is not None:
            # if we are connected and have a disconnection point, then we need to fetch the parts that were missed
            for contract in self._contracts:
                filter_msg_template = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_newFilter",
                    "params": [{"fromBlock": disconnection_point, "address": contract}],
                }
                self.context.logger.info(f"Creating filter to: {contract}")
                self._create_call(
                    bytes(json.dumps(filter_msg_template), DEFAULT_ENCODING)
                )
            self.context.logger.info(
                "Getting parts that were missed while disconnected."
            )
            return

        if (
            not is_connected
            and not self._subscription_required
            and disconnection_point is not None
        ):
            self.context.logger.warning(
                f"Disconnection detected on block {disconnection_point}."
            )

        if not is_connected:
            self._subscription_required = True

    def _create_call(self, content: bytes) -> None:
        """Create a call."""
        msg, _ = self.context.default_dialogues.create(
            counterparty=str(CONNECTION_ID),
            performative=DefaultMessage.Performative.BYTES,
            content=content,
        )
        # pylint: disable=W0212
        msg._sender = str(self.context.skill_id)
        envelope = Envelope(to=msg.to, sender=msg._sender, message=msg)
        self.context.outbox.put(envelope)

    def __init__(self, **kwargs):
        """Initialise the agent."""
        self._contracts: List[str] = kwargs.pop("contracts", [])
        self._ws_client_connection: Optional[WebSocketClient] = None
        self._subscription_required: bool = True
        super().__init__(**kwargs)
