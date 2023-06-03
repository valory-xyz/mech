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

"""This package contains a scaffold of a behaviour."""

import json
from typing import Any, List, Optional, cast

from aea.mail.base import Envelope
from aea.skills.behaviours import SimpleBehaviour

from packages.fetchai.protocols.default.message import DefaultMessage
from packages.valory.connections.websocket_client.connection import (
    PUBLIC_ID,
    WebSocketClient,
)
from packages.valory.skills.contract_subscription.handlers import DISCONNECTION_POINT


DEFAULT_ENCODING = "utf-8"
WEBSOCKET_CLIENT_CONNECTION_NAME = "websocket_client"


class SubscriptionBehaviour(SimpleBehaviour):
    """This class scaffolds a behaviour."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the agent."""
        self._contracts: List[str] = kwargs.pop("contracts", [])
        self._ws_client_connection: Optional[WebSocketClient] = None
        self._subscription_required: bool = True
        self._missed_parts: bool = False
        super().__init__(**kwargs)

    def setup(self) -> None:
        """Implement the setup."""
        use_polling = self.context.params.use_polling
        if use_polling:
            # if we are using polling, then we don't set up an contract subscription
            return
        for (
            connection
        ) in self.context.outbox._multiplexer.connections:  # pylint: disable=W0212
            if connection.component_id.name == WEBSOCKET_CLIENT_CONNECTION_NAME:
                self._ws_client_connection = cast(WebSocketClient, connection)

    def act(self) -> None:
        """Implement the act."""
        use_polling = self.context.params.use_polling
        if use_polling:
            # do nothing if we are polling
            return
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
            if disconnection_point is not None:
                self._missed_parts = True

        if is_connected and self._missed_parts:
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
            counterparty=str(PUBLIC_ID),
            performative=DefaultMessage.Performative.BYTES,
            content=content,
        )
        # pylint: disable=W0212
        msg._sender = str(self.context.skill_id)
        envelope = Envelope(to=msg.to, sender=msg._sender, message=msg)
        self.context.outbox.put(envelope)
