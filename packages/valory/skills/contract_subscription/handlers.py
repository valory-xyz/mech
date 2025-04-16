# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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
from typing import Any

from web3 import Web3
from web3.types import TxReceipt

from packages.valory.protocols.websocket_client.message import WebsocketClientMessage
from packages.valory.skills.websocket_client.handlers import (
    SubscriptionStatus,
    WEBSOCKET_SUBSCRIPTION_STATUS,
)
from packages.valory.skills.websocket_client.handlers import (
    WebSocketHandler as BaseWebSocketHandler,
)


JOB_QUEUE = "pending_tasks"
DISCONNECTION_POINT = "disconnection_point"


class WebSocketHandler(BaseWebSocketHandler):
    """This class scaffolds a handler."""

    SUPPORTED_PROTOCOL = WebsocketClientMessage.protocol_id
    w3: Web3 = None
    contract = None

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the handler."""
        self.websocket_provider = kwargs.pop("websocket_provider")
        self.contract_to_monitor = kwargs.pop("contract_to_monitor")
        self.debounce_seconds = kwargs.pop("debounce_seconds")
        self.fallback_interval = kwargs.pop("fallback_interval")
        self.last_processed_time = 0

        super().__init__(**kwargs)

    def setup(self) -> None:
        """Implement the setup."""
        super().setup()
        self.context.shared_state["event_detected"] = False
        self.context.shared_state[JOB_QUEUE] = []
        self.context.shared_state[DISCONNECTION_POINT] = None
        self._last_processed_block = None

        # loads the contracts from the config file
        with open(
            "vendor/valory/contracts/agent_mech/build/AgentMech.json",
            "r",
            encoding="utf-8",
        ) as file:
            abi = json.load(file)["abi"]

        self.w3 = Web3(
            Web3.WebsocketProvider(
                self.websocket_provider,
                websocket_timeout=60,
                websocket_kwargs={"ping_interval": 20},
            )
        )
        self.contract = self.w3.eth.contract(address=self.contract_to_monitor, abi=abi)

    def handle(self, message: WebsocketClientMessage) -> None:
        """Handle message."""
        super().handle(message)

        # Track disconnection
        if self.context.shared_state[WEBSOCKET_SUBSCRIPTION_STATUS][
            message.subscription_id
        ] in (SubscriptionStatus.UNSUBSCRIBED, SubscriptionStatus.SUBSCRIBING):
            self.context.logger.info(
                f"Setting disconnection point to {self._last_processed_block}"
            )
            self.context.shared_state[DISCONNECTION_POINT] = self._last_processed_block

        now = time.time()

        # Debounce logic
        if now - self.last_processed_time >= self.debounce_seconds:
            if not self.context.shared_state.get("event_detected", False):
                self.context.logger.info("🔔 Debounced event detected. Setting flag to pull.")
                self.context.shared_state["event_detected"] = True
            else:
                self.context.logger.info("⏸ Event already detected. Waiting for it to be handled.")
            self.last_processed_time = now
        else:
            remaining = int(self.debounce_seconds - (now - self.last_processed_time))
            self.context.logger.info(f"⏳ Debounced: skipping event (wait {remaining}s)")
        # Fallback logic
        if (
            not self.context.shared_state.get("event_detected", False)
            and now - self.last_processed_time > self.fallback_interval
        ):
            self.context.logger.warning("🛟 Fallback triggered: no event in fallback interval. Forcing pull.")
            self.context.shared_state["event_detected"] = True
            self.last_processed_time = now
