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

from aea.mail.base import Envelope
from aea.skills.behaviours import OneShotBehaviour

from packages.eightballer.connections.websocket_client.connection import \
    CONNECTION_ID
from packages.fetchai.protocols.default.message import DefaultMessage

DEFAULT_ENCODING = "utf-8"


class SubscriptionBehaviour(OneShotBehaviour):
    """This class scaffolds a behaviour."""

    def setup(self) -> None:
        """Implement the setup."""

    def act(self) -> None:
        """Implement the act."""
        for contract in self._contracts:
            subscription_msg_template = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_subscribe",
                "params": ["logs", {"address": contract}]
            }
            self.context.logger.info(f"Sending subscription to: {contract}")
            self._create_subscription(bytes(json.dumps(subscription_msg_template), DEFAULT_ENCODING))
        self.context.logger.info("Act completed.")

    def teardown(self) -> None:
        """Implement the task teardown."""

    def _create_subscription(self, content: bytes):
        """Create a subscription."""
        msg, _ = self.context.default_dialogues.create(
            counterparty=str(CONNECTION_ID),
            performative=DefaultMessage.Performative.BYTES,
            content=content,
        )
        # pylint: disable=W0212
        msg._sender = str(self.context.skill_id)
        envelope = Envelope(to=msg.to, sender=msg._sender, message=msg)
        self.context.outbox.put(envelope)

    def __init__(self, *args, **kwargs):
        """Initialise the agent."""
        self._contracts = kwargs.pop('contracts', [])
        super().__init__(*args, **kwargs)
