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

"""Websocket client connection."""

import asyncio
from threading import Thread, Event
from typing import Any, Optional

import websocket
from aea.configurations.base import PublicId
from aea.connections.base import Connection, ConnectionStates
from aea.mail.base import Envelope

from packages.fetchai.protocols.default.message import DefaultMessage

CONNECTION_ID = PublicId.from_str("eightballer/websocket_client:0.1.0")


class WebSocketClient(Connection):
    """Proxy to the functionality of the SDK or API."""

    connection_id = CONNECTION_ID
    _new_messages: list
    _endpoint: str
    _thread: Thread
    _wss: websocket.WebSocket

    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the connection.

        The configuration must be specified if and only if the following
        parameters are None: connection_id, excluded_protocols or restricted_to_protocols.

        Possible keyword arguments:
        - configuration: the connection configuration.
        - data_dir: directory where to put local files.
        - identity: the identity object held by the agent.
        - crypto_store: the crypto store for encrypted communication.
        - restricted_to_protocols: the set of protocols ids of the only supported protocols for this connection.
        - excluded_protocols: the set of protocols ids that we want to exclude for this connection.

        :param kwargs: keyword arguments passed to component base
        """
        self._new_messages = []
        self._event: Optional[Event] = None
        self._endpoint = kwargs["configuration"].config["endpoint"]
        self._target_skill_id = kwargs["configuration"].config["target_skill_id"]
        assert self._endpoint is not None, "Endpoint must be provided!"
        super().__init__(**kwargs)  # pragma: no cover

    async def connect(self) -> None:
        """
        Set up the connection.

        In the implementation, remember to update 'connection_status' accordingly.
        """
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                self._wss = websocket.create_connection(self._endpoint)
                self.state = ConnectionStates.connected
                self.logger.info("Websocket connection established.")
                self._event = Event()
                self._thread = Thread(target=self._run, args=(self._event,))
                self._thread.start()
                return
            except Exception as exception:  # pylint: disable=W0718
                self.logger.error(f"Failed to establish WebSocket connection: {exception}")
                self.state = ConnectionStates.disconnected
                retries += 1
                await asyncio.sleep(self.RETRY_DELAY)

        raise Exception(f"Failed to establish connection after {self.MAX_RETRIES} attempts.")  # pylint: disable=W0719

    async def disconnect(self) -> None:
        """
        Tear down the connection.

        In the implementation, remember to update 'connection_status' accordingly.
        """
        self.logger.debug("Disconnecting...")  # pragma: no cover
        self._wss.close()
        self._event.set()
        self.state = ConnectionStates.disconnected

    async def send(self, envelope: Envelope):
        """
        Send an envelope.

        :param envelope: the envelope to send.
        """
        if self.state != ConnectionStates.connected:
            raise Exception("Cannot send message. Connection is not established.")  # pylint: disable=W0719

        self.logger.debug("Sending content from envelope...")
        context = envelope.message.content
        try:
            self._wss.send(context)
        except websocket.WebSocketConnectionClosedException:
            self.logger.error("Websocket connection closed.")
            self.state = ConnectionStates.disconnected

    async def receive(self, *args: Any, **kwargs: Any):  # noqa: V107
        """
        Receive an envelope. Blocking.

        :param args: arguments to receive
        :param kwargs: keyword arguments to receive
        :return: the envelope received, if present.  # noqa: DAR202
        """
        if self.state != ConnectionStates.connected:
            raise Exception("Cannot receive message. Connection is not established.")  # pylint: disable=W0719

        if self._new_messages:
            new_msg = self._new_messages.pop()
            self.logger.debug(f"Received message from wss connection: {new_msg}")
            return self._from_wss_msg_to_envelope(new_msg)

    def _from_wss_msg_to_envelope(self, msg: str):
        """Convert a message from the wss to an envelope."""
        msg = DefaultMessage(
            performative=DefaultMessage.Performative.BYTES,
            content=bytes(msg, "utf-8"),
        )
        envelope = Envelope(
            to=self._target_skill_id, sender=str(self.connection_id), message=msg
        )
        return envelope

    async def _run(self, event: Event) -> None:
        """Run a loop to receive messages from the wss."""
        while True:
            try:
                msg = self._wss.recv()
                self._new_messages.append(msg)
            except websocket.WebSocketConnectionClosedException:
                event.set()
                self.logger.error("Websocket connection closed.")
                self.state = ConnectionStates.disconnected

            # Start the reconnection process
            if self.state == ConnectionStates.disconnected:
                self.logger.info("Attempting to reconnect...")
                await self._reconnect()

    async def _reconnect(self):
        """Attempt to reconnect."""
        retries = 0
        # these are the retry attempts for reconnecting; also the called connection logic has its own retry logic
        while retries < self.MAX_RETRIES:
            try:
                self.state = ConnectionStates.connecting
                await self.connect()
                self.state = ConnectionStates.connected
                self.logger.info("Reconnected successfully.")
                return
            except Exception as exception:  # pylint: disable=W0718
                self.state = ConnectionStates.disconnected
                self.logger.error(f"Failed to reconnect: {exception}")
                retries += 1
                await asyncio.sleep(self.RETRY_DELAY)
        self.logger.error(f"Failed to reconnect after {self.MAX_RETRIES} attempts.")
        raise Exception(f"Failed to reconnect after {self.MAX_RETRIES} attempts.")  # pylint: disable=W0719
