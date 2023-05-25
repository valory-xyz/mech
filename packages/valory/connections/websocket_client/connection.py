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

"""Websocket client connection."""

import asyncio
from contextlib import suppress
from time import time
from typing import Any, Optional

import aiohttp
from aea.configurations.base import PublicId
from aea.connections.base import Connection, ConnectionStates
from aea.mail.base import Envelope

from packages.fetchai.protocols.default.message import DefaultMessage

CONNECTION_ID = PublicId.from_str("valory/websocket_client:0.1.0")


class WebSocketHandler:
    CONNECT_ATTEMPTS = 5
    RECONNECT_DELAY = 0.1

    def __init__(self, url: str):
        self.url = url

        self._ws = None
        self._connected_future = None
        self._connection_exception = None
        self._recv_queue = asyncio.Queue()
        self._send_queue = asyncio.Queue()
        self._connection_task = None

    async def _wait_connected(self):
        if not self._connected_future:
            raise ValueError("Call connect first")
        await self._connected_future
        if self._connection_exception:
            raise self._connection_exception

    async def send(self, msg):
        await self._wait_connected()
        await self._send_queue.put(msg)

    async def receive(self):
        await self._wait_connected()
        msg = await self._recv_queue.get()
        return msg

    async def _set_connection(self):
        exc = None
        for _ in range(self.CONNECT_ATTEMPTS):
            try:
                self._ws = await self._session.ws_connect(self.url)
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                exc = e
                await asyncio.sleep(self.RECONNECT_DELAY)

        self._connection_exception = exc

    async def _connection_loop(self):
        send_task = None
        try:
            await self._set_connection()
            self._connected_future.set_result(True)

            if self._connection_exception:
                return

            send_task = asyncio.ensure_future(self._send_loop())

            while True:  # till cancelled
                try:
                    while True:
                        msg = await self._ws.receive()
                        if msg.type in (
                            aiohttp.WSMsgType.ERROR,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSE,
                        ):
                            raise ValueError("socket closed. will reconnect")
                        await self._recv_queue.put(msg.data)

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    await self._set_connection()
                    if self._connection_exception:  # failed!
                        break
        finally:
            if send_task and not send_task.done():
                send_task.cancel()
                with suppress(asyncio.CancelledError):
                    await send_task

            if self._ws:
                # stupid trick
                with suppress(asyncio.exceptions.TimeoutError):
                    await asyncio.wait_for(self._ws.close(), timeout=0.1)

            await self._session.close()

    async def _send_loop(self):
        """
        Just keep trying to get and send message till loop cancelled. wait connection restored for 1 second and try again.
        """
        while True:  # till not cancelled
            msg = await self._send_queue.get()
            while True:  # till not cancelled
                try:
                    await self._ws.send_str(msg)
                except asyncio.CancelledError:  # exit
                    raise
                except Exception as e:
                    await asyncio.sleep(0.1)

    async def connect(self):
        self._session = aiohttp.ClientSession()
        self._connected_future = asyncio.Future()
        self._connection_task = asyncio.ensure_future(self._connection_loop())
        try:
            await self._wait_connected()
        except:
            if not self._connection_task.done():
                self._connection_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._connection_task
            raise  # raise original exception

    async def disconnect(self):
        if not self._connection_task:
            return
        if self._connection_task.done():
            return
        self._connection_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._connection_task


class WebSocketClient(Connection):
    """Proxy to the functionality of the SDK or API."""

    connection_id = CONNECTION_ID
    _endpoint: str

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

        self._endpoint = kwargs["configuration"].config["endpoint"]
        self._target_skill_id = kwargs["configuration"].config["target_skill_id"]
        assert self._endpoint is not None, "Endpoint must be provided!"

        self._websocket_handler = WebSocketHandler(self._endpoint)
        super().__init__(**kwargs)  # pragma: no cover

    async def connect(self) -> None:
        """
        Set up the connection.

        In the implementation, remember to update 'connection_status' accordingly.
        """

        try:
            await self._websocket_handler.connect()
            self.state = ConnectionStates.connected
        except:
            self.state = ConnectionStates.disconnected
            raise

    async def disconnect(self) -> None:
        """
        Tear down the connection.

        In the implementation, remember to update 'connection_status' accordingly.
        """
        self.logger.debug("Disconnecting...")  # pragma: no cover
        try:
            await self._websocket_handler.disconnect()
        finally:
            self.state = ConnectionStates.disconnected

    async def send(self, envelope: Envelope):
        """
        Send an envelope.

        :param envelope: the envelope to send.
        """
        if self.state == ConnectionStates.disconnected:
            raise Exception(  # pylint: disable=W0719
                "Cannot receive message. Connection is not established."
            )

        self.logger.debug("Sending content from envelope...")
        context = envelope.message.content
        try:
            await self._websocket_handler.send(context)
        except Exception:
            self.logger.error("Websocket connection closed.")
            self.state = ConnectionStates.disconnected

    async def receive(self, *args: Any, **kwargs: Any):  # noqa: V107
        """
        Receive an envelope. Blocking.

        :param args: arguments to receive
        :param kwargs: keyword arguments to receive
        :return: the envelope received, if present.  # noqa: DAR202
        """
        if self.state == ConnectionStates.disconnected:
            raise Exception(  # pylint: disable=W0719
                "Cannot receive message. Connection is not established."
            )

        try:
            new_msg = await self._websocket_handler.receive()
            return self._from_wss_msg_to_envelope(new_msg)
        except Exception:
            self.logger.error("Websocket connection closed.")
            self.state = ConnectionStates.disconnected
            raise

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
