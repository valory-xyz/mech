# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2026 Valory AG
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

"""Tests for the WebSocketClient connection."""

# pylint: skip-file

from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.valory.connections.websocket_client.connection import WebSocketClient


@pytest.fixture
def connection() -> WebSocketClient:
    """Create a WebSocketClient with mocked internals for disconnect testing."""
    conn = WebSocketClient.__new__(WebSocketClient)
    conn._executor = MagicMock()
    conn._outbox = MagicMock()
    conn._manager = AsyncMock()
    conn._state = MagicMock()
    conn.logger = MagicMock()
    return conn


class TestDisconnectShutdownExecutor:
    """Tests that disconnect() properly releases the ThreadPoolExecutor."""

    @pytest.mark.asyncio
    async def test_disconnect_shuts_down_executor(
        self, connection: WebSocketClient
    ) -> None:
        """disconnect() calls executor.shutdown(wait=True) to release thread resources."""
        await connection.disconnect()

        connection._executor.shutdown.assert_called_once_with(wait=True)

    @pytest.mark.asyncio
    async def test_disconnect_order_of_operations(
        self, connection: WebSocketClient
    ) -> None:
        """disconnect() removes subscriptions and empties outbox before shutting down the executor."""
        call_order = []

        async def track_remove() -> None:
            call_order.append("remove_subscriptions")

        def track_empty() -> None:
            call_order.append("empty_outbox")

        def track_shutdown(wait: bool = False) -> None:
            call_order.append("shutdown_executor")

        connection._manager.remove_all_subscriptions = track_remove
        connection._outbox.empty = track_empty
        connection._executor.shutdown = track_shutdown

        await connection.disconnect()

        assert call_order == [
            "remove_subscriptions",
            "empty_outbox",
            "shutdown_executor",
        ]
