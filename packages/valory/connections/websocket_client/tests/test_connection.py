import asyncio
import socket
from contextlib import asynccontextmanager, closing
from copy import deepcopy
from unittest.mock import Mock

import aiohttp
import pytest
from aiohttp import web

from packages.valory.connections.websocket_client.connection import WebSocketHandler


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


DROP_CONN = object()


@asynccontextmanager
async def websocket_server(responses):
    port = find_free_port()
    url = f"http://localhost:{port}"

    responses = deepcopy(responses)

    async def websocket_handler(request):
        try:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            while responses:
                resp = responses.pop(0)
                if resp is DROP_CONN:
                    await ws.close()
                    return ws
                await ws.send_str(resp)
        except Exception as e:
            print(e)
        return ws

    app = web.Application()
    app.add_routes([web.get("/", websocket_handler)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", port)
    await site.start()
    try:
        yield url
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_conn():
    resps = ["1", "2", "3", "4", DROP_CONN, "5"]

    async with websocket_server(resps) as url:
        try:
            wsh = WebSocketHandler(url)
            await asyncio.wait_for(wsh.connect(), timeout=1)
            await asyncio.wait_for(wsh.send("some"), timeout=1)
            for resp in resps:
                if resp is DROP_CONN:
                    continue
                assert await asyncio.wait_for(wsh.receive(), timeout=1) == resp
        finally:
            await wsh.disconnect()

    wsh = WebSocketHandler(url)
    wsh.RECONNECT_DELAY = 0
    with pytest.raises(aiohttp.client_exceptions.ClientConnectorError):
        await asyncio.wait_for(wsh.connect(), timeout=10)

    wsh = WebSocketHandler(url)
    wsh.RECONNECT_DELAY = 1
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wsh.connect(), timeout=1)
