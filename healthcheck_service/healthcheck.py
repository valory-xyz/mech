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
"""Contains a small healthcheck server"""

import http.server
import json
import os
import socketserver
from pathlib import Path
from time import time
from typing import Dict, Any, List, Optional

from web3.types import BlockIdentifier

from web3 import Web3


class MechContract:
    def __init__(self, rpc_endpoint: str, contract_address: str) -> None:
        """Setup the base event tracker"""
        self.rpc_endpoint = rpc_endpoint
        self.web3 = Web3(Web3.HTTPProvider(self.rpc_endpoint))
        self.contract = self.web3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=self._get_abi(),
        )

    def _get_abi(self) -> Dict[str, Any]:
        """Get the abi of the contract."""
        path = Path(__file__).parent / "web3" / "abi.json"
        with open(str(path)) as f:
            abi = json.load(f)
        return abi

    def get_deliver_events(self, from_block: BlockIdentifier) -> List[Dict[str, Any]]:
        """Get the deliver events."""
        return self.contract.events.Deliver.create_filter(
            fromBlock=from_block
        ).get_all_entries()

    def get_request_events(self, from_block: BlockIdentifier) -> List[Dict[str, Any]]:
        """Get the request events."""
        return self.contract.events.Request.create_filter(
            fromBlock=from_block
        ).get_all_entries()

    def get_unfulfilled_request(self) -> List[Dict[str, Any]]:
        """Get the unfulfilled events."""
        from_block = self.web3.eth.block_number - 50_000  # ~ 3.5 days back
        delivers = self.get_deliver_events(from_block)
        requests = self.get_request_events(from_block)
        undeleted_requests = []
        deliver_req_ids = [deliver["args"]["requestId"] for deliver in delivers]

        for request in requests:
            if request["args"]["requestId"] not in deliver_req_ids:
                undeleted_requests.append(request)
        return undeleted_requests

    def get_block_timestamp(self, block_number: int) -> int:
        """Get the block timestamp."""
        return self.web3.eth.get_block(block_number)["timestamp"]

    def earliest_unfulfilled_request_timestamp(self) -> Optional[int]:
        """Get the earliest unfulfilled request."""
        unfulfilled_requests = self.get_unfulfilled_request()
        earliest_request = None
        for request in unfulfilled_requests:
            if (
                earliest_request is None
                or request["blockNumber"] < earliest_request["blockNumber"]
            ):
                earliest_request = request
        if earliest_request is not None:
            timestamp = self.get_block_timestamp(earliest_request["blockNumber"])
            return timestamp
        return None


class HealthCheckHandler(http.server.SimpleHTTPRequestHandler):
    """Healthcheck server handler."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the healthcheck server handler."""
        self.mech_contract = MechContract(
            rpc_endpoint=os.getenv("RPC_ENDPOINT", "http://localhost:8545"),
            contract_address=os.getenv("MECH_CONTRACT_ADDRESS"),
        )
        self.grace_period = int(os.getenv("GRACE_PERIOD", 600))
        super().__init__(*args, **kwargs)

    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        req_timestamp = self.mech_contract.earliest_unfulfilled_request_timestamp()
        if req_timestamp is None:
            return True
        return req_timestamp + self.grace_period > time()

    def do_GET(self) -> None:
        """
        Handle GET requests and send a health check response with a 200 status code.

        Returns:
            None
        """
        is_healthy = self.is_healthy()
        code, message = (200, "OK") if is_healthy else (500, "NOT OK")
        self.send_response(code)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(message.encode())


def run_healthcheck_server() -> None:
    """
    Run the health check server.

    Returns:
        None
    """
    port = os.getenv("PORT", 8080)
    handler = HealthCheckHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Health check server started on port {port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.shutdown()
            print("\nHealth check server stopped.")


if __name__ == "__main__":
    run_healthcheck_server()
