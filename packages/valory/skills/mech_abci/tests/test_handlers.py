# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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

"""Test the handlers.py module of the mech_abci skill."""


from dataclasses import dataclass
from pathlib import Path
from typing import Union
from unittest.mock import MagicMock


from packages.valory.connections.http_server.connection import (
    PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
)
from packages.valory.skills.task_execution.handlers import (
    MechHttpHandler,
)
from packages.valory.skills.mech_abci.handlers import HttpHandler, HttpMethod


PACKAGE_DIR = Path(__file__).parent.parent

HTTP_SERVER_SENDER = str(HTTP_SERVER_PUBLIC_ID.without_hash())


@dataclass
class HandlerTestCase:
    """HandlerTestCase"""

    name: str
    message_performative: str
    message_sender: str
    get_handler_return_value: tuple
    message_url: Union[str, None] = None
    message_method: Union[str, None] = None
    update_return_value: Union[MagicMock, None] = None
    expected_logger_call: Union[str, None] = None
    expected_handler_call: bool = False


class TestHttpHandler:
    """Test HttpHandler of mech_abci."""

    path_to_skill = PACKAGE_DIR

    @classmethod
    def setup_class(self):
        """Setup the test class."""
        self.context = MagicMock()
        self.context.logger = MagicMock()
        self.handler = HttpHandler(name="", skill_context=self.context)
        self.mech_handler = MechHttpHandler(name="", skill_context=self.context)
        self.mech_handler.context.shared_state = {}
        self.handler.context.params.service_endpoint_base = "http://localhost:8080/"
        self.mech_handler.setup()
        self.handler.setup()

    def test_setup(self):
        """Test the setup method of the handler."""
        service_endpoint_base = "localhost"
        propel_uri_base_hostname = (
            r"https?:\/\/[a-zA-Z0-9]{16}.agent\.propel\.(staging\.)?autonolas\.tech"
        )
        hostname_regex = rf".*({service_endpoint_base}|{propel_uri_base_hostname}|localhost|127.0.0.1|0.0.0.0)(:\d+)?"
        health_url_regex = rf"{hostname_regex}\/healthcheck"
        send_signed_url = rf"{hostname_regex}\/send_signed_requests"
        fetch_offchain_info_url = rf"{hostname_regex}\/fetch_offchain_info"

        assert self.handler.handler_url_regex == rf"{hostname_regex}\/.*"
        assert self.handler.routes == {
            (HttpMethod.GET.value, HttpMethod.HEAD.value): [
                (health_url_regex, self.handler._handle_get_health),
                (
                    fetch_offchain_info_url,
                    self.mech_handler._handle_offchain_request_info,
                ),
            ],
            (HttpMethod.POST.value,): [
                (send_signed_url, self.mech_handler._handle_signed_requests)
            ],
        }
        assert self.handler.json_content_header == "Content-Type: application/json\n"

    # @pytest.mark.parametrize(
    #     "url, method, expected_handler_name",
    #     [
    #         ("wrong_url", "get", None),
    #         (
    #             "http://mech.staging.autonolas.tech/send_signed_requests",
    #             "post",
    #             "_handle_signed_requests",
    #         ),
    #         (
    #             "http://mech.staging.autonolas.tech/health_check",
    #             "get",
    #             "_handle_get_health",
    #         ),
    #         (
    #             "http://mech.staging.autonolas.tech/fetch_offchain_info",
    #             "get",
    #             "_handle_offchain_request_info",
    #         ),
    #         (
    #             "http://mech.staging.autonolas.tech/fetch_offchain_info/1",
    #             "get",
    #             "_handle_offchain_request_info",
    #         ),
    #     ],
    # )
    # def test_get_handler(self, url, method, expected_handler_name):
    #     """Test check_url"""
    #     expected_handler = (
    #         getattr(self.http_handler, expected_handler_name)
    #         if expected_handler_name
    #         else None
    #     )
    #     actual_handler, _ = self.http_handler._get_handler(url, method)
    #     assert (
    #         actual_handler == expected_handler
    #     ), f"Wrong value for {url}. Expected {expected_handler}, got {actual_handler}"
