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
from typing import Any, Dict, Optional, Union
from unittest.mock import MagicMock

import pytest

from packages.valory.skills.mech_abci.handlers import HttpHandler, HttpMethod
from packages.valory.skills.task_execution.handlers import MechHttpHandler


PACKAGE_DIR = Path(__file__).parents[1]


@dataclass
class GetHandlerTestCase:
    """Get Handler test case."""

    name: str
    url: str
    method: str
    expected_handler: Union[str, None]
    is_mech_handler: Optional[bool] = False


class TestHttpHandler:
    """Test HttpHandler of mech_abci."""

    path_to_skill = PACKAGE_DIR

    def setup_class(self) -> None:
        """Setup the test class."""
        self.context = MagicMock()
        self.context.logger = MagicMock()
        self.handler = HttpHandler(name="", skill_context=self.context)
        self.mech_handler = MechHttpHandler(name="", skill_context=self.context)
        self.mech_handler.context.shared_state = {}
        self.handler.context.params.service_endpoint_base = "http://localhost:8080/"
        self.mech_handler.setup()
        self.handler.setup()

    def test_setup(self) -> None:
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

    @pytest.mark.parametrize(
        "test_case",
        [
            GetHandlerTestCase(
                name="Happy Path",
                url="http://localhost:8080/healthcheck",
                method=HttpMethod.GET.value,
                expected_handler="_handle_get_health",
            ),
            GetHandlerTestCase(
                name="Happy Path",
                url="http://localhost:8080/send_signed_requests",
                method=HttpMethod.POST.value,
                expected_handler="_handle_signed_requests",
                is_mech_handler=True,
            ),
            GetHandlerTestCase(
                name="Happy Path",
                url="http://localhost:8080/fetch_offchain_info",
                method=HttpMethod.GET.value,
                expected_handler="_handle_offchain_request_info",
                is_mech_handler=True,
            ),
            GetHandlerTestCase(
                name="Happy Path",
                url="http://localhost:8080/fetch_offchain_info/1",
                method=HttpMethod.GET.value,
                expected_handler="_handle_offchain_request_info",
                is_mech_handler=True,
            ),
            GetHandlerTestCase(
                name="No url match",
                url="http://invalid.url/not/matching",
                method=HttpMethod.GET.value,
                expected_handler=None,
            ),
            GetHandlerTestCase(
                name="No method match",
                url="http://localhost:8080/some/path",
                method=HttpMethod.POST.value,
                expected_handler="_handle_bad_request",
            ),
        ],
    )
    def test_mech_get_handler(self, test_case: GetHandlerTestCase) -> None:
        """Test _get_handler."""
        url = test_case.url
        method = test_case.method
        is_mech_handler = test_case.is_mech_handler

        if test_case.expected_handler is not None:
            if is_mech_handler:
                expected_handler = getattr(
                    self.mech_handler, test_case.expected_handler
                )
            else:
                expected_handler = getattr(self.handler, test_case.expected_handler)
        else:
            expected_handler = test_case.expected_handler
        expected_captures: Dict[Any, Any] = {}

        handler, captures = self.handler._get_handler(url, method)

        assert handler == expected_handler
        assert captures == expected_captures
