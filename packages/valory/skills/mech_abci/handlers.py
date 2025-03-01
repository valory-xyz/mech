# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
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

"""This module contains the handlers for the skill of AutonomousFundAbciApp."""

import json
import re
import time
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union, cast
from urllib.parse import urlparse

from aea.protocols.base import Message

from packages.valory.connections.http_server.connection import (
    PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
)
from packages.valory.protocols.http.message import HttpMessage
from packages.valory.skills.abstract_round_abci.handlers import (
    ABCIRoundHandler as BaseABCIRoundHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    ContractApiHandler as BaseContractApiHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    HttpHandler as BaseHttpHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    IpfsHandler as BaseIpfsHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    LedgerApiHandler as BaseLedgerApiHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    SigningHandler as BaseSigningHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    TendermintHandler as BaseTendermintHandler,
)
from packages.valory.skills.mech_abci.dialogues import HttpDialogue, HttpDialogues
from packages.valory.skills.task_submission_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.task_submission_abci.rounds import SynchronizedData


ABCIRoundHandler = BaseABCIRoundHandler
SigningHandler = BaseSigningHandler
LedgerApiHandler = BaseLedgerApiHandler
ContractApiHandler = BaseContractApiHandler
TendermintHandler = BaseTendermintHandler
IpfsHandler = BaseIpfsHandler

LAST_SUCCESSFUL_READ = "last_successful_read"
LAST_SUCCESSFUL_EXECUTED_TASK = "last_successful_executed_task"
WAS_LAST_READ_SUCCESSFUL = "was_last_read_successful"
LAST_TX = "last_tx"


class HttpCode(Enum):
    """Http codes"""

    OK_CODE = 200
    NOT_FOUND_CODE = 404
    BAD_REQUEST_CODE = 400


class HttpMethod(Enum):
    """Http methods"""

    GET = "get"
    HEAD = "head"
    POST = "post"


class HttpHandler(BaseHttpHandler):
    """This implements the HTTP handler."""

    SUPPORTED_PROTOCOL = HttpMessage.protocol_id

    @property
    def last_successful_read(self) -> Optional[Tuple[int, float]]:
        """Get the last successful read."""
        return cast(
            Optional[Tuple[int, float]],
            self.context.shared_state.get(LAST_SUCCESSFUL_READ),
        )

    @property
    def last_successful_executed_task(self) -> Optional[Tuple[int, float]]:
        """Get the last successful executed task."""
        return cast(
            Optional[Tuple[int, float]],
            self.context.shared_state.get(LAST_SUCCESSFUL_EXECUTED_TASK),
        )

    @property
    def was_last_read_successful(self) -> bool:
        """Get the last read status."""
        return self.context.shared_state.get(WAS_LAST_READ_SUCCESSFUL) is not False

    @property
    def last_tx(self) -> Optional[Tuple[str, float]]:
        """Get the last transaction."""
        return cast(Optional[Tuple[str, float]], self.context.shared_state.get(LAST_TX))

    def setup(self) -> None:
        """Implement the setup."""

        # Custom hostname (set via params)
        service_endpoint_base = urlparse(
            self.context.params.service_endpoint_base
        ).hostname

        # Propel hostname regex
        propel_uri_base_hostname = (
            r"https?:\/\/[a-zA-Z0-9]{16}.agent\.propel\.(staging\.)?autonolas\.tech"
        )

        # Route regexes
        hostname_regex = rf".*({service_endpoint_base}|{propel_uri_base_hostname}|localhost|127.0.0.1|0.0.0.0)(:\d+)?"
        self.handler_url_regex = rf"{hostname_regex}\/.*"
        health_url_regex = rf"{hostname_regex}\/healthcheck"

        # update the route for mech http handler
        routes_data = self.context.shared_state["routes_info"]
        routes = list(routes_data.keys())
        funcs = list(routes_data.values())

        send_signed_url = rf"{hostname_regex}\/{routes[0]}"
        fetch_offchain_info = rf"{hostname_regex}\/{routes[1]}"

        # Routes
        self.routes = {
            (HttpMethod.POST.value,): [],
            (HttpMethod.GET.value, HttpMethod.HEAD.value): [
                (health_url_regex, self._handle_get_health),
            ],
            (HttpMethod.POST.value,): [(send_signed_url, funcs[0])],
            (HttpMethod.GET.value, HttpMethod.HEAD.value): [
                (fetch_offchain_info, funcs[1])
            ],
        }

        self.json_content_header = "Content-Type: application/json\n"

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return SynchronizedData(
            db=self.context.state.round_sequence.latest_synchronized_data.db
        )

    def _get_handler(self, url: str, method: str) -> Tuple[Optional[Callable], Dict]:
        """Check if an url is meant to be handled in this handler

        We expect url to match the pattern {hostname}/.*,
        where hostname is allowed to be localhost, 127.0.0.1 or the token_uri_base's hostname.
        Examples:
            localhost:8000/0
            127.0.0.1:8000/100
            https://pfp.staging.autonolas.tech/45
            http://pfp.staging.autonolas.tech/120

        :param url: the url to check
        :param method: the http method
        :returns: the handling method if the message is intended to be handled by this handler, None otherwise, and the regex captures
        """
        # Check base url
        if not re.match(self.handler_url_regex, url):
            self.context.logger.info(
                f"The url {url} does not match the HttpHandler's pattern"
            )
            return None, {}

        # Check if there is a route for this request
        for methods, routes in self.routes.items():
            if method not in methods:
                continue

            for route in routes:  # type: ignore
                # Routes are tuples like (route_regex, handle_method)
                m = re.match(route[0], url)
                if m:
                    return route[1], m.groupdict()

        # No route found
        self.context.logger.info(
            f"The message [{method}] {url} is intended for the HttpHandler but did not match any valid pattern"
        )
        return self._handle_bad_request, {}

    def handle(self, message: Message) -> None:
        """
        Implement the reaction to an envelope.

        :param message: the message
        """
        http_msg = cast(HttpMessage, message)

        # Check if this is a request sent from the http_server skill
        if (
            http_msg.performative != HttpMessage.Performative.REQUEST
            or message.sender != str(HTTP_SERVER_PUBLIC_ID.without_hash())
        ):
            super().handle(message)
            return

        # Check if this message is for this skill. If not, send to super()
        handler, kwargs = self._get_handler(http_msg.url, http_msg.method)
        if not handler:
            super().handle(message)
            return

        # Retrieve dialogues
        http_dialogues = cast(HttpDialogues, self.context.http_dialogues)
        http_dialogue = cast(HttpDialogue, http_dialogues.update(http_msg))

        # Invalid message
        if http_dialogue is None:
            self.context.logger.info(
                "Received invalid http message={}, unidentified dialogue.".format(
                    http_msg
                )
            )
            return

        # Handle message
        self.context.logger.info(
            "Received http request with method={}, url={} and body={!r}".format(
                http_msg.method,
                http_msg.url,
                http_msg.body,
            )
        )
        handler(http_msg, http_dialogue, **kwargs)

    def _handle_bad_request(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle a Http bad request.

        :param http_msg: the http message
        :param http_dialogue: the http dialogue
        """
        http_response = http_dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=http_msg,
            version=http_msg.version,
            status_code=HttpCode.BAD_REQUEST_CODE.value,
            status_text="Bad request",
            headers=http_msg.headers,
            body=b"",
        )

        # Send response
        self.context.logger.info("Responding with: {}".format(http_response))
        self.context.outbox.put_message(message=http_response)

    def _send_ok_response(
        self,
        http_msg: HttpMessage,
        http_dialogue: HttpDialogue,
        data: Union[Dict, List],
    ) -> None:
        """Send an OK response with the provided data"""
        http_response = http_dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=http_msg,
            version=http_msg.version,
            status_code=HttpCode.OK_CODE.value,
            status_text="Success",
            headers=f"{self.json_content_header}{http_msg.headers}",
            body=json.dumps(data).encode("utf-8"),
        )

        # Send response
        self.context.logger.info("Responding with: {}".format(http_response))
        self.context.outbox.put_message(message=http_response)

    def _send_not_found_response(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """Send an not found response"""
        http_response = http_dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=http_msg,
            version=http_msg.version,
            status_code=HttpCode.NOT_FOUND_CODE.value,
            status_text="Not found",
            headers=http_msg.headers,
            body=b"",
        )
        # Send response
        self.context.logger.info("Responding with: {}".format(http_response))
        self.context.outbox.put_message(message=http_response)

    def _handle_get_health(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle a Http request of verb GET.

        :param http_msg: the http message
        :param http_dialogue: the http dialogue
        """
        seconds_since_last_transition = None
        is_tm_unhealthy = None
        is_transitioning_fast = None
        current_round = None
        previous_rounds = None

        round_sequence = cast(BaseSharedState, self.context.state).round_sequence

        if round_sequence._last_round_transition_timestamp:
            is_tm_unhealthy = cast(
                BaseSharedState, self.context.state
            ).round_sequence.block_stall_deadline_expired

            current_time = datetime.now().timestamp()
            seconds_since_last_transition = current_time - datetime.timestamp(
                round_sequence._last_round_transition_timestamp
            )

            is_transitioning_fast = (
                not is_tm_unhealthy
                and seconds_since_last_transition
                < 2 * self.context.params.reset_pause_duration
            )

        if round_sequence._abci_app:
            current_round = round_sequence._abci_app.current_round.round_id
            previous_rounds = [
                r.round_id for r in round_sequence._abci_app._previous_rounds[-10:]
            ]

        # ensure we are delivering
        grace_period = 300  # 5 min
        last_executed_task = (
            self.last_successful_executed_task[1]
            if self.last_successful_executed_task
            else time.time() + grace_period * 2
        )
        last_tx_made = self.last_tx[1] if self.last_tx else time.time()
        we_are_delivering = last_executed_task < last_tx_made + grace_period

        # ensure we can get new reqs
        last_successful_read = (
            self.last_successful_read[1] if self.last_successful_read else time.time()
        )
        grace_period = 300  # 5 min
        we_can_get_new_reqs = last_successful_read > time.time() - grace_period

        data = {
            "seconds_since_last_transition": seconds_since_last_transition,
            "is_tm_healthy": not is_tm_unhealthy,
            "period": self.synchronized_data.period_count,
            "reset_pause_duration": self.context.params.reset_pause_duration,
            "current_round": current_round,
            "previous_rounds": previous_rounds,
            "is_transitioning_fast": is_transitioning_fast,
            "last_successful_read": (
                {
                    "block_number": self.last_successful_read[0],
                    "timestamp": self.last_successful_read[1],
                }
                if self.last_successful_read
                else None
            ),
            "last_successful_executed_task": (
                {
                    "request_id": self.last_successful_executed_task[0],
                    "timestamp": self.last_successful_executed_task[1],
                }
                if self.last_successful_executed_task
                else None
            ),
            "was_last_read_successful": self.was_last_read_successful,
            "last_tx": (
                {
                    "tx_hash": self.last_tx[0],
                    "timestamp": self.last_tx[1],
                }
                if self.last_tx
                else None
            ),
            "is_ok": (we_are_delivering and we_can_get_new_reqs),
        }

        self._send_ok_response(http_msg, http_dialogue, data)
