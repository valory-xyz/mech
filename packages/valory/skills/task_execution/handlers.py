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

"""This package contains a scaffold of a handler."""

import json
import threading
import time
import urllib.parse
from enum import Enum
from typing import Any, Dict, List, Optional, Union, cast

from aea.protocols.base import Message
from aea.skills.base import Handler
from prometheus_client import start_http_server

from packages.valory.connections.ledger.connection import (
    PUBLIC_ID as LEDGER_CONNECTION_PUBLIC_ID,
)
from packages.valory.protocols.acn_data_share import AcnDataShareMessage
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.http.message import HttpMessage
from packages.valory.protocols.ipfs import IpfsMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.abstract_round_abci.handlers import AbstractResponseHandler
from packages.valory.skills.task_execution.dialogues import HttpDialogue
from packages.valory.skills.task_execution.models import Params


PENDING_TASKS = "pending_tasks"
DONE_TASKS = "ready_tasks"
IPFS_TASKS = "ipfs_tasks"
DONE_TASKS_LOCK = "lock"
TIMED_OUT_TASKS = "timed_out_tasks"
UNPROCESSED_TIMED_OUT_TASKS = "unprocessed_timed_out_tasks"
WAIT_FOR_TIMEOUT = "wait_for_timeout"
LAST_SUCCESSFUL_READ = "last_successful_read"
LAST_READ_ATTEMPT_TS = "last_read_attempt_ts"
INFLIGHT_READ_TS = "inflight_read_ts"
REQUEST_ID_TO_DELIVERY_RATE_INFO = "request_id_to_delivery_rate_info"
WAS_LAST_READ_SUCCESSFUL = "was_last_read_successful"
PAYMENT_MODEL = "payment_model"
PAYMENT_INFO = "payment_info"
TIMED_OUT_STATUS = 2
WAIT_FOR_TIMEOUT_STATUS = 1
DELIVERED_STATUS = 3
PROMETHEUS_PORT = 9000

LEDGER_API_ADDRESS = str(LEDGER_CONNECTION_PUBLIC_ID)


class BaseHandler(Handler):
    """Base Handler"""

    def setup(self) -> None:
        """Set up the handler."""
        self.context.logger.info(f"{self.__class__.__name__}: setup method called.")

    def cleanup_dialogues(self) -> None:
        """Clean up all dialogues."""
        self.context.logger.info("Cleaning up dialogues.")
        for handler_name in self.context.handlers.__dict__.keys():
            dialogues_name = handler_name.replace("_handler", "_dialogues")
            dialogues = getattr(self.context, dialogues_name)
            dialogues.cleanup()

    @property
    def params(self) -> Params:
        """Get the parameters."""
        return cast(Params, self.context.params)

    @property
    def mech_address(self) -> str:
        """Return the mech address from the list of contract addresses."""
        return self.params.agent_mech_contract_addresses[0]

    @property
    def from_block(self) -> Optional[int]:
        """Get the block from which we should search for new requests."""
        return self.params.req_params.from_block.get(
            cast(str, self.params.req_type), None
        )

    @from_block.setter
    def from_block(self, block_number: int) -> None:
        """Set the block from which we should search for new requests."""
        self.params.req_params.from_block[cast(str, self.params.req_type)] = (
            block_number
        )

    def teardown(self) -> None:
        """Teardown the handler."""
        self.context.logger.info(f"{self.__class__.__name__}: teardown called.")

    def on_message_handled(self, _message: Message) -> None:
        """Callback after a message has been handled."""
        self.params.request_count += 1
        self.context.logger.info(
            f"Message handled. {self.params.request_count=} {self.params.cleanup_freq=}"
        )

        if self.params.request_count % self.params.cleanup_freq == 0:
            self.context.logger.info(
                f"{self.params.request_count} requests processed. Cleaning up dialogues."
            )
            self.cleanup_dialogues()


class AcnHandler(BaseHandler):
    """ACN API message handler."""

    SUPPORTED_PROTOCOL = AcnDataShareMessage.protocol_id

    def handle(self, message: Message) -> None:
        """Handle the message."""
        # we don't respond to ACN messages at this point
        self.context.logger.info(f"Received ACN message: {message}")
        self.on_message_handled(message)


class IpfsHandler(BaseHandler):
    """IPFS API message handler."""

    SUPPORTED_PROTOCOL = IpfsMessage.protocol_id

    def handle(self, message: Message) -> None:
        """
        Implement the reaction to an IPFS message.

        :param message: the message
        """
        self.context.logger.info(f"Received IPFS message: {message}")
        ipfs_msg = cast(IpfsMessage, message)
        if ipfs_msg.performative == IpfsMessage.Performative.ERROR:
            self.context.logger.warning(
                f"IPFS Message performative not recognized: {ipfs_msg.performative}"
            )
            self.params.in_flight_req = False
            return

        dialogue = self.context.ipfs_dialogues.update(ipfs_msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        callback = self.params.req_to_callback.pop(nonce)
        deadline = self.params.req_to_deadline.pop(nonce)

        now = time.time()
        self.context.logger.info(f"IPFS response mapped. {nonce=} {deadline=} {now=}")

        if deadline and now > deadline:
            self.context.logger.info(
                f"Deadline reached for task with nonce {nonce} while handling IPFS message."
            )
            return

        self.context.logger.info(f"Invoking IPFS callback. {nonce=}")
        callback(ipfs_msg, dialogue)
        self.params.in_flight_req = False
        self.params.is_cold_start = False
        self.on_message_handled(message)


class ContractHandler(BaseHandler):
    """Contract API message handler."""

    SUPPORTED_PROTOCOL = ContractApiMessage.protocol_id

    def setup(self) -> None:
        """Setup the contract handler."""
        self.context.shared_state[PENDING_TASKS] = []
        self.context.shared_state[WAIT_FOR_TIMEOUT] = []
        self.context.shared_state[UNPROCESSED_TIMED_OUT_TASKS] = []
        self.context.shared_state[TIMED_OUT_TASKS] = []
        self.context.shared_state[DONE_TASKS] = []
        self.context.shared_state[DONE_TASKS_LOCK] = threading.Lock()
        self.context.shared_state[REQUEST_ID_TO_DELIVERY_RATE_INFO] = {}
        super().setup()

    def set_last_successful_read(self, block_number: Optional[int]) -> None:
        """Set the last successful read."""
        self.context.shared_state[LAST_SUCCESSFUL_READ] = (block_number, time.time())
        self.context.logger.info(
            f"Last successful read set to {self.context.shared_state[LAST_SUCCESSFUL_READ]}."
        )

    def set_was_last_read_successful(self, was_successful: bool) -> None:
        """Set the last successful read."""
        self.context.shared_state[WAS_LAST_READ_SUCCESSFUL] = was_successful
        self.context.logger.info(f"Last read success flag set to {was_successful}.")

    @property
    def pending_tasks(self) -> List[Dict[str, Any]]:
        """Get pending_tasks."""
        return self.context.shared_state[PENDING_TASKS]

    @property
    def wait_for_timeout_tasks(self) -> List[Dict[str, Any]]:
        """Get pending_tasks from other mechs"""
        return self.context.shared_state[WAIT_FOR_TIMEOUT]

    @property
    def mech_to_max_delivery_rate(self) -> int:
        """Get the max delivery rate of the mech"""
        mech_to_max_delivery_rate_dict = {
            k.lower(): v for k, v in self.params.mech_to_max_delivery_rate.items()
        }
        mech_address = self.mech_address.lower()
        return mech_to_max_delivery_rate_dict[mech_address]

    @property
    def unprocessed_timed_out_tasks(self) -> List[Dict[str, Any]]:
        """Get unprocessed timed_out_tasks for other mechs"""
        return self.context.shared_state[UNPROCESSED_TIMED_OUT_TASKS]

    @unprocessed_timed_out_tasks.setter
    def unprocessed_timed_out_tasks(self, value: List[Dict[str, Any]]) -> None:
        """Set unprocessed timed_out_tasks for other mechs"""
        self.context.shared_state[UNPROCESSED_TIMED_OUT_TASKS] = value

    @property
    def step_in_list_size(self) -> int:
        """Get step_in_list_size"""
        return self.params.step_in_list_size

    def handle(self, message: Message) -> None:
        """
        Implement the reaction to a contract message.

        :param message: the message
        """
        self.context.logger.info(f"Received ContractApi message: {message}")
        contract_api_msg = cast(ContractApiMessage, message)
        if contract_api_msg.performative != ContractApiMessage.Performative.STATE:
            self.context.logger.warning(
                f"Contract API Message performative not recognized: {contract_api_msg.performative}"
            )
            self.set_was_last_read_successful(False)
            self.params.in_flight_req = False
            return

        body = contract_api_msg.state.body
        self.context.logger.info(f"Contract state body keys={list(body.keys())}.")

        if body.get("data") or body.get("wait_for_timeout_tasks"):
            # handle the undelivered requests response from data and wait_for_timeout_tasks
            self._handle_get_undelivered_reqs(body)
        if body.get("request_ids"):
            # handle the request id status check response
            self._update_pending_list(body)
        if body.get("mech_type"):
            # handle the mech type response
            self.context.shared_state[PAYMENT_MODEL] = body["mech_type"]
            self.context.logger.info(f"Found payment model {body['mech_type']!r}.")
        if body.get("mech_types"):
            # handle the mech types response
            self.context.shared_state[PAYMENT_INFO] = body["mech_types"]
            self.context.logger.info(
                f"The cache was updated with the new mech types: {body['mech_types']}."
            )

        self.params.in_flight_req = False
        self.set_was_last_read_successful(True)
        self.on_message_handled(message)

    def _handle_get_undelivered_reqs(self, body: Dict[str, Any]) -> None:
        """Handle get undelivered reqs."""
        self.context.logger.info("Handling undelivered requests.")
        self.context.logger.info(
            f"State: "
            f"pending={len(self.pending_tasks)} "
            f"wait_for_timeout={len(self.wait_for_timeout_tasks)} "
            f"unprocessed_timed_out={len(self.unprocessed_timed_out_tasks)}",
        )

        # Reset lists.
        self.context.shared_state[INFLIGHT_READ_TS] = None
        self.wait_for_timeout_tasks.clear()
        self.unprocessed_timed_out_tasks = body.get("timed_out_requests", [])
        self.set_last_successful_read(self.from_block)

        self.context.logger.info(
            f"Loaded {len(self.unprocessed_timed_out_tasks)} timed out requests from contract.",
        )
        # collect items to process: fresh + previously waiting
        reqs = list(body.get("data", []))
        reqs.extend(body.get("wait_for_timeout_tasks", []))

        reqs_count = len(reqs)
        if reqs_count == 0:
            self.context.logger.info("No new requests returned from contract.")
            return

        old_block = self.from_block
        self.from_block = max(req["block_number"] for req in reqs) + 1
        self.context.logger.info(
            f"Received {reqs_count} requests. Advanced from_block {old_block} -> {self.from_block}."
        )

        filtered = [
            req
            for req in reqs
            if req["block_number"] % self.params.num_agents == self.params.agent_index
        ]
        self.context.logger.info(
            f"After agent sharding: {len(filtered)}/{reqs_count} requests selected."
        )
        self.filter_requests(filtered)

        self.context.logger.info(
            f"Post-filtering state: "
            f"pending={len(self.pending_tasks)} "
            f"wait_for_timeout={len(self.wait_for_timeout_tasks)} "
            f"unprocessed_timed_out={len(self.unprocessed_timed_out_tasks)}",
        )

    def _update_pending_list(self, body: Dict[str, List]) -> None:
        before = len(self.pending_tasks)
        self.context.shared_state[PENDING_TASKS] = [
            req for req in self.pending_tasks if req["requestId"] in body["request_ids"]
        ]
        after = len(self.pending_tasks)
        self.context.logger.info(
            f"Pending list updated via status check. {before} -> {after}"
        )

    def filter_requests(self, reqs: List[Dict[str, Any]]) -> None:
        """Filtering requests based on priority mech and status."""
        for req in reqs:
            rid = req.get("requestId")
            status = req.get("status")

            self.context.logger.info(f"Evaluating request {req}.")

            if (
                req["priorityMech"].lower() == self.mech_address.lower()
                and status != DELIVERED_STATUS
            ):
                self.context.logger.info(
                    f"Adding request with id {rid} to pending_tasks."
                )
                self.pending_tasks.append(req)

            elif status == TIMED_OUT_STATUS:
                self.context.logger.info(
                    f"Adding request with id {rid} to unprocessed_timed_out_tasks."
                )
                self.unprocessed_timed_out_tasks.append(req)

            elif (
                status == WAIT_FOR_TIMEOUT_STATUS
                and req["request_delivery_rate"] >= self.mech_to_max_delivery_rate
            ):
                self.context.logger.info(
                    f"Adding request with id {rid} to wait_for_timeout_tasks."
                )
                # no len check necessary as wait_for_timeout_tasks is
                # cleared everytime we handle new requests
                self.wait_for_timeout_tasks.append(req)

            else:
                self.context.logger.info(f"Request with id {rid} skipped.")


class LedgerHandler(BaseHandler):
    """Ledger API message handler."""

    SUPPORTED_PROTOCOL = LedgerApiMessage.protocol_id

    def handle(self, message: Message) -> None:
        """
        Implement the reaction to a ledger message.

        :param message: the message
        """
        self.context.logger.info(f"Received LedgerApi message: {message}")
        ledger_api_msg = cast(LedgerApiMessage, message)
        if ledger_api_msg.performative != LedgerApiMessage.Performative.STATE:
            self.context.logger.warning(
                f"Ledger API Message performative not recognized: {ledger_api_msg.performative}"
            )
            self.params.in_flight_req = False
            return

        block_number = ledger_api_msg.state.body["number"]
        old_from_block = self.from_block
        self.from_block = block_number - self.params.from_block_range
        self.context.logger.info(
            f"Block with number {block_number} received. Updated from_block: {old_from_block} -> {self.from_block}"
        )

        self.params.in_flight_req = False
        self.on_message_handled(message)


class HttpCode(Enum):
    """Http codes"""

    OK_CODE = 200
    NOT_FOUND_CODE = 404
    BAD_REQUEST_CODE = 400


class MechHttpHandler(AbstractResponseHandler):
    """Mech HTTP message handler."""

    SUPPORTED_PROTOCOL = HttpMessage.protocol_id

    @property
    def pending_tasks(self) -> List[Dict[str, Any]]:
        """Get pending_tasks."""
        return self.context.shared_state[PENDING_TASKS]

    @property
    def done_tasks(self) -> List[Dict[str, Any]]:
        """Get done_tasks."""
        return self.context.shared_state[DONE_TASKS]

    @property
    def ipfs_tasks(self) -> List[Dict[str, Any]]:
        """Get ipfs_tasks."""
        return self.context.shared_state[IPFS_TASKS]

    @property
    def params(self) -> Params:
        """Get the parameters."""
        return cast(Params, self.context.params)

    def setup(self) -> None:
        """Setup the mech http handler."""
        self.context.shared_state["routes_info"] = {
            "send_signed_requests": self._handle_signed_requests,
            "fetch_offchain_info": self._handle_offchain_request_info,
        }
        self.context.shared_state[IPFS_TASKS] = []
        self.json_content_header = "Content-Type: application/json\n"
        self.start_prometheus_server()
        super().setup()

    def start_prometheus_server(self) -> None:
        """Starts the prometheus server"""
        start_http_server(PROMETHEUS_PORT)
        self.context.logger.info(
            f"Prometheus server started on port {PROMETHEUS_PORT}."
        )

    def _handle_signed_requests(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle POST requests to send signed tx to mech.

        :param http_msg: the HttpMessage instance
        :param http_dialogue: the HttpDialogue instance
        """

        try:
            # Parse incoming data
            request_data = http_msg.body.decode("utf-8")
            parsed_data = urllib.parse.parse_qs(request_data)
            data = {key: value[0] for key, value in parsed_data.items()}
            request_id = data["request_id"]
            ipfs_hash = data["ipfs_hash"]
            request_delivery_rate = data["delivery_rate"]

            self.context.logger.info(
                f"Received signed offchain request with {request_id=} and {request_delivery_rate=}."
            )

            req = {
                "requestId": request_id,
                "data": bytes.fromhex(ipfs_hash[2:]),
                "is_offchain": True,
                "request_delivery_rate": request_delivery_rate,
                **data,
            }
            self.pending_tasks.append(req)
            self.ipfs_tasks.append(
                {"request_id": request_id, "ipfs_data": data["ipfs_data"]}
            )
            self.context.logger.info(
                f"Offchain task added with data: {req}. "
                f"pending_tasks={len(self.pending_tasks)} ipfs_tasks={len(self.ipfs_tasks)}."
            )

            self._send_ok_response(
                http_msg,
                http_dialogue,
                data={"request_id": request_id},
            )

        except Exception as e:
            self.context.logger.error(
                f"Error processing signed request. body={http_msg.body!r} error={str(e)}."
            )
            self._handle_bad_request(http_msg, http_dialogue)

    def _handle_offchain_request_info(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle GET requests to fetch offchain request info.

        :param http_msg: the HttpMessage instance
        :param http_dialogue: the HttpDialogue instance
        """

        try:
            # Parse incoming data
            request_data = http_msg.body.decode("utf-8")
            parsed_data = urllib.parse.parse_qs(request_data)
            data = {key: value[0] for key, value in parsed_data.items()}

            request_id = data["request_id"]
            self.context.logger.info(f"Fetching offchain info for {request_id=}.")

            done_tasks_list = self.done_tasks

            requested_done_tasks_list = [
                data for data in done_tasks_list if data.get("request_id") == request_id
            ]

            self._send_ok_response(
                http_msg,
                http_dialogue,
                data=requested_done_tasks_list[0] if requested_done_tasks_list else {},
            )

        except Exception as e:
            self.context.logger.error(f"Error getting offchain request info: {str(e)}")
            self._handle_bad_request(http_msg, http_dialogue)

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
