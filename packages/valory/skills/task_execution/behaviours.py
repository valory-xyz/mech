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

"""This package contains the implementation of ."""

import json
import threading
import time
from asyncio import Future
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

from aea.helpers.cid import to_v1
from aea.mail.base import EnvelopeContext
from aea.protocols.base import Message
from aea.protocols.dialogue.base import Dialogue
from aea.skills.behaviours import SimpleBehaviour
from pebble import ProcessPool
from prometheus_client import Counter, Gauge, Histogram

from packages.valory.connections.ipfs.connection import IpfsDialogues
from packages.valory.connections.ipfs.connection import PUBLIC_ID as IPFS_CONNECTION_ID
from packages.valory.connections.ledger.connection import (
    PUBLIC_ID as LEDGER_CONNECTION_PUBLIC_ID,
)
from packages.valory.connections.p2p_libp2p_client.connection import (
    PUBLIC_ID as P2P_CLIENT_PUBLIC_ID,
)
from packages.valory.contracts.mech_marketplace.contract import MechMarketplaceContract
from packages.valory.contracts.olas_mech.contract import OlasMechContract
from packages.valory.protocols.acn_data_share import AcnDataShareMessage
from packages.valory.protocols.acn_data_share.dialogues import AcnDataShareDialogues
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ipfs import IpfsMessage
from packages.valory.protocols.ipfs.dialogues import IpfsDialogue
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.task_execution.models import Params
from packages.valory.skills.task_execution.utils.apis import KeyChain
from packages.valory.skills.task_execution.utils.benchmarks import TokenCounterCallback
from packages.valory.skills.task_execution.utils.cost_calculation import (
    get_cost_for_done_task,
)
from packages.valory.skills.task_execution.utils.ipfs import (
    ComponentPackageLoader,
    get_ipfs_file_hash,
    to_multihash,
)
from packages.valory.skills.task_execution.utils.local_cid import compute_cidv1
from packages.valory.skills.task_execution.utils.task import AnyToolAsTask

PENDING_TASKS = "pending_tasks"
WAIT_FOR_TIMEOUT = "wait_for_timeout"
UNPROCESSED_TIMED_OUT_TASKS = "unprocessed_timed_out_tasks"
TIMED_OUT_TASKS = "timed_out_tasks"
DONE_TASKS = "ready_tasks"
IPFS_TASKS = "ipfs_tasks"
DONE_TASKS_LOCK = "lock"
PAYMENT_MODEL = "payment_model"
PAYMENT_INFO = "payment_info"
LAST_SUCCESSFUL_EXECUTED_TASK = "last_successful_executed_task"
LAST_READ_ATTEMPT_TS = "last_read_attempt_ts"
INFLIGHT_READ_TS = "inflight_read_ts"
REQUEST_ID_TO_DELIVERY_RATE_INFO = "request_id_to_delivery_rate_info"
# Shared-state keys owned by the MechHttpHandler (handlers.py); mirrored here
# because the off-chain finalize path writes the rejection signal and clears the
# buffered request metadata from the behaviour side.
OFFCHAIN_REQUEST_RESPONSES = "offchain_request_responses"
IN_MEMORY_REQUESTS = "in_memory_requests"
INITIAL_DEADLINE = 1200.0  # 20mins of deadline
SUBSEQUENT_DEADLINE = 300.0  # 5min of deadline
STATUS_CHECK_INTERVAL = 600.0  # 10min interval
RESPONSE_SCHEMA_VERSION = "2.0"
IPFS_MAX_TASK_BYTES = 1_048_576  # 1MB cap on attacker-controlled task payload
MAX_PROMPT_BYTES = 100_000  # 100KB cap on the prompt field

# Cap on the JSON-serialised size of each ``raw_content`` blob attached to
# a wildcard event. The blob rides Tendermint consensus replication into
# ``synchronized_data`` on every agent and the field is requester-influenced
# (``params``, ``model``, response body), so an uncapped large payload would
# inflate per-node state. 256 KiB matches the wildcard server's ``_MAX_TEXT``
# cap on individual TEXT columns; the analytics ETL drops the row to a
# truncated sentinel when the cap is hit so the rest of the event still
# lands.
MAX_RAW_CONTENT_BYTES = 262_144  # 256 KiB


def _cap_raw_content(blob: Dict[str, Any]) -> Dict[str, Any]:
    """Bound the JSON-serialised size of a ``raw_content`` blob.

    The full request/response payload rides Tendermint consensus
    replication via ``synchronized_data.done_tasks`` to every agent, and
    its contents (``params``, ``model``, response body) are
    requester-influenced. An uncapped large payload would inflate
    per-node state on every offchain delivery. When the serialised size
    exceeds :data:`MAX_RAW_CONTENT_BYTES`, return a sentinel dict noting
    the truncation and the size so the analytics ETL can flag the row
    rather than silently store half of it.

    :param blob: the raw request- or response-side payload dict.
    :return: ``blob`` itself when under the cap, or a truncated sentinel.
    """
    try:
        size = len(json.dumps(blob).encode("utf-8"))
    except (TypeError, ValueError):
        return {"truncated": True, "reason": "non_json_serialisable"}
    if size <= MAX_RAW_CONTENT_BYTES:
        return blob
    return {"truncated": True, "size_bytes": size, "cap_bytes": MAX_RAW_CONTENT_BYTES}


PAYMENT_MODEL_REQUEST_TIMEOUT = 60.0  # reset stuck payment-model in_flight_req

LEDGER_API_ADDRESS = str(LEDGER_CONNECTION_PUBLIC_ID)


class RequestType(Enum):
    """Request Types"""

    LEGACY = "legacy"
    MARKETPLACE = "marketplace"


@dataclass(init=False)
class MechMetrics:
    """Prometheus Metrics for mech"""

    mech_pending_queue_len: Gauge
    mech_timed_out_queue_len: Gauge
    mech_wait_for_time_out_queue_len: Gauge
    mech_tasks_started_total: Counter
    mech_tasks_completed_total: Counter
    mech_tasks_failed_total: Counter
    mech_tasks_timed_out_total: Counter
    mech_tasks_inflight: Gauge
    mech_tool_preparation_time: Histogram
    mech_tool_execution_time: Histogram

    def __init__(self) -> None:
        """Define Prometheus metrics"""
        self.mech_pending_queue_len = Gauge(
            "mech_pending_queue_len", "Total pending tasks in the mech agent"
        )
        self.mech_timed_out_queue_len = Gauge(
            "mech_timed_out_queue_len", "Total timed out tasks in the mech agent"
        )
        self.mech_wait_for_time_out_queue_len = Gauge(
            "mech_wait_for_time_out_queue_len",
            "Total wait for time out tasks in the mech agent",
        )
        self.mech_tasks_started_total = Counter(
            "mech_tasks_started_total", "Total tasks worked on by the mech"
        )
        self.mech_tasks_completed_total = Counter(
            "mech_tasks_completed_total",
            "Total tasks completed by the mech",
            labelnames=["tool"],
        )
        self.mech_tasks_failed_total = Counter(
            "mech_tasks_failed_total",
            "Total tasks failed in mech with tool and reason",
            labelnames=["tool", "reason"],
        )
        self.mech_tasks_timed_out_total = Counter(
            "mech_tasks_timed_out_total",
            "Total tasks timed out during execution",
            labelnames=["tool"],
        )
        self.mech_tasks_inflight = Gauge(
            "mech_tasks_inflight",
            "Current task in execution",
        )
        self.mech_tool_preparation_time = Histogram(
            "mech_tool_preparation_time",
            "Duration taken by tool from preparation till execution",
            labelnames=["tool"],
            buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600),
        )
        self.mech_tool_execution_time = Histogram(
            "mech_tool_execution_time",
            "Duration taken by tool from execution till completion",
            labelnames=["tool"],
            buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600),
        )

    def set_gauge(self, metric: Gauge, value: int, **labels: Any) -> None:
        """Set the Prometheus' guage metric"""
        if labels:
            metric.labels(**labels).set_to_current_time()
            metric.labels(**labels).set(value)
        else:
            metric.set_to_current_time()
            metric.set(value)

    def inc_counter(self, metric: Counter, value: float = 1, **labels: Any) -> None:
        """Increment the Prometheus' counter metric"""
        if labels:
            metric.labels(**labels).inc(value)
        else:
            metric.inc(value)

    def observe_histogram(self, metric: Histogram, value: float, **labels: Any) -> None:
        """Observe the Prometheus' histogram metric"""
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)


class TaskExecutionBehaviour(SimpleBehaviour):
    """A class to execute tasks."""

    def __init__(self, **kwargs: Any):
        """Initialise the agent."""
        super().__init__(**kwargs)
        # we only want to execute one task at a time, for the time being
        self._executor = ProcessPool(max_workers=1)
        self._executing_task: Optional[Dict[str, Any]] = None
        self._current_request_id: Optional[int] = None
        self._tools_to_package_hash: Dict[str, str] = {}
        self._tools_to_pricing: Dict[str, int] = {}
        self._all_tools: Dict[str, Tuple[str, str, Dict[str, Any]]] = {}
        self._inflight_tool_req: Optional[str] = None
        self._inflight_ipfs_req: Optional[str] = None
        self._done_task: Optional[Dict[str, Any]] = None
        self._request_handling_deadline: Optional[float] = None
        self._invalid_request = False
        self._ipfs_error_reason: Optional[str] = None
        self._async_result: Optional[Future] = None
        self._keychain: Optional[KeyChain] = None
        self._ignored_request_ids: Set[int] = set()
        self._payment_model_request_sent_at: Optional[float] = None
        # We fetch the requests and their status on the startup so this should be fairly accurate
        self.last_status_check_time: float = time.time()
        self.tool_preparation_start_time: float = 0.0
        self.tool_execution_start_time: float = 0.0

        # Prometheus metrics
        self.mech_metrics = MechMetrics()

    def _request_payment_model(self) -> None:
        """Request the mech's payment model."""
        contract_api_msg, _ = self.context.contract_dialogues.create(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=self.params.agent_mech_contract_address,
            contract_id=str(OlasMechContract.contract_id),
            callable="get_mech_type",
            kwargs=ContractApiMessage.Kwargs(
                dict(
                    chain_id=self.params.default_chain_id,
                )
            ),
            counterparty=LEDGER_API_ADDRESS,
            ledger_id=self.context.default_ledger_id,
        )
        self.context.outbox.put_message(message=contract_api_msg)
        self.params.in_flight_req = True
        self._payment_model_request_sent_at = time.time()

    def setup(self) -> None:
        """Implement the setup."""
        self.context.logger.info("Setting up TaskExecutionBehaviour")
        self._tools_to_package_hash = self.params.tools_to_package_hash
        self._tools_to_pricing = self.params.tools_to_pricing
        self._keychain = KeyChain(self.params.api_keys)

    def _ensure_payment_model(self) -> bool:
        """Set the mech's payment model."""
        if not self.params.use_mech_marketplace:
            return True

        if self.payment_model:
            self._payment_model_request_sent_at = None
            return True

        if self.params.in_flight_req:
            sent_at = self._payment_model_request_sent_at
            if (
                sent_at is not None
                and time.time() - sent_at > PAYMENT_MODEL_REQUEST_TIMEOUT
            ):
                self.context.logger.warning(
                    f"Payment-model request has been in flight for "
                    f">{PAYMENT_MODEL_REQUEST_TIMEOUT}s with no response. "
                    f"Resetting in_flight_req so it can be retried."
                )
                self.params.in_flight_req = False
                self._payment_model_request_sent_at = None
            return False

        self.context.logger.info("Setting the mech's payment model...")
        self._request_payment_model()
        return False

    def act(self) -> None:
        """Implement the act."""
        self._download_tools()
        if not self._ensure_payment_model():
            return
        self._execute_ipfs_tasks()
        self._execute_task()
        self._check_for_new_marketplace_reqs()
        self._filter_out_incompatible_reqs()
        self._update_pending_tasks()

    @property
    def payment_model(self) -> Optional[Any]:
        """Get the mech's payment model."""
        return self.context.shared_state.get(PAYMENT_MODEL)

    @property
    def payment_info(self) -> Dict[str, Any]:
        """Get the cached mechs' payment info."""
        return self.context.shared_state.get(PAYMENT_INFO, {})

    @property
    def done_tasks_lock(self) -> threading.Lock:
        """Get done_tasks_lock."""
        return self.context.shared_state[DONE_TASKS_LOCK]

    @property
    def params(self) -> Params:
        """Get the parameters."""
        return cast(Params, self.context.params)

    @property
    def request_id_to_num_timeouts(self) -> Dict[int, int]:
        """Maps the request id to the number of times it has timed out."""
        return self.params.request_id_to_num_timeouts

    def count_timeout(self, request_id: int) -> None:
        """Increase the timeout for a request."""
        self.request_id_to_num_timeouts[request_id] += 1

    def timeout_limit_reached(self, request_id: int) -> bool:
        """Check if the timeout limit has been reached."""
        return self.params.timeout_limit <= self.request_id_to_num_timeouts[request_id]

    @property
    def pending_tasks(self) -> List[Dict[str, Any]]:
        """Get pending_tasks."""
        return self.context.shared_state[PENDING_TASKS]

    @property
    def wait_for_timeout_tasks(self) -> List[Dict[str, Any]]:
        """Get pending_tasks from other mechs"""
        return self.context.shared_state[WAIT_FOR_TIMEOUT]

    @property
    def timed_out_tasks(self) -> List[Dict[str, Any]]:
        """Get timed_out_tasks for other mechs."""
        return self.context.shared_state[TIMED_OUT_TASKS]

    @timed_out_tasks.setter
    def timed_out_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        """Set timed_out_tasks for other mechs."""
        self.context.shared_state[TIMED_OUT_TASKS] = tasks

    @property
    def unprocessed_timed_out_tasks(self) -> List[Dict[str, Any]]:
        """Get unprocessed timed_out_tasks for other mechs."""
        return self.context.shared_state[UNPROCESSED_TIMED_OUT_TASKS]

    @unprocessed_timed_out_tasks.setter
    def unprocessed_timed_out_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        """Set unprocessed timed_out_tasks for other mechs."""
        self.context.shared_state[UNPROCESSED_TIMED_OUT_TASKS] = tasks

    @property
    def done_tasks(self) -> List[Dict[str, Any]]:
        """Get done_tasks."""
        return self.context.shared_state[DONE_TASKS]

    @property
    def ipfs_tasks(self) -> List[Dict[str, Any]]:
        """Get ipfs_tasks."""
        return self.context.shared_state[IPFS_TASKS]

    @property
    def last_status_check(self) -> float:
        """Get last status check time."""
        return self.last_status_check_time

    @property
    def request_id_to_delivery_rate_info(self) -> Dict[str, int]:
        """Get request_id_to_delivery_rate_info."""
        return self.context.shared_state[REQUEST_ID_TO_DELIVERY_RATE_INFO]

    def _should_poll(self, req_type: str) -> bool:
        """If we should poll the contract."""
        last_polling = self.params.req_params.last_polling.get(req_type, None)

        if last_polling is None:
            return True
        return last_polling + self.params.polling_interval <= time.time()

    def _fetch_deadline(self) -> float:
        if self.params.is_cold_start:
            return time.time() + INITIAL_DEADLINE
        return time.time() + SUBSEQUENT_DEADLINE

    def _is_executing_task_ready(self) -> bool:
        """Check if the executing task is ready."""
        if self._executing_task is None or self._async_result is None:
            return False
        return self._async_result.done()

    def _has_executing_task_timed_out(self) -> bool:
        """Check if the executing task timed out."""
        if self._executing_task is None:
            return False
        timeout_deadline = self._executing_task.get("timeout_deadline", None)
        if timeout_deadline is None:
            return False
        return timeout_deadline <= time.time()

    def set_last_executed_task(self, request_id: int) -> None:
        """Set the last executed task."""
        self.context.shared_state[LAST_SUCCESSFUL_EXECUTED_TASK] = (
            request_id,
            time.time(),
        )

    def _download_tools(self) -> None:
        """Download tools."""
        if self._inflight_tool_req is not None:
            # there already is a req in flight
            return
        if len(self._tools_to_package_hash) == len(self._all_tools):
            # we already have all the tools
            return
        for tool, file_hash in self._tools_to_package_hash.items():
            if tool in self._all_tools:
                continue
            # read one at a time
            ipfs_msg, message = self._build_ipfs_get_file_req(file_hash)
            self._inflight_tool_req = tool
            self.send_message(ipfs_msg, message, self._handle_get_tool)
            return

    def _handle_get_tool(self, message: IpfsMessage, dialogue: Dialogue) -> None:
        """Handle get tool response"""
        component_yaml, tool_py, callable_method = ComponentPackageLoader.load(
            message.files
        )
        tool_req = cast(str, self._inflight_tool_req)
        self._all_tools[tool_req] = tool_py, callable_method, component_yaml
        self._inflight_tool_req = None

    def _populate_from_block(self) -> None:
        """Populate from_block"""
        ledger_api_msg, _ = self.context.ledger_dialogues.create(
            performative=LedgerApiMessage.Performative.GET_STATE,
            callable="get_block",
            kwargs=LedgerApiMessage.Kwargs(
                dict(block_identifier="latest", chain_id=self.params.default_chain_id)
            ),
            counterparty=LEDGER_API_ADDRESS,
            ledger_id=self.context.default_ledger_id,
            args=(),
        )
        self.context.outbox.put_message(message=ledger_api_msg)
        self.params.in_flight_req = True

    def _get_payment_types(self) -> bool:
        """Get the payment types of the mechs in the timed out tasks, if not available in the cache."""
        if self.params.in_flight_req:
            return False

        to_request = {
            mech_address
            for task in self.unprocessed_timed_out_tasks
            if (mech_address := task["priorityMech"]) not in self.payment_info
            and mech_address != self.params.agent_mech_contract_address
        }

        if not to_request:
            return True

        self.context.logger.info(
            f"Getting mech types for addresses {to_request} not found in cache."
        )

        contract_api_msg, _ = self.context.contract_dialogues.create(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=self.params.agent_mech_contract_address,
            contract_id=str(OlasMechContract.contract_id),
            callable="get_mech_types",
            kwargs=ContractApiMessage.Kwargs(
                dict(mech_addresses=to_request, chain_id=self.params.default_chain_id)
            ),
            counterparty=LEDGER_API_ADDRESS,
            ledger_id=self.context.default_ledger_id,
        )
        self.context.outbox.put_message(message=contract_api_msg)
        self.params.in_flight_req = True
        return False

    def _filter_out_incompatible_reqs(self) -> None:
        """Filter out incompatible requests based on the payment model."""
        if not self.params.use_mech_marketplace:
            return

        done = self._get_payment_types()
        if not done:
            return

        same_payment_type = []
        while self.unprocessed_timed_out_tasks:
            task = self.unprocessed_timed_out_tasks.pop()
            req_mech = task["priorityMech"]
            if req_mech not in self.payment_info:
                # not stepping in for self
                if req_mech != self.params.agent_mech_contract_address:
                    # this should not happen
                    self.context.logger.warning(
                        f"A mech address for which there is no payment type information was found in pending {task=}! Dropping the task."
                    )
                continue

            req_mech_pm = self.payment_info[req_mech]
            if req_mech_pm == self.payment_model:
                # the timed out task can be processed by this mech as they share the same payment type model
                same_payment_type.append(task)
                continue

            self.context.logger.info(
                f"Filtering out incompatible request {task}. "
                f"Requested mech's payment model: {req_mech_pm} != {self.payment_model}."
            )

        self.timed_out_tasks = same_payment_type[: self.params.step_in_list_size]

    def _check_for_new_marketplace_reqs(self) -> None:
        """Check for new reqs."""
        now = time.time()

        # This prevents readiness from going stale while we're legitimately busy.
        if self.params.in_flight_req:
            self.context.shared_state[INFLIGHT_READ_TS] = now
            return

        # If we're within our polling cadence, record an "attempt tick" so readiness
        # can treat this as fresh enough even though we didn't touch the dependency yet.
        if not self._should_poll(RequestType.MARKETPLACE.value):
            self.context.shared_state[LAST_READ_ATTEMPT_TS] = now
            return

        from_block = self.params.req_params.from_block.get(
            RequestType.MARKETPLACE.value, None
        )
        self.context.logger.info(
            f"Checking for new marketplace requests from block {from_block}..."
        )
        if from_block is None:
            # set the initial from block
            self._populate_from_block()
            self.params.req_type = RequestType.MARKETPLACE.value
            self.context.shared_state[LAST_READ_ATTEMPT_TS] = now
            return

        # We are actually going to poll → stamp both attempt and inflight.
        self.context.shared_state[LAST_READ_ATTEMPT_TS] = now
        self.context.shared_state[INFLIGHT_READ_TS] = now

        self._check_undelivered_reqs_marketplace()
        self.params.in_flight_req = True
        self.params.req_params.last_polling[RequestType.MARKETPLACE.value] = time.time()

    def _check_undelivered_reqs_marketplace(self) -> None:
        """Check for undelivered mech reqs."""
        if not self.params.use_mech_marketplace:
            return

        contract_api_msg, _ = self.context.contract_dialogues.create(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=self._get_designated_marketplace_mech_address(),
            contract_id=str(MechMarketplaceContract.contract_id),
            callable="get_marketplace_undelivered_reqs",
            kwargs=ContractApiMessage.Kwargs(
                dict(
                    from_block=self.params.req_params.from_block.get(
                        RequestType.MARKETPLACE.value
                    ),
                    chain_id=self.params.default_chain_id,
                    max_block_window=self.params.max_block_window,
                    marketplace_address=self.params.mech_marketplace_address,
                    wait_for_timeout_tasks=self.wait_for_timeout_tasks,
                    timeout_tasks=self.timed_out_tasks,
                )
            ),
            counterparty=LEDGER_API_ADDRESS,
            ledger_id=self.context.default_ledger_id,
        )
        self.params.req_type = RequestType.MARKETPLACE.value
        self.context.outbox.put_message(message=contract_api_msg)

    def _execute_ipfs_tasks(self) -> None:
        """Execute IPFS tasks."""

        if (
            self._inflight_ipfs_req
            or self._inflight_tool_req
            or self.params.in_flight_req
            or self.ipfs_tasks is None
            or len(self.ipfs_tasks) == 0
        ):
            return

        self.context.logger.info(f"Found {len(self.ipfs_tasks)} IPFS tasks.")
        ipfs_task = self.ipfs_tasks.pop(0)
        request_id = ipfs_task["request_id"]
        ipfs_data = ipfs_task["ipfs_data"]
        self.context.logger.info(
            f"Preparing ipfs task for request id {request_id} with data: {ipfs_data}"
        )
        self._inflight_ipfs_req = request_id
        msg, dialogue = self._build_ipfs_store_file_req({"metadata.json": ipfs_data})
        self.send_message(msg, dialogue, self._handle_ipfs_tasks_response)

    def _ensure_deadline(self) -> None:
        """Set a deadline if not set already, otherwise continue."""
        if self._request_handling_deadline is None:
            self._request_handling_deadline = self._fetch_deadline()
            self.context.logger.info(
                f"Deadline set to {self._request_handling_deadline} for task {self._executing_task}."
            )

    def _execute_task(self) -> None:
        """Execute tasks."""
        # check if there is a task already executing
        if self.params.in_flight_req:
            # there is an in flight request

            if self._executing_task:
                self._ensure_deadline()

            # check if the executing task is within deadline or not
            if self._executing_task and time.time() > cast(
                float, self._request_handling_deadline
            ):
                # Deadline reached, restart the task execution
                self.context.logger.info(
                    f"Request handling deadline reached for task {self._executing_task}. Restarting task execution..."
                )
                self._handle_timeout_task()
            return

        if self._executing_task is not None:
            req_id = self._executing_task.get("requestId", None)
            if self._current_request_id != req_id:
                self._current_request_id = req_id
                self.context.logger.info(f"Waiting for task: {self._executing_task}")

            if self._is_executing_task_ready() or self._invalid_request:
                task_result = self._get_executing_task_result()
                self._handle_done_task(task_result)
            elif self._has_executing_task_timed_out():
                self._handle_timeout_task()
            return

        self.mech_metrics.set_gauge(
            self.mech_metrics.mech_pending_queue_len, len(self.pending_tasks)
        )
        self.mech_metrics.set_gauge(
            self.mech_metrics.mech_timed_out_queue_len, len(self.timed_out_tasks)
        )
        self.mech_metrics.set_gauge(
            self.mech_metrics.mech_wait_for_time_out_queue_len,
            len(self.wait_for_timeout_tasks),
        )

        if len(self.pending_tasks) == 0:
            if len(self.timed_out_tasks) == 0:
                return
            task_data = self.timed_out_tasks.pop(0)
        else:
            task_data = self.pending_tasks.pop(0)
        self.context.logger.info(f"Preparing task with data: {task_data}")
        # Start the time counter to measure time taken to prepare the task
        self.tool_preparation_start_time = time.perf_counter()
        self.mech_metrics.inc_counter(self.mech_metrics.mech_tasks_started_total)
        # convert request id to int if it's bytes
        if type(task_data.get("requestId")) == bytes:
            request_id = task_data["requestId"]
            task_data["requestId"] = int.from_bytes(request_id, byteorder="big")

        request_id = task_data["requestId"]
        self.mech_metrics.set_gauge(
            self.mech_metrics.mech_tasks_inflight,
            request_id,
        )
        delivery_rate = task_data["request_delivery_rate"]
        self.request_id_to_delivery_rate_info[request_id] = delivery_rate
        self._executing_task = task_data
        self._request_handling_deadline = None
        try:
            ipfs_hash = get_ipfs_file_hash(task_data["data"])
        except Exception as e:  # pylint: disable=W0718
            self.context.logger.warning(
                f"Malformed IPFS data for request {request_id}: {e}. "
                f"Skipping request."
            )
            self._invalid_request = True
            return
        self.context.logger.info(f"IPFS hash: {ipfs_hash}")
        ipfs_msg, message = self._build_ipfs_get_file_req(ipfs_hash)
        self.send_message(
            ipfs_msg,
            message,
            self._handle_get_task,
            self._handle_ipfs_error,
        )

    def _update_pending_tasks(self) -> None:
        if not self.params.use_mech_marketplace:
            return

        # there is an in flight request
        if self.params.in_flight_req:
            return

        # status check interval not reached
        if self.last_status_check + STATUS_CHECK_INTERVAL > time.time():
            return

        pending_tasks_count = len(self.pending_tasks)
        # no pending tasks to check
        if pending_tasks_count == 0:
            return

        self.context.logger.info(
            f"Checking status change for {pending_tasks_count} pending tasks..."
        )
        pending_tasks_request_ids = [t["requestId"] for t in self.pending_tasks]

        contract_api_msg, _ = self.context.contract_dialogues.create(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=self.params.mech_marketplace_address,
            contract_id=str(MechMarketplaceContract.contract_id),
            callable="fetch_batch_request_id_status",
            kwargs=ContractApiMessage.Kwargs(
                dict(
                    request_ids=pending_tasks_request_ids,
                    chain_id=self.params.default_chain_id,
                )
            ),
            counterparty=LEDGER_API_ADDRESS,
            ledger_id=self.context.default_ledger_id,
        )
        self.params.req_type = RequestType.MARKETPLACE.value
        self.context.outbox.put_message(message=contract_api_msg)
        self.params.in_flight_req = True
        self.last_status_check_time = time.time()

    def send_message(
        self,
        msg: Message,
        dialogue: Dialogue,
        callback: Callable,
        error_callback: Optional[Callable] = None,
    ) -> None:
        """Send message."""
        self.context.outbox.put_message(message=msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        self.params.req_to_callback[nonce] = callback
        if error_callback is not None:
            self.params.req_to_error_callback[nonce] = error_callback
        self._ensure_deadline()
        self.params.req_to_deadline[nonce] = cast(
            float, self._request_handling_deadline
        )
        self.params.in_flight_req = True

    def _handle_ipfs_error(self, reason: str) -> None:
        """Handle an IPFS error reported by the handler.

        Classify the error: a reason containing the substring "timed out"
        is treated as a transient socket-level timeout from the IPFS
        client and routed through :meth:`_handle_timeout_task`, so the
        task is retried up to ``timeout_limit`` times before a terminal
        error is delivered on-chain. Everything else (malformed content,
        gateway HTTP errors, missing content) is a terminal failure and
        marks the request invalid on the first attempt.

        The "timed out" substring is the message forwarded verbatim by
        ``aea_cli_ipfs.ipfs_client.TimeoutError``, which wraps the
        underlying ``socket.timeout`` or ``urllib`` timeout. See
        ``plugins/aea-cli-ipfs/aea_cli_ipfs/ipfs_client.py`` in open-aea
        for the producing sites.

        :param reason: the error reason from the IPFS connection.
        """
        full_reason = (
            f"Request data could not be retrieved from IPFS (detail: {reason})"
        )
        if reason and "timed out" in reason.lower():
            # Transient timeout: retry via the existing timeout machinery.
            # Pass full_reason so terminal delivery (if timeout_limit is
            # reached) reflects the IPFS detail.
            self._handle_timeout_task(error_reason=full_reason)
            return
        self._ipfs_error_reason = full_reason
        self._invalid_request = True

    def _get_designated_marketplace_mech_address(self) -> str:
        """Get the designated mech address."""
        for mech, config in self.params.mech_to_config.items():
            if config.is_marketplace_mech:
                return mech

        raise ValueError("No marketplace mech address found")

    def _handle_done_task(self, task_result: Any) -> None:
        """Handle done tasks"""
        executed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        executing_task = cast(Dict[str, Any], self._executing_task)
        self.context.logger.info(f"Handling done task {executing_task}.")
        req_id = executing_task.get("requestId", None)
        request_id_nonce = executing_task.get("requestIdWithNonce", None)
        mech_address = (
            executing_task.get("contract_address", None)
            or self._get_designated_marketplace_mech_address()
        )
        tool = executing_task.get("tool", None)
        model = executing_task.get("model", None)
        tool_params = executing_task.get("params", None)
        is_offchain = executing_task.get("is_offchain", False)
        result_msg = self._ipfs_error_reason or "Invalid response"
        response = {
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "requestId": req_id,
            "result": result_msg,
            "tool": tool,
            "executed_at": executed_at,
        }
        task_executor = self.context.agent_address
        self._done_task = {
            "request_id": req_id,
            "mech_address": mech_address,
            "task_executor_address": task_executor,
            "tool": tool,
            "request_id_nonce": request_id_nonce,
            "is_offchain": is_offchain,
            **executing_task,
        }

        # compute tool execution duration before building metadata
        # if tool exec start time is 0.0, set to current time
        # it can be 0.0 if _prepare_task was not called due to other checks such as tool not valid
        # or stepping in but tool not found or tool to pricing not found for dynamic mechs
        tool_exec_time_duration = time.perf_counter() - (
            self.tool_execution_start_time or time.perf_counter()
        )

        if task_result is not None and len(task_result) >= 5:
            # task succeeded — unpack based on tuple length
            # 6-tuple: tool returned used_params (new contract)
            # 5-tuple: tool did not return used_params (old contract)
            if len(task_result) == 6:
                (
                    deliver_msg,
                    prompt,
                    transaction,
                    counter_callback,
                    used_params,
                    keychain,
                ) = task_result
            else:
                deliver_msg, prompt, transaction, counter_callback, keychain = (
                    task_result
                )
                used_params = None
            cost_dict = {}
            actual_model = None
            if counter_callback is not None:
                cost_dict = cast(TokenCounterCallback, counter_callback).cost_dict
                actual_model = cast(TokenCounterCallback, counter_callback).actual_model
            # prefer runtime params reported by the tool, fall back to component.yaml
            resolved_params = used_params if used_params is not None else tool_params
            metadata = {
                "model": actual_model or model,
                "tool": tool,
                "tool_hash": self._tools_to_package_hash.get(tool or ""),
                "execution_latency_ms": int(tool_exec_time_duration * 1000),
            }
            if resolved_params is not None:
                metadata["params"] = resolved_params
            response = {
                **response,
                "result": deliver_msg,
                "prompt": prompt,
                "cost_dict": cost_dict,
                "metadata": metadata,
                "is_offchain": is_offchain,
            }
            self._done_task["transaction"] = transaction

            # update the keychain, it's possible that rotations happened
            # we want to use the most up-to-date key priority
            self._keychain = keychain
        self.context.logger.info(
            f"Request id {req_id!r} with {tool=}, took {tool_exec_time_duration} seconds to complete execution. "
            f"Request's result:\n{task_result}"
        )
        # reset the time counter used to measure time taken to execute the task
        self.tool_execution_start_time = 0.0
        self.mech_metrics.observe_histogram(
            self.mech_metrics.mech_tool_execution_time,
            tool_exec_time_duration,
            tool=tool,
        )
        # Start the time counter to measure time taken to deliver the task
        self.tool_deliver_start_time = time.perf_counter()
        self.mech_metrics.inc_counter(
            self.mech_metrics.mech_tasks_completed_total, tool=tool
        )
        if self._invalid_request:
            self.mech_metrics.inc_counter(
                metric=self.mech_metrics.mech_tasks_failed_total,
                tool=tool,
                reason=response["result"],
            )

        if is_offchain:
            # Off-chain path: skip the IPFS upload so the response stays private
            # and commit a locally-computed CIDv1 instead. This is a bare
            # single-block file CID; it intentionally differs from the on-chain
            # path's directory-wrapped CID (see utils/local_cid.py). The on-chain
            # commitment derivation (to_multihash, inside _finalize_done_task) is
            # otherwise identical.
            if self._invalid_request:
                # The task ran but produced no valid result, so `response`
                # carries an error string rather than an answer. Route it through
                # the same terminal-failure channel as the CID / done-task
                # failures below instead of serving it as a success: a paying
                # requester keys refund / retry / dispute on `status`, so a "ran
                # but failed" delivery must be distinguishable from a successful
                # one — otherwise the client accepts and pays for a failure.
                self._record_offchain_failure(
                    cast(str, req_id),
                    cast(str, response.get("result") or "task execution failed"),
                )
                return
            try:
                # content_bytes is exactly what the CID commits to. The serving
                # payload below carries this same object verbatim under
                # "response" (envelope fields hang outside it), so a client that
                # re-derives compute_cidv1(json.dumps(stored["response"]))
                # reproduces content_cid byte-for-byte and can verify the
                # delivery against the on-chain commitment. json.dumps round-trips
                # stably here (default separators, dict insertion order is
                # preserved through JSON), matching the on-chain path which also
                # content-addresses json.dumps(response).
                content_bytes = json.dumps(response).encode("utf-8")
                local_cid = compute_cidv1(content_bytes)
            except (ValueError, TypeError) as exc:
                # compute_cidv1 raises ValueError above the single-block bound;
                # json.dumps raises TypeError (non-serializable) / ValueError
                # (circular ref). Both must stay inside this guard, or the crash
                # this fix targets still escapes through act(). Fail cleanly so
                # the executing-task slot resets and the requester gets a
                # definitive rejection, rather than crashing the agent
                # (propagate policy) or stalling forever (just_log policy).
                self.context.logger.error(
                    f"Off-chain CID computation failed for request {req_id}: {exc}"
                )
                self._record_offchain_failure(
                    cast(str, req_id), f"cid computation failed: {exc}"
                )
                return
            self.context.logger.info(
                f"Off-chain response for request {req_id} kept private; "
                f"local CID {local_cid}."
            )
            # Off-chain return channel: the response was not uploaded to IPFS, so
            # persist it under offchain_request_responses for /fetch_offchain_info
            # to serve. The committed object is kept verbatim under "response"
            # (the exact bytes hashed into content_cid); request_id / status /
            # content_cid are envelope fields kept outside it so they don't
            # perturb the hash. Without this the poll falls back to the done_task,
            # which carries the CID/multihash but not the result/prompt/cost_dict.
            self.context.shared_state.setdefault(OFFCHAIN_REQUEST_RESPONSES, {})[
                cast(str, req_id)
            ] = {
                "request_id": cast(str, req_id),
                "status": "ok",
                "content_cid": local_cid,
                "response": response,
            }
            self._finalize_done_task(local_cid)
            return

        msg, dialogue = self._build_ipfs_store_file_req(
            {str(req_id): json.dumps(response)}
        )
        self.send_message(msg, dialogue, self._handle_store_response)

    def _restart_executor(self) -> None:
        """Restarts the executor."""
        self._executor.stop()
        self._executor.join(timeout=10.0)
        # create a new executor
        self._executor = ProcessPool(max_workers=1)

    def _handle_timeout_task(self, error_reason: Optional[str] = None) -> None:
        """Handle timeout tasks.

        :param error_reason: optional underlying error reason (e.g. an IPFS
            timeout detail) to include in the terminal delivery when
            ``timeout_limit`` has been reached. When provided, this takes
            precedence over any previously set ``_ipfs_error_reason``.
        :returns: None
        """
        # Preserve the reason across the reset below so that, if we end
        # up delivering a terminal result due to timeout_limit, the
        # message reflects the underlying cause instead of only the
        # generic "timed out N times" string.
        preserved_reason = error_reason or self._ipfs_error_reason
        self.params.in_flight_req = False
        self.params.is_cold_start = False
        self._request_handling_deadline = None
        self._ipfs_error_reason = None
        # reset all times
        self.tool_preparation_start_time = 0.0
        self.tool_execution_start_time = 0.0

        executing_task = cast(Dict[str, Any], self._executing_task)
        req_id = executing_task.get("requestId", None)

        # This should never be the case but added to handle all cases for latest mypy updates
        if not req_id:
            self.context.logger.error(
                "Request id not found inside executing task for handle timedout task."
            )
            self._executing_task = None
            self._async_result = None
            return None

        # Prometheus has no way to remove/clear metrics, so we set to default 0
        self.mech_metrics.set_gauge(
            self.mech_metrics.mech_tasks_inflight,
            0,
        )
        tool = executing_task.get("tool", None)
        self.count_timeout(req_id)
        self.context.logger.info(f"Task timed out for request {req_id}")
        self.context.logger.info(
            f"Task {req_id} has timed out {self.request_id_to_num_timeouts[req_id]} times."
        )
        self.mech_metrics.inc_counter(
            metric=self.mech_metrics.mech_tasks_timed_out_total, tool=tool
        )
        if self._async_result:
            async_result = cast(Future, self._async_result)
            async_result.cancel()

        # we restart the executor in case of a timeout.
        # we do this because its possible the .cancel() call above is not respected
        # by the executor. Since we only have 1 process running at a time, this would
        # mean that the task being executed next would be queued. We want to avoid this.
        self.context.logger.info("Restarting executor.")
        self._restart_executor()

        # check if we can add the task to the end of the queue
        if self.timeout_limit_reached(req_id):
            # added to end of queue
            self.context.logger.info(
                f"Task {req_id} has reached the timeout limit of{self.params.timeout_limit}. "
                f"It won't be added to the end of the queue again."
            )
            base_msg = (
                f"Task timed out {self.params.timeout_limit} times during execution."
            )
            terminal_msg = (
                f"{base_msg} Last detail: {preserved_reason}"
                if preserved_reason
                else f"{base_msg} "
            )
            # _handle_done_task uses _ipfs_error_reason as the "result"
            # field of the on-chain response on the failure path.
            self._ipfs_error_reason = terminal_msg
            task_result = (
                terminal_msg,
                "",
                None,
                None,
            )
            return self._handle_done_task(task_result)

        self.context.logger.info(f"Adding task {req_id} to the end of the queue.")
        self.pending_tasks.append(executing_task)
        self._executing_task = None
        self._async_result = None
        return None

    def _handle_get_task(self, message: IpfsMessage, dialogue: Dialogue) -> None:
        """Handle the response from ipfs for a task request."""
        if (
            self._request_handling_deadline
            and time.time() > self._request_handling_deadline
        ):
            self.context.logger.warning(
                "Task data arrived after deadline. "
                "Skipping tool execution and cleaning up."
            )
            self._executing_task = None
            self._async_result = None
            self._request_handling_deadline = None
            self.tool_preparation_start_time = 0.0
            return

        executing_task = cast(Dict[str, Any], self._executing_task)
        req_id = executing_task.get("requestId", "unknown")
        total_bytes = sum(
            (
                len(content)
                if isinstance(content, (bytes, bytearray))
                else len(content.encode("utf-8"))
            )
            for content in message.files.values()
        )
        if total_bytes > IPFS_MAX_TASK_BYTES:
            self.context.logger.warning(
                f"IPFS task payload for request {req_id} is "
                f"{total_bytes} bytes, exceeds cap of {IPFS_MAX_TASK_BYTES}. "
                f"Skipping request."
            )
            self._invalid_request = True
            return

        try:
            task_data = [json.loads(content) for content in message.files.values()][0]
        except (json.JSONDecodeError, IndexError, TypeError, UnicodeDecodeError) as e:
            self.context.logger.warning(
                f"Malformed IPFS content for request {req_id}: {e}. "
                f"Skipping request."
            )
            self._invalid_request = True
            return
        is_data_valid = (
            task_data
            and isinstance(task_data, dict)
            and isinstance(task_data.get("prompt"), str)
            and isinstance(task_data.get("tool"), str)
        )  # pylint: disable=C0301

        if not is_data_valid:
            self.context.logger.warning(f"Invalid {task_data=} for {executing_task=}.")
            self._invalid_request = True
            return

        prompt_bytes = len(task_data["prompt"].encode("utf-8"))
        if prompt_bytes > MAX_PROMPT_BYTES:
            self.context.logger.warning(
                f"Prompt for request {req_id} is {prompt_bytes} bytes, "
                f"exceeds cap of {MAX_PROMPT_BYTES}. Skipping request."
            )
            self._invalid_request = True
            return

        my_mech = self._get_designated_marketplace_mech_address().lower()
        exec_prio = str(executing_task.get("priorityMech", "")).lower()
        stepping_in = exec_prio != my_mech
        tool_name = task_data["tool"]
        if stepping_in and tool_name not in self._tools_to_package_hash:
            rid = int(executing_task["requestId"])
            self._ignored_request_ids.add(rid)
            reason = f"Cannot step in. Tool {tool_name} is not installed. Ignoring request {rid}."
            self.context.logger.info(reason)
            self.mech_metrics.inc_counter(
                metric=self.mech_metrics.mech_tasks_failed_total,
                tool=tool_name,
                reason=reason,
            )
            # Prometheus has no way to remove/clear metrics, so we set to default 0
            self.mech_metrics.set_gauge(
                self.mech_metrics.mech_tasks_inflight,
                0,
            )
            self._executing_task = None
            self._request_handling_deadline = None
            self._async_result = None
            # reset the time counter used to measure time taken to prepare the task
            self.tool_preparation_start_time = 0.0
            return

        if tool_name in self._tools_to_package_hash:
            if self._tools_to_pricing:
                tool_pricing = self._tools_to_pricing[tool_name]
                req_id_delivery_rate = self.request_id_to_delivery_rate_info[
                    executing_task["requestId"]
                ]
                if req_id_delivery_rate < tool_pricing:
                    reason = f"Requested pricing invalid. Actual {req_id_delivery_rate}, needed {tool_pricing}."
                    self.context.logger.warning(reason)
                    self.mech_metrics.inc_counter(
                        metric=self.mech_metrics.mech_tasks_failed_total,
                        tool=tool_name,
                        reason=reason,
                    )
                    self._invalid_request = True
                    # reset the time counter used to measure time taken to prepare the task
                    self.tool_preparation_start_time = 0.0
                    return

            rid = int(executing_task["requestId"])
            # fetch the time duration for tool preparation to complete
            tool_prep_time_duration = (
                time.perf_counter() - self.tool_preparation_start_time
            )
            self.context.logger.info(
                f"Request id {rid} with {tool_name=} took {tool_prep_time_duration} seconds to prepare."
            )
            # reset the time counter used to measure time taken to prepare the task
            self.tool_preparation_start_time = 0.0
            self.mech_metrics.observe_histogram(
                self.mech_metrics.mech_tool_preparation_time,
                tool_prep_time_duration,
                tool=tool_name,
            )
            # Start the time counter to measure time taken to execute the task
            self.tool_execution_start_time = time.perf_counter()
            self._prepare_task(task_data)
        else:
            # Unknown tool and we're the priority mech -> store stub (existing behavior)
            executing_task["tool"] = tool_name
            reason = f"Tool {tool_name} is not valid."
            self.context.logger.warning(reason)
            self.mech_metrics.inc_counter(
                metric=self.mech_metrics.mech_tasks_failed_total,
                tool=tool_name,
                reason=reason,
            )
            self._invalid_request = True
            # reset the time counter used to measure time taken to prepare the task
            self.tool_preparation_start_time = 0.0

    def _submit_task(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Submit a task with a hard per-call timeout via Pebble."""
        try:
            return self._executor.schedule(  # type: ignore[attr-defined]
                fn,
                args=args,
                kwargs=kwargs,
                timeout=float(self.params.task_deadline),
            )
        except Exception:
            self.context.logger.warning("Executor is broken. Restarting...")
            self._restart_executor()
            return self._executor.schedule(  # type: ignore[attr-defined]
                fn,
                args=args,
                kwargs=kwargs,
                timeout=float(self.params.task_deadline),
            )

    def _get_executing_task_result(self) -> Any:
        """Get the executing task result (convert Pebble errors to your 5-tuple)."""
        if self._executing_task is None:
            self.context.logger.warning("No executing task found to get a result from.")
            return None

        if self._invalid_request:
            self.context.logger.warning(
                "Cannot get executing task result for invalid request."
            )
            return None

        if self._async_result is None:
            self.context.logger.warning(
                f"No result found for the executing task: {self._executing_task}."
            )
            return None

        try:
            return self._async_result.result()
        except TimeoutError as e:
            self.context.logger.warning(f"Task expired: {e}")
        except Exception as e:
            self.context.logger.error(f"Exception during task: {e}")

    def _prepare_task(self, task_data: Dict[str, Any]) -> None:
        """Prepare the task."""
        executing_task = cast(Dict[str, Any], self._executing_task)
        self.context.logger.info(
            f"Preparing tool task: {executing_task} with data: {task_data}"
        )
        tool_task = AnyToolAsTask()
        tool_py, callable_method, component_yaml = self._all_tools[task_data["tool"]]
        tool_params = component_yaml.get("params", {})
        task_data["tool_py"] = tool_py
        task_data["callable_method"] = callable_method
        task_data["api_keys"] = self._keychain
        task_data["counter_callback"] = TokenCounterCallback()
        task_data["model"] = task_data.get(
            "model", tool_params.get("default_model", None)
        )
        future = self._submit_task(tool_task.execute, **task_data)
        executing_task["timeout_deadline"] = time.time() + self.params.task_deadline
        executing_task["tool"] = task_data["tool"]
        executing_task["model"] = task_data.get(
            "model", tool_params.get("default_model", None)
        )
        executing_task["params"] = tool_params
        self._async_result = cast(Optional[Future], future)

    def _build_ipfs_message(
        self,
        performative: IpfsMessage.Performative,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[IpfsMessage, IpfsDialogue]:
        """Builds an IPFS message."""
        ipfs_dialogues = cast(IpfsDialogues, self.context.ipfs_dialogues)
        message, dialogue = ipfs_dialogues.create(
            counterparty=str(IPFS_CONNECTION_ID),
            performative=performative,
            timeout=timeout,
            **kwargs,
        )
        return message, dialogue

    def _build_ipfs_store_file_req(  # pylint: disable=too-many-arguments
        self,
        filename_to_obj: Dict[str, str],
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[IpfsMessage, IpfsDialogue]:
        """Builds a STORE_FILES ipfs message."""
        message, dialogue = self._build_ipfs_message(
            performative=IpfsMessage.Performative.STORE_FILES,  # type: ignore
            files=filename_to_obj,
            timeout=timeout,
            **kwargs,
        )
        return message, dialogue

    def _build_ipfs_get_file_req(
        self,
        ipfs_hash: str,
        timeout: Optional[float] = None,
    ) -> Tuple[IpfsMessage, IpfsDialogue]:
        """
        Builds a GET_FILES IPFS request.

        :param ipfs_hash: the ipfs hash of the file/dir to download.
        :param timeout: timeout for the request.
        :returns: the ipfs message, and its corresponding dialogue.
        """
        message, dialogue = self._build_ipfs_message(
            performative=IpfsMessage.Performative.GET_FILES,  # type: ignore
            ipfs_hash=ipfs_hash,
            timeout=timeout,
        )
        return message, dialogue

    def _handle_store_response(self, message: IpfsMessage, dialogue: Dialogue) -> None:
        """Handle the response from ipfs for a store response request."""
        if not self._executing_task:
            self.context.logger.error(
                "No executing task found while trying to handle a store response from IPFS."
            )
            return

        ipfs_hash = to_v1(message.ipfs_hash)
        req_id = self._executing_task["requestId"]
        self.context.logger.info(
            f"Response for request {req_id} stored on IPFS with hash {ipfs_hash}."
        )
        self._finalize_done_task(ipfs_hash)

    def _finalize_done_task(self, cid: str) -> None:
        """Apply the CID to the in-flight done_task and publish it.

        Shared by the on-chain path (CID from a real IPFS store response) and the
        off-chain path (CID computed locally). Updates pricing / mech address /
        task_result, appends to done_tasks under the lock, and resets the
        executing-task slot. The CID is converted to the multihash shape only
        after the done_task validity check passes, so an invalid done_task takes
        the early-reset branch cleanly.

        :param cid: the multibase-encoded CIDv1 to record on chain via
            ``task_result``.
        """
        executing_task = self._executing_task
        if executing_task is None:
            # Two entry points call this now; if the slot was already cleared
            # there is nothing to finalize. Guard turns a None-deref into a log.
            self.context.logger.warning(
                "_finalize_done_task called with no executing task; skipping."
            )
            return
        req_id = cast(Dict[str, Any], executing_task)["requestId"]
        is_offchain = bool(executing_task.get("is_offchain", False))
        self.set_last_executed_task(req_id)
        done_task = cast(Dict[str, Any], self._done_task)
        if done_task is None or not isinstance(done_task, Dict):
            self.context.logger.error(
                f"Invalid done task format. Expected Dict. Actual: {done_task}"
            )
            if is_offchain:
                # On-chain has other fallbacks; off-chain the polling client would
                # otherwise never learn this failed, so emit a definitive
                # rejection (which also resets the task slot).
                self._record_offchain_failure(req_id, "invalid done task")
                return
            self._reset_executing_task()
            return

        task_result = to_multihash(cid)
        tool = str(done_task.get("tool"))
        dynamic_tool_cost = self._tools_to_pricing.get(tool)
        if dynamic_tool_cost is not None:
            self.context.logger.info(
                f"Tools to pricing found for tool {tool}. "
                f"Adding dynamic pricing of {dynamic_tool_cost} for request id {req_id}."
            )
            done_task["dynamic_tool_cost"] = dynamic_tool_cost
        else:
            cost = get_cost_for_done_task(done_task)
            self.context.logger.info(f"Cost for task {req_id}: {cost}.")
        mech_address = self._get_designated_marketplace_mech_address()
        mech_config = self.params.mech_to_config[mech_address.lower()]
        done_task["is_marketplace_mech"] = mech_config.is_marketplace_mech
        done_task["task_result"] = task_result
        # store the time the task was added to done task list
        done_task["start_time"] = time.perf_counter()
        # pop the data key value as it's bytes which causes issues
        # with json dumps and not required anywhere
        done_task.pop("data", None)
        # Attach the offchain wildcard-event payload before the
        # IN_MEMORY_REQUESTS pop below; the post-settlement behaviour in
        # task_submission_abci reads it off ``synchronized_data.done_tasks``
        # (i.e. after consensus replication) and posts it to the wildcard
        # data lake, but the buffered request metadata it needs is local
        # to this agent's shared_state and goes away at the pop.
        #
        # Gated on BOTH ``is_offchain`` AND the ``mech_events_enabled`` flag
        # so Phase 1 is genuinely dark: when the flag is off, no payload is
        # built and nothing rides Tendermint consensus replication. The flag
        # check lives here as well as in the post-settlement behaviour so
        # the consensus-state cost is gated, not just the HTTP write.
        if is_offchain and getattr(self.params, "mech_events_enabled", False):
            done_task["wildcard_event"] = self._build_wildcard_event(
                done_task=done_task,
                cid=cid,
                executing_task=cast(Dict[str, Any], executing_task),
            )
        # add to done tasks, in thread safe way
        with self.done_tasks_lock:
            self.done_tasks.append(done_task)
        self.context.logger.info(f"Done task added to done tasks list: {done_task}.")
        # Off-chain success: drop the buffered request metadata so it can't grow
        # unbounded (on-chain tasks never populate it, so this is a no-op there).
        # .get keeps this safe if the handler-owned key was never initialised.
        self.context.shared_state.get(IN_MEMORY_REQUESTS, {}).pop(req_id, None)
        self._reset_executing_task()

    def _build_wildcard_event(
        self,
        *,
        done_task: Dict[str, Any],
        cid: str,
        executing_task: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the structured wildcard-event payload from on-hand data.

        The post-settlement behaviour batches every event from one FSM round
        into a single signed POST to the wildcard data lake. Fields are
        populated best-effort from the buffered request metadata
        (``IN_MEMORY_REQUESTS``), the offchain response
        (``OFFCHAIN_REQUEST_RESPONSES``), the in-flight ``executing_task``,
        and skill params. Optional fields that aren't readily available are
        left ``None``; the wildcard server accepts a NULL there. Required
        fields default to safe placeholders rather than blocking the row: a
        malformed event surfaces as a 422 when the wildcard write fires (and
        lands in the local replay buffer so it's recoverable), which is
        preferable to silently dropping analytics data.

        Datetimes are emitted as ISO 8601 strings with UTC offset so the
        wildcard ``AwareDatetime`` parser accepts them; mech ints (Unix
        seconds) are converted here so the structured event can ride the
        FSM consensus channel without round-tripping through a custom
        codec.

        :param done_task: the done-task dict appended to ``done_tasks``.
        :param cid: the locally-computed multibase CIDv1 for the response.
        :param executing_task: the original request dict from the handler.
        :return: a ``MechSettlementEvent``-shaped dict ready for FSM consensus
            replication and downstream signing in the post-tx behaviour.
        """
        req_id = done_task["request_id"]
        request_id_str = str(req_id)
        request_data_raw = self.context.shared_state.get(IN_MEMORY_REQUESTS, {}).get(
            req_id, {}
        )
        # IPFS_DATA can land as either a parsed dict (normal handler path)
        # or a JSON-encoded string (some test fixtures and the historical
        # backfill path); normalise to dict here so downstream ``.get``
        # calls don't crash on a string.
        if isinstance(request_data_raw, (bytes, bytearray)):
            try:
                request_data_raw = request_data_raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                self.context.logger.warning(
                    "request_data for req_id=%s is non-UTF-8 bytes; "
                    "using empty payload: %s",
                    req_id,
                    exc,
                )
                request_data_raw = ""
        if isinstance(request_data_raw, str):
            try:
                request_data = json.loads(request_data_raw) if request_data_raw else {}
                if not isinstance(request_data, dict):
                    request_data = {"raw": request_data_raw}
            except (ValueError, TypeError):
                request_data = {"raw": request_data_raw}
        elif isinstance(request_data_raw, dict):
            request_data = request_data_raw
        else:
            request_data = {}
        # OFFCHAIN_REQUEST_RESPONSES stores an envelope
        # ``{"request_id":..., "status":"ok", "content_cid":..., "response":<inner>}``;
        # the tool's actual ``result`` / ``executed_at`` live on the inner
        # ``response`` dict, not on the envelope. Reading the envelope
        # directly (the original code) returned ``None`` for both fields,
        # forcing every completed task to be tagged ``status="failed"`` in
        # the analytics row.
        response_envelope_raw = self.context.shared_state.get(
            OFFCHAIN_REQUEST_RESPONSES, {}
        ).get(req_id, {})
        response_envelope = (
            response_envelope_raw if isinstance(response_envelope_raw, dict) else {}
        )
        inner_response = response_envelope.get("response", {})
        response_data = inner_response if isinstance(inner_response, dict) else {}
        tool = str(done_task.get("tool") or "unknown")
        delivery_mech = str(
            done_task.get("mech_address") or self.params.mech_marketplace_address or ""
        )

        # `start_time` is a perf_counter() value (monotonic, not wall-clock),
        # not suitable as a delivered_at timestamp. Use the current wall-clock
        # at finalize as the best-effort delivered_at; the analytics ETL can
        # cross-correlate against the on-chain block timestamp if needed.
        now_iso = datetime.now(timezone.utc).isoformat()
        executed_at_unix = response_data.get("executed_at")
        executed_at_iso = (
            datetime.fromtimestamp(executed_at_unix, tz=timezone.utc).isoformat()
            if isinstance(executed_at_unix, (int, float))
            else now_iso
        )
        # Requested_at: prefer the signed timestamp on the buffered request;
        # the handler validates and stores it before enqueueing. Fall back to
        # the executed_at on the response so the time-leading index entry is
        # always populated for an offchain row. The requester can store the
        # value as a Unix int or as an ISO 8601 string — ``str(int)`` would
        # produce ``"1700000000"`` which the wildcard's ``AwareDatetime``
        # parser rejects with a 422, so coerce int / float explicitly.
        requested_at_raw = request_data.get("datetime") or request_data.get(
            "requested_at"
        )
        if isinstance(requested_at_raw, (int, float)):
            requested_at_iso = datetime.fromtimestamp(
                requested_at_raw, tz=timezone.utc
            ).isoformat()
        elif isinstance(requested_at_raw, str) and requested_at_raw:
            requested_at_iso = requested_at_raw
        else:
            requested_at_iso = executed_at_iso

        # ``or``-chain would discard a legitimate ``0`` value (which is
        # both falsy and a valid nonce / delivery rate); use an explicit
        # ``in`` probe so a zero in the canonical key wins over a fallback
        # ``None`` in the legacy key.
        if "request_delivery_rate" in executing_task:
            delivery_rate_raw = executing_task["request_delivery_rate"]
        else:
            delivery_rate_raw = executing_task.get("delivery_rate")
        delivery_rate_str = (
            str(delivery_rate_raw) if delivery_rate_raw is not None else None
        )
        if "request_id_nonce" in executing_task:
            nonce_raw = executing_task["request_id_nonce"]
        else:
            nonce_raw = executing_task.get("nonce")
        try:
            nonce_int = int(nonce_raw) if nonce_raw is not None else None
        except (TypeError, ValueError):
            nonce_int = None
        # The wildcard server requires ``prompt`` to be a non-empty string;
        # placeholder rather than dropping the row. Enforce the cap on the
        # UTF-8 BYTE length, not on ``len()`` (code points) — a CJK prompt
        # of 50k code points encodes to ~150 KB and would otherwise sail
        # past the 100 KB cap; matches how ``IPFS_MAX_TASK_BYTES`` is
        # enforced elsewhere in this file.
        prompt = str(request_data.get("prompt") or "[offchain request]")
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) > MAX_PROMPT_BYTES:
            prompt = prompt_bytes[:MAX_PROMPT_BYTES].decode("utf-8", errors="ignore")
        invalid = bool(self._invalid_request)
        result_text = response_data.get("result") if response_data else None
        status = "failed" if invalid or result_text is None else "complete"
        # Failed-row ``error`` carries the available diagnostic, regardless
        # of which branch put us on the failed arm. When ``invalid`` is set
        # but no ``result_text`` is available (the tool produced no
        # response object), the field is ``None``; the wildcard server's
        # CHECK allows a ``failed`` row with empty error so the row still
        # lands as a settled failure.
        error_text = str(result_text) if status == "failed" and result_text else None
        # Strip the result from the failed arm so the wildcard's
        # status/error/result shape CHECK accepts the row.
        result_value: Optional[str] = (
            None if status == "failed" else (str(result_text) if result_text else "")
        )
        tool_hash = self._tools_to_package_hash.get(tool)
        cost_dict = (
            done_task.get("cost_dict")
            if isinstance(done_task.get("cost_dict"), dict)
            else None
        )

        return {
            "request": {
                "request_id": request_id_str,
                "chain_id": int(getattr(self.params, "mech_events_chain_id", 0) or 0),
                "marketplace_address": str(
                    getattr(self.params, "mech_marketplace_address", "") or ""
                ),
                "requester": str(executing_task.get("sender") or ""),
                "priority_mech": str(
                    executing_task.get("priority_mech") or delivery_mech
                ),
                "delivery_mech": delivery_mech,
                "payment_type": str(executing_task.get("payment_type") or "") or None,
                "delivery_rate": delivery_rate_str,
                "nonce": nonce_int,
                "content_cid": cid,
                "prompt": prompt,
                "tool": tool,
                "model": (
                    str(request_data.get("model"))
                    if request_data.get("model")
                    else None
                ),
                "tool_params": (
                    request_data.get("params")
                    if isinstance(request_data.get("params"), dict)
                    else None
                ),
                "raw_content": _cap_raw_content(
                    request_data
                    if isinstance(request_data, dict)
                    else {"prompt": prompt, "tool": tool}
                ),
                "requested_at": requested_at_iso,
            },
            "response": {
                "request_id": request_id_str,
                "delivery_mech": delivery_mech,
                "schema_version": RESPONSE_SCHEMA_VERSION,
                "result": result_value,
                "status": status,
                "error": error_text,
                "executed_at": executed_at_iso,
                "cost_dict": cost_dict,
                "is_offchain": True,
                "tool_hash": tool_hash,
                "execution_latency_ms": None,
                "params_used": (
                    done_task.get("used_params")
                    if isinstance(done_task.get("used_params"), dict)
                    else None
                ),
                "raw_content": _cap_raw_content(
                    response_data
                    if isinstance(response_data, dict)
                    else {"result": result_value or "", "status": status}
                ),
                "response_cid": None,
                "delivered_at": now_iso,
            },
        }

    def _reset_executing_task(self) -> None:
        """Reset the in-flight task slot so the next task can be picked up."""
        # Prometheus has no clear; set the in-flight gauge back to 0.
        self.mech_metrics.set_gauge(self.mech_metrics.mech_tasks_inflight, 0)
        self._executing_task = None
        self._done_task = None
        self._invalid_request = False
        self._ipfs_error_reason = None
        self._request_handling_deadline = None
        self._async_result = None

    def _record_offchain_failure(self, req_id: str, reason: str) -> None:
        """Record an off-chain request failure and reset the task slot.

        Writes the same ``{request_id, status, reason}`` shape the HTTP handler's
        rejection path uses, so a poll of ``GET /fetch_offchain_info`` returns a
        definitive rejection instead of an empty "still processing" body forever.

        :param req_id: the off-chain request id that failed.
        :param reason: a short human-readable failure reason.
        """
        # setdefault / get keep this safe if the handler-owned shared-state keys
        # were never initialised (e.g. on-chain-only deployments, unit tests).
        self.context.shared_state.setdefault(OFFCHAIN_REQUEST_RESPONSES, {})[req_id] = {
            "request_id": req_id,
            "status": "rejected",
            "reason": reason,
        }
        self.context.shared_state.get(IN_MEMORY_REQUESTS, {}).pop(req_id, None)
        self._reset_executing_task()

    def _handle_ipfs_tasks_response(
        self, message: IpfsMessage, dialogue: Dialogue
    ) -> None:
        """Handle the response from ipfs for a stored request."""
        request_id = cast(str, self._inflight_ipfs_req)
        ipfs_hash = to_v1(message.ipfs_hash)
        self.context.logger.info(
            f"Response for request {request_id} stored on IPFS with hash {ipfs_hash}."
        )
        # remove the uploaded request from the IPFS tasks
        self.context.shared_state[IPFS_TASKS] = [
            t for t in self.ipfs_tasks if t.get("request_id") != request_id
        ]
        self._inflight_ipfs_req = None

    def send_data_via_acn(
        self,
        sender_address: str,
        request_id: str,
        data: Any,
    ) -> None:
        """Handle callbacks."""
        self.context.logger.info(
            f"Sending data to {sender_address} via ACN for request ID {request_id}"
        )
        response, _ = cast(
            AcnDataShareDialogues, self.context.acn_data_share_dialogues
        ).create(
            counterparty=sender_address,
            performative=AcnDataShareMessage.Performative.DATA,
            request_id=request_id,
            content=data,
        )
        self.context.outbox.put_message(
            message=response,
            context=EnvelopeContext(connection_id=P2P_CLIENT_PUBLIC_ID),
        )
