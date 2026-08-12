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

import base64
import concurrent.futures
import json
import re
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

from aea.protocols.base import Message
from aea.skills.base import Handler
from aea_ledger_ethereum import EthereumApi
from prometheus_client import start_http_server
from requests.exceptions import RequestException
from web3.exceptions import BadFunctionCallOutput, ContractLogicError, Web3RPCError

from packages.valory.connections.ledger.connection import (
    PUBLIC_ID as LEDGER_CONNECTION_PUBLIC_ID,
)
from packages.valory.contracts.balance_tracker.contract import BalanceTrackerContract
from packages.valory.contracts.mech_marketplace.contract import MechMarketplaceContract
from packages.valory.contracts.olas_mech.contract import OlasMechContract
from packages.valory.protocols.acn_data_share import AcnDataShareMessage
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.http.message import HttpMessage
from packages.valory.protocols.ipfs import IpfsMessage
from packages.valory.protocols.kv_store.message import KvStoreMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.abstract_round_abci.handlers import AbstractResponseHandler
from packages.valory.skills.task_execution.dialogues import HttpDialogue
from packages.valory.skills.task_execution.models import Params
from packages.valory.skills.task_execution.utils import preimage as preimage_buffer
from packages.valory.skills.task_execution.utils.eip1271 import (
    Eip1271Verdict,
    check_eip1271_signature,
    get_marketplace_domain_separator,
    get_marketplace_request_id_view,
)
from packages.valory.skills.task_execution.utils.ipfs import to_multihash
from packages.valory.skills.task_execution.utils.local_cid import compute_cidv1
from packages.valory.skills.task_execution.utils.request_id import (
    compute_request_id,
    recover_eoa_signer,
)

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
ROUTES_INFO = "routes_info"
OFFCHAIN_REQUEST_RESPONSES = "offchain_request_responses"
IN_MEMORY_REQUESTS = "in_memory_requests"
# Accepted-but-not-yet-done wire nonces per sender. Populated on
# ``_enqueue_offchain_request`` and removed on rollback / done / rejection.
# ``done`` here means the task completed on the mech side (
# ``_finalize_done_task``) — at that moment the nonce transitions from
# the accepted set to the settling set, where it stays until the batch
# on-chain settlement advances ``MechMarketplace.mapNonces[sender]``
# past the nonce and the pruning pass in ``_bind_wire_nonce_to_chain``
# removes it. See ``SETTLING_NONCES_BY_SENDER`` for the settlement-gap
# half of the split.
ACCEPTED_NONCES_BY_SENDER = "accepted_nonces_by_sender"
# Wire nonces that have moved to done_tasks but whose batch settlement
# has not yet landed on chain. Populated when
# ``_release_outstanding_nonce`` fires on ``_finalize_done_task`` and
# drained when ``_bind_wire_nonce_to_chain`` reads a
# ``mapNonces[sender]`` value at or past the nonce (pruning below).
# Required so the admission gate's next-expected-slot formula stays
# monotonic across the release/settlement gap: without this half, the
# formula would double-count the nonce (once in ``mapNonces`` which
# hasn't advanced yet, once nowhere because ``accepted`` was already
# cleared) and admit a duplicate under a colliding request_id or 503
# the sender's next honest sequential slot until settlement lands.
SETTLING_NONCES_BY_SENDER = "settling_nonces_by_sender"
# Retained for backward compat with tests / consumers still keyed on
# the pre-split name. Kept identical to ``ACCEPTED_NONCES_BY_SENDER``
# so a shared-state seed against the old key still populates the
# accepted-half of the admission gate.
OUTSTANDING_NONCES_BY_SENDER = ACCEPTED_NONCES_BY_SENDER
JSON_CONTENT_HEADER = "Content-Type: application/json\n"
ENCODING_UTF8 = "utf-8"

BALANCE_LOG_DECISION_ACCEPTED = "accepted"
BALANCE_LOG_DECISION_REJECTED = "rejected"
TIMED_OUT_STATUS = 2
WAIT_FOR_TIMEOUT_STATUS = 1
DELIVERED_STATUS = 3
PROMETHEUS_PORT = 9000

# Off-chain HTTP hardening
MAX_HTTP_BODY_BYTES = 1_048_576  # 1MB cap on inbound HTTP bodies
MIN_DELIVERY_RATE = 0  # 0 allowed for free tasks; rejects only negatives
MAX_DELIVERY_RATE = 2**256 - 1  # uint256 upper bound
# 0x + 32-byte (64 hex) or 34-byte multihash (68 hex) payload
IPFS_HASH_RE = re.compile(r"^0x([0-9a-fA-F]{64}|[0-9a-fA-F]{68})$")
# Canonical ASCII decimal for uint256 ``request_id``: single ``0`` OR a
# non-zero leading digit followed by ASCII decimals. ``str.isdigit()``
# alone admits non-canonical decimals (Unicode superscripts, Arabic-Indic
# digits, Thai digits, etc.) that ``int()`` may accept and re-stringify
# to something entirely different — reopening the very silent-money-loss
# path this file's coercion sweep is meant to close. Leading zeros are
# rejected so ``str(int(x)) == x`` holds for every accepted value; this
# keeps the wire ``request_id`` collision-free with the writer's
# ``str(int(request_id))`` key.
#
# The end anchor is ``\Z``, NOT ``$``: Python's ``$`` matches immediately
# before a trailing ``\n``, so ``re.match(r"^...$", "42\n")`` would pass
# while ``int("42\n") == 42`` — the same roundtrip divergence the guard
# exists to close. The call site also uses ``.fullmatch()``, both to
# make full consumption explicit and as a defense against a future
# accidental ``.match()``.
REQUEST_ID_RE = re.compile(r"\A(0|[1-9][0-9]*)\Z")
# uint256 upper bound on ``request_id``. ``REQUEST_ID_RE`` bounds the
# alphabet but not the magnitude: on CPython 3.11+ ``int()`` raises
# ``ValueError`` for a decimal string longer than 4300 digits, which
# would fire deep inside ``_enqueue_offchain_request`` and reintroduce
# the "opaque ``ValueError`` at the coercion site" outcome the upfront
# guard exists to close. Mirrors ``MAX_DELIVERY_RATE``.
MAX_REQUEST_ID = 2**256 - 1

# Rejection reasons carried on ``SignatureVerdict`` for the nonce-bind
# outcomes. Kept as module-level constants so tests and log-search
# tooling can pin the exact string rather than fishing it out of the
# enum by index. ``NONCE_BELOW_EXPECTED`` covers a wire value that lands
# strictly before the sender's next contiguous slot (settlement would
# derive a different ``request_id`` for it and revert the whole batch),
# and ``NONCE_ABOVE_EXPECTED`` covers a wire value that skips a slot
# (same batch-revert outcome from the settlement side, but the sender
# can retry as soon as their earlier requests drain — so this branch
# routes to 503 rather than 401).
NONCE_BELOW_EXPECTED = "wire nonce below sender's next expected slot"
NONCE_ABOVE_EXPECTED = "wire nonce above sender's next expected slot"
NONCE_READ_FAILED = "on-chain mapNonces read failed"
# Non-transient companion of ``NONCE_READ_FAILED`` used when the
# ``mapNonces`` read fails deterministically (ABI drift, wrong
# ``mech_marketplace_address``, unwrapped ledger return). Same 503
# status code so the client backs off, but the reason string names the
# fault as unrecoverable so an operator paging on the log line can
# distinguish an infra blip from a config error.
NONCE_READ_UNRECOVERABLE = "on-chain mapNonces read unrecoverable"
# Distinct reason surfaced when checksum conversion of the wire
# ``sender`` fails — a config-side problem (missing / malformed RPC
# settings) rather than an RPC round-trip failure. Kept separate from
# ``NONCE_READ_FAILED`` so operators can attribute the outcome to the
# right subsystem.
SENDER_RESOLUTION_FAILED = "sender address resolution failed"
# Reason surfaced when the sender exceeds the per-sender in-flight cap
# (``MAX_ACCEPTED_PER_SENDER``). ``is_infra=True`` at the caller so
# mech-client's 503 handling backs off rather than treating the
# outcome as a credential rejection — nothing about the client's
# signature is wrong, they simply have too much unsettled work.
SENDER_INFLIGHT_LIMIT = "sender in-flight request limit reached"
# Reason surfaced when the EIP-1271 ``isValidSignature`` view call
# fails with an infra-side error distinct from a genuine credential
# rejection (``Web3RPCError``, ``BadResponseFormat``,
# ``Web3ValidationError``, ``RequestException``, ``ValueError``, or
# the wall-clock deadline). ``is_infra=True`` at the caller so the
# client sees 503 with a "try again later" verdict rather than 401
# "unauthorized" during an infra failure. A revert, an out-of-gas,
# or a codeless target still returns the ``DECLINED`` verdict below
# and stays 401.
EIP1271_CALL_FAILED = "eip1271 isValidSignature call failed"

# Timeout applied to the sender's ``isValidSignature`` view via the
# provider-level HTTP timeout on the dedicated ``EthereumApi`` used for
# EIP-1271 verification. Kept short so a slow or hostile Safe
# ``isValidSignature`` cannot pin the AEA main thread past the
# ``http_server`` reply budget (default ``RESPONSE_TIMEOUT=5s``). The
# balance-check path uses ``_BALANCE_RPC_DEADLINE_SECONDS`` under the
# same executor to bound its own three unbounded balance-read RPCs.
_EIP1271_CALL_TIMEOUT_SECONDS = 2.0

# ``gas`` cap applied to the ``isValidSignature`` ``eth_call``. Bounds
# the amount of work a sender contract can force the RPC node to do
# per verify. A well-behaved Safe returns in a few thousand gas; a
# contract whose view loops to the block gas limit reverts here with
# out-of-gas and the caller flattens the outcome to a rejected
# signature — no accept, no runaway node work.
_EIP1271_CALL_GAS_CAP = 500_000

# Rejection reason for a Safe whose ``isValidSignature`` view exceeded
# the provider-level HTTP timeout. Distinct from the generic
# "signature verification failed" reason so operators can tell an
# unresponsive Safe apart from an actively-declined signature.
EIP1271_CALL_TIMEOUT = "eip1271 isValidSignature call timed out"

# Wall-clock deadline enforced on the on-path RPCs the accept path
# makes (EIP-1271 ``isValidSignature``, marketplace ``mapNonces``, and
# the three balance-check reads). The per-HTTP-request timeout on
# ``EthereumApi`` is not a wall-clock bound: both ``web3.HTTPProvider``'s
# ``exception_retry_configuration`` (five retries with backoff on
# ``requests.Timeout``) and open-aea's ``RotatingHTTPProvider``
# (``min(MAX_RETRIES=6, url_count * 2)`` outer retries with
# ``time.sleep(min(2**a, 5.0))`` between attempts) multiply the
# per-request timeout well past the ``http_server`` reply budget
# (default ``RESPONSE_TIMEOUT=5s``). Wrapping each on-path call in a
# thread with an explicit ``future.result(timeout=...)`` returns
# control to the AEA main loop within the budget so a slow / hostile
# RPC or ``isValidSignature`` cannot stall ABCI processing. The
# underlying HTTP call keeps running in the worker thread but the
# handler-thread returns and the outcome maps to the same infra
# verdicts (``EIP1271_CALL_TIMEOUT`` / ``NONCE_READ_FAILED``) as any
# other transport failure. Two seconds fits inside the accept-path
# per-request budget: worst-case 2 (sig) + 2 (mapNonces) + 3 * 2
# (balance path) = 10s of RPC work under complete infra collapse.
# The healthy case (all RPCs <500ms) still returns inside the 5s
# ``http_server`` reply budget with room to spare. The pathological
# all-timeout case gives the client a 408 from the server-side
# ``http_server`` timeout — accepted trade-off since no accept
# verdict is coming from the mech either under total RPC collapse.
_RPC_WALL_CLOCK_DEADLINE_SECONDS = 2.0

# Wall-clock deadline for each of the three balance-check RPCs
# (``paymentType`` on the mech, ``getBalanceTrackerForMechType`` on the
# marketplace, ``getRequesterBalance`` on the balance tracker). Each
# call was previously unbounded at the ledger-default 30 s per HTTP
# request times the ``RotatingHTTPProvider`` retry loop. Two seconds
# per call caps the healthy-case aggregate under a second and the
# worst-case aggregate at six seconds, matching the same trade-off
# ``_RPC_WALL_CLOCK_DEADLINE_SECONDS`` picks for the sig-verify calls.
_BALANCE_RPC_DEADLINE_SECONDS = 2.0

# Per-sender cap on ``accepted + settling`` in-flight nonces. Guards
# the previously-unbounded growth of ``accepted_nonces_by_sender`` and
# ``pending_tasks`` after the deletion of the arbitrary
# ``MAX_OUTSTANDING_NONCE_WINDOW`` protocol constraint. 64 is a
# generous burst budget for a well-behaved trader (mech-client rarely
# fires more than a handful of parallel requests) while keeping the
# handler's post-auth pre-payment memory footprint bounded on the DoS
# path. Operators tuning this need to keep it above the peak
# legitimate burst and well below any absolute per-agent memory bound.
MAX_ACCEPTED_PER_SENDER = 64

# Bound on the RPC executor's pending-work queue. ``submit()`` on an
# unbounded queue never blocks, so a wave of hung RPCs draining the
# worker pool (all workers stuck on slow / hostile calls) still
# accepts new submissions; ``future.result(deadline)`` then fires
# WITHOUT the callable ever having run and the caller reports a
# retry-storm-inducing ``NONCE_READ_FAILED`` on every request. Sizing
# this at 4 * ``_RPC_EXECUTOR_MAX_WORKERS`` gives one queue slot per
# worker plus three deep so a short burst can still land while the
# saturation branch fires early on a real overload.
_RPC_EXECUTOR_MAX_QUEUE = 16

# Reason surfaced when the RPC executor's queue is saturated at
# submit-time or the callable has not started running by the deadline.
# ``is_infra=True`` at the caller — 503 with a distinct log line so
# operators can distinguish "we couldn't even try" from "the RPC
# returned late" (which flattens to ``NONCE_READ_FAILED`` or
# ``EIP1271_CALL_TIMEOUT`` as before).
RPC_QUEUE_SATURATED = "on-path RPC executor saturated"

# ``max_workers`` for the ``ThreadPoolExecutor`` cached on
# :class:`MechHttpHandler` and used to enforce
# ``_RPC_WALL_CLOCK_DEADLINE_SECONDS`` on on-path RPCs. The pool is
# used exclusively for I/O-bound web3 calls and is sized to absorb a
# small burst of concurrent accepts without queueing while still
# leaving headroom against the AEA main-thread pool.
_RPC_EXECUTOR_MAX_WORKERS = 4

# ``signature`` on the wire is the hex encoding of the packed signature
# bytes with a mandatory ``0x`` prefix. The settlement path
# (``task_submission_abci.behaviours``) does an unconditional
# ``bytes.fromhex(sig[2:])`` when it packs the on-chain payload; a
# body sent without the ``0x`` would have its first hex byte silently
# truncated there. The lower bound is 65 bytes for a canonical
# secp256k1 signature; the upper bound is generous so multi-signer
# Safe EIP-1271 payloads (n * 65 bytes plus contract signature data)
# fit while capping payload growth to a fixed constant. The alphabet
# is case-insensitive hex and the length is always even
# (``bytes.fromhex`` would otherwise raise on the settlement path).
SIGNATURE_BYTES_MIN = 65
SIGNATURE_BYTES_MAX = 2048
SIGNATURE_RE = re.compile(
    r"\A0x(?:[0-9a-fA-F]{2}){"
    + str(SIGNATURE_BYTES_MIN)
    + ","
    + str(SIGNATURE_BYTES_MAX)
    + r"}\Z"
)

# ``sender`` on the wire is the requester address (EOA or Safe) — a
# 0x-prefixed 20-byte hex string, no checksum requirement (the address
# preparation step in ``_verify_offchain_request_signature`` calls
# ``to_checksum_address`` on it). A malformed value would otherwise
# raise inside the address-preparation block whose ``except``
# classifies the outcome as infra (``is_infra=True`` → 503) because
# the same block also converts the deployment-scoped
# ``mech_marketplace_address`` and ``_marketplace_mech_address``,
# whose failures ARE genuine infra. Guarding at ingress keeps that
# classification honest: a client-supplied bad address 400s here
# alongside the other body-shape violations rather than being
# reported to the client as "server broken, please retry" — which
# a well-behaved auto-retry client would loop on forever.
ADDRESS_RE = re.compile(r"\A0x[0-9a-fA-F]{40}\Z")

# Boot-time self-check inputs for the marketplace ``getRequestId`` view
# vs the local ``compute_request_id`` reimplementation. Any well-formed
# tuple is fine because we compare two bytes32 outputs, not the
# semantic meaning of the inputs. Dummy requester + 32-byte data blob
# + delivery_rate=1 + nonce=0 keep the call cheap and deterministic; a
# real client is never affected because the values never enter the
# accept path.
_SELFCHECK_REQUESTER = "0x0000000000000000000000000000000000000001"
_SELFCHECK_REQUEST_DATA = b"\x00" * 32
_SELFCHECK_DELIVERY_RATE = 1
_SELFCHECK_NONCE = 0

# Rejection reason for a body whose local CID does not match the
# posted ``ipfs_hash``: the request would enqueue arbitrary work under
# a signature that authorised different content.
IPFS_HASH_BODY_MISMATCH = "ipfs_hash does not match ipfs_data content"

# Rejection reason for a body whose ``ipfs_data`` exceeds the 256 KiB
# single-block CID bound (``compute_cidv1`` raises above it). Distinct
# from ``IPFS_HASH_BODY_MISMATCH`` so a caller can tell a size overflow
# apart from a genuine content-hash disagreement.
IPFS_DATA_OVERSIZE = "ipfs_data exceeds the single-block CID bound"

# Single-block CID bound (256 KiB). Mirrors ``local_cid._MAX_BLOCK_BYTES``
# and is used to range-check ``ipfs_data`` before ``compute_cidv1`` is
# called so an oversize body surfaces as ``IPFS_DATA_OVERSIZE`` (accept
# rejection) rather than a misleading CID-mismatch reason.
MAX_IPFS_DATA_BYTES = 256 * 1024

# Reason surfaced on the idempotent-retry response (HTTP 200) when the
# handler already knows the ``request_id`` from the pending queue, the
# in-flight executor, the done batch, or the rejected-response cache.
# Kept a 200 so mech-client's own retry loop doesn't treat it as a
# different response than the first accept.
REQUEST_ALREADY_ACCEPTED = "already accepted"

# Multihash function-code + digest-length prefix a caller may include on
# the wire when writing a full 34-byte multihash (``0x1220<digest>``)
# in ``ipfs_hash``. The 32-byte form (bare SHA-256 digest, 64 hex
# chars) is the wire default. Kept as a hex ``str`` so the strip is a
# ``str.startswith`` on the ``0x``-stripped remainder — no ``bytes``
# round-trip needed for the compare.
IPFS_HASH_MULTIHASH_PREFIX = "1220"

LEDGER_API_ADDRESS = str(LEDGER_CONNECTION_PUBLIC_ID)


class Route(str, Enum):
    """Supported HTTP route names."""

    SEND_SIGNED_REQUESTS = "send_signed_requests"
    FETCH_OFFCHAIN_INFO = "fetch_offchain_info"


class ResponseStatus(str, Enum):
    """Internal/API response status."""

    REJECTED = "rejected"
    OK = "ok"
    UNAVAILABLE = "unavailable"


class ChainId(int, Enum):
    """Supported chain ids."""

    GNOSIS = 100
    POLYGON = 137
    BASE = 8453
    OPTIMISM = 10


class BodyKey(str, Enum):
    """Contract/body keys."""

    DATA = "data"
    WAIT_FOR_TIMEOUT_TASKS = "wait_for_timeout_tasks"
    REQUEST_IDS = "request_ids"
    MECH_TYPE = "mech_type"
    MECH_TYPES = "mech_types"
    TIMED_OUT_REQUESTS = "timed_out_requests"
    REQUESTER_BALANCE = "requester_balance"


class RequestKey(str, Enum):
    """Offchain request keys."""

    REQUEST_ID = "request_id"
    REQUEST_ID_CAMEL = "requestId"
    IPFS_HASH = "ipfs_hash"
    IPFS_DATA = "ipfs_data"
    SENDER = "sender"
    DELIVERY_RATE = "delivery_rate"
    REQUEST_DELIVERY_RATE = "request_delivery_rate"
    IS_OFFCHAIN = "is_offchain"
    SIGNATURE = "signature"
    NONCE = "nonce"


class ResponseKey(str, Enum):
    """Offchain response keys."""

    STATUS = "status"
    REASON = "reason"
    ERROR_CODE = "error_code"
    REQUIRED_AMOUNT = "required_amount"
    AVAILABLE_AMOUNT = "available_amount"
    RPC_ADDRESS = "rpc_address"
    CHAIN_ID = "chain_id"
    BALANCE_TRACKER_ADDRESS = "balance_tracker_address"
    PAYMENT_TYPE = "payment_type"


# 402 challenge constants — surfaced so clients can branch on a stable label.
PAYMENT_SCHEME = "olas-prepay"
DEPOSIT_FN_ABI = "depositFor(address requester, uint256 amount)"
SETTLEMENT_STATUS_PENDING = "pending"
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


class SignatureVerdict(NamedTuple):
    """Outcome of ``_verify_offchain_request_signature``.

    ``is_infra`` distinguishes verification failures caused by our
    side (boot constants unset, ledger settings missing, address
    preparation failure) from a genuinely bad caller signature. The
    caller routes ``is_infra=True`` outcomes to a 503 ("try again
    later") so a well-behaved client backs off instead of giving up
    on the assumption its credentials are wrong; ``is_infra=False``
    outcomes stay 401 ("unauthorized").
    """

    ok: bool
    reason: str
    is_infra: bool


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

        # Update dialogue and pop bookkeeping for ALL performatives (including ERROR).
        # ERROR is a valid TERMINAL_PERFORMATIVE in the IPFS dialogue protocol,
        # so ipfs_dialogues.update() will succeed for error replies.
        dialogue = self.context.ipfs_dialogues.update(ipfs_msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        callback = self.params.req_to_callback.pop(nonce)
        error_callback = self.params.req_to_error_callback.pop(nonce, None)
        deadline = self.params.req_to_deadline.pop(nonce)

        if ipfs_msg.performative == IpfsMessage.Performative.ERROR:
            reason = ipfs_msg.reason
            self.context.logger.warning(
                f"IPFS request failed for nonce {nonce}: {reason}"
            )
            if error_callback is not None:
                error_callback(reason)
            self.params.in_flight_req = False
            self.on_message_handled(message)
            return

        now = time.time()
        self.context.logger.info(f"IPFS response mapped. {nonce=} {deadline=} {now=}")

        if deadline and now > deadline:
            self.context.logger.warning(
                f"Deadline reached for task with nonce {nonce} while handling IPFS message. "
                f"Invoking callback for cleanup."
            )

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

        if body.get(BodyKey.DATA.value) or body.get(
            BodyKey.WAIT_FOR_TIMEOUT_TASKS.value
        ):
            # handle the undelivered requests response from data and wait_for_timeout_tasks
            self._handle_get_undelivered_reqs(body)
        if body.get(BodyKey.REQUEST_IDS.value):
            # handle the request id status check response
            self._update_pending_list(body)
        if body.get(BodyKey.MECH_TYPE.value):
            # handle the mech type response
            self.context.shared_state[PAYMENT_MODEL] = body[BodyKey.MECH_TYPE.value]
            self.context.logger.info(
                f"Found payment model {body[BodyKey.MECH_TYPE.value]!r}."
            )
        if body.get(BodyKey.MECH_TYPES.value):
            # handle the mech types response
            self.context.shared_state[PAYMENT_INFO] = body[BodyKey.MECH_TYPES.value]
            self.context.logger.info(
                f"The cache was updated with the new mech types: {body[BodyKey.MECH_TYPES.value]}."
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
        self.unprocessed_timed_out_tasks = body.get(
            BodyKey.TIMED_OUT_REQUESTS.value, []
        )
        self.set_last_successful_read(self.from_block)

        self.context.logger.info(
            f"Loaded {len(self.unprocessed_timed_out_tasks)} timed out requests from contract.",
        )
        # collect items to process: fresh + previously waiting
        reqs = list(body.get(BodyKey.DATA.value, []))
        reqs.extend(body.get(BodyKey.WAIT_FOR_TIMEOUT_TASKS.value, []))

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
        """Rewrite ``pending_tasks`` to only ids the on-chain status check returned.

        Off-chain tasks (``is_offchain=True``) are NOT present in the
        on-chain status body — their ``request_id`` is not visible on
        chain until settlement lands. Filtering by
        ``body[REQUEST_IDS]`` alone would silently drop every
        outstanding off-chain task on every polling cycle and leak
        their outstanding-nonce entries. Retain off-chain tasks by
        construction so this method only prunes on-chain entries the
        status check did not return.

        :param body: the on-chain status response body.
        """
        before = len(self.pending_tasks)
        on_chain_ids = body[BodyKey.REQUEST_IDS.value]
        self.context.shared_state[PENDING_TASKS] = [
            req
            for req in self.pending_tasks
            if req.get(RequestKey.IS_OFFCHAIN.value)
            or req[RequestKey.REQUEST_ID_CAMEL.value] in on_chain_ids
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

            priority_mech = req.get("priorityMech", "")
            if (
                priority_mech.lower() == self.mech_address.lower()
                and status != DELIVERED_STATUS
            ):
                # Stamp the local enqueue time so PostTxSettlement's
                # undelivered-sweep (see task_submission_abci.behaviours
                # ``_sweep_pending_undelivered``) can detect tasks that
                # have sat in the pending queue past the operator-configured
                # sweep window without paying a per-task RPC at sweep time.
                # The contract's RequestInfo.responseTimeout is the
                # authoritative timeout; this local stamp is the proxy the
                # mech uses to decide when to emit a request-only event
                # to the predict-api data lake. See
                # ``autonolas-marketplace/docs/onchain_write_path_scope.md``
                # §3.2 for the design.
                req.setdefault("enqueued_at_local", time.time())
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
                and req.get("request_delivery_rate", 0)
                >= self.mech_to_max_delivery_rate
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


class KvStoreHandler(BaseHandler):
    """Handler for kv_store replies backing the off-chain preimage buffer."""

    SUPPORTED_PROTOCOL = KvStoreMessage.protocol_id

    def handle(self, message: Message) -> None:
        """Process a kv_store reply (CREATE_OR_UPDATE / LIST / DELETE outcome).

        Clears the single-in-flight flag so the behaviour can issue the next
        preimage op. On LIST_RESPONSE it queues the keys past the retention
        window for deletion; on ERROR it re-queues a failed write so the replay
        survives a transient kv_store failure.

        :param message: the incoming kv_store message.
        """
        kv_msg = cast(KvStoreMessage, message)
        dialogue = self.context.kv_store_dialogues.update(kv_msg)
        shared_state = self.context.shared_state

        # Unrecognised message: open-aea returns None from .update() when
        # the incoming envelope doesn't match any tracked dialogue. Drop
        # it WITHOUT running the cleanup block at the end — that block
        # would zero PREIMAGE_INFLIGHT_DIALOGUE, which is the very signal
        # the late-reply guard below relies on to reject a stale reply
        # against the next op. Standard AEA handler pattern.
        if dialogue is None:
            self.context.logger.warning(
                "KvStoreHandler: received message with no matching dialogue; "
                "dropping."
            )
            return

        # Late-reply guard: the watchdog (PREIMAGE_KV_REQUEST_TIMEOUT, 5s)
        # may have given up on a stuck reply and started the next kv op
        # already. If a reply for the OLD op finally arrives, applying it
        # to the NEW op's bookkeeping would corrupt counters and clear
        # the in-flight gate while another op is still in flight. Compare
        # the INITIATOR nonce only — open-aea's .update() completes the
        # responder slot from "" to the connection's reference on the
        # first reply, so the full tuple stamped at send time
        # ``(nonce, "")`` would never equal the incoming
        # ``(nonce, responder_ref)``. The initiator nonce is the part we
        # generate and the part that uniquely identifies the op, so
        # that's the right thing to compare.
        expected = shared_state.get(preimage_buffer.PREIMAGE_INFLIGHT_DIALOGUE)
        actual = dialogue.dialogue_label.dialogue_reference
        if expected is not None and tuple(actual)[0] != tuple(expected)[0]:
            self.context.logger.warning(
                "Ignoring kv_store reply for a previously-timed-out op "
                "(expected initiator=%s, got=%s); current op state untouched.",
                tuple(expected)[0],
                tuple(actual)[0],
            )
            return
        performative = kv_msg.performative

        if performative == KvStoreMessage.Performative.LIST_RESPONSE:
            now = time.time()
            expired = preimage_buffer.expired_keys(
                dict(kv_msg.data),
                now,
                self.params.preimage_retention_seconds,
            )
            if expired:
                shared_state.setdefault(
                    preimage_buffer.PREIMAGE_DELETE_QUEUE, []
                ).extend(expired)
                self.context.logger.info(
                    f"Preimage sweep: queued {len(expired)} expired entries "
                    f"for deletion."
                )
            # The LIST response carries next_cursor when the kv_store has more
            # pages past preimage_list_page_size; the behaviour loop reads
            # this on the next tick to keep paging. Only when the page is
            # final (next_cursor == "") do we consider the sweep complete and
            # stamp PREIMAGE_LAST_SWEEP — otherwise a multi-page sweep
            # would reset the clock mid-walk and the next interval would
            # start over from page 0, never finishing.
            next_cursor = kv_msg.next_cursor or ""
            if next_cursor:
                shared_state[preimage_buffer.PREIMAGE_LIST_CURSOR] = next_cursor
            else:
                shared_state[preimage_buffer.PREIMAGE_LIST_CURSOR] = None
                shared_state[preimage_buffer.PREIMAGE_LAST_SWEEP] = now
            # Successful LIST reply — reset the consecutive-error counter so a
            # transient failure earlier in the walk doesn't carry over.
            shared_state[preimage_buffer.PREIMAGE_LIST_ATTEMPTS] = 0
        elif performative == KvStoreMessage.Performative.SUCCESS:
            # A successful write of a terminal (delivered/rejected) record means
            # the kv_store now owns it; drop the in-process copy. Non-terminal
            # (processing) records are kept so the settle update can merge.
            #
            # Race guard: a settle can happen between when ``_send_kv_write``
            # serialized the processing record and when this SUCCESS arrives.
            # ``record_settlement`` mutates the record in place to terminal AND
            # re-enqueues the request id. If we pop now, the kv only ever saw
            # the processing snapshot — the terminal write the next tick tries
            # to flush hits ``record is None`` and is silently skipped, losing
            # the delivered/rejected preimage. ``inflight in write_queue`` is
            # exactly the signal a re-enqueue happened: leave the record so
            # the next flush persists the terminal state.
            inflight = shared_state.get(preimage_buffer.PREIMAGE_INFLIGHT_WRITE)
            if inflight is not None:
                records = shared_state.get(preimage_buffer.PREIMAGE_RECORDS, {})
                record = records.get(inflight)
                write_queue = shared_state.get(preimage_buffer.PREIMAGE_WRITE_QUEUE, [])
                if (
                    record is not None
                    and record.get("settlement_status")
                    in preimage_buffer.TERMINAL_STATUSES
                    and inflight not in write_queue
                ):
                    records.pop(inflight, None)
                    # Clear the retry counter so a future re-use of the same
                    # request_id starts fresh — keeps PREIMAGE_WRITE_ATTEMPTS
                    # bounded by the live record set, not by lifetime traffic.
                    shared_state.get(preimage_buffer.PREIMAGE_WRITE_ATTEMPTS, {}).pop(
                        inflight, None
                    )
        elif performative == KvStoreMessage.Performative.ERROR:
            self.context.logger.warning(f"kv_store error: {kv_msg.message}")
            inflight_op = shared_state.get(preimage_buffer.PREIMAGE_INFLIGHT_OP)
            if inflight_op == preimage_buffer.OP_LIST:
                # Bound the LIST hot-loop: a persistently failing kv_store
                # would otherwise re-LIST every act() tick (initial-LIST and
                # mid-walk page failures both qualify — neither has an
                # in-flight write to retry, so the write-counter path above
                # doesn't apply). At the cap we clear the cursor + stamp
                # LAST_SWEEP + WARN so the next sweep_interval is the
                # natural backoff. Counter resets on any successful
                # LIST_RESPONSE.
                list_attempts = (
                    shared_state.get(preimage_buffer.PREIMAGE_LIST_ATTEMPTS, 0) + 1
                )
                shared_state[preimage_buffer.PREIMAGE_LIST_ATTEMPTS] = list_attempts
                if list_attempts >= self.params.preimage_max_list_attempts:
                    self.context.logger.warning(
                        "Preimage sweep LIST failed %d times in a row; "
                        "giving up the current walk. Next attempt in "
                        "preimage_sweep_interval. Last kv_store error: %s",
                        list_attempts,
                        kv_msg.message,
                    )
                    shared_state[preimage_buffer.PREIMAGE_LIST_CURSOR] = None
                    shared_state[preimage_buffer.PREIMAGE_LAST_SWEEP] = time.time()
                    shared_state[preimage_buffer.PREIMAGE_LIST_ATTEMPTS] = 0
            elif inflight_op == preimage_buffer.OP_DELETE:
                # A failed DELETE drops its key batch — the keys were
                # already sliced off PREIMAGE_DELETE_QUEUE in
                # _process_preimage_buffer. The path self-heals because
                # the next sweep re-LISTs and re-queues the same expired
                # keys (they're still in kv), but the loss is otherwise
                # unobservable beyond the generic WARN above. Surface it
                # explicitly so a degraded kv_store is visible in operator
                # logs even when LIST is still succeeding.
                dropped = shared_state.get(
                    preimage_buffer.PREIMAGE_INFLIGHT_DELETE_COUNT, 0
                )
                self.context.logger.warning(
                    "Preimage sweep DELETE failed; %d expired key(s) will "
                    "be retried on the next sweep. Last kv_store error: %s",
                    dropped,
                    kv_msg.message,
                )
            inflight = shared_state.get(preimage_buffer.PREIMAGE_INFLIGHT_WRITE)
            if inflight is not None:
                # Bound the retry loop: a persistently unhealthy kv_store
                # would otherwise hot-loop forever on the same record,
                # pinning PREIMAGE_KV_IN_FLIGHT and starving sweeps + new
                # writes. After preimage_max_write_attempts ERRORs for the
                # same id we drop the record + WARN — the buffer is a
                # best-effort audit copy, not a transactional store, so a
                # bounded loss beats an unbounded stall.
                attempts: Dict[str, int] = shared_state.setdefault(
                    preimage_buffer.PREIMAGE_WRITE_ATTEMPTS, {}
                )
                attempts[inflight] = attempts.get(inflight, 0) + 1
                if attempts[inflight] >= self.params.preimage_max_write_attempts:
                    self.context.logger.warning(
                        "Preimage write for request_id=%s failed "
                        "%d times; dropping record. Last kv_store error: %s",
                        inflight,
                        attempts[inflight],
                        kv_msg.message,
                    )
                    shared_state.get(preimage_buffer.PREIMAGE_RECORDS, {}).pop(
                        inflight, None
                    )
                    attempts.pop(inflight, None)
                else:
                    # Re-queue the failed write so the preimage isn't lost.
                    # Route through enqueue_write so the queue stays dedup'd
                    # (matches the discipline of every other enqueue in the
                    # buffer module).
                    preimage_buffer.enqueue_write(shared_state, inflight)

        shared_state[preimage_buffer.PREIMAGE_INFLIGHT_WRITE] = None
        shared_state[preimage_buffer.PREIMAGE_INFLIGHT_OP] = None
        shared_state[preimage_buffer.PREIMAGE_INFLIGHT_DELETE_COUNT] = 0
        shared_state[preimage_buffer.PREIMAGE_INFLIGHT_DIALOGUE] = None
        shared_state[preimage_buffer.PREIMAGE_KV_IN_FLIGHT] = False
        shared_state[preimage_buffer.PREIMAGE_INFLIGHT_SENT_AT] = None
        self.on_message_handled(message)


class HttpCode(Enum):
    """Http codes"""

    OK_CODE = 200
    NOT_FOUND_CODE = 404
    BAD_REQUEST_CODE = 400
    UNAUTHORIZED_CODE = 401
    PAYMENT_REQUIRED_CODE = 402
    SERVICE_UNAVAILABLE_CODE = 503
    INTERNAL_SERVER_ERROR_CODE = 500


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
    def offchain_request_responses(self) -> Dict[str, Dict[str, Any]]:
        """Get stored off-chain request responses by request id."""
        return self.context.shared_state[OFFCHAIN_REQUEST_RESPONSES]

    @property
    def in_memory_requests(self) -> Dict[str, str]:
        """Get in-memory off-chain request payloads buffered by request id.

        :return: a dict keyed by request_id whose values are the request's
            ``ipfs_data`` payload (not the full request envelope). Populated when
            the off-chain path skips the IPFS upload; cleared when the task
            finalizes.
        """
        return self.context.shared_state[IN_MEMORY_REQUESTS]

    @property
    def accepted_nonces_by_sender(self) -> Dict[str, Set[int]]:
        """Get the accepted-but-not-yet-done wire nonce set per sender.

        Populated on ``_enqueue_offchain_request`` and drained on
        rollback / done / rejection. Values are ``set`` instances so
        admission-gate reads and mutation share a single container
        without rebuilding on every mutation.

        :return: a dict keyed by sender checksum address. A missing key
            means "no accepted-but-not-yet-done requests from this
            sender" and the admission gate uses ``len()==0`` in that
            case.
        """
        return self.context.shared_state[ACCEPTED_NONCES_BY_SENDER]

    @property
    def settling_nonces_by_sender(self) -> Dict[str, Set[int]]:
        """Get the done-but-not-yet-settled wire nonce set per sender.

        Populated when a task moves from ``pending_tasks`` to
        ``done_tasks`` via ``_release_outstanding_nonce`` on the
        behaviour side, drained by the pruning pass in
        ``_bind_wire_nonce_to_chain`` when the on-chain
        ``mapNonces[sender]`` advances past the nonce.

        :return: a dict keyed by sender checksum address. A missing key
            means "no done-but-unsettled requests from this sender".
        """
        return self.context.shared_state[SETTLING_NONCES_BY_SENDER]

    @property
    def outstanding_nonces_by_sender(self) -> Dict[str, Set[int]]:
        """Retained alias for the pre-split ``accepted`` half.

        Older tests / call sites written against the pre-split state
        keep working: this property points at the same underlying
        dict as ``accepted_nonces_by_sender``.

        :return: the accepted-nonce map (same dict object as
            :attr:`accepted_nonces_by_sender`).
        """
        return self.context.shared_state[ACCEPTED_NONCES_BY_SENDER]

    @property
    def params(self) -> Params:
        """Get the parameters."""
        return cast(Params, self.context.params)

    def setup(self) -> None:
        """Setup the mech http handler."""
        self.context.shared_state[ROUTES_INFO] = {
            Route.SEND_SIGNED_REQUESTS.value: self._handle_signed_requests,
            Route.FETCH_OFFCHAIN_INFO.value: self._handle_offchain_request_info,
        }
        self.context.shared_state[IPFS_TASKS] = []
        self.context.shared_state[OFFCHAIN_REQUEST_RESPONSES] = {}
        self.context.shared_state[IN_MEMORY_REQUESTS] = {}
        # Split live in-flight wire nonces per sender into two sets so
        # the admission gate's next-expected-slot formula stays
        # monotonic across the release/settlement gap. See
        # ``ACCEPTED_NONCES_BY_SENDER`` / ``SETTLING_NONCES_BY_SENDER``
        # module docstrings for the semantics; both are keyed by the
        # sender's checksum address and hold ``set[int]`` values.
        #
        # Invariant: this state is per-agent-process. The off-chain
        # accept path assumes single-agent ingress at
        # ``/send_signed_requests``: multi-agent deployments MUST
        # terminate that route on one agent (usually the leader) or
        # the admission gate breaks (each agent sees an empty
        # accepted+settling set for a sender the other has already
        # served, admitting a duplicate under a colliding request_id
        # settlement will revert). A boot-time WARNING fires below if
        # ``use_offchain=True`` and ``num_agents>1``. Moving this
        # state to a shared kv_store / consensus round is a bigger
        # follow-up; the invariant + warning is the light insurance.
        self.context.shared_state[ACCEPTED_NONCES_BY_SENDER] = {}
        self.context.shared_state[SETTLING_NONCES_BY_SENDER] = {}
        self.json_content_header = JSON_CONTENT_HEADER
        # Deployment-scoped constants used by the offchain request-id
        # derivation: the marketplace EIP-712 ``domainSeparator`` and the
        # mech's ``paymentType``. Both are set once at contract deployment
        # and read on-chain here at handler init; the request-handling path
        # references the cached bytes so a normal accept never triggers an
        # extra RPC for these values.
        #
        # ``_marketplace_mech_address`` is resolved by scanning
        # ``params.mech_to_config`` for the entry flagged
        # ``is_marketplace_mech``, NOT by taking
        # ``agent_mech_contract_addresses[0]``. A deployment whose
        # ``mech_to_config`` lists a non-marketplace mech first would
        # otherwise cache the wrong ``paymentType`` at setup, derive the
        # wrong ``request_id`` from every accept, and 401 every offchain
        # request. Falling back to None leaves the constants unloaded so
        # the request-handling path 401s with a clear reason.
        self._domain_separator: Optional[bytes] = None
        self._payment_type: Optional[bytes] = None
        self._marketplace_mech_address: Optional[str] = (
            self._resolve_marketplace_mech_address()
        )
        # Cache ``EthereumApi`` instances keyed by
        # ``(rpc_address, chain_id, timeout_seconds)``. Every construction
        # of ``EthereumApi`` builds a fresh ``RotatingHTTPProvider`` and
        # connection pool; caching keeps one instance per (rpc, chain,
        # timeout) tuple for the process lifetime so the sig-verify and
        # balance-check paths reuse pooled connections instead of opening
        # a new pool per accept. Keyed on ``timeout_seconds`` as well as
        # ``(rpc, chain_id)`` so the short-timeout instance used for the
        # EIP-1271 view coexists with the ledger-default instance used
        # for the balance-check reads without either overwriting the
        # other's provider timeout.
        self._ledger_api_cache: Dict[Tuple[str, int, float], EthereumApi] = {}
        # Executor for on-path RPCs whose wall-clock duration must be
        # capped independently of the per-HTTP-request provider timeout.
        # See ``_RPC_WALL_CLOCK_DEADLINE_SECONDS`` for the multiplication
        # rationale (web3 + RotatingHTTPProvider retry loops). Threads
        # are used (not processes) because the wrapped calls are I/O
        # bound and share the ``EthereumApi`` cache above. Constructed
        # lazily on first call so a deployment with ``use_offchain=False``
        # never spins the pool up.
        self._rpc_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        if self.params.use_offchain:
            self._initialise_offchain_verification_constants()
            # Per-agent ingress invariant. The admission gate above
            # keys on process-local state; multi-agent deployments
            # must terminate /send_signed_requests at a single agent
            # or the gate misbehaves. Warn loudly on boot so the
            # invariant is alertable rather than invisible when the
            # deployment fans out ingress. ``isinstance`` guard so
            # test harnesses that stand up ``skill_context`` with a
            # ``MagicMock`` params namespace don't trip the ``>``
            # comparison — real ``Params.num_agents`` is always an
            # ``int``.
            num_agents = getattr(self.params, "num_agents", 1)
            if isinstance(num_agents, int) and num_agents > 1:
                self.context.logger.warning(
                    "Offchain accept path is enabled with num_agents=%d; "
                    "deployment MUST ensure /send_signed_requests ingress "
                    "terminates at a single agent (the outstanding-nonce "
                    "admission gate is per-agent-process state; fanning "
                    "out ingress lets one agent admit a wire nonce that "
                    "another has already served).",
                    num_agents,
                )
        self.start_prometheus_server()
        super().setup()

    def _get_rpc_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return the lazily-constructed on-path RPC executor.

        Kept lazy so a deployment with the off-chain path disabled never
        spins the pool up. The pool is used to bound the wall-clock
        time of every on-path RPC (EIP-1271 ``isValidSignature``,
        marketplace ``mapNonces``, and the three balance-check reads
        under ``_check_offchain_requester_balance``).

        :return: the cached ``ThreadPoolExecutor`` instance.
        """
        if self._rpc_executor is None:
            self._rpc_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=_RPC_EXECUTOR_MAX_WORKERS,
                thread_name_prefix="mech-rpc-sigverify",
            )
        return self._rpc_executor

    def teardown(self) -> None:
        """Shut down the RPC executor without waiting for in-flight callables.

        ``ThreadPoolExecutor`` workers are non-daemon: a bare process
        exit blocks on ``executor._threads.join()`` for any worker
        still running an abandoned HTTP read, adding seconds to the
        SIGTERM → SIGKILL window. ``cancel_futures=True`` (Python 3.9+)
        drops queued callables that never started, and ``wait=False``
        returns immediately so shutdown does not have to wait on a
        hung provider read.
        """
        if self._rpc_executor is not None:
            try:
                self._rpc_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:  # pylint: disable=broad-exception-caught
                # Best-effort; teardown must not raise into the AEA
                # shutdown path.
                self.context.logger.exception(
                    "Error shutting down RPC executor during teardown."
                )
        super().teardown()

    def _make_late_outcome_callback(
        self, started_at: float, label: str
    ) -> Callable[["concurrent.futures.Future[Any]"], None]:
        """Bind ``started_at`` and ``label`` for use in ``add_done_callback``.

        Extracted so the caller can annotate the callback with a stable
        one-arg signature (mypy cannot infer the lambda's parameter
        types at the ``add_done_callback`` site).

        :param started_at: monotonic timestamp of the ``submit()`` call.
        :param label: short human-readable RPC name for the log line.
        :return: a one-arg callable suitable for
            :meth:`concurrent.futures.Future.add_done_callback`.
        """

        def _cb(fut: "concurrent.futures.Future[Any]") -> None:
            self._log_late_rpc_outcome(fut, started_at, label)

        return _cb

    def _log_late_rpc_outcome(
        self,
        future: "concurrent.futures.Future[Any]",
        started_at: float,
        label: str,
    ) -> None:
        """Log the late completion of an RPC whose deadline already fired.

        ``concurrent.futures.Future`` (unlike ``asyncio.Future``) never
        logs an unretrieved exception on its own, so an eth_call that
        eventually fails after the caller has moved on loses both the
        error and the traceback silently — exactly the path where the
        root cause matters most for operator triage.

        :param future: the completed future for the abandoned call.
        :param started_at: monotonic timestamp of the ``submit()`` call.
        :param label: short human-readable RPC name for the log line.
        :return: None.
        """
        elapsed = time.monotonic() - started_at
        try:
            exc = future.exception()
        except (
            concurrent.futures.CancelledError,
            concurrent.futures.InvalidStateError,
        ):
            return
        if exc is None:
            self.context.logger.debug(
                "on-path RPC %s completed after deadline (elapsed=%.3fs); "
                "outcome was discarded",
                label,
                elapsed,
            )
            return
        self.context.logger.warning(
            "on-path RPC %s completed after deadline (elapsed=%.3fs) with "
            "an error the caller had already moved past: %r",
            label,
            elapsed,
            exc,
        )

    def _run_with_wall_clock_deadline(
        self,
        fn: Callable[[], Any],
        deadline_seconds: float,
        label: str = "unnamed",
    ) -> Any:
        """Run ``fn`` on the RPC executor and enforce a wall-clock deadline.

        Delegates the call to the dedicated ``ThreadPoolExecutor`` and
        waits at most ``deadline_seconds`` for the result. On timeout
        the underlying HTTP call keeps running in the worker thread but
        this method raises ``concurrent.futures.TimeoutError`` so the
        caller can return an infra verdict to the client and free the
        AEA main thread to process the next message. Any exception the
        wrapped call raises propagates to the caller so existing
        exception handling (``TimeoutError`` from the EIP-1271 wrapper,
        the broad ``except`` around the ``mapNonces`` read) keeps
        working unchanged.

        Queue saturation is a distinct outcome from "the RPC took too
        long": a wave of hung callables donates all worker slots to
        never-returning HTTP reads, ``submit()`` still lands on the
        unbounded queue, and ``future.result(timeout=...)`` fires
        WITHOUT the callable ever having started. Detect this with
        ``future.running()`` after the timeout — if the callable hasn't
        started, raise :class:`RuntimeError` labelled ``RPC_QUEUE_SATURATED``
        so the caller can route to a queue-saturation verdict instead
        of the generic "read failed" one that a real RPC timeout uses.
        Also short-circuit ``submit()`` on queue depth so a genuine
        overload rejects fast rather than piling up.

        :param fn: zero-arg callable that performs the RPC.
        :param deadline_seconds: wall-clock upper bound in seconds.
        :param label: short human-readable RPC name used on late-
            outcome log lines and saturation errors.
        :return: whatever ``fn`` returns.
        :raises concurrent.futures.TimeoutError: when the deadline
            elapses before ``fn`` returns.
        :raises RuntimeError: on executor shutdown or on saturated
            queue at submit time.
        """
        executor = self._get_rpc_executor()
        # Reject fast on queue overload. ``_work_queue`` is CPython
        # private API but its ``qsize()`` is stable and behaviorally
        # equivalent to ``deque.__len__`` (see cpython:Lib/concurrent/
        # futures/thread.py — the queue is always ``queue.SimpleQueue``
        # or ``queue.Queue`` and both expose ``qsize()``). We keep the
        # bound conservative (``_RPC_EXECUTOR_MAX_QUEUE``) so a real
        # burst still lands.
        try:
            queue_depth = executor._work_queue.qsize()  # noqa: SLF001
        except Exception:  # pylint: disable=broad-exception-caught
            queue_depth = 0
        if queue_depth >= _RPC_EXECUTOR_MAX_QUEUE:
            self.context.logger.warning(
                "RPC executor queue saturated at %d for %s; failing fast "
                "instead of piling on more work.",
                queue_depth,
                label,
            )
            raise RuntimeError(RPC_QUEUE_SATURATED)
        started_at = time.monotonic()
        try:
            future = executor.submit(fn)
        except RuntimeError:
            # ``submit()`` raises on ``executor._shutdown`` or when
            # ``_adjust_thread_count`` fails. Both are infra signals;
            # let the caller surface the same 503 verdict.
            self.context.logger.exception("RPC executor submit failed for %s.", label)
            raise
        try:
            return future.result(timeout=deadline_seconds)
        except concurrent.futures.TimeoutError:
            # ``TimeoutError`` is aliased to
            # ``concurrent.futures.TimeoutError`` in Python 3.11+, so
            # this branch fires for BOTH the wall-clock deadline (the
            # callable never returned) AND a callable that raised its
            # own ``TimeoutError`` (which ``future.result`` re-raises).
            # Disambiguate on ``future.done()``: done + timeout means
            # the callable finished and raised, and we should let the
            # exception propagate as a normal callable-side result.
            if future.done():
                raise
            if not future.running():
                # Callable never got a worker: the pool is drained by
                # earlier hung calls. Distinguish this from a slow-RPC
                # timeout so operators see the real cause.
                self.context.logger.warning(
                    "RPC callable %s never started before deadline "
                    "(pool saturated with in-flight abandoned callables); "
                    "reporting queue saturation.",
                    label,
                )
                _log_cb = self._make_late_outcome_callback(started_at, label)
                future.add_done_callback(_log_cb)
                raise RuntimeError(RPC_QUEUE_SATURATED)
            # ``future.cancel()`` returns False for a RUNNING future —
            # it does NOT interrupt the underlying HTTP call and it
            # does NOT protect the worker slot. The comment used to
            # claim otherwise; corrected here. What we DO get: an
            # ``add_done_callback`` fires when the abandoned call
            # finally returns so its exception (if any) lands in the
            # log rather than being silently discarded by
            # ``concurrent.futures``.
            _log_cb = self._make_late_outcome_callback(started_at, label)
            future.add_done_callback(_log_cb)
            raise

    def _get_ledger_api(
        self,
        rpc_address: str,
        chain_id: int,
        timeout_seconds: Optional[float] = None,
    ) -> EthereumApi:
        """Return a cached ``EthereumApi`` for ``(rpc_address, chain_id, timeout)``.

        A missing entry constructs a new ``EthereumApi`` (which builds a
        ``RotatingHTTPProvider`` + connection pool) and stores it. Subsequent
        calls for the same key return the same instance. When
        ``timeout_seconds`` is ``None`` the ledger default HTTP timeout is
        used and the cache key stores that as ``0.0``; a positive value
        pins the provider-level HTTP timeout for calls made through the
        returned instance.

        :param rpc_address: the RPC endpoint URL (or comma-separated list
            consumed by ``RotatingHTTPProvider``).
        :param chain_id: the ledger chain id.
        :param timeout_seconds: optional provider-level HTTP timeout in
            seconds. ``None`` uses the ledger default (30s).
        :return: the cached (or freshly-constructed) ``EthereumApi``.
        """
        cache_key: Tuple[str, int, float] = (
            rpc_address,
            chain_id,
            0.0 if timeout_seconds is None else float(timeout_seconds),
        )
        cached = self._ledger_api_cache.get(cache_key)
        if cached is not None:
            return cached
        kwargs: Dict[str, Any] = {"address": rpc_address, "chain_id": chain_id}
        if timeout_seconds is not None:
            kwargs["timeout"] = float(timeout_seconds)
        ledger_api = EthereumApi(**kwargs)
        self._ledger_api_cache[cache_key] = ledger_api
        return ledger_api

    def _resolve_marketplace_mech_address(self) -> Optional[str]:
        """Return the mech address flagged ``is_marketplace_mech`` in config.

        Mirrors ``TaskExecutionBehaviour._get_designated_marketplace_mech_address``
        so the offchain accept path and the on-chain delivery path pick the
        same mech even when ``mech_to_config`` lists a non-marketplace mech
        first. Returns ``None`` when no marketplace mech is configured; the
        caller treats that as "offchain unavailable" and refuses subsequent
        accepts with a clear reason instead of caching a wrong address.

        :return: the marketplace mech address (lowercase, as stored in
            ``mech_to_config``), or ``None`` if no marketplace mech is
            configured.
        """
        for mech, config in self.params.mech_to_config.items():
            if config.is_marketplace_mech:
                return mech
        return None

    def _initialise_offchain_verification_constants(self) -> None:
        """Read the marketplace ``domainSeparator`` and mech ``paymentType``.

        Any read failure logs a warning and leaves the constants unset; the
        request-handling path treats an unset value as an internal error and
        replies 401 so the client sees a clear rejection instead of a
        mis-derived request_id.

        After the constants are loaded, a boot-time self-check calls the
        marketplace ``getRequestId`` view and compares its bytes32 output
        with the local ``compute_request_id`` reimplementation on the same
        dummy inputs. On mismatch the boot-cached constants are flipped off
        and a fatal-level log line is emitted; every subsequent accept
        401s. Guards against a marketplace upgrade that silently changes
        either the ``getRequestId`` layout or the EIP-712 ``domainSeparator``
        derivation (the marketplace sits behind ``MechMarketplaceProxy``).
        """
        if self._marketplace_mech_address is None:
            self.context.logger.warning(
                "Offchain verification constants unavailable: no "
                "marketplace mech configured in mech_to_config."
            )
            return
        ledger_settings = self._get_ledger_settings()
        if ledger_settings[ResponseKey.STATUS.value] != ResponseStatus.OK.value:
            self.context.logger.warning(
                "Offchain verification constants unavailable: %s",
                ledger_settings.get(ResponseKey.REASON.value, "unknown"),
            )
            return
        try:
            rpc_address = cast(str, ledger_settings[ResponseKey.RPC_ADDRESS.value])
            chain_id = cast(int, ledger_settings[ResponseKey.CHAIN_ID.value])
            ledger_api = self._get_ledger_api(rpc_address, chain_id)
            self._domain_separator = get_marketplace_domain_separator(
                ledger_api=ledger_api,
                marketplace_address=self.params.mech_marketplace_address,
            )
            # Read the mech ``paymentType`` through the packaged
            # ``OlasMechContract`` wrapper (same selector,
            # ``paymentType()``, same ``bytes32`` return) so this
            # module does not carry a second ABI fragment for the
            # same view.
            payment_type_res = OlasMechContract.get_mech_type(
                ledger_api, self._marketplace_mech_address
            )
            payment_type_raw = payment_type_res.get(BodyKey.MECH_TYPE.value)
            self._payment_type = bytes(cast(bytes, payment_type_raw))
            if len(self._payment_type) != 32:
                raise ValueError(
                    f"payment_type length is {len(self._payment_type)}, expected 32"
                )
            self.context.logger.info(
                "Offchain verification constants loaded: "
                "domain_separator=0x%s payment_type=0x%s",
                self._domain_separator.hex(),
                self._payment_type.hex(),
            )
        except Exception:  # pylint: disable=broad-exception-caught
            self.context.logger.exception(
                "Failed to load offchain verification constants; "
                "offchain requests will be refused until the next restart."
            )
            return
        self._selfcheck_marketplace_request_id_derivation(ledger_api)

    def _selfcheck_marketplace_request_id_derivation(
        self, ledger_api: EthereumApi
    ) -> None:
        """Cross-check the local request-id derivation against the marketplace.

        Calls the marketplace ``getRequestId`` view with fixed dummy inputs
        and compares its bytes32 output with ``compute_request_id`` run on
        the same inputs and the boot-cached ``domain_separator`` and
        ``payment_type``. Outcomes:

        - On value mismatch (or a genuine contract revert / decode
          error): treat as a real incompat, flip the boot-cached
          constants to ``None`` (subsequent accepts route to 503 via
          the sig-verify infra branch), and log fatal-level.
        - On a transport failure (RPC unreachable, timeout, socket
          error): leave the constants loaded so a transient blip does
          not permanently disable the endpoint. The sig-verify path's
          own EIP-1271 call will hit the same RPC on the next request;
          if it stays down, per-request 503s and standard alerts fire
          without the boot handler locking the endpoint open-loop.

        :param ledger_api: the ledger API object used for the view call.
        """
        # Precondition: the caller loaded both constants before invoking.
        # Belt-and-braces to keep the type checker happy for the two
        # dereferences below.
        if self._domain_separator is None or self._payment_type is None:
            return
        marketplace_address = self.params.mech_marketplace_address
        mech_address = cast(str, self._marketplace_mech_address)
        try:
            onchain_request_id = get_marketplace_request_id_view(
                ledger_api=ledger_api,
                marketplace_address=marketplace_address,
                mech=mech_address,
                requester=_SELFCHECK_REQUESTER,
                data=_SELFCHECK_REQUEST_DATA,
                delivery_rate=_SELFCHECK_DELIVERY_RATE,
                payment_type=self._payment_type,
                nonce=_SELFCHECK_NONCE,
            )
            local_request_id = compute_request_id(
                marketplace=marketplace_address,
                mech=mech_address,
                requester=_SELFCHECK_REQUESTER,
                request_data=_SELFCHECK_REQUEST_DATA,
                delivery_rate=_SELFCHECK_DELIVERY_RATE,
                payment_type=self._payment_type,
                nonce=_SELFCHECK_NONCE,
                domain_separator=self._domain_separator,
            )
        except (RequestException, TimeoutError, Web3RPCError):
            # Transport-level failure: leave the constants loaded so a
            # transient RPC blip does not permanently disable the
            # endpoint. Per-request sig verification will re-attempt
            # against the same RPC.
            self.context.logger.warning(
                "Marketplace request-id self-check hit a transport error; "
                "keeping the boot-cached constants and continuing.",
                exc_info=True,
            )
            return
        except (ContractLogicError, BadFunctionCallOutput, ValueError):
            # Genuine incompat: a revert, an undecodable return, or the
            # local derivation raising. Flip the constants off so
            # subsequent accepts refuse with a clear infra-side reason.
            self.context.logger.exception(
                "Marketplace request-id self-check failed with a "
                "contract-level error; disabling offchain accepts "
                "until the next restart."
            )
            self._domain_separator = None
            self._payment_type = None
            return
        if onchain_request_id != local_request_id:
            self.context.logger.error(
                "FATAL: marketplace getRequestId does not match the local "
                "compute_request_id reimplementation "
                "(onchain=0x%s local=0x%s). The marketplace may have been "
                "upgraded to a layout this handler does not mirror; "
                "offchain accepts will 401 until the mech is redeployed "
                "against a matching handler.",
                onchain_request_id.hex(),
                local_request_id.hex(),
            )
            self._domain_separator = None
            self._payment_type = None
            return
        self.context.logger.info("Marketplace request-id self-check passed.")

    def start_prometheus_server(self) -> None:
        """Starts the prometheus server"""
        start_http_server(PROMETHEUS_PORT)
        self.context.logger.info(
            f"Prometheus server started on port {PROMETHEUS_PORT}."
        )

    def _handle_signed_requests(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """Handle POST requests to send signed tx to mech.

        Top-level entry point. Delegates to
        :meth:`_handle_signed_requests_impl` inside a defensive
        ``try/except Exception`` so any un-caught error inside the
        accept path (a fresh RPC exception class the inner tuples
        don't cover, an ``OSError`` from a mid-flight ``ConnectionReset``,
        anything raised out of an executor callback) surfaces as a
        503 to the client instead of propagating out to the AEA
        framework's default ``propagate`` handler and stopping the
        agent. The narrower catches inside the impl still classify
        outcomes for the log lines and per-branch verdict reasons.

        :param http_msg: the HttpMessage instance.
        :param http_dialogue: the HttpDialogue instance.
        """
        try:
            self._handle_signed_requests_impl(http_msg, http_dialogue)
        except Exception:  # pylint: disable=broad-exception-caught
            # A defensive backstop for anything the inner
            # classification branches missed. ``ConnectionResetError``
            # is an ``OSError`` (not covered by the inner tuple around
            # ``future.result()``), a fresh ``Web3Exception`` subclass
            # could be added in a future web3 release, or an executor
            # callback could raise. Any of those would otherwise
            # propagate out of the handler and — with the default
            # ``propagate`` skill exception policy — stop the agent.
            # ``request_id`` is unknown here (parse may have raised);
            # the client's own timeout has almost certainly fired.
            self.context.logger.exception("Unhandled error on offchain accept path")
            try:
                self._send_rejection_response(
                    http_msg,
                    http_dialogue,
                    "unknown",
                    reason="internal error",
                    status_code=HttpCode.SERVICE_UNAVAILABLE_CODE.value,
                    status_text="Service unavailable",
                )
            except Exception:  # pylint: disable=broad-exception-caught
                # The rejection sender itself failed — nothing more we
                # can do; the client will time out its own request.
                self.context.logger.exception(
                    "Failed to emit defensive 503 on unhandled accept-path error"
                )

    def _handle_signed_requests_impl(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """Body of the signed-request handler; called under the outer guard.

        Split from ``_handle_signed_requests`` so the outer try/except
        wraps every accept-path branch by construction. See the outer
        method for the rationale.

        :param http_msg: the HttpMessage instance.
        :param http_dialogue: the HttpDialogue instance.
        """
        # Phase 1 ships dark: the off-chain HTTP path is disabled by default and
        # enabled per deployment in the Phase 2 rollout (use_offchain). When off,
        # nothing about the on-chain + IPFS flow changes; off-chain requests are
        # refused so none of the new off-chain code path runs.
        if not self.params.use_offchain:
            self.context.logger.info(
                "Off-chain request received but the off-chain path is disabled "
                "(use_offchain=false); refusing."
            )
            http_response = http_dialogue.reply(
                performative=HttpMessage.Performative.RESPONSE,
                target_message=http_msg,
                version=http_msg.version,
                status_code=HttpCode.SERVICE_UNAVAILABLE_CODE.value,
                status_text="Service unavailable",
                headers=f"{self.json_content_header}{http_msg.headers}",
                body=json.dumps({"error": "offchain path disabled"}).encode(
                    ENCODING_UTF8
                ),
            )
            self.context.outbox.put_message(message=http_response)
            return

        try:
            data = self._parse_http_body(http_msg)
            request_id = data[RequestKey.REQUEST_ID.value]
            ipfs_hash = data[RequestKey.IPFS_HASH.value]
            sender = data[RequestKey.SENDER.value]
            request_delivery_rate = int(data[RequestKey.DELIVERY_RATE.value])
            signature_hex = data[RequestKey.SIGNATURE.value]
            # ``nonce`` on the wire is form-urlencoded — always ``str``.
            # Coerce with ``int(...)`` inside the ingress try/except so a
            # non-numeric value 400s alongside the other body-shape
            # violations. Downstream (see the ``_enqueue_offchain_request``
            # writeback and the sort in
            # ``task_submission_abci.behaviours._get_offchain_tasks_deliver_data``)
            # gets the coerced ``int`` value.
            wire_nonce = data[RequestKey.NONCE.value]
            request_nonce = int(wire_nonce)
        except Exception as e:
            self.context.logger.error(
                f"Error processing signed request. body_len={len(http_msg.body)} "
                f"error={str(e)}."
            )
            self._handle_bad_request(http_msg, http_dialogue)
            return

        # ``request_id`` on the wire is the decimal encoding of the uint256
        # marketplace ``getRequestId`` return; ``nonce`` is the decimal
        # encoding of the requester's uint256 ``mapNonces`` counter at
        # signing time. Both must be canonical ASCII decimals inside the
        # uint256 upper bound — validated by the shared helper below so
        # the two call sites stay in lock-step. See the helper's
        # docstring for the rationale on the alphabet regex plus the
        # length short-circuit before ``int()``.
        if not self._reject_unless_uint256_decimal(request_id, "request_id"):
            self._handle_bad_request(http_msg, http_dialogue)
            return
        if not self._reject_unless_uint256_decimal(wire_nonce, "nonce"):
            self._handle_bad_request(http_msg, http_dialogue)
            return

        if not IPFS_HASH_RE.match(ipfs_hash):
            self.context.logger.error(
                f"Rejecting offchain request {request_id}: invalid ipfs_hash "
                f"format (len={len(ipfs_hash)})."
            )
            self._handle_bad_request(http_msg, http_dialogue)
            return

        # ``signature`` on the wire is the one signed field with no format
        # guard at the parser layer. The settlement path does an
        # unconditional ``bytes.fromhex(sig[2:])`` when packing the
        # on-chain payload; a body without the mandatory ``0x`` prefix
        # would have its first hex byte silently truncated there, the
        # marketplace would revert with ``IncorrectSignatureLength``, and
        # the whole per-sender batch would be dropped. Validate the
        # format alongside the other body-shape checks so a
        # missing-prefix or bad-hex signature 400s at ingress rather
        # than surfacing as an on-chain revert at settlement time.
        if not SIGNATURE_RE.fullmatch(signature_hex or ""):
            self.context.logger.error(
                f"Rejecting offchain request {request_id}: signature must be "
                f"a 0x-prefixed even-length hex string within "
                f"[{SIGNATURE_BYTES_MIN}, {SIGNATURE_BYTES_MAX}] bytes "
                f"(len={len(signature_hex) if signature_hex else 0})."
            )
            self._handle_bad_request(http_msg, http_dialogue)
            return

        if not ADDRESS_RE.fullmatch(sender or ""):
            self.context.logger.error(
                f"Rejecting offchain request {request_id}: sender must be a "
                f"0x-prefixed 20-byte hex address "
                f"(len={len(sender) if sender else 0})."
            )
            self._handle_bad_request(http_msg, http_dialogue)
            return

        if (
            request_delivery_rate < MIN_DELIVERY_RATE
            or request_delivery_rate > MAX_DELIVERY_RATE
        ):
            self.context.logger.error(
                f"Rejecting offchain request {request_id}: "
                f"request_delivery_rate={request_delivery_rate} out of range "
                f"[{MIN_DELIVERY_RATE}, {MAX_DELIVERY_RATE}]."
            )
            self._handle_bad_request(http_msg, http_dialogue)
            return

        self.context.logger.info(
            f"Received signed offchain request with {request_id=} and {request_delivery_rate=}."
        )

        # Verify the request signature before charging the requester's balance.
        # EIP-1271 for Safe senders, plain ecrecover for EOAs. Sequential by
        # design: a failure here rejects before the balance-check RPC runs.
        sig_verdict = self._verify_offchain_request_signature(
            sender=sender,
            ipfs_hash=ipfs_hash,
            delivery_rate=request_delivery_rate,
            nonce=request_nonce,
            wire_request_id=request_id,
            signature_hex=signature_hex,
        )
        if not sig_verdict.ok:
            # Route infra-side failures (boot constants unset, ledger
            # settings missing, address preparation failure) to 503 so a
            # well-behaved client backs off and retries instead of
            # concluding its credentials are wrong. Bad-caller
            # signatures still 401.
            #
            # Pre-auth: do NOT persist a rejection payload keyed by
            # the caller-supplied ``request_id`` on either branch — a
            # caller could otherwise poison an arbitrary id that a
            # legitimate caller would later read via the polling
            # endpoint. See
            # ``_send_rejection_response(record_response=...)``.
            if sig_verdict.is_infra:
                status_code = HttpCode.SERVICE_UNAVAILABLE_CODE.value
                status_text = "Service unavailable"
            else:
                status_code = HttpCode.UNAUTHORIZED_CODE.value
                status_text = "Unauthorized"
            self._send_rejection_response(
                http_msg,
                http_dialogue,
                request_id,
                reason=sig_verdict.reason,
                status_code=status_code,
                status_text=status_text,
            )
            return

        # Bind the executed payload to the signature. The trader signs
        # a ``request_id`` derived from ``keccak(ipfs_hash)`` and posts
        # the ``ipfs_data`` body inline; without a match check the
        # signature would authorise "work described by CID X" while
        # the mech executes body Y. Re-derive the CID over the bytes
        # the mech will actually run and reject the request outright
        # if it does not equal the posted ``ipfs_hash``.
        cid_bind_ok, cid_reject_reason = self._verify_ipfs_hash_binding(
            request_id=request_id,
            ipfs_hash=ipfs_hash,
            ipfs_data=data.get(RequestKey.IPFS_DATA.value, ""),
        )
        if not cid_bind_ok:
            self._send_rejection_response(
                http_msg,
                http_dialogue,
                request_id,
                reason=cid_reject_reason,
                status_code=HttpCode.BAD_REQUEST_CODE.value,
                status_text="Bad request",
            )
            return

        # Replay protection. Once the sig + CID gates have passed, the
        # accept-time trust base is enough to dedup exactly: every field
        # that changes the ``request_id`` is signed, so the same body
        # posted twice hashes to the same id, and any change to a signed
        # field would have produced a different id. Return 200 with a
        # note rather than a 409 so mech-client's idempotent-retry
        # semantics are preserved on a genuine network retry. Dedup
        # runs BEFORE the nonce-bind admission gate so a client that
        # re-posts an already-delivered request still sees a 200 with
        # ``REQUEST_ALREADY_ACCEPTED`` instead of a nonce mismatch.
        if self._is_duplicate_request(request_id):
            self.context.logger.info(
                "Duplicate offchain request %s already known; returning 200 "
                "without a second enqueue.",
                request_id,
            )
            self._send_ok_response(
                http_msg=http_msg,
                http_dialogue=http_dialogue,
                data={
                    RequestKey.REQUEST_ID.value: request_id,
                    ResponseKey.STATUS.value: ResponseStatus.OK.value,
                    ResponseKey.REASON.value: REQUEST_ALREADY_ACCEPTED,
                },
            )
            return

        # Admission gate: bind the wire nonce to the sender's
        # ``MechMarketplace.mapNonces`` counter combined with the live
        # outstanding accepted-but-unsettled nonces for the same
        # sender. This is where the contiguity requirement settlement
        # imposes (``_deliverMarketplaceWithSignatures`` recomputes
        # each request_id from its own ``nonce, nonce+1, ...`` counter
        # and reverts the whole per-sender batch on the first mismatch)
        # is enforced on the way in. Runs AFTER dedup so a genuine
        # idempotent retry never reaches this branch. Also runs BEFORE
        # the balance check so a nonce mismatch — which is a hard
        # settlement-side rejection, not a transient state — short-
        # circuits without an RPC round-trip for balance.
        sender_checksum = self._resolve_sender_checksum(sender)
        if sender_checksum is None:
            # Sig-verify above already succeeded, so the checksum
            # conversion should be deterministic; a failure here means
            # ledger settings were reconfigured between the two calls
            # (settings dict is rebuilt per call). Route to a distinct
            # ``SENDER_RESOLUTION_FAILED`` reason so config-side
            # failures are not mis-reported as RPC failures.
            self._send_rejection_response(
                http_msg,
                http_dialogue,
                request_id,
                reason=SENDER_RESOLUTION_FAILED,
                status_code=HttpCode.SERVICE_UNAVAILABLE_CODE.value,
                status_text="Service unavailable",
            )
            return
        nonce_verdict = self._bind_wire_nonce_to_chain(
            sender_checksum=sender_checksum,
            wire_nonce=request_nonce,
            wire_request_id=request_id,
        )
        if not nonce_verdict.ok:
            if nonce_verdict.is_infra:
                status_code = HttpCode.SERVICE_UNAVAILABLE_CODE.value
                status_text = "Service unavailable"
            else:
                status_code = HttpCode.UNAUTHORIZED_CODE.value
                status_text = "Unauthorized"
            self._send_rejection_response(
                http_msg,
                http_dialogue,
                request_id,
                reason=nonce_verdict.reason,
                status_code=status_code,
                status_text=status_text,
            )
            return

        balance_check = self._check_offchain_requester_balance(
            sender=sender,
            delivery_rate=request_delivery_rate,
        )
        if balance_check[ResponseKey.STATUS.value] != ResponseStatus.OK.value:
            # Pre-enqueue 503: same pre-auth-write rule as the 401
            # branch above. The caller has proven signature ownership,
            # but the ledger read failed before any state was committed;
            # do not persist a rejection payload for a request that
            # never made it into pending_tasks.
            self._send_rejection_response(
                http_msg,
                http_dialogue,
                request_id,
                reason="balance check unavailable",
                status_code=HttpCode.SERVICE_UNAVAILABLE_CODE.value,
                status_text="Service unavailable",
            )
            return

        available_amount = cast(int, balance_check[ResponseKey.AVAILABLE_AMOUNT.value])
        if available_amount < request_delivery_rate:
            try:
                extra_headers = self._build_www_authenticate_header()
                challenge_body = self._build_402_challenge(
                    balance_check, error_msg="insufficient balance"
                )
            except Exception:  # pylint: disable=broad-exception-caught
                # The 402 / header builders raise on malformed internal state
                # (missing balance-tracker address, mis-terminated header).
                # Reply 500 rather than letting it escape with no HTTP response.
                self.context.logger.exception(
                    f"Failed to build 402 challenge for {request_id}."
                )
                self._send_internal_error(http_msg, http_dialogue, request_id)
                return
            # 402 is the one rejection path that DOES persist the
            # response payload keyed by ``request_id``: the caller has
            # authenticated (signature ownership proven) and needs the
            # deposit challenge to be readable via the polling endpoint
            # (``_handle_offchain_request_info``). Every other pre-auth
            # rejection returns the HTTP response only.
            self._send_rejection_response(
                http_msg,
                http_dialogue,
                request_id,
                reason="insufficient balance",
                status_code=HttpCode.PAYMENT_REQUIRED_CODE.value,
                status_text="Payment required",
                extra_headers=extra_headers,
                body_extras=challenge_body,
                record_response=True,
            )
            return

        # ``nonce`` moves from the wire ``str`` to the coerced ``int``
        # before the enqueue merges the parsed body dict into the
        # pending task. Without this, ``_enqueue_offchain_request``
        # would spread the original ``str`` through as ``data["nonce"]``
        # and the sort key in
        # ``task_submission_abci.behaviours._get_offchain_tasks_deliver_data``
        # would fall back to lexicographic order (``"10"`` before
        # ``"9"``) and mis-order the batch against the on-chain
        # ``mapNonces`` array consume.
        # Copy at the call site rather than mutating in place: the
        # parsed body is typed ``Dict[str, str]`` and other consumers
        # (``preimage_buffer.record_accept``) read the same dict; a
        # spread copy keeps the source dict immutable and lets the
        # enqueue accept the wider ``Dict[str, Any]`` type without a
        # ``type: ignore``.
        enqueue_data = {**data, RequestKey.NONCE.value: request_nonce}

        try:
            req = self._enqueue_offchain_request(
                request_id=request_id,
                ipfs_hash=ipfs_hash,
                request_delivery_rate=request_delivery_rate,
                data=enqueue_data,
                sender_checksum=sender_checksum,
                wire_nonce=request_nonce,
            )
        except Exception as e:
            self.context.logger.error(
                f"Error enqueuing offchain request {request_id}: {str(e)}."
            )
            self._handle_bad_request(http_msg, http_dialogue)
            return

        self.context.logger.info(
            f"Offchain task added with data: {req}. "
            f"pending_tasks={len(self.pending_tasks)} "
            f"in_memory_requests={len(self.in_memory_requests)}."
        )

        self.offchain_request_responses.pop(request_id, None)
        try:
            receipt_header = self._build_payment_receipt_header(
                request_id, request_delivery_rate
            )
            self._send_ok_response(
                http_msg=http_msg,
                http_dialogue=http_dialogue,
                data={RequestKey.REQUEST_ID.value: request_id},
                extra_headers=receipt_header,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            # The receipt-header builder / OK sender raise only on malformed
            # internal state. The task is already enqueued, so reply 500 rather
            # than hanging the client; the mech still processes the request.
            self.context.logger.exception(
                f"Failed to send OK response for {request_id}."
            )
            self._send_internal_error(http_msg, http_dialogue, request_id)

    def _reject_unless_uint256_decimal(self, value: str, name: str) -> bool:
        """Return True iff ``value`` is a canonical ASCII decimal in uint256.

        Combined alphabet + magnitude guard for the two wire fields
        the handler treats as uint256 decimals (``request_id`` and
        ``nonce``). The alphabet regex catches ``"9abc"``, empty
        strings, leading zeros, Unicode digits (``๙``, ``٩``,
        superscripts), and any string ``int()`` would happily coerce
        to a value that doesn't round-trip via ``str(int(x)) == x``.
        The uint256 range check catches ``2**256``. The length check
        MUST short-circuit before ``int(value)``: on CPython 3.11+,
        ``int()`` on a string longer than
        ``sys.get_int_max_str_digits()`` (4300 by default) raises
        ``ValueError`` — a 5000-digit canonical decimal would pass
        ``fullmatch`` but blow up here, outside any try/except, and
        bypass the caller's own 400 return. ``len(str(2**256-1))`` is
        78, well under the CPython cap, so the length branch is the
        cheap guard that makes the value branch safe.

        Emits a specific error log naming ``name`` on rejection so
        operators can attribute a 400 to the right field.

        :param value: the wire-format value being validated. May be
            ``None``/empty; the alphabet regex rejects both.
        :param name: the wire field name for the rejection log line.
        :return: True on a canonical uint256 decimal; False otherwise.
        """
        # ``str(2**256 - 1)`` is 78 chars; both request_id and nonce
        # share the same magnitude bound. Kept as a single check
        # instead of parametrising on ``MAX_REQUEST_ID`` vs
        # ``MAX_NONCE`` because the two are numerically identical and
        # any drift would be a spec regression, not a design change.
        max_len = len(str(MAX_REQUEST_ID))
        if not value or not REQUEST_ID_RE.fullmatch(value):
            self.context.logger.error(
                f"Rejecting offchain request: {name} must be a canonical "
                f"ASCII decimal ({name}={value!r})."
            )
            return False
        if len(value) > max_len or int(value) > MAX_REQUEST_ID:
            self.context.logger.error(
                f"Rejecting offchain request: {name} exceeds uint256 upper "
                f"bound ({name}={value})."
            )
            return False
        return True

    def _verify_ipfs_hash_binding(
        self,
        request_id: str,
        ipfs_hash: str,
        ipfs_data: str,
    ) -> "tuple[bool, str]":
        """Return (True, "") iff the local CID of ``ipfs_data`` matches ``ipfs_hash``.

        The trader signs a ``request_id`` derived from ``keccak(ipfs_hash)``
        where ``ipfs_hash`` is the on-chain content commitment. The
        signature therefore binds the ``request_id`` to a content hash,
        not to any particular body — a client could sign the CID of a
        benign prompt and paste an expensive one into the same POST.
        This method re-derives the CID over the raw ``ipfs_data`` bytes
        the mech will feed to ``_handle_get_task`` and compares against
        the caller-posted ``ipfs_hash`` before the request enqueues.

        The wire ``ipfs_hash`` is a 0x-prefixed hex string of either
        32 bytes (bare SHA-256 digest, 64 hex chars — matches
        ``to_multihash`` on the locally-derived CID) or 34 bytes
        (SHA-256 multihash prefix ``0x1220`` + 32-byte digest, 68 hex
        chars). Both forms reduce to the same 32-byte digest for the
        compare; the multihash prefix is stripped only from the 68-char
        form (a bare 64-char digest whose bytes happen to start with
        ``1220`` is content, not a prefix).

        :param request_id: the caller-supplied off-chain request id
            (used only for the log line).
        :param ipfs_hash: the caller-posted ``ipfs_hash`` from the
            signed body, already ``IPFS_HASH_RE``-validated.
        :param ipfs_data: the raw request-metadata JSON string carried
            inline in the signed body under ``ipfs_data``.
        :return: ``(True, "")`` on match; ``(False, reason)`` on
            rejection, where ``reason`` is one of
            ``IPFS_HASH_BODY_MISMATCH`` (empty body or digest
            disagreement) or ``IPFS_DATA_OVERSIZE`` (body over the
            single-block CID bound).
        """
        if not ipfs_data:
            self.context.logger.error(
                "Rejecting offchain request %s: ipfs_data is empty; "
                "cannot re-derive the CID for the binding check.",
                request_id,
            )
            return False, IPFS_HASH_BODY_MISMATCH
        encoded_body = ipfs_data.encode(ENCODING_UTF8)
        # Range-check the encoded body before ``compute_cidv1`` so an
        # oversize body surfaces as a distinct rejection reason
        # (``IPFS_DATA_OVERSIZE``) instead of being folded into the
        # ``ValueError`` catch below alongside actual CID-mismatch
        # cases. The 256 KiB single-block bound applies to the encoded
        # bytes, not the source string length.
        if len(encoded_body) > MAX_IPFS_DATA_BYTES:
            self.context.logger.error(
                "Rejecting offchain request %s: ipfs_data length %d bytes "
                "exceeds the single-block CID bound (%d bytes).",
                request_id,
                len(encoded_body),
                MAX_IPFS_DATA_BYTES,
            )
            return False, IPFS_DATA_OVERSIZE
        try:
            local_cid = compute_cidv1(encoded_body)
        except ValueError:
            # Reachable only via ``local_cid._varint`` (rejects negative
            # sizes it does not see today because the size guard above
            # short-circuits first). Kept as a defensive fallback so a
            # future ``compute_cidv1`` refactor cannot silently 500 on
            # an accept-time input.
            self.context.logger.exception(
                "Rejecting offchain request %s: local CID derivation "
                "raised ValueError; treating as a content-hash mismatch.",
                request_id,
            )
            return False, IPFS_HASH_BODY_MISMATCH
        local_digest_hex = to_multihash(local_cid)
        # Strip a leading ``0x`` (guaranteed by ``IPFS_HASH_RE``) and,
        # only for the 68-char multihash form, the ``1220`` SHA-256
        # function-code + digest-length prefix. The 64-char bare-digest
        # form is left as-is: a bare digest whose bytes happen to start
        # with ``1220`` is content, not a wrapper, and stripping four
        # chars there would deterministically break every valid payload
        # whose digest starts with that nibble sequence.
        posted_hex = ipfs_hash[2:]
        if len(posted_hex) == 68 and posted_hex.startswith(IPFS_HASH_MULTIHASH_PREFIX):
            posted_hex = posted_hex[len(IPFS_HASH_MULTIHASH_PREFIX) :]
        # Case-fold: ``IPFS_HASH_RE`` accepts both cases; ``to_multihash``
        # returns lowercase. Compare lowercase to lowercase.
        if posted_hex.lower() != local_digest_hex.lower():
            self.context.logger.error(
                "Rejecting offchain request %s: local CID digest "
                "0x%s does not match posted ipfs_hash %s.",
                request_id,
                local_digest_hex,
                ipfs_hash,
            )
            return False, IPFS_HASH_BODY_MISMATCH
        return True, ""

    def _is_duplicate_request(self, request_id: str) -> bool:
        """Return True iff ``request_id`` is already in an accepted state.

        Trace of the off-chain request lifecycle:

        1. ``_enqueue_offchain_request`` writes
           ``in_memory_requests[str(request_id)]`` at accept time.
        2. The behaviour picks the task, executes it, and pops
           ``in_memory_requests[str(req_id)]`` in ``_handle_done_task``
           (see ``behaviours.py:1604``). Between accept and pop the
           entry is present through every intermediate state (pending
           list, ``_executing_task`` attribute).
        3. ``offchain_request_responses[str(request_id)]`` is populated
           at settlement (delivered), on 402 challenges (pre-enqueue,
           the caller must retry after depositing), on defensive 500
           responses (post-auth), and on post-auth rejection
           (``_record_offchain_failure``).

        Only settled entries — those whose ``status`` is not
        ``REJECTED`` — count as "already accepted" for dedup. A stored
        rejection means the request was NOT enqueued, so the retry
        that follows a 402 deposit (mech-client's ``auto_deposit`` path
        re-posts the identical body) must not be short-circuited into
        an ``already accepted`` reply that never runs the balance
        check. Keys on both sides are ``str``; the wire ``request_id``
        is also ``str`` after the ``REQUEST_ID_RE`` guard, so no
        coercion is needed for the lookup.

        :param request_id: the wire-format request id (already
            ``REQUEST_ID_RE``-validated and inside the uint256 bound).
        :return: True if the id is present in ``in_memory_requests``
            (pending or executing), or the ``offchain_request_responses``
            entry for it is a settled/accepted payload rather than a
            rejection.
        """
        if request_id in self.in_memory_requests:
            return True
        stored = self.offchain_request_responses.get(request_id)
        if stored is None:
            return False
        return stored.get(ResponseKey.STATUS.value) != ResponseStatus.REJECTED.value

    def _handle_offchain_request_info(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle GET requests to fetch offchain request info.

        :param http_msg: the HttpMessage instance
        :param http_dialogue: the HttpDialogue instance
        """

        try:
            data = self._parse_http_body(http_msg)
            request_id = data[RequestKey.REQUEST_ID.value]
        except Exception as e:
            self.context.logger.error(f"Error getting offchain request info: {str(e)}")
            self._handle_bad_request(http_msg, http_dialogue)
            return

        self.context.logger.info(f"Fetching offchain info for {request_id=}.")

        # ``OFFCHAIN_REQUEST_RESPONSES`` is ``str``-keyed on the writer
        # side (see :mod:`task_execution.behaviours._handle_done_task`
        # and ``_record_offchain_failure`` — both convert the coerced
        # ``int`` ``req_id`` back to ``str`` before writing). ``request_id``
        # from the GET body is already ``str``, but wrap in ``str(...)``
        # so a future writer regression that lands an ``int`` key is
        # still findable by the requester.
        stored_response = self.offchain_request_responses.get(str(request_id))
        if stored_response is not None:
            self._send_ok_response(
                http_msg,
                http_dialogue,
                data=stored_response,
            )
            return

        done_tasks_list = self.done_tasks

        # Same reason as above: ``done_task["request_id"]`` is ``int`` for
        # off-chain tasks post the ingress coercion, but ``request_id``
        # from the GET body is ``str``. Normalize both sides so the
        # fallback scan doesn't silently miss.
        requested_done_tasks_list = [
            item
            for item in done_tasks_list
            if str(item.get(RequestKey.REQUEST_ID.value)) == str(request_id)
        ]

        self._send_ok_response(
            http_msg,
            http_dialogue,
            data=requested_done_tasks_list[0] if requested_done_tasks_list else {},
        )

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
        extra_headers: str = "",
    ) -> None:
        r"""Send an OK response with the provided data.

        :param http_msg: the incoming HTTP request message.
        :param http_dialogue: the HTTP dialogue used to reply.
        :param data: the response body payload, serialized as JSON.
        :param extra_headers: optional pre-formatted header block (each header
            terminated by ``\n``) prepended to the response. Callers use this to
            add audit headers (e.g. ``Payment-Receipt``) without rewriting body.
        """
        # Each header line must be newline-terminated, else it would silently
        # merge with the Content-Type line that follows.
        if extra_headers and not extra_headers.endswith("\n"):
            raise ValueError("extra_headers must be empty or newline-terminated")
        http_response = http_dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=http_msg,
            version=http_msg.version,
            status_code=HttpCode.OK_CODE.value,
            status_text="Success",
            headers=f"{extra_headers}{self.json_content_header}{http_msg.headers}",
            body=json.dumps(data).encode(ENCODING_UTF8),
        )

        # Send response
        self.context.logger.info("Responding with: {}".format(http_response))
        self.context.outbox.put_message(message=http_response)

    def _send_rejection_response(
        self,
        http_msg: HttpMessage,
        http_dialogue: HttpDialogue,
        request_id: str,
        reason: str,
        status_code: int,
        status_text: str,
        extra_headers: str = "",
        body_extras: Optional[Dict[str, Any]] = None,
        record_response: bool = False,
    ) -> None:
        """Build a rejection payload, optionally persist it, and reply.

        :param http_msg: the incoming HTTP request message.
        :param http_dialogue: the HTTP dialogue used to reply.
        :param request_id: the off-chain request id being rejected.
        :param reason: a short human-readable rejection reason.
        :param status_code: the HTTP status code to emit.
        :param status_text: the HTTP status text to emit.
        :param extra_headers: optional pre-formatted header block to prepend.
        :param body_extras: optional dict merged into the JSON response body.
        :param record_response: when True, persist the rejection payload in
            ``offchain_request_responses`` keyed by ``request_id`` so the
            polling endpoint can surface it. Callers on pre-authentication
            paths (bad signature, malformed body, ledger unavailable)
            must leave this False so a caller cannot pre-populate an
            arbitrary payload under an id they do not own. Only the 402
            balance-insufficient path passes True: the caller has already
            proven signature ownership and needs the challenge readable.
        """
        # Each header line must be newline-terminated, else it would silently
        # merge with the Content-Type line that follows.
        if extra_headers and not extra_headers.endswith("\n"):
            raise ValueError("extra_headers must be empty or newline-terminated")
        # ``body_extras`` is merged into the JSON body alongside the canonical
        # ``{request_id, status, reason}`` keys, so legacy clients that only
        # read ``reason`` keep working while new clients can pick up structured
        # fields such as the 402 challenge.
        response_payload: Dict[str, Any] = {
            RequestKey.REQUEST_ID.value: request_id,
            ResponseKey.STATUS.value: ResponseStatus.REJECTED.value,
            ResponseKey.REASON.value: reason,
        }
        if body_extras:
            response_payload.update(body_extras)
        if record_response:
            self.offchain_request_responses[request_id] = response_payload
        http_response = http_dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=http_msg,
            version=http_msg.version,
            status_code=status_code,
            status_text=status_text,
            headers=f"{extra_headers}{self.json_content_header}{http_msg.headers}",
            body=json.dumps(response_payload).encode(ENCODING_UTF8),
        )
        self.context.logger.info("Responding with: {}".format(http_response))
        self.context.outbox.put_message(message=http_response)

    def _send_internal_error(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue, request_id: str
    ) -> None:
        """Send a 500 with no extra headers / body extras.

        Used when a defensive guard in the 402 / header builders fires on the
        request-handling path: a bare ``raise`` there would leave the client
        with no HTTP reply (hung). This emits a definitive 500 instead; passing
        no extra_headers keeps the senders' own newline guard from re-raising.

        Both current 500 call sites are post-authentication (the sig-verify
        gate has already passed and the caller has proven ownership of
        ``request_id``), so persisting the rejection payload for the
        polling endpoint is safe: it cannot be used to pre-poison an id
        the caller does not own.

        :param http_msg: the incoming HTTP request message.
        :param http_dialogue: the HTTP dialogue used to reply.
        :param request_id: the off-chain request id being rejected.
        """
        self._send_rejection_response(
            http_msg,
            http_dialogue,
            request_id,
            reason="internal error",
            status_code=HttpCode.INTERNAL_SERVER_ERROR_CODE.value,
            status_text="Internal server error",
            record_response=True,
        )

    def _build_402_challenge(
        self,
        balance_check: Dict[str, Any],
        error_msg: str,
    ) -> Dict[str, Any]:
        """Build the structured 402 challenge body.

        :param balance_check: a successful result dict from
            ``_check_offchain_requester_balance``; it must carry the
            balance-tracker address. An error-shaped dict is rejected rather than
            silently emitting zero-address deposit instructions.
        :param error_msg: a short human-readable error string echoed in the body.
        :return: the structured 402 challenge as a dict ready for JSON encoding.
        :raises ValueError: if ``balance_check`` lacks the balance-tracker address.
        """
        # Native-asset payment models surface the zero address for ``asset``;
        # clients that see it must skip the ERC20 approve step.
        if ResponseKey.BALANCE_TRACKER_ADDRESS.value not in balance_check:
            raise ValueError(
                "cannot build a 402 challenge from a balance check without a "
                "balance_tracker_address; refusing to emit zero-address deposit "
                "instructions"
            )
        balance_tracker_address = cast(
            str, balance_check[ResponseKey.BALANCE_TRACKER_ADDRESS.value]
        )
        # Normalize the lookup key — `or ""` guards a present-but-None
        # payment_type (a bare cast would be a runtime no-op and .lower() would
        # then AttributeError); the map keys are lower-cased at load time
        # (models.Params) so a checksummed payment_type still resolves.
        payment_type = (balance_check.get(ResponseKey.PAYMENT_TYPE.value) or "").lower()
        asset_address = self.params.payment_type_to_asset_address.get(
            payment_type, ZERO_ADDRESS
        )
        return {
            "scheme": PAYMENT_SCHEME,
            "payTo": balance_tracker_address,
            "asset": asset_address,
            "chainId": int(balance_check.get(ResponseKey.CHAIN_ID.value, 0)),
            "currentBalance": str(
                balance_check.get(ResponseKey.AVAILABLE_AMOUNT.value, 0)
            ),
            "required": str(balance_check.get(ResponseKey.REQUIRED_AMOUNT.value, 0)),
            "depositInstructions": {
                "contract": balance_tracker_address,
                "abi": DEPOSIT_FN_ABI,
            },
            "error": error_msg,
        }

    def _build_www_authenticate_header(self) -> str:
        """Build the ``WWW-Authenticate`` header line for a 402 response.

        :return: a single header line terminated with newline.
        :raises ValueError: defensive-only. The 402 branch that calls
            this is only reached after ``_check_offchain_requester_balance``
            returned OK, and that method returns ``UNAVAILABLE`` when
            ``_marketplace_mech_address is None`` — so the guard below
            is unreachable on today's paths. Kept so a future refactor
            that widens the balance-check contract cannot silently emit
            a blank-realm challenge.
        """
        # The ``realm`` carries the mech address so clients with multiple mechs
        # configured can route the deposit to the right balance tracker.
        if self._marketplace_mech_address is None:
            raise ValueError(
                "cannot build a WWW-Authenticate header without a "
                "marketplace mech address"
            )
        return (
            f'WWW-Authenticate: Payment scheme="{PAYMENT_SCHEME}" '
            f'realm="{self._marketplace_mech_address}"\n'
        )

    def _build_payment_receipt_header(
        self, request_id: str, accepted_amount: int
    ) -> str:
        """Build the ``Payment-Receipt`` header line for a 200 response.

        :param request_id: the off-chain request id being acknowledged.
        :param accepted_amount: the requester-signed delivery rate being committed.
        :return: a single header line terminated with newline.
        """
        # The base64-encoded JSON payload is intentionally a snapshot at HTTP
        # accept time, NOT a settlement confirmation. ``settlement_status`` is
        # always ``"pending"`` here; on-chain settlement happens later via the
        # task_submission flow.
        receipt = {
            "request_id": request_id,
            "accepted_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "accepted_amount": str(accepted_amount),
            "settlement_status": SETTLEMENT_STATUS_PENDING,
        }
        encoded = base64.b64encode(json.dumps(receipt).encode(ENCODING_UTF8))
        return f"Payment-Receipt: {encoded.decode('ascii')}\n"

    def _rollback_offchain_enqueue(
        self,
        request_id: str,
        sender_checksum: Optional[str] = None,
        wire_nonce: Optional[int] = None,
    ) -> None:
        """Rollback a partial off-chain enqueue in case of unexpected failure.

        :param request_id: the wire-format request id whose partial
            entries must be removed from the pending queue and the
            in-memory buffer.
        :param sender_checksum: the sender address in checksum form.
            When supplied together with ``wire_nonce``, the wire nonce
            is also removed from the sender's outstanding set so the
            admission gate on subsequent accepts stays coherent with
            the actual live queue. Optional so pre-existing callers
            that did not track sender / nonce still work.
        :param wire_nonce: the wire nonce recorded in the outstanding
            set at enqueue time. See ``sender_checksum``.
        """
        # The pending task's ``requestId`` is stored as ``int`` after the
        # ingress coercion in :meth:`_enqueue_offchain_request`; the local
        # ``request_id`` here is still the wire-format ``str`` used as the
        # in-memory dict key. Compare on ``int`` so the filter actually
        # matches — a mixed ``str`` vs ``int`` comparison would silently
        # leave the partial entry in the queue and defeat the rollback.
        target_id_int = int(request_id)
        self.context.shared_state[PENDING_TASKS] = [
            req
            for req in self.pending_tasks
            if req.get(RequestKey.REQUEST_ID_CAMEL.value) != target_id_int
        ]
        self.in_memory_requests.pop(request_id, None)
        if sender_checksum is not None and wire_nonce is not None:
            # Rollback drops from the accepted set only. The settling
            # set is populated by ``_release_outstanding_nonce`` on the
            # done path — a rollback fires strictly before that
            # transition so the entry is guaranteed to still be in the
            # accepted half.
            accepted = self.accepted_nonces_by_sender.get(sender_checksum)
            if accepted is not None:
                accepted.discard(wire_nonce)
                if not accepted:
                    self.accepted_nonces_by_sender.pop(sender_checksum, None)
        self.context.logger.error(
            f"Queue rollback applied for {request_id=}. "
            f"pending_tasks={len(self.pending_tasks)} "
            f"in_memory_requests={len(self.in_memory_requests)}"
        )

    def _parse_http_body(self, http_msg: HttpMessage) -> Dict[str, str]:
        """Parse form-urlencoded HTTP body into a flat key-value dictionary."""
        body = http_msg.body
        if len(body) > MAX_HTTP_BODY_BYTES:
            raise ValueError(
                f"HTTP body is {len(body)} bytes, exceeds cap of "
                f"{MAX_HTTP_BODY_BYTES}"
            )
        request_data = body.decode(ENCODING_UTF8)
        parsed_data = urllib.parse.parse_qs(request_data)
        return {key: value[0] for key, value in parsed_data.items()}

    def _enqueue_offchain_request(
        self,
        request_id: str,
        ipfs_hash: str,
        request_delivery_rate: int,
        data: Dict[str, Any],
        sender_checksum: str,
        wire_nonce: int,
    ) -> Dict[str, Any]:
        """Enqueue the off-chain task and buffer its request metadata locally.

        :param request_id: the off-chain request id.
        :param ipfs_hash: the requester-supplied content hash (0x-prefixed hex).
        :param request_delivery_rate: the requester-signed delivery rate.
        :param data: the request-body dict. Caller passes a copy of the
            parsed HTTP body with ``nonce`` coerced to ``int`` so the
            downstream lexicographic-vs-numeric sort discipline holds;
            the type is ``Dict[str, Any]`` (not ``Dict[str, str]``)
            because the coerced ``nonce`` is an ``int``.
        :param sender_checksum: the sender address in checksum form.
            Used to record the wire nonce in ``outstanding_nonces_by_sender``
            so the admission gate on subsequent accepts from the same
            sender computes the correct next-expected slot.
        :param wire_nonce: the wire ``nonce`` (already coerced to
            ``int`` and admitted by the nonce-bind gate). Recorded in
            the outstanding set alongside the pending task.
        :return: the queued task dict.
        :raises Exception: if either queue write fails; partial state is rolled back.
        """
        # The off-chain path skips the IPFS upload entirely and keeps the
        # request JSON in process memory under ``in_memory_requests``. The
        # content commitment on chain still comes from the locally-computed
        # CID (response side), so the mech's on-chain receipt is unchanged.
        #
        # ``requestId`` is normalized to ``int`` so on-chain tasks (already
        # ``int`` via ``int.from_bytes(bytes32, "big")`` in
        # :mod:`task_execution.behaviours`) and off-chain tasks agree on
        # the type once they land in ``shared_state[DONE_TASKS]``. Without
        # this, ``TaskPoolingRound.end_block`` crashed on
        # ``sorted(..., key=lambda x: x["request_id"])`` and
        # ``TaskExecutionBaseBehaviour.remove_tasks`` silently failed its
        # equality check on a mixed-type batch.
        #
        # The client-supplied body (``data``) is stripped of any key we
        # own before merge, so a request carrying ``request_id=...``,
        # ``is_offchain=...``, ``data=...``, or ``request_delivery_rate=...``
        # can't override the trusted values above. This closes two
        # separate problems: (a) a body-supplied ``request_id`` would
        # re-plant the wire-format ``str`` and defeat the coercion, and
        # (b) a body-supplied ``request_delivery_rate`` would land in
        # ``request_id_to_delivery_rate_info`` and bypass the tool-
        # minimum-price gate in
        # :meth:`task_execution.behaviours.TaskExecutionBehaviour._handle_get_task`
        # (see the ``req_id_delivery_rate < tool_pricing`` check).
        # The signature verifies the request-id and delivery-rate at the
        # marketplace on chain, not here, so pre-settlement trust in
        # ``data`` values other than these four must be nil.
        #
        # The local ``request_id`` variable stays ``str`` — it is used as
        # a dict key in ``in_memory_requests`` and
        # ``offchain_request_responses`` (both typed ``Dict[str, ...]``)
        # and matched against the caller-visible response body. Only the
        # value that flows into the pending-task dict (and from there
        # into ``done_tasks``) is coerced.
        # ``reserved_keys`` also covers three keys the behaviour later reads
        # off ``_executing_task`` (``mech_address``, ``task_executor_address``,
        # ``request_id_nonce``) and the ``tool`` used for pricing / metrics.
        # The behaviour re-spreads ``**executing_task`` when it builds
        # ``done_task`` in ``_handle_done_task``, so any of these leaking in
        # from a client body would ride all the way to consensus and either
        # (a) reroute the on-chain delivery (``mech_address``), (b)
        # misattribute the executor for karma accounting
        # (``task_executor_address``), (c) desync the on-chain signature
        # (``request_id_nonce``, read as ``requestIdWithNonce`` upstream), or
        # (d) point the tool-price gate at the wrong tool.
        reserved_keys = {
            RequestKey.REQUEST_ID.value,
            RequestKey.REQUEST_ID_CAMEL.value,
            RequestKey.IS_OFFCHAIN.value,
            BodyKey.DATA.value,
            RequestKey.REQUEST_DELIVERY_RATE.value,
            "mech_address",
            "task_executor_address",
            "request_id_nonce",
            "requestIdWithNonce",
            "tool",
        }
        # ``RequestKey.REQUEST_ID.value`` is mandatory on the body (read at
        # ``_handle_signed_requests`` and 400 on missing), so it is present
        # on every accepted request; excluding it here keeps the log
        # a real anomaly signal instead of firing on 100% of requests.
        dropped_keys = [
            k
            for k in data.keys()
            if k in reserved_keys and k != RequestKey.REQUEST_ID.value
        ]
        if dropped_keys:
            # Log at info: dropping is the safe outcome, but an integration
            # sending one of these needs a way to notice their field was
            # ignored rather than silently accepted.
            self.context.logger.info(
                "Dropping client-supplied reserved keys from offchain "
                "request %r: %s",
                request_id,
                sorted(dropped_keys),
            )
        req = {
            RequestKey.REQUEST_ID_CAMEL.value: int(request_id),
            BodyKey.DATA.value: bytes.fromhex(ipfs_hash[2:]),
            RequestKey.IS_OFFCHAIN.value: True,
            RequestKey.REQUEST_DELIVERY_RATE.value: request_delivery_rate,
            **{k: v for k, v in data.items() if k not in reserved_keys},
        }
        try:
            self.pending_tasks.append(req)
            self.in_memory_requests[request_id] = data[RequestKey.IPFS_DATA.value]
            # Track the wire nonce in the sender's accepted set so the
            # admission gate on subsequent accepts computes the correct
            # next-expected slot. Moves to the settling set at done
            # time via ``_release_outstanding_nonce``.
            self.accepted_nonces_by_sender.setdefault(sender_checksum, set()).add(
                wire_nonce
            )
        except Exception:
            self._rollback_offchain_enqueue(
                request_id,
                sender_checksum=sender_checksum,
                wire_nonce=wire_nonce,
            )
            raise
        # Buffer the request half of the durable preimage (no-op unless off-chain
        # preimage retention is enabled). The response half is added at
        # settlement; the behaviour flushes both to the kv_store asynchronously.
        if self.params.preimage_retention_enabled:
            preimage_buffer.record_accept(
                self.context.shared_state,
                request_id,
                data[RequestKey.IPFS_DATA.value],
                time.time(),
            )
        return req

    @staticmethod
    def _make_unavailable_balance_response(
        required_amount: int,
        reason: str,
    ) -> Dict[str, Union[str, int]]:
        """Build a standardised 'unavailable' balance-check response."""
        return {
            ResponseKey.STATUS.value: ResponseStatus.UNAVAILABLE.value,
            ResponseKey.REQUIRED_AMOUNT.value: required_amount,
            ResponseKey.AVAILABLE_AMOUNT.value: 0,
            ResponseKey.REASON.value: reason,
        }

    def _check_offchain_requester_balance(
        self,
        sender: str,
        delivery_rate: int,
    ) -> Dict[str, Union[str, int]]:
        """Check requester balance in balance tracker against requested delivery rate."""
        required_amount = int(delivery_rate)
        if self._marketplace_mech_address is None:
            return self._make_unavailable_balance_response(
                required_amount,
                "No marketplace mech configured for the offchain handler.",
            )
        ledger_settings = self._get_ledger_settings()
        if ledger_settings[ResponseKey.STATUS.value] != ResponseStatus.OK.value:
            return self._make_unavailable_balance_response(
                required_amount,
                cast(str, ledger_settings[ResponseKey.REASON.value]),
            )

        try:
            rpc_address = cast(str, ledger_settings[ResponseKey.RPC_ADDRESS.value])
            chain_id = cast(int, ledger_settings[ResponseKey.CHAIN_ID.value])
            ledger_api = self._get_ledger_api(rpc_address, chain_id)

            requester = ledger_api.api.to_checksum_address(sender)
            mech_address = ledger_api.api.to_checksum_address(
                self._marketplace_mech_address
            )
            marketplace_address = ledger_api.api.to_checksum_address(
                self.params.mech_marketplace_address
            )

            payment_type = self._get_mech_payment_type(ledger_api, mech_address)
            if payment_type is None:
                return self._make_unavailable_balance_response(
                    required_amount, "Unable to fetch mech payment type."
                )

            balance_tracker_address = (
                self._get_balance_tracker_address_for_payment_type(
                    ledger_api=ledger_api,
                    marketplace_address=marketplace_address,
                    payment_type=payment_type,
                )
            )
            if not balance_tracker_address or int(balance_tracker_address, 16) == 0:
                return self._make_unavailable_balance_response(
                    required_amount,
                    "No balance tracker configured for mech payment type.",
                )

            balance_tracker_address = ledger_api.api.to_checksum_address(
                balance_tracker_address
            )
            available_amount = self._get_requester_balance(
                ledger_api=ledger_api,
                balance_tracker_address=balance_tracker_address,
                requester=requester,
            )
            decision = (
                BALANCE_LOG_DECISION_ACCEPTED
                if available_amount >= required_amount
                else BALANCE_LOG_DECISION_REJECTED
            )
            self.context.logger.info(
                f"offchain_balance_check sender={sender} required={required_amount} "
                f"available={available_amount} decision={decision}"
            )
            return {
                ResponseKey.STATUS.value: ResponseStatus.OK.value,
                ResponseKey.REQUIRED_AMOUNT.value: required_amount,
                ResponseKey.AVAILABLE_AMOUNT.value: int(available_amount),
                ResponseKey.REASON.value: "balance check completed",
                ResponseKey.BALANCE_TRACKER_ADDRESS.value: balance_tracker_address,
                ResponseKey.PAYMENT_TYPE.value: payment_type,
                ResponseKey.CHAIN_ID.value: chain_id,
            }
        except Exception as e:
            return self._make_unavailable_balance_response(
                required_amount, f"Balance check failed: {str(e)}"
            )

    def _get_mech_payment_type(
        self, ledger_api: EthereumApi, mech_address: str
    ) -> Optional[str]:
        """Get the mech payment type from the mech contract.

        Wall-clock bounded through the on-path RPC executor so a slow
        RPC does not stall the AEA main thread past the http_server
        reply budget. See ``_BALANCE_RPC_DEADLINE_SECONDS``.

        :param ledger_api: the ledger API object.
        :param mech_address: the mech contract address.
        :return: the mech's payment type, or None if unavailable.
        """
        payment_type_res = self._run_with_wall_clock_deadline(
            lambda: OlasMechContract.get_mech_type(ledger_api, mech_address),
            deadline_seconds=_BALANCE_RPC_DEADLINE_SECONDS,
            label="mech_paymentType",
        )
        return cast(Optional[str], payment_type_res.get(BodyKey.MECH_TYPE.value))

    def _get_balance_tracker_address_for_payment_type(
        self,
        ledger_api: EthereumApi,
        marketplace_address: str,
        payment_type: str,
    ) -> str:
        """Get the balance tracker address for the provided payment type.

        Wall-clock bounded through the on-path RPC executor. See
        ``_BALANCE_RPC_DEADLINE_SECONDS``.

        :param ledger_api: the ledger API object.
        :param marketplace_address: the mech marketplace contract address.
        :param payment_type: the mech's payment type.
        :return: the balance tracker contract address for the payment type.
        """
        balance_tracker_res = self._run_with_wall_clock_deadline(
            lambda: (
                MechMarketplaceContract.get_balance_tracker_for_mech_type(
                    ledger_api=ledger_api,
                    contract_address=marketplace_address,
                    mech_type=payment_type,
                )
            ),
            deadline_seconds=_BALANCE_RPC_DEADLINE_SECONDS,
            label="marketplace_balanceTrackerForMechType",
        )
        return cast(str, balance_tracker_res.get(BodyKey.DATA.value))

    def _get_requester_balance(
        self, ledger_api: EthereumApi, balance_tracker_address: str, requester: str
    ) -> int:
        """Get requester balance from the balance tracker.

        Wall-clock bounded through the on-path RPC executor. See
        ``_BALANCE_RPC_DEADLINE_SECONDS``.

        :param ledger_api: the ledger API object.
        :param balance_tracker_address: the balance tracker address.
        :param requester: the requester (Safe) address.
        :return: the requester's balance in the balance tracker (0 if unavailable).
        """
        requester_balance_res = self._run_with_wall_clock_deadline(
            lambda: BalanceTrackerContract.get_requester_balance(
                ledger_api=ledger_api,
                contract_address=balance_tracker_address,
                requester=requester,
            ),
            deadline_seconds=_BALANCE_RPC_DEADLINE_SECONDS,
            label="balance_tracker_getRequesterBalance",
        )
        return cast(int, requester_balance_res.get(BodyKey.REQUESTER_BALANCE.value, 0))

    def _verify_offchain_request_signature(
        self,
        sender: str,
        ipfs_hash: str,
        delivery_rate: int,
        nonce: int,
        wire_request_id: str,
        signature_hex: str,
    ) -> SignatureVerdict:
        """Verify the caller signed the marketplace-derived request_id.

        Sequential path:

        1. Infra failure if the deployment-scoped constants
           (marketplace ``domainSeparator``, mech ``paymentType``) are
           missing. The caller replies 503.
        2. Client failure if the ``ipfs_hash`` or ``signature`` bytes
           are malformed hex, if the derivation fails, or if the wire
           request_id disagrees with the locally-derived value. The
           caller replies 401.
        3. Try local ``ecrecover`` against ``sender``. Match wins with
           zero RPC for the sig-recover step.
        4. Fall back to the sender's EIP-1271 ``isValidSignature`` view
           via a cached short-timeout ``EthereumApi``. The view
           returns one of three verdicts: ``VALID`` (accept),
           ``DECLINED`` (401 signature verification failed), or
           ``CALL_FAILED`` (503 with the ``EIP1271_CALL_FAILED``
           reason so an infra failure is not mis-reported as a
           credential rejection). A transport timeout on the view
           surfaces as 503 with the ``EIP1271_CALL_TIMEOUT`` reason.
        5. Bind the wire ``nonce`` to the sender's on-chain
           ``MechMarketplace.mapNonces[sender]`` value combined with
           the sender's live ``accepted + settling`` sets. A wire
           value below that expected next-slot rejects 401; a wire
           value above it, or a per-sender in-flight cap breach,
           rejects 503. A transport-level failure on the ``mapNonces``
           read fails closed as 503 rather than accepting under an
           unknown counter.

        :param sender: the requester address, EOA or Safe.
        :param ipfs_hash: 0x-prefixed hex string used as the on-chain
            ``requestData`` blob.
        :param delivery_rate: requester-signed delivery rate.
        :param nonce: requester nonce as tracked by
            ``MechMarketplace.mapNonces[sender]`` at signing time.
        :param wire_request_id: the ASCII decimal request_id supplied on
            the wire; must equal the locally-derived value.
        :param signature_hex: hex-encoded signature bytes, with or without
            a leading ``0x``.
        :return: a :class:`SignatureVerdict`. ``ok=True`` on a valid
            signature whose wire nonce sits inside the accepted
            on-chain window; ``ok=False`` with ``is_infra=True`` on
            server-side prerequisites (constants unset, ledger config
            missing, address preparation failure, ``mapNonces`` read
            failure) or ``is_infra=False`` on a bad-caller outcome
            (malformed hex, derivation mismatch, sig recovery failure,
            EIP-1271 view timeout, wire nonce below the on-chain
            counter or above the accepted window).
        """
        if self._domain_separator is None or self._payment_type is None:
            self.context.logger.error(
                "Cannot verify offchain signature: marketplace constants "
                "unavailable; rejecting request %s.",
                wire_request_id,
            )
            return SignatureVerdict(
                ok=False,
                reason="marketplace verification constants unavailable",
                is_infra=True,
            )

        try:
            request_data = bytes.fromhex(ipfs_hash[2:])
            signature_bytes = bytes.fromhex(
                signature_hex[2:] if signature_hex.startswith("0x") else signature_hex
            )
        except ValueError:
            self.context.logger.error(
                "Malformed hex in signature or ipfs_hash for request %s.",
                wire_request_id,
            )
            return SignatureVerdict(
                ok=False,
                reason="malformed signature or ipfs_hash hex",
                is_infra=False,
            )

        ledger_settings = self._get_ledger_settings()
        if ledger_settings[ResponseKey.STATUS.value] != ResponseStatus.OK.value:
            self.context.logger.error(
                "Cannot verify offchain signature: ledger settings unavailable "
                "(%s); rejecting request %s.",
                ledger_settings.get(ResponseKey.REASON.value, "unknown"),
                wire_request_id,
            )
            return SignatureVerdict(
                ok=False,
                reason="ledger settings unavailable",
                is_infra=True,
            )

        try:
            rpc_address = cast(str, ledger_settings[ResponseKey.RPC_ADDRESS.value])
            chain_id = cast(int, ledger_settings[ResponseKey.CHAIN_ID.value])
            ledger_api = self._get_ledger_api(rpc_address, chain_id)
            sender_checksum = ledger_api.api.to_checksum_address(sender)
            marketplace_address = ledger_api.api.to_checksum_address(
                self.params.mech_marketplace_address
            )
            # The cached ``_marketplace_mech_address`` is non-None here
            # because ``_domain_separator`` / ``_payment_type`` are both
            # None when the address is None (see
            # ``_initialise_offchain_verification_constants``), and the
            # None-guard on those constants at the top of this method
            # would have already returned an infra verdict.
            mech_address = ledger_api.api.to_checksum_address(
                cast(str, self._marketplace_mech_address)
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.context.logger.error(
                "Address preparation failed for request %s: %s.",
                wire_request_id,
                e,
            )
            return SignatureVerdict(
                ok=False,
                reason="address preparation failed",
                is_infra=True,
            )

        try:
            derived = compute_request_id(
                marketplace=marketplace_address,
                mech=mech_address,
                requester=sender_checksum,
                request_data=request_data,
                delivery_rate=delivery_rate,
                payment_type=self._payment_type,
                nonce=nonce,
                domain_separator=self._domain_separator,
            )
        except ValueError as e:
            self.context.logger.error(
                "Request-id derivation failed for %s: %s.", wire_request_id, e
            )
            return SignatureVerdict(
                ok=False,
                reason="request-id derivation failed",
                is_infra=False,
            )

        # ``wire_request_id`` has already passed the canonical ASCII decimal
        # + uint256 guards above, so ``int(...)`` is safe.
        if int.from_bytes(derived, "big") != int(wire_request_id):
            self.context.logger.warning(
                "Wire request_id %s does not match locally-derived value "
                "0x%s for sender %s.",
                wire_request_id,
                derived.hex(),
                sender_checksum,
            )
            return SignatureVerdict(
                ok=False,
                reason="wire request_id does not match local derivation",
                is_infra=False,
            )

        recovered = recover_eoa_signer(derived, signature_bytes)
        if recovered is not None and recovered == sender_checksum:
            self.context.logger.info(
                "offchain_auth accepted request %s sender=%s "
                "mechanism=ecrecover recovered=%s",
                wire_request_id,
                sender_checksum,
                recovered,
            )
            return SignatureVerdict(ok=True, reason="", is_infra=False)
        # Log the EOA branch outcome for auditability: recovery either
        # returned ``None`` (malformed / malleable / bad-v signature) or
        # yielded an address other than the declared sender. The
        # EIP-1271 fallback runs next.
        self.context.logger.debug(
            "offchain_auth eoa_recovery_failed request=%s sender=%s "
            "recovered=%s (falls through to EIP-1271 branch)",
            wire_request_id,
            sender_checksum,
            recovered,
        )

        # Sender is either a Safe or an EOA that produced an unrecognised
        # signature. Defer to the sender's EIP-1271 view for the final call.
        # Use a separately-cached ``EthereumApi`` with a short provider
        # HTTP timeout so a slow ``isValidSignature`` view cannot pin the
        # AEA main thread past the ``http_server`` reply budget; the
        # balance-check path continues to run against the ledger-default
        # ``EthereumApi`` cached above.
        eip1271_ledger_api = self._get_ledger_api(
            rpc_address, chain_id, timeout_seconds=_EIP1271_CALL_TIMEOUT_SECONDS
        )
        try:
            # Wall-clock bound the call so a slow / hostile
            # ``isValidSignature`` combined with the multiplying
            # web3 + RotatingHTTPProvider retry loops cannot stall the
            # AEA main thread past the http_server reply budget. See
            # ``_RPC_WALL_CLOCK_DEADLINE_SECONDS``.
            eip1271_verdict = self._run_with_wall_clock_deadline(
                lambda: check_eip1271_signature(
                    ledger_api=eip1271_ledger_api,
                    contract_address=sender_checksum,
                    message_hash=derived,
                    signature=signature_bytes,
                    gas=_EIP1271_CALL_GAS_CAP,
                    logger=self.context.logger,
                ),
                deadline_seconds=_RPC_WALL_CLOCK_DEADLINE_SECONDS,
                label="eip1271_isValidSignature",
            )
        except (TimeoutError, concurrent.futures.TimeoutError) as exc:
            self.context.logger.warning(
                "EIP-1271 isValidSignature timed out for sender %s on "
                "request %s: %s.",
                sender_checksum,
                wire_request_id,
                exc,
            )
            # Provider-side slowness — the gas cap already bounds a
            # hostile ``isValidSignature`` view to sub-millisecond CPU
            # (that path surfaces as ``ContractLogicError`` and returns
            # ``DECLINED`` on the other branch), so a timeout at this
            # depth is an infrastructure signal, mirroring
            # ``NONCE_READ_FAILED`` under identical conditions.
            return SignatureVerdict(
                ok=False,
                reason=EIP1271_CALL_TIMEOUT,
                is_infra=True,
            )
        except RuntimeError as exc:
            # Queue saturation or executor shutdown. Both are infra.
            self.context.logger.warning(
                "EIP-1271 isValidSignature could not be dispatched for "
                "sender %s on request %s: %s.",
                sender_checksum,
                wire_request_id,
                exc,
            )
            return SignatureVerdict(
                ok=False,
                reason=RPC_QUEUE_SATURATED,
                is_infra=True,
            )
        if eip1271_verdict == Eip1271Verdict.VALID:
            self.context.logger.info(
                "offchain_auth accepted request %s sender=%s mechanism=eip1271",
                wire_request_id,
                sender_checksum,
            )
            return SignatureVerdict(ok=True, reason="", is_infra=False)
        if eip1271_verdict == Eip1271Verdict.CALL_FAILED:
            # Infra-side failure inside ``isValidSignature`` (RPC error,
            # transport error, ABI decode failure). Route to 503 with a
            # dedicated reason so a legitimate Safe requester is not
            # 401'd during an RPC outage.
            self.context.logger.warning(
                "EIP-1271 isValidSignature call failed for sender %s on "
                "request %s; routing to 503.",
                sender_checksum,
                wire_request_id,
            )
            return SignatureVerdict(
                ok=False,
                reason=EIP1271_CALL_FAILED,
                is_infra=True,
            )
        return SignatureVerdict(
            ok=False,
            reason="signature verification failed",
            is_infra=False,
        )

    def _reconcile_stale_accepted(
        self,
        sender_checksum: str,
        on_chain_nonce: int,
    ) -> Set[int]:
        """Evict orphaned accepted entries and return what remains.

        An accepted entry with nonce **strictly below** ``on_chain_nonce``
        cannot settle: ``deliverMarketplaceWithSignatures`` derives
        each ``requestId`` from its own ``mapNonces[sender]`` at
        settlement time, and a task signed against an older slot does
        not match. This state is reached when the same sender uses
        both the on-chain and off-chain rails — an on-chain
        ``request()`` bumps ``mapNonces`` past a slot whose off-chain
        task is still in ``pending_tasks``.

        For each such entry the method evicts the corresponding
        pending task, pops its in-memory payload, and writes a
        rejection into ``offchain_request_responses`` so the polling
        client sees a definitive result. A WARNING per eviction
        carries sender + nonce + on_chain for operator triage.

        :param sender_checksum: the sender's checksum key.
        :param on_chain_nonce: current ``mapNonces[sender]`` value.
        :return: the surviving set for ``sender_checksum`` (empty
            set if the sender had none or every entry was orphaned).
            Also pops the sender key from the map if nothing survives.
        """
        entries = self.accepted_nonces_by_sender.get(sender_checksum)
        if not entries:
            return set()
        orphaned = sorted(n for n in entries if n < on_chain_nonce)
        if not orphaned:
            return entries
        for wire_nonce in orphaned:
            try:
                matched = self._evict_orphaned_pending(
                    sender_checksum, wire_nonce, on_chain_nonce
                )
            except Exception:  # pylint: disable=broad-exception-caught
                # Per-nonce eviction is isolated: a corrupt entry in
                # ``pending_tasks`` / ``done_tasks`` (non-dict, missing
                # keys) must not abort the loop and leave later orphans
                # in the accepted set. Log with traceback and discard
                # this nonce so the gate's slot count stays truthful.
                self.context.logger.warning(
                    "Failure while evicting orphaned nonce %d for sender %s "
                    "(on_chain=%d); dropping the gate entry regardless.",
                    wire_nonce,
                    sender_checksum,
                    on_chain_nonce,
                    exc_info=True,
                )
                matched = False
            if matched:
                self.context.logger.warning(
                    "Evicted orphaned accepted nonce %d for sender %s: "
                    "on-chain mapNonces has advanced to %d "
                    "(concurrent on-chain activity from this sender "
                    "consumed the slot).",
                    wire_nonce,
                    sender_checksum,
                    on_chain_nonce,
                )
            else:
                self.context.logger.warning(
                    "No pending/done task matched orphaned nonce %d for "
                    "sender %s (on_chain=%d); the task may be executing "
                    "or awaiting settlement. Dropping the gate entry.",
                    wire_nonce,
                    sender_checksum,
                    on_chain_nonce,
                )
            entries.discard(wire_nonce)
        if not entries:
            self.accepted_nonces_by_sender.pop(sender_checksum, None)
        return entries

    def _reconcile_stale_settling(
        self,
        sender_checksum: str,
        on_chain_nonce: int,
    ) -> Set[int]:
        """Drop settling entries the on-chain counter has already consumed.

        Called on ``settling`` before computing the admission gate's
        next-expected-slot formula. A settling entry with nonce
        **strictly below** ``on_chain_nonce`` corresponds to a slot
        the marketplace has already consumed (``mapNonces`` advances
        via ``deliverMarketplaceWithSignatures`` finishing a batch,
        or via a same-sender on-chain ``request()``). The drain is
        logged at INFO so the transition is observable.

        An entry AT ``on_chain_nonce`` is the next slot to be
        consumed and stays: the same wire nonce must remain
        recognisable as a duplicate while its batch is in flight.

        :param sender_checksum: the sender's checksum key.
        :param on_chain_nonce: current ``mapNonces[sender]`` value.
        :return: the surviving set for ``sender_checksum``.
        """
        entries = self.settling_nonces_by_sender.get(sender_checksum)
        if not entries:
            return set()
        drained = {n for n in entries if n < on_chain_nonce}
        if not drained:
            return entries
        self.context.logger.info(
            "Draining %d settled nonce(s) from sender %s "
            "(on-chain mapNonces advanced to %d): %s",
            len(drained),
            sender_checksum,
            on_chain_nonce,
            sorted(drained),
        )
        entries -= drained
        if not entries:
            self.settling_nonces_by_sender.pop(sender_checksum, None)
        return entries

    def _evict_orphaned_pending(
        self,
        sender_checksum: str,
        wire_nonce: int,
        on_chain_nonce: int,
    ) -> bool:
        """Remove an orphaned task from pending / done and record the rejection.

        Scans both ``pending_tasks`` and ``done_tasks`` for a matching
        off-chain task (case-insensitive on sender, exact match on
        wire_nonce). When found, drops it from the list, removes its
        in-memory payload, writes a rejection to
        ``offchain_request_responses`` so the polling client sees a
        terminal result, and records the rejection in the preimage
        buffer so the durable audit row moves to the ``rejected``
        terminal state instead of staying at ``processing``.

        Handler-side counterpart to
        ``behaviours.TaskExecutionBehaviour._record_offchain_failure``.

        :param sender_checksum: the sender's checksum key.
        :param wire_nonce: the orphaned wire nonce.
        :param on_chain_nonce: current ``mapNonces[sender]``, used in
            the eviction log for operator triage.
        :return: ``True`` if a matching task was found and evicted
            from either queue, ``False`` otherwise (task may be
            mid-flight between ``pending_tasks`` and ``done_tasks``).
        """
        sender_lower = sender_checksum.lower()
        rejection_reason = (
            "on-chain nonce advanced past this slot; concurrent on-chain "
            "activity from this sender consumed it before the off-chain "
            "batch could settle"
        )
        now = time.time()

        def _match(task: Dict[str, Any]) -> bool:
            if not isinstance(task, dict):
                return False
            if not task.get(RequestKey.IS_OFFCHAIN.value):
                return False
            task_sender = str(task.get(RequestKey.SENDER.value, "")).lower()
            if task_sender != sender_lower:
                return False
            raw_nonce = task.get(RequestKey.NONCE.value)
            if raw_nonce is None:
                return False
            try:
                return int(raw_nonce) == wire_nonce
            except (TypeError, ValueError):
                return False

        matched = False
        for source_label, storage in (
            ("pending_tasks", self.pending_tasks),
            ("done_tasks", self.done_tasks),
        ):
            for idx in range(len(storage) - 1, -1, -1):
                task = storage[idx]
                if not _match(task):
                    continue
                request_id = str(task.get(RequestKey.REQUEST_ID_CAMEL.value, ""))
                del storage[idx]
                if request_id:
                    self.in_memory_requests.pop(request_id, None)
                    self.offchain_request_responses[request_id] = {
                        RequestKey.REQUEST_ID.value: request_id,
                        ResponseKey.STATUS.value: ResponseStatus.REJECTED.value,
                        ResponseKey.REASON.value: rejection_reason,
                    }
                    if self.params.preimage_retention_enabled:
                        preimage_buffer.record_settlement(
                            self.context.shared_state,
                            request_id,
                            rejection_reason,
                            None,
                            preimage_buffer.STATUS_REJECTED,
                            now,
                        )
                self.context.logger.info(
                    "Removed orphaned task from %s: request_id=%s "
                    "sender=%s nonce=%d on_chain=%d.",
                    source_label,
                    request_id,
                    sender_checksum,
                    wire_nonce,
                    on_chain_nonce,
                )
                matched = True
        return matched

    def _bind_wire_nonce_to_chain(
        self,
        sender_checksum: str,
        wire_nonce: int,
        wire_request_id: str,
    ) -> SignatureVerdict:
        """Admission gate: bind the wire ``nonce`` to the sender's next slot.

        Reads ``MechMarketplace.mapNonces[sender]`` and combines the
        result with the sender's ``accepted`` and ``settling`` sets to
        compute the single next-expected slot:
        ``on_chain + len(accepted) + len(settling)``. This is the
        contiguity check settlement's
        ``_deliverMarketplaceWithSignatures`` already enforces on
        chain — it recomputes each request_id from its own
        ``nonce, nonce+1, ...`` counter and reverts the whole
        per-sender batch on the first mismatch. Admitting a request
        that skips a slot would enqueue work that settlement is
        guaranteed to revert, dragging every legitimately co-batched
        request down with it.

        Splitting ``accepted`` from ``settling`` is required for the
        formula to stay monotonic across the release/settlement gap.
        ``accepted`` drains on the mech side when the task hits
        ``_finalize_done_task``; ``mapNonces[sender]`` only advances
        several ABCI rounds later when
        ``_deliverMarketplaceWithSignatures`` lands on chain. Without
        the ``settling`` half, the nonce is counted by NEITHER term
        during that window and the gate lets a duplicate re-enter
        (`accepted` cleared, `mapNonces` not yet advanced) OR 503s
        every honest sequential request the sender fires until the
        batch settles.

        Both sets are pruned of entries ``< on_chain_nonce`` on every
        call so a settlement that lands between two accepts drops
        the corresponding entry from ``settling`` and the formula
        collapses back to the pre-accept baseline.

        Outcomes:

        * ``wire == expected``: accept (pending per-sender cap).
        * ``wire in accepted`` or ``wire in settling``: reject 401
          (``NONCE_BELOW_EXPECTED``); a duplicate submission of a
          nonce that is either in flight or settling.
        * ``wire < expected``: reject 401 (``NONCE_BELOW_EXPECTED``).
        * ``wire > expected``: reject 503 (``NONCE_ABOVE_EXPECTED``).
        * ``accepted + settling >= MAX_ACCEPTED_PER_SENDER``: reject
          503 (``SENDER_INFLIGHT_LIMIT``); the sender has too much
          unsettled work.
        * Read failure: reject 503 (``NONCE_READ_FAILED`` or
          ``NONCE_READ_UNRECOVERABLE``).

        :param sender_checksum: the sender address in checksum form.
        :param wire_nonce: the ``nonce`` supplied on the wire (already
            coerced to ``int`` and range-checked at ingress).
        :param wire_request_id: the wire request_id, for logging only.
        :return: a :class:`SignatureVerdict`.
        """
        ledger_settings = self._get_ledger_settings()
        if ledger_settings[ResponseKey.STATUS.value] != ResponseStatus.OK.value:
            # Sig-verify has already gated on the same settings above
            # by the time this method runs, but a reconfiguration
            # between the two calls (settings dict rebuilt per call) is
            # possible in principle; treat it as an infra failure so
            # the caller returns 503 and the request is retried.
            self.context.logger.warning(
                "Cannot bind wire nonce: ledger settings unavailable "
                "(%s); rejecting request %s.",
                ledger_settings.get(ResponseKey.REASON.value, "unknown"),
                wire_request_id,
            )
            return SignatureVerdict(
                ok=False,
                reason=NONCE_READ_FAILED,
                is_infra=True,
            )
        rpc_address = cast(str, ledger_settings[ResponseKey.RPC_ADDRESS.value])
        chain_id = cast(int, ledger_settings[ResponseKey.CHAIN_ID.value])
        ledger_api = self._get_ledger_api(rpc_address, chain_id)
        try:
            # Wall-clock bound the read for the same reason as the
            # EIP-1271 view above; see ``_RPC_WALL_CLOCK_DEADLINE_SECONDS``.
            on_chain_result = self._run_with_wall_clock_deadline(
                lambda: MechMarketplaceContract.get_nonce(
                    ledger_api=ledger_api,
                    contract_address=self.params.mech_marketplace_address,
                    sender_address=sender_checksum,
                ),
                deadline_seconds=_RPC_WALL_CLOCK_DEADLINE_SECONDS,
                label="mapNonces_read",
            )
            on_chain_nonce = int(cast(int, on_chain_result.get(BodyKey.DATA.value, 0)))
        except concurrent.futures.TimeoutError as exc:
            # Wall-clock deadline elapsed. Kept separate from the
            # broader exception branch below so operators can tell a
            # slow RPC apart from a deterministic ABI / config drift.
            self.context.logger.warning(
                "mapNonces read timed out for sender %s on request %s: %s.",
                sender_checksum,
                wire_request_id,
                exc,
            )
            return SignatureVerdict(
                ok=False,
                reason=NONCE_READ_FAILED,
                is_infra=True,
            )
        except RuntimeError as exc:
            # Queue saturation or executor shutdown. Both are infra.
            self.context.logger.warning(
                "mapNonces read could not be dispatched for sender %s on "
                "request %s: %s.",
                sender_checksum,
                wire_request_id,
                exc,
            )
            return SignatureVerdict(
                ok=False,
                reason=RPC_QUEUE_SATURATED,
                is_infra=True,
            )
        except (
            Web3RPCError,
            RequestException,
            ConnectionError,
            OSError,
        ) as exc:
            # Transient RPC / transport failure — a flaky node, a
            # network blip, a rate-limiter mid-flight reset. Route to
            # the retryable ``NONCE_READ_FAILED`` reason and let the
            # caller's 503 nudge mech-client's retry loop.
            self.context.logger.warning(
                "mapNonces read failed (transient) for sender %s on " "request %s: %r",
                sender_checksum,
                wire_request_id,
                exc,
                exc_info=True,
            )
            return SignatureVerdict(
                ok=False,
                reason=NONCE_READ_FAILED,
                is_infra=True,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Deterministic failure classes: ``BadFunctionCallOutput``
            # (wrong ``mech_marketplace_address`` or ABI drift),
            # ``AttributeError`` (non-dict wrapper return),
            # ``ValueError`` (ABI decode failure), ``ContractLogicError``
            # (revert on a proxy pointing at a legacy implementation).
            # None recover without an operator config change; keep the
            # 503 so the client backs off, but surface a distinct
            # reason so the log line names the fault as unrecoverable
            # rather than collapsing into the same warning that a
            # flaky node produces. ``exc_info=True`` so the traceback
            # lands in the log for triage.
            self.context.logger.warning(
                "mapNonces read failed unrecoverably for sender %s on "
                "request %s: %r",
                sender_checksum,
                wire_request_id,
                exc,
                exc_info=True,
            )
            return SignatureVerdict(
                ok=False,
                reason=NONCE_READ_UNRECOVERABLE,
                is_infra=True,
            )
        # Reconcile both maps against the current on-chain counter
        # before computing the slot count. Accepted entries below
        # the counter carry their pending tasks with them to the
        # eviction path; settling entries below the counter are
        # drained (their slot has already been consumed on chain).
        accepted = self._reconcile_stale_accepted(sender_checksum, on_chain_nonce)
        settling = self._reconcile_stale_settling(sender_checksum, on_chain_nonce)
        expected_next_nonce = on_chain_nonce + len(accepted) + len(settling)
        # Duplicate detection covers both sets: a wire nonce already
        # accepted (and either still pending on the mech side or moved
        # to settling awaiting on-chain settlement) is a duplicate
        # under whichever half currently holds it.
        if wire_nonce in accepted or wire_nonce in settling:
            self.context.logger.warning(
                "Wire nonce %d already in-flight for sender %s on request %s "
                "(expected next %d, accepted=%d, settling=%d).",
                wire_nonce,
                sender_checksum,
                wire_request_id,
                expected_next_nonce,
                len(accepted),
                len(settling),
            )
            return SignatureVerdict(
                ok=False,
                reason=NONCE_BELOW_EXPECTED,
                is_infra=False,
            )
        if wire_nonce < expected_next_nonce:
            self.context.logger.warning(
                "Wire nonce %d below expected next slot %d for sender %s on "
                "request %s (on_chain=%d, accepted=%d, settling=%d).",
                wire_nonce,
                expected_next_nonce,
                sender_checksum,
                wire_request_id,
                on_chain_nonce,
                len(accepted),
                len(settling),
            )
            return SignatureVerdict(
                ok=False,
                reason=NONCE_BELOW_EXPECTED,
                is_infra=False,
            )
        if wire_nonce > expected_next_nonce:
            self.context.logger.warning(
                "Wire nonce %d above expected next slot %d for sender %s on "
                "request %s (on_chain=%d, accepted=%d, settling=%d).",
                wire_nonce,
                expected_next_nonce,
                sender_checksum,
                wire_request_id,
                on_chain_nonce,
                len(accepted),
                len(settling),
            )
            # 503 (is_infra=True) — the sender is racing its own
            # in-flight queue. mech-client retries 503 in the next
            # round, so the earlier requests get a chance to drain and
            # the same body settles when the slot opens.
            return SignatureVerdict(
                ok=False,
                reason=NONCE_ABOVE_EXPECTED,
                is_infra=True,
            )
        # Per-sender in-flight cap. Guards the previously-unbounded
        # growth of ``accepted_nonces_by_sender`` (and by extension
        # ``pending_tasks`` and ``in_memory_requests``) on the post-
        # auth pre-payment surface. See ``MAX_ACCEPTED_PER_SENDER``.
        if len(accepted) + len(settling) >= MAX_ACCEPTED_PER_SENDER:
            self.context.logger.warning(
                "Sender %s hit in-flight cap on request %s "
                "(accepted=%d, settling=%d, cap=%d); rejecting 503.",
                sender_checksum,
                wire_request_id,
                len(accepted),
                len(settling),
                MAX_ACCEPTED_PER_SENDER,
            )
            return SignatureVerdict(
                ok=False,
                reason=SENDER_INFLIGHT_LIMIT,
                is_infra=True,
            )
        return SignatureVerdict(ok=True, reason="", is_infra=False)

    def _resolve_sender_checksum(self, sender: str) -> Optional[str]:
        """Return the sender's checksum address using the cached ledger api.

        Extracted from the sig-verify path so the caller in
        ``_handle_signed_requests`` can compute the checksum once for
        both the dedup key (unchanged) and the outstanding-nonces
        admission gate. Returns ``None`` on a failed checksum
        conversion; the caller routes that to a distinct
        ``SENDER_RESOLUTION_FAILED`` verdict so a config-side problem
        (missing / malformed RPC settings) does not get reported to
        the client as an RPC round-trip failure.

        :param sender: the raw sender string from the wire body.
        :return: the checksum-cased address, or ``None`` on failure.
        """
        ledger_settings = self._get_ledger_settings()
        if ledger_settings[ResponseKey.STATUS.value] != ResponseStatus.OK.value:
            self.context.logger.warning(
                "Cannot resolve sender checksum: ledger settings unavailable "
                "(%s); rejecting.",
                ledger_settings.get(ResponseKey.REASON.value, "unknown"),
            )
            return None
        try:
            rpc_address = cast(str, ledger_settings[ResponseKey.RPC_ADDRESS.value])
            chain_id = cast(int, ledger_settings[ResponseKey.CHAIN_ID.value])
            ledger_api = self._get_ledger_api(rpc_address, chain_id)
            return cast(str, ledger_api.api.to_checksum_address(sender))
        except Exception:  # pylint: disable=broad-exception-caught
            self.context.logger.warning(
                "Sender checksum resolution failed for %r.",
                sender,
                exc_info=True,
            )
            return None

    def _get_ledger_settings(self) -> Dict[str, Union[str, int]]:
        """Read ledger RPC settings from skill params using default_chain_id."""
        chain = str(self.params.default_chain_id).lower()
        rpc_address = cast(
            Optional[str], getattr(self.context.params, f"{chain}_ledger_rpc", None)
        )
        if not rpc_address:
            return {
                ResponseKey.STATUS.value: ResponseStatus.UNAVAILABLE.value,
                ResponseKey.REASON.value: f"Missing RPC config for chain: {chain}.",
            }

        try:
            chain_id = ChainId[chain.upper()].value
        except KeyError:
            return {
                ResponseKey.STATUS.value: ResponseStatus.UNAVAILABLE.value,
                ResponseKey.REASON.value: f"Unsupported chain: {chain}.",
            }

        return {
            ResponseKey.STATUS.value: ResponseStatus.OK.value,
            ResponseKey.RPC_ADDRESS.value: rpc_address,
            ResponseKey.CHAIN_ID.value: chain_id,
        }
