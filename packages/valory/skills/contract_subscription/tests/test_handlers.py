# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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
"""Tests for the WebSocket contract-subscription handler.

The WebSocket ingest is the primary path for on-chain request events
(``use_polling: false`` across every agent / service / skill config in
this repo). ``_get_tx_args`` is the choke point that decides whether a
freshly-observed on-chain request lands at predict-api with a
``request_tx_hash`` — the whole feature the PR builds on. Without a
test here, a future refactor of ``_get_tx_args`` that drops the
``"tx_hash": tx_hash`` key would silently kill the feature on the
primary ingest with no test failure.
"""

from types import SimpleNamespace
from typing import Any, Dict

from packages.valory.skills.contract_subscription.handlers import WebSocketHandler


def _logger_stub() -> SimpleNamespace:
    """Minimal logger — every level accepts and drops the call."""
    return SimpleNamespace(
        info=lambda *a, **k: None,
        error=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )


def _make_handler_bypassing_init(
    tx_receipt: Dict[str, Any],
    rich_logs: Any,
) -> WebSocketHandler:
    """Build a ``WebSocketHandler`` with just the fields ``_get_tx_args`` reads.

    ``WebSocketHandler.__init__`` walks the aea framework's kwargs
    contract (``websocket_provider``, ``contract_to_monitor``,
    plus base-class kwargs), which requires a real Skill context.
    ``_get_tx_args`` reads only ``self.w3``, ``self.contract``, and
    ``self.context.logger``, so bypass ``__init__`` via
    ``__new__`` and inject just those three. Same approach the
    sibling test helpers in this repo use for handler unit tests.
    """
    handler = WebSocketHandler.__new__(WebSocketHandler)

    handler.w3 = SimpleNamespace(
        eth=SimpleNamespace(
            get_transaction_receipt=lambda _tx: tx_receipt,
        ),
    )
    handler.contract = SimpleNamespace(
        events=SimpleNamespace(
            Request=lambda: SimpleNamespace(
                processReceipt=lambda _receipt: rich_logs,
            ),
        ),
    )
    # ``context`` is a read-only property backed by ``_context``.
    # Set the backing attribute directly so ``handler.context`` resolves.
    handler._context = SimpleNamespace(logger=_logger_stub())
    return handler


class TestGetTxArgsThreadsTxHash:
    """The ``request_tx_hash`` on-chain feature is anchored here.

    Every assertion below fails if a refactor drops the ``tx_hash``
    key from the returned dict. That's the single behaviour the rest
    of the request-tx-hash rollout depends on for the WebSocket path.
    """

    def test_returns_event_args_plus_tx_hash_and_false_flag(self) -> None:
        tx_hash = "0x" + "ab" * 32
        tx_receipt = {"blockNumber": 12345}
        request_args = {"requestId": 42, "requester": "0x" + "aa" * 20}
        rich_logs = [{"args": request_args}]

        handler = _make_handler_bypassing_init(tx_receipt, rich_logs)

        args, no_request = handler._get_tx_args(tx_hash)

        # tx_hash lands under the ``tx_hash`` key on the returned
        # dict — the key predict-api / mech-analytics read via
        # ``executing_task.get("tx_hash")`` in
        # ``_build_predict_api_event``.
        assert args["tx_hash"] == tx_hash
        # No collision with contract-emitted args (neither Request
        # nor MarketplaceRequest ABI defines a tx_hash arg).
        assert args["requestId"] == 42
        assert args["requester"] == "0x" + "aa" * 20
        # no_request is False on a legit Request emission — the
        # caller uses this to break out of the polling retry loop.
        assert no_request is False
        # The last-block cursor advances even on the WebSocket path
        # so a follow-on reconnect knows where to resume from.
        assert handler._last_processed_block == 12345

    def test_non_request_event_returns_empty_args_and_true_flag(self) -> None:
        """A tx that doesn't emit a Request event lands as ``([], True)``.

        The tx receipt lookup succeeds but ``processReceipt`` returns
        an empty list — indexing at ``[0]`` raises, and the broad
        except in ``_get_tx_args`` catches it, returning ``({}, True)``
        so the caller skips the tx as "not a Request" rather than
        looping on empty args.
        """
        tx_hash = "0x" + "cd" * 32
        tx_receipt = {"blockNumber": 67890}

        handler = _make_handler_bypassing_init(tx_receipt, rich_logs=[])

        args, no_request = handler._get_tx_args(tx_hash)

        assert args == {}
        assert no_request is True

    def test_get_receipt_failure_returns_empty_args_and_true_flag(self) -> None:
        """A receipt lookup failure (RPC blip) reports no_request=True.

        The broad except keeps the ingest loop alive; the caller
        breaks out on the True flag rather than sleeping+retrying
        against a receipt that will keep failing.
        """
        tx_hash = "0x" + "ef" * 32

        handler = WebSocketHandler.__new__(WebSocketHandler)

        def _raise_rpc(_tx: str) -> None:
            raise RuntimeError("rpc reset")

        handler.w3 = SimpleNamespace(
            eth=SimpleNamespace(get_transaction_receipt=_raise_rpc),
        )
        # ``contract`` is unused on the exception path but must exist so
        # attribute lookup doesn't precede the try block.
        handler.contract = SimpleNamespace(events=SimpleNamespace(Request=lambda: None))
        # ``context`` is a read-only property backed by ``_context``.
        # Set the backing attribute directly so ``handler.context`` resolves.
        handler._context = SimpleNamespace(logger=_logger_stub())

        args, no_request = handler._get_tx_args(tx_hash)

        assert args == {}
        assert no_request is True
