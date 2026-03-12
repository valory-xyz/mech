# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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
"""Tests for task_submission_abci.payloads."""

from packages.valory.skills.task_submission_abci.payloads import (
    TaskPoolingPayload,
    TransactionPayload,
)


def test_task_pooling_payload() -> None:
    """TaskPoolingPayload stores sender and content."""
    p = TaskPoolingPayload(sender="agent-0", content='[{"request_id": "1"}]')
    assert p.content == '[{"request_id": "1"}]'
    assert p.sender == "agent-0"


def test_transaction_payload() -> None:
    """TransactionPayload stores sender and content."""
    p = TransactionPayload(sender="agent-0", content="0xabc123")
    assert p.content == "0xabc123"
    assert p.sender == "agent-0"
