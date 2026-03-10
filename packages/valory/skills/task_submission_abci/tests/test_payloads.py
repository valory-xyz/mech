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

import dataclasses
from typing import Any

import pytest

from packages.valory.skills.abstract_round_abci.base import BaseTxPayload
from packages.valory.skills.task_submission_abci.payloads import (
    TaskPoolingPayload,
    TransactionPayload,
)


class TestTaskPoolingPayload:
    """Tests for TaskPoolingPayload."""

    def test_creation_with_content(self) -> None:
        """Test TaskPoolingPayload is created with content."""
        p = TaskPoolingPayload(sender="agent-0", content='[{"request_id": "1"}]')
        assert p.content == '[{"request_id": "1"}]'
        assert p.sender == "agent-0"

    def test_inherits_from_base_tx_payload(self) -> None:
        """Test TaskPoolingPayload inherits from BaseTxPayload."""
        assert issubclass(TaskPoolingPayload, BaseTxPayload)

    def test_is_frozen_dataclass(self) -> None:
        """Test TaskPoolingPayload is a frozen dataclass."""
        p = TaskPoolingPayload(sender="agent-0", content="[]")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            p.content = "new"  # type: ignore

    def test_empty_content(self) -> None:
        """Test TaskPoolingPayload accepts empty content."""
        p = TaskPoolingPayload(sender="agent-0", content="")
        assert p.content == ""

    @pytest.mark.parametrize("content", ["[]", "[1,2]", '{"key": "val"}'])
    def test_various_content_strings(self, content: Any) -> None:
        """Test TaskPoolingPayload accepts various content strings."""
        p = TaskPoolingPayload(sender="agent-0", content=content)
        assert p.content == content


class TestTransactionPayload:
    """Tests for TransactionPayload."""

    def test_creation_with_tx_hash(self) -> None:
        """Test TransactionPayload is created with a tx hash."""
        tx = "0xabc123"
        p = TransactionPayload(sender="agent-0", content=tx)
        assert p.content == tx

    def test_creation_with_error_sentinel(self) -> None:
        """Test TransactionPayload is created with an error sentinel."""
        p = TransactionPayload(sender="agent-0", content="error")
        assert p.content == "error"

    def test_inherits_from_base_tx_payload(self) -> None:
        """Test TransactionPayload inherits from BaseTxPayload."""
        assert issubclass(TransactionPayload, BaseTxPayload)

    def test_is_frozen_dataclass(self) -> None:
        """Test TransactionPayload is a frozen dataclass."""
        p = TransactionPayload(sender="agent-0", content="0xhash")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            p.content = "tampered"  # type: ignore
