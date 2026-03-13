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
"""Tests for task_submission_abci.dialogues."""

from typing import Any
from unittest.mock import MagicMock, patch

from packages.valory.protocols.acn_data_share.dialogues import (
    AcnDataShareDialogue as BaseAcnDataShareDialogue,
)
from packages.valory.protocols.acn_data_share.dialogues import (
    AcnDataShareDialogues as BaseAcnDataShareDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    AbciDialogue as BaseAbciDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    AbciDialogues as BaseAbciDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    ContractApiDialogue as BaseContractApiDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    ContractApiDialogues as BaseContractApiDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    HttpDialogue as BaseHttpDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    HttpDialogues as BaseHttpDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    IpfsDialogue as BaseIpfsDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    IpfsDialogues as BaseIpfsDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    LedgerApiDialogue as BaseLedgerApiDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    LedgerApiDialogues as BaseLedgerApiDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    SigningDialogue as BaseSigningDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    SigningDialogues as BaseSigningDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    TendermintDialogue as BaseTendermintDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    TendermintDialogues as BaseTendermintDialogues,
)
from packages.valory.skills.task_submission_abci import dialogues as task_dialogues


def test_dialogue_aliases_re_exported() -> None:
    """All dialogue type aliases point to their expected base classes."""
    assert task_dialogues.AbciDialogue is BaseAbciDialogue
    assert task_dialogues.AbciDialogues is BaseAbciDialogues
    assert task_dialogues.HttpDialogue is BaseHttpDialogue
    assert task_dialogues.HttpDialogues is BaseHttpDialogues
    assert task_dialogues.SigningDialogue is BaseSigningDialogue
    assert task_dialogues.SigningDialogues is BaseSigningDialogues
    assert task_dialogues.LedgerApiDialogue is BaseLedgerApiDialogue
    assert task_dialogues.LedgerApiDialogues is BaseLedgerApiDialogues
    assert task_dialogues.ContractApiDialogue is BaseContractApiDialogue
    assert task_dialogues.ContractApiDialogues is BaseContractApiDialogues
    assert task_dialogues.TendermintDialogue is BaseTendermintDialogue
    assert task_dialogues.TendermintDialogues is BaseTendermintDialogues
    assert task_dialogues.IpfsDialogue is BaseIpfsDialogue
    assert task_dialogues.IpfsDialogues is BaseIpfsDialogues
    assert task_dialogues.AcnDataShareDialogue is BaseAcnDataShareDialogue


def test_acn_data_share_dialogues_init() -> None:
    """AcnDataShareDialogues.__init__ initialises both Model and BaseAcnDataShareDialogues."""
    ctx = MagicMock()
    ctx.agent_address = "agent-address-0"

    captured_role_fn: dict = {}

    def capture_base_init(
        self_: Any,
        self_address: Any = None,
        role_from_first_message: Any = None,
    ) -> None:
        captured_role_fn["fn"] = role_from_first_message

    with (
        patch(
            "packages.valory.skills.task_submission_abci.dialogues.Model.__init__",
            return_value=None,
        ) as mock_model_init,
        patch.object(
            BaseAcnDataShareDialogues, "__init__", side_effect=capture_base_init
        ),
    ):
        obj = task_dialogues.AcnDataShareDialogues.__new__(
            task_dialogues.AcnDataShareDialogues
        )
        object.__setattr__(obj, "_context", ctx)
        task_dialogues.AcnDataShareDialogues.__init__(obj, skill_context=ctx)

    mock_model_init.assert_called_once()
    assert captured_role_fn["fn"] is not None
    assert callable(captured_role_fn["fn"])


def test_acn_role_from_first_message_returns_agent() -> None:
    """The role_from_first_message closure returns AcnDataShareDialogue.Role.AGENT."""
    ctx = MagicMock()
    ctx.agent_address = "agent-address-0"

    captured_role_fn: dict = {}

    def capture_base_init(
        self_: Any,
        self_address: Any = None,
        role_from_first_message: Any = None,
    ) -> None:
        captured_role_fn["fn"] = role_from_first_message

    with (
        patch(
            "packages.valory.skills.task_submission_abci.dialogues.Model.__init__",
            return_value=None,
        ),
        patch.object(
            BaseAcnDataShareDialogues, "__init__", side_effect=capture_base_init
        ),
    ):
        obj = task_dialogues.AcnDataShareDialogues.__new__(
            task_dialogues.AcnDataShareDialogues
        )
        object.__setattr__(obj, "_context", ctx)
        task_dialogues.AcnDataShareDialogues.__init__(obj, skill_context=ctx)

    role = captured_role_fn["fn"](MagicMock(), "some-address")
    assert role == BaseAcnDataShareDialogue.Role.AGENT
