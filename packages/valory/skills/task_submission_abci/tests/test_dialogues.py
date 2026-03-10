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

from unittest.mock import MagicMock, patch

from packages.valory.protocols.acn_data_share.dialogues import (
    AcnDataShareDialogue as BaseAcnDataShareDialogue,
    AcnDataShareDialogues as BaseAcnDataShareDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    AbciDialogue as BaseAbciDialogue,
    AbciDialogues as BaseAbciDialogues,
    ContractApiDialogue as BaseContractApiDialogue,
    ContractApiDialogues as BaseContractApiDialogues,
    HttpDialogue as BaseHttpDialogue,
    HttpDialogues as BaseHttpDialogues,
    IpfsDialogue as BaseIpfsDialogue,
    IpfsDialogues as BaseIpfsDialogues,
    LedgerApiDialogue as BaseLedgerApiDialogue,
    LedgerApiDialogues as BaseLedgerApiDialogues,
    SigningDialogue as BaseSigningDialogue,
    SigningDialogues as BaseSigningDialogues,
    TendermintDialogue as BaseTendermintDialogue,
    TendermintDialogues as BaseTendermintDialogues,
)
from packages.valory.skills.task_submission_abci import dialogues as task_dialogues


class TestDialogueAliases:
    """Verify simple type alias re-exports in task_submission_abci.dialogues."""

    def test_abci_dialogue(self):
        assert task_dialogues.AbciDialogue is BaseAbciDialogue

    def test_abci_dialogues(self):
        assert task_dialogues.AbciDialogues is BaseAbciDialogues

    def test_http_dialogue(self):
        assert task_dialogues.HttpDialogue is BaseHttpDialogue

    def test_http_dialogues(self):
        assert task_dialogues.HttpDialogues is BaseHttpDialogues

    def test_signing_dialogue(self):
        assert task_dialogues.SigningDialogue is BaseSigningDialogue

    def test_signing_dialogues(self):
        assert task_dialogues.SigningDialogues is BaseSigningDialogues

    def test_ledger_api_dialogue(self):
        assert task_dialogues.LedgerApiDialogue is BaseLedgerApiDialogue

    def test_ledger_api_dialogues(self):
        assert task_dialogues.LedgerApiDialogues is BaseLedgerApiDialogues

    def test_contract_api_dialogue(self):
        assert task_dialogues.ContractApiDialogue is BaseContractApiDialogue

    def test_contract_api_dialogues(self):
        assert task_dialogues.ContractApiDialogues is BaseContractApiDialogues

    def test_tendermint_dialogue(self):
        assert task_dialogues.TendermintDialogue is BaseTendermintDialogue

    def test_tendermint_dialogues(self):
        assert task_dialogues.TendermintDialogues is BaseTendermintDialogues

    def test_ipfs_dialogue(self):
        assert task_dialogues.IpfsDialogue is BaseIpfsDialogue

    def test_ipfs_dialogues(self):
        assert task_dialogues.IpfsDialogues is BaseIpfsDialogues

    def test_acn_data_share_dialogue(self):
        assert task_dialogues.AcnDataShareDialogue is BaseAcnDataShareDialogue


class TestAcnDataShareDialogues:
    """Tests for the AcnDataShareDialogues class."""

    def test_inherits_from_base_acn_dialogues(self):
        assert issubclass(task_dialogues.AcnDataShareDialogues, BaseAcnDataShareDialogues)

    def test_init_sets_up_dialogues(self):
        """AcnDataShareDialogues.__init__ initialises both Model and BaseAcnDataShareDialogues."""
        ctx = MagicMock()
        ctx.agent_address = "agent-address-0"

        captured_role_fn = {}

        def capture_base_init(self_, self_address=None, role_from_first_message=None):
            captured_role_fn["fn"] = role_from_first_message

        with (
            patch("packages.valory.skills.task_submission_abci.dialogues.Model.__init__", return_value=None) as mock_model_init,
            patch.object(BaseAcnDataShareDialogues, "__init__", side_effect=capture_base_init),
        ):
            obj = task_dialogues.AcnDataShareDialogues.__new__(task_dialogues.AcnDataShareDialogues)
            object.__setattr__(obj, "_context", ctx)
            task_dialogues.AcnDataShareDialogues.__init__(obj, skill_context=ctx)

        mock_model_init.assert_called_once()
        assert captured_role_fn["fn"] is not None
        assert callable(captured_role_fn["fn"])

    def test_role_from_first_message_returns_agent_role(self):
        """The role_from_first_message closure returns AcnDataShareDialogue.Role.AGENT."""
        ctx = MagicMock()
        ctx.agent_address = "agent-address-0"

        captured_role_fn = {}

        def capture_base_init(self_, self_address=None, role_from_first_message=None):
            captured_role_fn["fn"] = role_from_first_message

        with (
            patch("packages.valory.skills.task_submission_abci.dialogues.Model.__init__", return_value=None),
            patch.object(BaseAcnDataShareDialogues, "__init__", side_effect=capture_base_init),
        ):
            obj = task_dialogues.AcnDataShareDialogues.__new__(task_dialogues.AcnDataShareDialogues)
            object.__setattr__(obj, "_context", ctx)
            task_dialogues.AcnDataShareDialogues.__init__(obj, skill_context=ctx)

        # Call the captured closure to cover line 129
        role = captured_role_fn["fn"](MagicMock(), "some-address")
        assert role == BaseAcnDataShareDialogue.Role.AGENT
