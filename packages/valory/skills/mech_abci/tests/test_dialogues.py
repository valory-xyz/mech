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
"""Tests for mech_abci.dialogues."""

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
from packages.valory.skills.mech_abci import dialogues as mech_dialogues
from packages.valory.skills.task_submission_abci.dialogues import (
    AcnDataShareDialogue as BaseAcnDataShareDialogue,
    AcnDataShareDialogues as BaseAcnDataShareDialogues,
)


class TestDialogueAliases:
    """Verify all dialogue type aliases in mech_abci.dialogues are correct re-exports."""

    def test_abci_dialogue(self):
        assert mech_dialogues.AbciDialogue is BaseAbciDialogue

    def test_abci_dialogues(self):
        assert mech_dialogues.AbciDialogues is BaseAbciDialogues

    def test_http_dialogue(self):
        assert mech_dialogues.HttpDialogue is BaseHttpDialogue

    def test_http_dialogues(self):
        assert mech_dialogues.HttpDialogues is BaseHttpDialogues

    def test_signing_dialogue(self):
        assert mech_dialogues.SigningDialogue is BaseSigningDialogue

    def test_signing_dialogues(self):
        assert mech_dialogues.SigningDialogues is BaseSigningDialogues

    def test_ledger_api_dialogue(self):
        assert mech_dialogues.LedgerApiDialogue is BaseLedgerApiDialogue

    def test_ledger_api_dialogues(self):
        assert mech_dialogues.LedgerApiDialogues is BaseLedgerApiDialogues

    def test_contract_api_dialogue(self):
        assert mech_dialogues.ContractApiDialogue is BaseContractApiDialogue

    def test_contract_api_dialogues(self):
        assert mech_dialogues.ContractApiDialogues is BaseContractApiDialogues

    def test_tendermint_dialogue(self):
        assert mech_dialogues.TendermintDialogue is BaseTendermintDialogue

    def test_tendermint_dialogues(self):
        assert mech_dialogues.TendermintDialogues is BaseTendermintDialogues

    def test_ipfs_dialogue(self):
        assert mech_dialogues.IpfsDialogue is BaseIpfsDialogue

    def test_ipfs_dialogues(self):
        assert mech_dialogues.IpfsDialogues is BaseIpfsDialogues

    def test_acn_data_share_dialogue(self):
        assert mech_dialogues.AcnDataShareDialogue is BaseAcnDataShareDialogue

    def test_acn_data_share_dialogues(self):
        assert mech_dialogues.AcnDataShareDialogues is BaseAcnDataShareDialogues
