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
from packages.valory.skills.mech_abci import dialogues as mech_dialogues
from packages.valory.skills.task_submission_abci.dialogues import (
    AcnDataShareDialogue as BaseAcnDataShareDialogue,
)
from packages.valory.skills.task_submission_abci.dialogues import (
    AcnDataShareDialogues as BaseAcnDataShareDialogues,
)


def test_all_dialogue_aliases_re_exported() -> None:
    """All dialogue type aliases point to their expected base classes."""
    assert mech_dialogues.AbciDialogue is BaseAbciDialogue
    assert mech_dialogues.AbciDialogues is BaseAbciDialogues
    assert mech_dialogues.HttpDialogue is BaseHttpDialogue
    assert mech_dialogues.HttpDialogues is BaseHttpDialogues
    assert mech_dialogues.SigningDialogue is BaseSigningDialogue
    assert mech_dialogues.SigningDialogues is BaseSigningDialogues
    assert mech_dialogues.LedgerApiDialogue is BaseLedgerApiDialogue
    assert mech_dialogues.LedgerApiDialogues is BaseLedgerApiDialogues
    assert mech_dialogues.ContractApiDialogue is BaseContractApiDialogue
    assert mech_dialogues.ContractApiDialogues is BaseContractApiDialogues
    assert mech_dialogues.TendermintDialogue is BaseTendermintDialogue
    assert mech_dialogues.TendermintDialogues is BaseTendermintDialogues
    assert mech_dialogues.IpfsDialogue is BaseIpfsDialogue
    assert mech_dialogues.IpfsDialogues is BaseIpfsDialogues
    assert mech_dialogues.AcnDataShareDialogue is BaseAcnDataShareDialogue
    assert mech_dialogues.AcnDataShareDialogues is BaseAcnDataShareDialogues
