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


class TestDialogueAliases:
    """Verify simple type alias re-exports in task_submission_abci.dialogues."""

    def test_abci_dialogue(self) -> None:
        """Test AbciDialogue is re-exported correctly."""
        assert task_dialogues.AbciDialogue is BaseAbciDialogue

    def test_abci_dialogues(self) -> None:
        """Test AbciDialogues is re-exported correctly."""
        assert task_dialogues.AbciDialogues is BaseAbciDialogues

    def test_http_dialogue(self) -> None:
        """Test HttpDialogue is re-exported correctly."""
        assert task_dialogues.HttpDialogue is BaseHttpDialogue

    def test_http_dialogues(self) -> None:
        """Test HttpDialogues is re-exported correctly."""
        assert task_dialogues.HttpDialogues is BaseHttpDialogues

    def test_signing_dialogue(self) -> None:
        """Test SigningDialogue is re-exported correctly."""
        assert task_dialogues.SigningDialogue is BaseSigningDialogue

    def test_signing_dialogues(self) -> None:
        """Test SigningDialogues is re-exported correctly."""
        assert task_dialogues.SigningDialogues is BaseSigningDialogues

    def test_ledger_api_dialogue(self) -> None:
        """Test LedgerApiDialogue is re-exported correctly."""
        assert task_dialogues.LedgerApiDialogue is BaseLedgerApiDialogue

    def test_ledger_api_dialogues(self) -> None:
        """Test LedgerApiDialogues is re-exported correctly."""
        assert task_dialogues.LedgerApiDialogues is BaseLedgerApiDialogues

    def test_contract_api_dialogue(self) -> None:
        """Test ContractApiDialogue is re-exported correctly."""
        assert task_dialogues.ContractApiDialogue is BaseContractApiDialogue

    def test_contract_api_dialogues(self) -> None:
        """Test ContractApiDialogues is re-exported correctly."""
        assert task_dialogues.ContractApiDialogues is BaseContractApiDialogues

    def test_tendermint_dialogue(self) -> None:
        """Test TendermintDialogue is re-exported correctly."""
        assert task_dialogues.TendermintDialogue is BaseTendermintDialogue

    def test_tendermint_dialogues(self) -> None:
        """Test TendermintDialogues is re-exported correctly."""
        assert task_dialogues.TendermintDialogues is BaseTendermintDialogues

    def test_ipfs_dialogue(self) -> None:
        """Test IpfsDialogue is re-exported correctly."""
        assert task_dialogues.IpfsDialogue is BaseIpfsDialogue

    def test_ipfs_dialogues(self) -> None:
        """Test IpfsDialogues is re-exported correctly."""
        assert task_dialogues.IpfsDialogues is BaseIpfsDialogues

    def test_acn_data_share_dialogue(self) -> None:
        """Test AcnDataShareDialogue is re-exported correctly."""
        assert task_dialogues.AcnDataShareDialogue is BaseAcnDataShareDialogue


class TestAcnDataShareDialogues:
    """Tests for the AcnDataShareDialogues class."""

    def test_inherits_from_base_acn_dialogues(self) -> None:
        """Test AcnDataShareDialogues inherits from BaseAcnDataShareDialogues."""
        assert issubclass(
            task_dialogues.AcnDataShareDialogues, BaseAcnDataShareDialogues
        )

    def test_init_sets_up_dialogues(self) -> None:
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

    def test_role_from_first_message_returns_agent_role(self) -> None:
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

        # Call the captured closure to cover line 129
        role = captured_role_fn["fn"](MagicMock(), "some-address")
        assert role == BaseAcnDataShareDialogue.Role.AGENT
