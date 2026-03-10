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

"""Tests for the task_execution skill's dialogues wiring and self-addresses."""

from typing import Any
from unittest.mock import MagicMock, patch

import packages.valory.skills.task_execution.dialogues as dmod
from packages.valory.protocols.acn_data_share.dialogues import (
    AcnDataShareDialogue as BaseAcnDataShareDialogue,
)
from packages.valory.protocols.acn_data_share.dialogues import (
    AcnDataShareDialogues as BaseAcnDataShareDialogues,
)
from packages.valory.protocols.contract_api.dialogues import (
    ContractApiDialogue as BaseContractApiDialogue,
)
from packages.valory.protocols.contract_api.dialogues import (
    ContractApiDialogues as BaseContractApiDialogues,
)
from packages.valory.protocols.ipfs.dialogues import IpfsDialogue as BaseIpfsDialogue
from packages.valory.protocols.ipfs.dialogues import IpfsDialogues as BaseIpfsDialogues
from packages.valory.protocols.ledger_api.dialogues import (
    LedgerApiDialogue as BaseLedgerApiDialogue,
)
from packages.valory.protocols.ledger_api.dialogues import (
    LedgerApiDialogues as BaseLedgerApiDialogues,
)


def _get_self_addr(dialogues_obj: Any) -> str:
    """
    Return the dialogue's configured self address.

    Handles differences across AEA versions by checking both `self_address`
    and `_self_address`.

    :param dialogues_obj: A dialogues instance under test.
    :return: The configured self address as a string.
    :raises AttributeError: If neither attribute exists.
    """
    for attr in ("self_address", "_self_address"):
        if hasattr(dialogues_obj, attr):
            return getattr(dialogues_obj, attr)
    raise AttributeError("Could not find self address on dialogues object")


def test_ipfs_dialogues_uses_skill_id_for_self_address(
    dialogue_skill_context: Any,
) -> None:
    """IPFS dialogues should use the skill_id as self address."""
    dlg = dmod.IpfsDialogues(
        name="ipfs_dialogues", skill_context=dialogue_skill_context
    )
    assert _get_self_addr(dlg) == str(dialogue_skill_context.skill_id)
    dlg.cleanup()  # should not raise


def test_contract_dialogues_uses_skill_id_for_self_address(
    dialogue_skill_context: Any,
) -> None:
    """Contract dialogues should use the skill_id as self address."""
    dlg = dmod.ContractDialogues(
        name="contract_dialogues", skill_context=dialogue_skill_context
    )
    assert _get_self_addr(dlg) == str(dialogue_skill_context.skill_id)
    dlg.cleanup()


def test_ledger_dialogues_uses_skill_id_for_self_address(
    dialogue_skill_context: Any,
) -> None:
    """Ledger dialogues should use the skill_id as self address."""
    dlg = dmod.LedgerDialogues(
        name="ledger_dialogues", skill_context=dialogue_skill_context
    )
    assert _get_self_addr(dlg) == str(dialogue_skill_context.skill_id)
    dlg.cleanup()


def test_acn_data_share_dialogues_uses_agent_address_for_self_address(
    dialogue_skill_context: Any,
) -> None:
    """ACN Data Share dialogues should use the agent_address from context as self address."""
    dlg = dmod.AcnDataShareDialogues(
        name="acn_dialogues", skill_context=dialogue_skill_context
    )
    assert _get_self_addr(dlg) == str(dialogue_skill_context.agent_address)
    dlg.cleanup()


# ---------------------------------------------------------------------------
# Closure coverage — role_from_first_message return lines
# ---------------------------------------------------------------------------


def _capture_role_fn(cls: Any, base_cls: Any, ctx: Any) -> Any:
    """Create instance, capturing the role_from_first_message closure via patched base init."""
    captured: dict = {}

    def capture_base_init(
        self_: Any,
        self_address: Any = None,
        role_from_first_message: Any = None,
        **kw: Any
    ) -> None:
        captured["fn"] = role_from_first_message

    with (
        patch(
            "packages.valory.skills.task_execution.dialogues.Model.__init__",
            return_value=None,
        ),
        patch.object(base_cls, "__init__", side_effect=capture_base_init),
    ):
        obj = cls.__new__(cls)
        object.__setattr__(obj, "_context", ctx)
        cls.__init__(obj, skill_context=ctx)

    return captured["fn"]


def test_ipfs_dialogues_role_from_first_message_returns_skill(
    dialogue_skill_context: Any,
) -> None:
    """The IpfsDialogues closure should return IpfsDialogue.Role.SKILL."""
    fn = _capture_role_fn(dmod.IpfsDialogues, BaseIpfsDialogues, dialogue_skill_context)
    role = fn(MagicMock(), "some-address")
    assert role == BaseIpfsDialogue.Role.SKILL


def test_contract_dialogues_role_from_first_message_returns_agent(
    dialogue_skill_context: Any,
) -> None:
    """The ContractDialogues closure should return ContractApiDialogue.Role.AGENT."""
    fn = _capture_role_fn(
        dmod.ContractDialogues, BaseContractApiDialogues, dialogue_skill_context
    )
    role = fn(MagicMock(), "some-address")
    assert role == BaseContractApiDialogue.Role.AGENT


def test_ledger_dialogues_role_from_first_message_returns_agent(
    dialogue_skill_context: Any,
) -> None:
    """The LedgerDialogues closure should return LedgerDialogue.Role.AGENT."""
    fn = _capture_role_fn(
        dmod.LedgerDialogues, BaseLedgerApiDialogues, dialogue_skill_context
    )
    role = fn(MagicMock(), "some-address")
    assert role == BaseLedgerApiDialogue.Role.AGENT


def test_acn_data_share_dialogues_role_from_first_message_returns_agent(
    dialogue_skill_context: Any,
) -> None:
    """The AcnDataShareDialogues closure should return AcnDataShareDialogue.Role.AGENT."""
    fn = _capture_role_fn(
        dmod.AcnDataShareDialogues, BaseAcnDataShareDialogues, dialogue_skill_context
    )
    role = fn(MagicMock(), "some-address")
    assert role == BaseAcnDataShareDialogue.Role.AGENT
