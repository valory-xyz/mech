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

import pytest

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
    """Return the dialogue's configured self address."""
    for attr in ("self_address", "_self_address"):
        if hasattr(dialogues_obj, attr):
            return getattr(dialogues_obj, attr)
    raise AttributeError("Could not find self address on dialogues object")


@pytest.mark.parametrize(
    "cls_name,expected_addr_attr",
    [
        ("IpfsDialogues", "skill_id"),
        ("ContractDialogues", "skill_id"),
        ("LedgerDialogues", "skill_id"),
        ("AcnDataShareDialogues", "agent_address"),
    ],
)
def test_dialogues_self_address(
    dialogue_skill_context: Any, cls_name: str, expected_addr_attr: str
) -> None:
    """Each dialogue class uses the correct self address from context."""
    cls = getattr(dmod, cls_name)
    dlg = cls(name=cls_name.lower(), skill_context=dialogue_skill_context)
    expected = str(getattr(dialogue_skill_context, expected_addr_attr))
    assert _get_self_addr(dlg) == expected
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


@pytest.mark.parametrize(
    "cls_name,base_cls,expected_role",
    [
        ("IpfsDialogues", BaseIpfsDialogues, BaseIpfsDialogue.Role.SKILL),
        ("ContractDialogues", BaseContractApiDialogues, BaseContractApiDialogue.Role.AGENT),
        ("LedgerDialogues", BaseLedgerApiDialogues, BaseLedgerApiDialogue.Role.AGENT),
        ("AcnDataShareDialogues", BaseAcnDataShareDialogues, BaseAcnDataShareDialogue.Role.AGENT),
    ],
)
def test_role_from_first_message(
    dialogue_skill_context: Any, cls_name: str, base_cls: Any, expected_role: Any
) -> None:
    """Each dialogue closure returns the correct role."""
    cls = getattr(dmod, cls_name)
    fn = _capture_role_fn(cls, base_cls, dialogue_skill_context)
    role = fn(MagicMock(), "some-address")
    assert role == expected_role
