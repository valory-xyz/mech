# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2025 Valory AG
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

import packages.valory.skills.task_execution.dialogues as dmod


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


def test_default_dialogues_uses_agent_address_for_self_address(
    dialogue_skill_context: Any,
) -> None:
    """Default dialogues should use the agent_address from context as self address."""
    dlg = dmod.DefaultDialogues(
        name="default_dialogues", skill_context=dialogue_skill_context
    )
    assert _get_self_addr(dlg) == dialogue_skill_context.agent_address
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
