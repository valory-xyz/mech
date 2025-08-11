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

import packages.valory.skills.task_execution.dialogues as dmod


def _get_self_addr(dialogues_obj):
    # tolerate different attribute names across AEA versions
    for attr in ("self_address", "_self_address"):
        if hasattr(dialogues_obj, attr):
            return getattr(dialogues_obj, attr)
    raise AttributeError("Could not find self address on dialogues object")


def test_ipfs_dialogues_uses_skill_id_for_self_address(dialogue_skill_context):
    dlg = dmod.IpfsDialogues(
        name="ipfs_dialogues", skill_context=dialogue_skill_context
    )
    assert _get_self_addr(dlg) == str(dialogue_skill_context.skill_id)
    # basic lifecycle
    dlg.cleanup()  # should not raise


def test_contract_dialogues_uses_skill_id_for_self_address(dialogue_skill_context):
    dlg = dmod.ContractDialogues(
        name="contract_dialogues", skill_context=dialogue_skill_context
    )
    assert _get_self_addr(dlg) == str(dialogue_skill_context.skill_id)
    dlg.cleanup()


def test_ledger_dialogues_uses_skill_id_for_self_address(dialogue_skill_context):
    dlg = dmod.LedgerDialogues(
        name="ledger_dialogues", skill_context=dialogue_skill_context
    )
    assert _get_self_addr(dlg) == str(dialogue_skill_context.skill_id)
    dlg.cleanup()


def test_default_dialogues_uses_agent_address_for_self_address(dialogue_skill_context):
    dlg = dmod.DefaultDialogues(
        name="default_dialogues", skill_context=dialogue_skill_context
    )
    # DefaultDialogues picks self_address from context.agent_address
    assert _get_self_addr(dlg) == dialogue_skill_context.agent_address
    dlg.cleanup()


def test_acn_data_share_dialogues_uses_agent_address_for_self_address(
    dialogue_skill_context,
):
    dlg = dmod.AcnDataShareDialogues(
        name="acn_dialogues", skill_context=dialogue_skill_context
    )
    # AcnDataShareDialogues uses str(self.context.agent_address)
    assert _get_self_addr(dlg) == str(dialogue_skill_context.agent_address)
    dlg.cleanup()
