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

"""
This module contains the classes required for kv_store dialogue management.

- KvStoreDialogue: The dialogue class maintains state of a dialogue and manages it.
- KvStoreDialogues: The dialogues class keeps track of all dialogues.
"""

from abc import ABC
from typing import Callable, Dict, FrozenSet, Type, cast

from aea.common import Address
from aea.protocols.base import Message
from aea.protocols.dialogue.base import Dialogue, DialogueLabel, Dialogues

from packages.valory.protocols.kv_store.message import KvStoreMessage


class KvStoreDialogue(Dialogue):
    """The kv_store dialogue class maintains state of a dialogue and manages it."""

    INITIAL_PERFORMATIVES: FrozenSet[Message.Performative] = frozenset(
        {
            KvStoreMessage.Performative.READ_REQUEST,
            KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST,
            KvStoreMessage.Performative.DELETE_REQUEST,
            KvStoreMessage.Performative.LIST_REQUEST,
        }
    )
    TERMINAL_PERFORMATIVES: FrozenSet[Message.Performative] = frozenset(
        {
            KvStoreMessage.Performative.READ_RESPONSE,
            KvStoreMessage.Performative.LIST_RESPONSE,
            KvStoreMessage.Performative.SUCCESS,
            KvStoreMessage.Performative.ERROR,
        }
    )
    VALID_REPLIES: Dict[Message.Performative, FrozenSet[Message.Performative]] = {
        KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST: frozenset(
            {KvStoreMessage.Performative.SUCCESS, KvStoreMessage.Performative.ERROR}
        ),
        KvStoreMessage.Performative.DELETE_REQUEST: frozenset(
            {KvStoreMessage.Performative.SUCCESS, KvStoreMessage.Performative.ERROR}
        ),
        KvStoreMessage.Performative.ERROR: frozenset(),
        KvStoreMessage.Performative.LIST_REQUEST: frozenset(
            {
                KvStoreMessage.Performative.LIST_RESPONSE,
                KvStoreMessage.Performative.ERROR,
            }
        ),
        KvStoreMessage.Performative.LIST_RESPONSE: frozenset(),
        KvStoreMessage.Performative.READ_REQUEST: frozenset(
            {
                KvStoreMessage.Performative.READ_RESPONSE,
                KvStoreMessage.Performative.ERROR,
            }
        ),
        KvStoreMessage.Performative.READ_RESPONSE: frozenset(),
        KvStoreMessage.Performative.SUCCESS: frozenset(),
    }

    class Role(Dialogue.Role):
        """This class defines the agent's role in a kv_store dialogue."""

        CONNECTION = "connection"
        SKILL = "skill"

    class EndState(Dialogue.EndState):
        """This class defines the end states of a kv_store dialogue."""

        SUCCESSFUL = 0

    def __init__(
        self,
        dialogue_label: DialogueLabel,
        self_address: Address,
        role: Dialogue.Role,
        message_class: Type[KvStoreMessage] = KvStoreMessage,
    ) -> None:
        """
        Initialize a dialogue.

        :param dialogue_label: the identifier of the dialogue
        :param self_address: the address of the entity for whom this dialogue is maintained
        :param role: the role of the agent this dialogue is maintained for
        :param message_class: the message class used
        """
        Dialogue.__init__(
            self,
            dialogue_label=dialogue_label,
            message_class=message_class,
            self_address=self_address,
            role=role,
        )


class KvStoreDialogues(Dialogues, ABC):
    """This class keeps track of all kv_store dialogues."""

    END_STATES = frozenset({KvStoreDialogue.EndState.SUCCESSFUL})

    _keep_terminal_state_dialogues = False

    def __init__(
        self,
        self_address: Address,
        role_from_first_message: Callable[[Message, Address], Dialogue.Role],
        dialogue_class: Type[KvStoreDialogue] = KvStoreDialogue,
    ) -> None:
        """
        Initialize dialogues.

        :param self_address: the address of the entity for whom dialogues are maintained
        :param dialogue_class: the dialogue class used
        :param role_from_first_message: the callable determining role from first message
        """
        Dialogues.__init__(
            self,
            self_address=self_address,
            end_states=cast(FrozenSet[Dialogue.EndState], self.END_STATES),
            message_class=KvStoreMessage,
            dialogue_class=dialogue_class,
            role_from_first_message=role_from_first_message,
        )
