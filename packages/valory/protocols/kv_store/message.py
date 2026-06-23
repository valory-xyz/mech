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

"""This module contains kv_store's message definition."""

# pylint: disable=too-many-statements,too-many-locals,no-member,too-few-public-methods,too-many-branches,not-an-iterable,unidiomatic-typecheck,unsubscriptable-object
import logging
from typing import Any, Dict, Set, Tuple, cast

from aea.configurations.base import PublicId
from aea.exceptions import AEAEnforceError, enforce
from aea.protocols.base import Message  # type: ignore

_default_logger = logging.getLogger("aea.packages.valory.protocols.kv_store.message")

DEFAULT_BODY_SIZE = 4


class KvStoreMessage(Message):
    """A protocol for simple key-value storage."""

    protocol_id = PublicId.from_str("valory/kv_store:0.1.0")
    protocol_specification_id = PublicId.from_str("valory/kv_store:0.1.0")

    class Performative(Message.Performative):
        """Performatives for the kv_store protocol."""

        CREATE_OR_UPDATE_REQUEST = "create_or_update_request"
        DELETE_REQUEST = "delete_request"
        ERROR = "error"
        LIST_REQUEST = "list_request"
        LIST_RESPONSE = "list_response"
        READ_REQUEST = "read_request"
        READ_RESPONSE = "read_response"
        SUCCESS = "success"

        def __str__(self) -> str:
            """Get the string representation."""
            return str(self.value)

    _performatives = {
        "create_or_update_request",
        "delete_request",
        "error",
        "list_request",
        "list_response",
        "read_request",
        "read_response",
        "success",
    }
    __slots__: Tuple[str, ...] = tuple()

    class _SlotsCls:
        __slots__ = (
            "cursor",
            "data",
            "dialogue_reference",
            "key_prefix",
            "keys",
            "limit",
            "message",
            "message_id",
            "next_cursor",
            "performative",
            "target",
        )

    def __init__(
        self,
        performative: Performative,
        dialogue_reference: Tuple[str, str] = ("", ""),
        message_id: int = 1,
        target: int = 0,
        **kwargs: Any,
    ):
        """
        Initialise an instance of KvStoreMessage.

        :param message_id: the message id.
        :param dialogue_reference: the dialogue reference.
        :param target: the message target.
        :param performative: the message performative.
        :param **kwargs: extra options.
        """
        super().__init__(
            dialogue_reference=dialogue_reference,
            message_id=message_id,
            target=target,
            performative=KvStoreMessage.Performative(performative),
            **kwargs,
        )

    @property
    def valid_performatives(self) -> Set[str]:
        """Get valid performatives."""
        return self._performatives

    @property
    def dialogue_reference(self) -> Tuple[str, str]:
        """Get the dialogue_reference of the message."""
        enforce(self.is_set("dialogue_reference"), "dialogue_reference is not set.")
        return cast(Tuple[str, str], self.get("dialogue_reference"))

    @property
    def message_id(self) -> int:
        """Get the message_id of the message."""
        enforce(self.is_set("message_id"), "message_id is not set.")
        return cast(int, self.get("message_id"))

    @property
    def performative(self) -> Performative:  # type: ignore # noqa: F821
        """Get the performative of the message."""
        enforce(self.is_set("performative"), "performative is not set.")
        return cast(KvStoreMessage.Performative, self.get("performative"))

    @property
    def target(self) -> int:
        """Get the target of the message."""
        enforce(self.is_set("target"), "target is not set.")
        return cast(int, self.get("target"))

    @property
    def cursor(self) -> str:
        """Get the 'cursor' content from the message."""
        enforce(self.is_set("cursor"), "'cursor' content is not set.")
        return cast(str, self.get("cursor"))

    @property
    def data(self) -> Dict[str, str]:
        """Get the 'data' content from the message."""
        enforce(self.is_set("data"), "'data' content is not set.")
        return cast(Dict[str, str], self.get("data"))

    @property
    def key_prefix(self) -> str:
        """Get the 'key_prefix' content from the message."""
        enforce(self.is_set("key_prefix"), "'key_prefix' content is not set.")
        return cast(str, self.get("key_prefix"))

    @property
    def keys(self) -> Tuple[str, ...]:
        """Get the 'keys' content from the message."""
        enforce(self.is_set("keys"), "'keys' content is not set.")
        return cast(Tuple[str, ...], self.get("keys"))

    @property
    def limit(self) -> int:
        """Get the 'limit' content from the message."""
        enforce(self.is_set("limit"), "'limit' content is not set.")
        return cast(int, self.get("limit"))

    @property
    def message(self) -> str:
        """Get the 'message' content from the message."""
        enforce(self.is_set("message"), "'message' content is not set.")
        return cast(str, self.get("message"))

    @property
    def next_cursor(self) -> str:
        """Get the 'next_cursor' content from the message."""
        enforce(self.is_set("next_cursor"), "'next_cursor' content is not set.")
        return cast(str, self.get("next_cursor"))

    def _is_consistent(self) -> bool:
        """Check that the message follows the kv_store protocol."""
        try:
            enforce(
                isinstance(self.dialogue_reference, tuple),
                "Invalid type for 'dialogue_reference'. Expected 'tuple'. Found '{}'.".format(
                    type(self.dialogue_reference)
                ),
            )
            enforce(
                isinstance(self.dialogue_reference[0], str),
                "Invalid type for 'dialogue_reference[0]'. Expected 'str'. Found '{}'.".format(
                    type(self.dialogue_reference[0])
                ),
            )
            enforce(
                isinstance(self.dialogue_reference[1], str),
                "Invalid type for 'dialogue_reference[1]'. Expected 'str'. Found '{}'.".format(
                    type(self.dialogue_reference[1])
                ),
            )
            enforce(
                type(self.message_id) is int,
                "Invalid type for 'message_id'. Expected 'int'. Found '{}'.".format(
                    type(self.message_id)
                ),
            )
            enforce(
                type(self.target) is int,
                "Invalid type for 'target'. Expected 'int'. Found '{}'.".format(
                    type(self.target)
                ),
            )

            # Light Protocol Rule 2
            # Check correct performative
            enforce(
                isinstance(self.performative, KvStoreMessage.Performative),
                "Invalid 'performative'. Expected either of '{}'. Found '{}'.".format(
                    self.valid_performatives, self.performative
                ),
            )

            # Check correct contents
            actual_nb_of_contents = len(self._body) - DEFAULT_BODY_SIZE
            expected_nb_of_contents = 0
            if self.performative == KvStoreMessage.Performative.READ_REQUEST:
                expected_nb_of_contents = 1
                enforce(
                    isinstance(self.keys, tuple),
                    "Invalid type for content 'keys'. Expected 'tuple'. Found '{}'.".format(
                        type(self.keys)
                    ),
                )
                enforce(
                    all(isinstance(element, str) for element in self.keys),
                    "Invalid type for tuple elements in content 'keys'. Expected 'str'.",
                )
            elif self.performative == KvStoreMessage.Performative.READ_RESPONSE:
                expected_nb_of_contents = 1
                enforce(
                    isinstance(self.data, dict),
                    "Invalid type for content 'data'. Expected 'dict'. Found '{}'.".format(
                        type(self.data)
                    ),
                )
                for key_of_data, value_of_data in self.data.items():
                    enforce(
                        isinstance(key_of_data, str),
                        "Invalid type for dictionary keys in content 'data'. Expected 'str'. Found '{}'.".format(
                            type(key_of_data)
                        ),
                    )
                    enforce(
                        isinstance(value_of_data, str),
                        "Invalid type for dictionary values in content 'data'. Expected 'str'. Found '{}'.".format(
                            type(value_of_data)
                        ),
                    )
            elif (
                self.performative
                == KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST
            ):
                expected_nb_of_contents = 1
                enforce(
                    isinstance(self.data, dict),
                    "Invalid type for content 'data'. Expected 'dict'. Found '{}'.".format(
                        type(self.data)
                    ),
                )
                for key_of_data, value_of_data in self.data.items():
                    enforce(
                        isinstance(key_of_data, str),
                        "Invalid type for dictionary keys in content 'data'. Expected 'str'. Found '{}'.".format(
                            type(key_of_data)
                        ),
                    )
                    enforce(
                        isinstance(value_of_data, str),
                        "Invalid type for dictionary values in content 'data'. Expected 'str'. Found '{}'.".format(
                            type(value_of_data)
                        ),
                    )
            elif self.performative == KvStoreMessage.Performative.DELETE_REQUEST:
                expected_nb_of_contents = 1
                enforce(
                    isinstance(self.keys, tuple),
                    "Invalid type for content 'keys'. Expected 'tuple'. Found '{}'.".format(
                        type(self.keys)
                    ),
                )
                enforce(
                    all(isinstance(element, str) for element in self.keys),
                    "Invalid type for tuple elements in content 'keys'. Expected 'str'.",
                )
            elif self.performative == KvStoreMessage.Performative.LIST_REQUEST:
                expected_nb_of_contents = 3
                enforce(
                    isinstance(self.key_prefix, str),
                    "Invalid type for content 'key_prefix'. Expected 'str'. Found '{}'.".format(
                        type(self.key_prefix)
                    ),
                )
                enforce(
                    type(self.limit) is int,
                    "Invalid type for content 'limit'. Expected 'int'. Found '{}'.".format(
                        type(self.limit)
                    ),
                )
                enforce(
                    self.limit >= 0,
                    "Invalid value for content 'limit'. Must be non-negative. Got: {}".format(
                        self.limit
                    ),
                )
                enforce(
                    isinstance(self.cursor, str),
                    "Invalid type for content 'cursor'. Expected 'str'. Found '{}'.".format(
                        type(self.cursor)
                    ),
                )
            elif self.performative == KvStoreMessage.Performative.LIST_RESPONSE:
                expected_nb_of_contents = 2
                enforce(
                    isinstance(self.data, dict),
                    "Invalid type for content 'data'. Expected 'dict'. Found '{}'.".format(
                        type(self.data)
                    ),
                )
                enforce(
                    isinstance(self.next_cursor, str),
                    "Invalid type for content 'next_cursor'. Expected 'str'. Found '{}'.".format(
                        type(self.next_cursor)
                    ),
                )
                for key_of_data, value_of_data in self.data.items():
                    enforce(
                        isinstance(key_of_data, str),
                        "Invalid type for dictionary keys in content 'data'. Expected 'str'. Found '{}'.".format(
                            type(key_of_data)
                        ),
                    )
                    enforce(
                        isinstance(value_of_data, str),
                        "Invalid type for dictionary values in content 'data'. Expected 'str'. Found '{}'.".format(
                            type(value_of_data)
                        ),
                    )
            elif self.performative == KvStoreMessage.Performative.SUCCESS:
                expected_nb_of_contents = 1
                enforce(
                    isinstance(self.message, str),
                    "Invalid type for content 'message'. Expected 'str'. Found '{}'.".format(
                        type(self.message)
                    ),
                )
            elif self.performative == KvStoreMessage.Performative.ERROR:
                expected_nb_of_contents = 1
                enforce(
                    isinstance(self.message, str),
                    "Invalid type for content 'message'. Expected 'str'. Found '{}'.".format(
                        type(self.message)
                    ),
                )

            # Check correct content count
            enforce(
                expected_nb_of_contents == actual_nb_of_contents,
                "Incorrect number of contents. Expected {}. Found {}".format(
                    expected_nb_of_contents, actual_nb_of_contents
                ),
            )

            # Light Protocol Rule 3
            if self.message_id == 1:
                enforce(
                    self.target == 0,
                    "Invalid 'target'. Expected 0 (because 'message_id' is 1). Found {}.".format(
                        self.target
                    ),
                )
        except (AEAEnforceError, ValueError, KeyError) as e:
            _default_logger.error(str(e))
            return False

        return True
