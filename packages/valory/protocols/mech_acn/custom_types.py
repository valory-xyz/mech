# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 valory
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

"""This module contains class representations corresponding to every custom type in the protocol specification."""

from enum import Enum


class StatusEnum(Enum):
    """StatusEnum for tx check."""

    REQUEST_NOT_FOUND = 0
    DATA_NOT_READY = 1
    READY = 2
    REQUEST_EXPIRED = 3


class Status:
    """This class represents an instance of Status."""

    def __init__(self, status: StatusEnum):
        """Initialise an instance of Status."""
        self.status = status

    @staticmethod
    def encode(status_protobuf_object, status_object: "Status") -> None:
        """
        Encode an instance of this class into the protocol buffer object.

        The protocol buffer object in the status_protobuf_object argument is matched with the instance of this class in the 'status_object' argument.

        :param status_protobuf_object: the protocol buffer object whose type corresponds with this class.
        :param status_object: an instance of this class to be encoded in the protocol buffer object.
        """
        status_protobuf_object.status = status_object.status.value

    @classmethod
    def decode(cls, status_protobuf_object) -> "Status":
        """
        Decode a protocol buffer object that corresponds with this class into an instance of this class.

        A new instance of this class is created that matches the protocol buffer object in the 'status_protobuf_object' argument.

        :param status_protobuf_object: the protocol buffer object whose type corresponds with this class.
        :return: A new instance of this class that matches the protocol buffer object in the 'status_protobuf_object' argument.
        """
        return Status(StatusEnum(status_protobuf_object.status))

    def __eq__(self, other):
        """Compare with another object."""
        return isinstance(other, Status) and self.status == other.status
