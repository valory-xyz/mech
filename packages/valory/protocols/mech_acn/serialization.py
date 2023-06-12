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

"""Serialization module for mech_acn protocol."""

# pylint: disable=too-many-statements,too-many-locals,no-member,too-few-public-methods,redefined-builtin
from typing import Any, Dict, cast

from aea.mail.base_pb2 import DialogueMessage
from aea.mail.base_pb2 import Message as ProtobufMessage
from aea.protocols.base import Message, Serializer

from packages.valory.protocols.mech_acn import mech_acn_pb2
from packages.valory.protocols.mech_acn.custom_types import Status
from packages.valory.protocols.mech_acn.message import MechAcnMessage


class MechAcnSerializer(Serializer):
    """Serialization for the 'mech_acn' protocol."""

    @staticmethod
    def encode(msg: Message) -> bytes:
        """
        Encode a 'MechAcn' message into bytes.

        :param msg: the message object.
        :return: the bytes.
        """
        msg = cast(MechAcnMessage, msg)
        message_pb = ProtobufMessage()
        dialogue_message_pb = DialogueMessage()
        mech_acn_msg = mech_acn_pb2.MechAcnMessage()

        dialogue_message_pb.message_id = msg.message_id
        dialogue_reference = msg.dialogue_reference
        dialogue_message_pb.dialogue_starter_reference = dialogue_reference[0]
        dialogue_message_pb.dialogue_responder_reference = dialogue_reference[1]
        dialogue_message_pb.target = msg.target

        performative_id = msg.performative
        if performative_id == MechAcnMessage.Performative.REQUEST:
            performative = mech_acn_pb2.MechAcnMessage.Request_Performative()  # type: ignore
            request_id = msg.request_id
            performative.request_id = request_id
            mech_acn_msg.request.CopyFrom(performative)
        elif performative_id == MechAcnMessage.Performative.RESPONSE:
            performative = mech_acn_pb2.MechAcnMessage.Response_Performative()  # type: ignore
            data = msg.data
            performative.data = data
            status = msg.status
            Status.encode(performative.status, status)
            mech_acn_msg.response.CopyFrom(performative)
        else:
            raise ValueError("Performative not valid: {}".format(performative_id))

        dialogue_message_pb.content = mech_acn_msg.SerializeToString()

        message_pb.dialogue_message.CopyFrom(dialogue_message_pb)
        message_bytes = message_pb.SerializeToString()
        return message_bytes

    @staticmethod
    def decode(obj: bytes) -> Message:
        """
        Decode bytes into a 'MechAcn' message.

        :param obj: the bytes object.
        :return: the 'MechAcn' message.
        """
        message_pb = ProtobufMessage()
        mech_acn_pb = mech_acn_pb2.MechAcnMessage()
        message_pb.ParseFromString(obj)
        message_id = message_pb.dialogue_message.message_id
        dialogue_reference = (
            message_pb.dialogue_message.dialogue_starter_reference,
            message_pb.dialogue_message.dialogue_responder_reference,
        )
        target = message_pb.dialogue_message.target

        mech_acn_pb.ParseFromString(message_pb.dialogue_message.content)
        performative = mech_acn_pb.WhichOneof("performative")
        performative_id = MechAcnMessage.Performative(str(performative))
        performative_content = dict()  # type: Dict[str, Any]
        if performative_id == MechAcnMessage.Performative.REQUEST:
            request_id = mech_acn_pb.request.request_id
            performative_content["request_id"] = request_id
        elif performative_id == MechAcnMessage.Performative.RESPONSE:
            data = mech_acn_pb.response.data
            performative_content["data"] = data
            pb2_status = mech_acn_pb.response.status
            status = Status.decode(pb2_status)
            performative_content["status"] = status
        else:
            raise ValueError("Performative not valid: {}.".format(performative_id))

        return MechAcnMessage(
            message_id=message_id,
            dialogue_reference=dialogue_reference,
            target=target,
            performative=performative,
            **performative_content
        )
