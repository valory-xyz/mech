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

"""Test messages module for mech_acn protocol."""

# pylint: disable=too-many-statements,too-many-locals,no-member,too-few-public-methods,redefined-builtin
from typing import List

from aea.test_tools.test_protocol import BaseProtocolMessagesTestCase

from packages.valory.protocols.mech_acn.custom_types import Status
from packages.valory.protocols.mech_acn.message import MechAcnMessage


class TestMessageMechAcn(BaseProtocolMessagesTestCase):
    """Test for the 'mech_acn' protocol message."""

    MESSAGE_CLASS = MechAcnMessage

    def build_messages(self) -> List[MechAcnMessage]:  # type: ignore[override]
        """Build the messages to be used for testing."""
        return [
            MechAcnMessage(
                performative=MechAcnMessage.Performative.REQUEST,
                request_id="some str",
            ),
            MechAcnMessage(
                performative=MechAcnMessage.Performative.RESPONSE,
                data="some str",
                status=Status(),  # check it please!
            ),
        ]

    def build_inconsistent(self) -> List[MechAcnMessage]:  # type: ignore[override]
        """Build inconsistent messages to be used for testing."""
        return [
            MechAcnMessage(
                performative=MechAcnMessage.Performative.REQUEST,
                # skip content: request_id
            ),
            MechAcnMessage(
                performative=MechAcnMessage.Performative.RESPONSE,
                # skip content: data
                status=Status(),  # check it please!
            ),
        ]
