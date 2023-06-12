# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 Valory AG
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

"""This module contains the handlers for the skill of TaskExecutionAbciApp."""

from typing import Any, cast

from aea.skills.base import Handler

from packages.valory.protocols.mech_acn.custom_types import (
    Status as AcnRequestStatusObj,
)
from packages.valory.protocols.mech_acn.custom_types import (
    StatusEnum as AcnRequestStatus,
)
from packages.valory.protocols.mech_acn.dialogues import (
    MechAcnDialogue,
    MechAcnDialogues,
)
from packages.valory.protocols.mech_acn.message import MechAcnMessage
from packages.valory.skills.abstract_round_abci.handlers import (
    ABCIRoundHandler as BaseABCIRoundHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    ContractApiHandler as BaseContractApiHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    HttpHandler as BaseHttpHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    IpfsHandler as BaseIpfsHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    LedgerApiHandler as BaseLedgerApiHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    SigningHandler as BaseSigningHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    TendermintHandler as BaseTendermintHandler,
)
from packages.valory.skills.task_execution_abci.models import AcnDataRequests


ABCIHandler = BaseABCIRoundHandler
HttpHandler = BaseHttpHandler
SigningHandler = BaseSigningHandler
LedgerApiHandler = BaseLedgerApiHandler
ContractApiHandler = BaseContractApiHandler
TendermintHandler = BaseTendermintHandler
IpfsHandler = BaseIpfsHandler

DATA_REQUESTS = "data_requests"


class MechAcnRequestsHandler(Handler):
    """ACN callback handler."""

    SUPPORTED_PROTOCOL = MechAcnMessage.protocol_id

    def setup(self) -> None:
        """Setup handler."""

    @property
    def data_requests(self) -> AcnDataRequests:
        """Return the params."""
        return cast(AcnDataRequests, self.context.acn_data_requests)

    def handle(self, message: MechAcnMessage) -> None:
        """Handle message."""
        if message.performative != MechAcnMessage.Performative.REQUEST:
            return None
        request_id = message.request_id
        dialogues = cast(MechAcnDialogues, self.context.mech_acn_requests_dialogues)
        dialogue = cast(MechAcnDialogue, dialogues.update(message))
        if not self.data_requests.request_exists(request_id=request_id):
            self.context.logger.info(
                f"ACN data request with ID {request_id} does not exist!"
            )
            return self.send_request_not_found(
                message=message,
                dialogue=dialogue,
            )

        if self.data_requests.request_ready(request_id=request_id):
            self.context.logger.info(
                f"Sending data for ACN request with ID {request_id}"
            )
            return self.send_data(
                message=message,
                dialogue=dialogue,
                data=self.data_requests.get_data(request_id=request_id),
            )

        self.context.logger.info(
            f"Adding callback for request ID {request_id} from address {message.to}"
        )
        self.data_requests.add_callback(
            request_id=request_id,
            callback_dialogue=dialogue,
        )

    def send_data(
        self, message: MechAcnMessage, dialogue: MechAcnDialogue, data: Any
    ) -> None:
        """Send error message."""
        response = dialogue.reply(
            performative=MechAcnMessage.Performative.RESPONSE,
            target_message=message,
            status=AcnRequestStatusObj(AcnRequestStatus.READY),
            data=data,
        )
        self.context.outbox.put_message(response)

    def send_request_not_found(
        self, message: MechAcnMessage, dialogue: MechAcnDialogue
    ) -> None:
        """Send error message."""
        response = dialogue.reply(
            performative=MechAcnMessage.Performative.RESPONSE,
            target_message=message,
            status=AcnRequestStatusObj(AcnRequestStatus.REQUEST_NOT_FOUND),
            data="Agent has not started processing the request yet",
        )
        self.context.outbox.put_message(response)

    def send_data_not_ready(
        self, message: MechAcnMessage, dialogue: MechAcnDialogue
    ) -> None:
        """Send error message."""
        response = dialogue.reply(
            performative=MechAcnMessage.Performative.RESPONSE,
            target_message=message,
            status=AcnRequestStatusObj(AcnRequestStatus.DATA_NOT_READY),
            data="Data not ready",
        )
        self.context.outbox.put_message(response)

    def teardown(self) -> None:
        """Teardown handler."""
