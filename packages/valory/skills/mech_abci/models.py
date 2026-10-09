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

"""This module contains the shared state for the abci skill of Mech."""

import re
from typing import Any, Optional

from packages.valory.skills.abstract_round_abci.models import (
    ApiSpecs,
)
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.delivery_rate_abci.models import (
    Params as SubscriptionParams,
)
from packages.valory.skills.mech_abci.composition import MechAbciApp
from packages.valory.skills.reset_pause_abci.rounds import Event as ResetPauseEvent
from packages.valory.skills.task_submission_abci.models import (
    Params as TaskExecutionAbciParams,
)
from packages.valory.skills.task_submission_abci.models import (
    SharedState as TaskExecSharedState,
)
from packages.valory.skills.task_submission_abci.rounds import (
    Event as TaskExecutionEvent,
)
from packages.valory.skills.termination_abci.models import TerminationParams
from packages.valory.skills.transaction_settlement_abci.rounds import (
    Event as TransactionSettlementEvent,
)

TaskExecutionParams = TaskExecutionAbciParams

# ERC-8004 IdentityRegistry, deployed at this address on every chain the
# mechs run on. Overridable for a chain where it lives elsewhere.
DEFAULT_ERC8004_IDENTITY_REGISTRY = "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"
_ADDRESS_REGEX = re.compile(r"^0x[0-9a-fA-F]{40}$")


def parse_erc8004_agent_id(value: Any) -> Optional[int]:
    """
    Validate the configured ERC-8004 agent id.

    The id is the one the identity registry assigned to this service, which
    is not in general the Olas service id. ``None`` means the mech publishes
    no domain proof.

    :param value: the raw ``erc8004_agent_id`` param
    :return: the agent id, or None when unset
    :raises ValueError: when the value is set but is not a non-negative integer
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"erc8004_agent_id must be a non-negative integer or null, got {value!r}"
        )
    return value


def parse_identity_registry_address(value: Any) -> str:
    """
    Validate the configured ERC-8004 identity registry address.

    :param value: the raw ``erc8004_identity_registry_address`` param
    :return: the address, or the default registry when unset
    :raises ValueError: when the value is set but is not a 20-byte hex address
    """
    if value is None or value == "":
        return DEFAULT_ERC8004_IDENTITY_REGISTRY
    if not isinstance(value, str) or not _ADDRESS_REGEX.match(value):
        raise ValueError(
            f"erc8004_identity_registry_address must be a 0x-prefixed 20-byte hex address, got {value!r}"
        )
    return value


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool


class RandomnessApi(ApiSpecs):
    """A model that wraps ApiSpecs for randomness api specifications."""


MARGIN = 5


class SharedState(TaskExecSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = MechAbciApp

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the shared state."""
        self.last_processed_request_block_number: int = 0
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        """Set up."""
        super().setup()

        MechAbciApp.event_to_timeout[TaskExecutionEvent.ROUND_TIMEOUT] = (
            self.context.params.round_timeout_seconds
        )

        MechAbciApp.event_to_timeout[
            TaskExecutionEvent.TASK_EXECUTION_ROUND_TIMEOUT
        ] = self.context.params.round_timeout_seconds

        MechAbciApp.event_to_timeout[ResetPauseEvent.ROUND_TIMEOUT] = (
            self.context.params.round_timeout_seconds
        )

        MechAbciApp.event_to_timeout[TransactionSettlementEvent.ROUND_TIMEOUT] = (
            self.context.params.round_timeout_seconds
        )

        MechAbciApp.event_to_timeout[TransactionSettlementEvent.VALIDATE_TIMEOUT] = (
            self.context.params.validate_timeout
        )

        MechAbciApp.event_to_timeout[TransactionSettlementEvent.FINALIZE_TIMEOUT] = (
            self.context.params.finalize_timeout
        )

        MechAbciApp.event_to_timeout[ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT] = (
            self.context.params.reset_pause_duration + MARGIN
        )


class Params(TaskExecutionParams, SubscriptionParams, TerminationParams):  # type: ignore
    """A model to represent params for multiple abci apps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the params, reading the ERC-8004 domain proof settings."""
        self.erc8004_agent_id: Optional[int] = parse_erc8004_agent_id(
            kwargs.get("erc8004_agent_id")
        )
        self.erc8004_identity_registry_address: str = parse_identity_registry_address(
            kwargs.get("erc8004_identity_registry_address")
        )
        super().__init__(*args, **kwargs)
