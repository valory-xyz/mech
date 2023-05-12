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

"""This package contains round behaviours of MultiplexerAbciApp."""

import json
from abc import ABC
from datetime import datetime, timezone
from typing import Generator, Set, Tuple, Type, cast

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.multiplexer_abci.models import Params, SharedState
from packages.valory.skills.multiplexer_abci.rounds import (
    MultiplexerAbciApp,
    MultiplexerPayload,
    MultiplexerRound,
    SynchronizedData,
)


class MultiplexerBaseBehaviour(BaseBehaviour, ABC):
    """Base behaviour for the multiplexer_abci skill."""

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return cast(SynchronizedData, super().synchronized_data)

    @property
    def params(self) -> Params:
        """Return the params."""
        return cast(Params, super().params)


class MultiplexerBehaviour(MultiplexerBaseBehaviour):
    """MultiplexerBehaviour"""

    matching_round: Type[AbstractRound] = MultiplexerRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():

            payload_content = MultiplexerRound.WAIT_PAYLOAD

            period_counter = self.synchronized_data.period_counter
            do_reset = period_counter % self.params.reset_period_count == 0

            if self.context.state.task_queue:
                payload_content = MultiplexerRound.EXECUTE_PAYLOAD
            elif do_reset:
                payload_content = MultiplexerRound.RESET_PAYLOAD

            sender = self.context.agent_address
            payload = MultiplexerPayload(sender=sender, content=payload_content)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()


class MultiplexerRoundBehaviour(AbstractRoundBehaviour):
    """MultiplexerRoundBehaviour"""

    initial_behaviour_cls = MultiplexerBehaviour
    abci_app_cls = MultiplexerAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = [MultiplexerBehaviour]
