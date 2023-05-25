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

"""This module contains the shared state for the abci skill of MultiplexerAbciApp."""

from typing import Any, List

from aea.skills.base import SkillContext

from packages.valory.skills.abstract_round_abci.models import BaseParams
from packages.valory.skills.abstract_round_abci.models import \
    BenchmarkTool as BaseBenchmarkTool
from packages.valory.skills.abstract_round_abci.models import \
    Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import \
    SharedState as BaseSharedState
from packages.valory.skills.multiplexer_abci.rounds import MultiplexerAbciApp


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = MultiplexerAbciApp

    def __init__(self, *args: Any, skill_context: SkillContext, **kwargs: Any) -> None:
        """Init"""
        super().__init__(*args, skill_context=skill_context, **kwargs)

        self.pending_tasks: List = []
        self.last_processed_request_block_number: int = 0


class Params(BaseParams):
    """Parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""

        self.reset_period_count = self._ensure("reset_period_count", kwargs, int)
        self.use_polling = self._ensure("use_polling", kwargs, bool)
        self.polling_interval = self._ensure("polling_interval", kwargs, int)
        self.agent_mech_contract_address = kwargs.get(
            "agent_mech_contract_address", None
        )
        if self.agent_mech_contract_address is None:
            raise ValueError("agent_mech_contract_address is required")
        super().__init__(*args, **kwargs)


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool
