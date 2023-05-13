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

"""This module contains the shared state for the abci skill of TaskExecutionAbciApp."""

from typing import Any

from packages.valory.skills.abstract_round_abci.models import BaseParams
from packages.valory.skills.abstract_round_abci.models import \
    BenchmarkTool as BaseBenchmarkTool
from packages.valory.skills.abstract_round_abci.models import \
    Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import \
    SharedState as BaseSharedState
from packages.valory.skills.task_execution_abci.rounds import \
    TaskExecutionAbciApp


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = TaskExecutionAbciApp


class Params(BaseParams):
    """Parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""

        self.openai_api_key = self._ensure("openai_api_key", kwargs, str)

        super().__init__(*args, **kwargs)

Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool
