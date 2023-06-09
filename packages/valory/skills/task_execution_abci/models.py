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

from typing import Any, Dict, List, Type, Union

from packages.valory.skills.abstract_round_abci.base import AbciApp
from packages.valory.skills.abstract_round_abci.models import BaseParams
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.task_execution_abci.rounds import TaskExecutionAbciApp


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls: Type[AbciApp] = TaskExecutionAbciApp

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the shared state object."""
        self.all_tools: Dict[str, str] = {}
        super().__init__(*args, **kwargs)


class Params(BaseParams):
    """Parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""

        self.api_keys: Dict[str, str] = self._nested_list_todict_workaround(
            kwargs, "api_keys_json", List[List[str]]
        )
        self.file_hash_to_tools: Dict[
            str, List[str]
        ] = self._nested_list_todict_workaround(
            kwargs, "file_hash_to_tools_json", List[List[Union[str, List[str]]]]
        )
        self.tools_to_file_hash = {
            value: key
            for key, values in self.file_hash_to_tools.items()
            for value in values
        }
        self.all_tools: Dict[str, str] = {}
        self.multisend_address = kwargs.get("multisend_address", None)
        if self.multisend_address is None:
            raise ValueError("No multisend_address specified!")
        self.agent_mech_contract_address = kwargs.get(
            "agent_mech_contract_address", None
        )
        if self.agent_mech_contract_address is None:
            raise ValueError("agent_mech_contract_address is required")
        self.ipfs_fetch_timeout = self._ensure(
            "ipfs_fetch_timeout", kwargs=kwargs, type_=float
        )
        super().__init__(*args, **kwargs)

    def _nested_list_todict_workaround(
        self, kwargs: Dict, key: str, type_: Any
    ) -> Dict:
        """
        Get a nested list from the kwargs and convert it to a dictionary.

        This is a workaround because we currently cannot input a json string on Propel.

        :param kwargs: the keyword arguments parsed from the yaml configuration.
        :param key: the key for which we want to try to retrieve its value from the kwargs.
        :param type_: the expected type of the retrieved value.
        :return: the nested list converted to a dictionary.
        """
        values = self._ensure(key, kwargs, type_)
        if len(values) == 0:
            raise ValueError(f"No {key} specified!")
        return {value[0]: value[1] for value in values}


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool
