# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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

"""This module contains the shared state for the abci skill of UpdateDeliveryRateAbciApp."""
from dataclasses import dataclass
from typing import Any, Optional, Type, Dict, List

from aea.exceptions import enforce

from packages.valory.skills.abstract_round_abci.base import AbciApp
from packages.valory.skills.abstract_round_abci.models import BaseParams
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.abstract_round_abci.models import TypeCheckMixin
from packages.valory.skills.abstract_round_abci.utils import check_type
from packages.valory.skills.delivery_rate_abci.rounds import DeliveryRateUpdateAbciApp


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls: Type[AbciApp] = DeliveryRateUpdateAbciApp


@dataclass
class MutableParams(TypeCheckMixin):
    """Collection for the mutable parameters."""

    latest_metadata_hash: Optional[bytes] = None


class Params(BaseParams):
    """Parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""

        self.mech_to_max_delivery_rate: Dict[str, int] = self._ensure_get(
            "mech_to_max_delivery_rate", kwargs, Dict[str, int]
        )
        self.manual_gas_limit = self._ensure_get("manual_gas_limit", kwargs, int)
        self.multisend_address = self._ensure_get("multisend_address", kwargs, str)
        super().__init__(*args, **kwargs)

    @classmethod
    def _ensure_get(cls, key: str, kwargs: Dict, type_: Any) -> Any:
        """Ensure that the parameters are set, and return them without popping the key."""
        enforce("skill_context" in kwargs, "Only use on models!")
        skill_id = kwargs["skill_context"].skill_id
        enforce(
            key in kwargs,
            f"'{key}' of type '{type_}' required, but it is not set in `models.params.args` of `skill.yaml` of `{skill_id}`",
        )
        value = kwargs.get(key, None)
        try:
            check_type(key, value, type_)
        except TypeError:  # pragma: nocover
            enforce(
                False,
                f"'{key}' must be a {type_}, but type {type(value)} was found in `models.params.args` of `skill.yaml` of `{skill_id}`",
            )
        return value


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool
