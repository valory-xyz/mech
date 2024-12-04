# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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
import dataclasses
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from aea.exceptions import enforce
from aea.skills.base import Model

from packages.valory.skills.abstract_round_abci.utils import check_type


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


@dataclasses.dataclass
class MechConfig:
    """Mech config dataclass."""

    use_dynamic_pricing: bool
    is_marketplace_mech: bool

    @staticmethod
    def from_dict(raw_dict: Dict[str, bool]) -> "MechConfig":
        """From dict."""
        print(f"{raw_dict=}")
        return MechConfig(
            use_dynamic_pricing=raw_dict.get("use_dynamic_pricing", False),
            is_marketplace_mech=raw_dict.get("is_marketplace_mech", False),
        )


@dataclasses.dataclass
class RequestParams:
    """Mech Req Params dataclass."""

    from_block: Dict[str, Optional[int]] = dataclasses.field(
        default_factory=lambda: {"legacy": None, "marketplace": None}
    )
    last_polling: Dict[str, Optional[float]] = dataclasses.field(
        default_factory=lambda: {"legacy": None, "marketplace": None}
    )


class Params(Model):
    """A model to represent params for multiple abci apps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""
        self.in_flight_req: bool = False
        self.req_params: RequestParams = RequestParams()
        self.req_type: Optional[str] = None
        self.req_to_callback: Dict[str, Callable] = {}
        self.api_keys: Dict[str, List[str]] = self._ensure_get(
            "api_keys", kwargs, Dict[str, List[str]]
        )
        self.tools_to_package_hash: Dict[str, str] = self._ensure_get(
            "tools_to_package_hash", kwargs, Dict[str, str]
        )
        self.polling_interval = kwargs.get("polling_interval", 30.0)
        self.task_deadline = kwargs.get("task_deadline", 240.0)
        self.num_agents = self._ensure_get("num_agents", kwargs, int)
        self.request_count: int = 0
        self.cleanup_freq = kwargs.get("cleanup_freq", 50)
        self.agent_index: int = self._ensure_get("agent_index", kwargs, int)
        self.from_block_range: int = self._ensure_get("from_block_range", kwargs, int)
        self.timeout_limit: int = self._ensure_get("timeout_limit", kwargs, int)
        self.max_block_window: int = self._ensure_get("max_block_window", kwargs, int)
        # maps the request id to the number of times it has timed out
        self.request_id_to_num_timeouts: Dict[int, int] = defaultdict(lambda: 0)
        mech_to_config_dict: Dict[str, Dict[str, bool]] = self._ensure_get(
            "mech_to_config", kwargs, Dict[str, Dict[str, bool]]
        )
        self.mech_to_config: Dict[str, MechConfig] = {
            key: MechConfig.from_dict(value)
            for key, value in mech_to_config_dict.items()
        }
        self.agent_mech_contract_addresses = list(self.mech_to_config.keys())
        self.mech_marketplace_address: str = self._ensure_get(
            "mech_marketplace_address", kwargs, str
        )
        self.use_mech_marketplace = (
            self.mech_marketplace_address is not None
            and self.mech_marketplace_address != ZERO_ADDRESS
        )
        self.offchain_tx_list = list()
        super().__init__(*args, **kwargs)

    @classmethod
    def _ensure_get(cls, key: str, kwargs: Dict, type_: Any) -> Any:
        """Ensure that the parameters are set, and return them without popping the key."""
        enforce("skill_context" in kwargs, "Only use on models!")
        skill_id = kwargs["skill_context"].skill_id
        enforce(
            key in kwargs,
            f"{key!r} of type {type_!r} required, but it is not set in `models.params.args` of `skill.yaml` of `{skill_id}`",
        )
        value = kwargs.get(key, None)
        try:
            check_type(key, value, type_)
        except TypeError:  # pragma: nocover
            enforce(
                False,
                f"{key!r} must be a {type_}, but type {type(value)} was found in `models.params.args` of `skill.yaml` of `{skill_id}`",
            )
        return value
