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
        self.is_cold_start: bool = True
        self.req_params: RequestParams = RequestParams()
        self.req_type: Optional[str] = None
        self.req_to_callback: Dict[str, Callable] = {}
        self.req_to_error_callback: Dict[str, Callable] = {}
        self.req_to_deadline: Dict[str, float] = {}
        self.mech_to_max_delivery_rate: Dict[str, int] = self._ensure_get(
            "mech_to_max_delivery_rate", kwargs, Dict[str, int]
        )
        self.api_keys: Dict[str, List[str]] = self._ensure_get(
            "api_keys", kwargs, Dict[str, List[str]]
        )
        self.tools_to_package_hash: Dict[str, str] = self._ensure_get(
            "tools_to_package_hash", kwargs, Dict[str, str]
        )
        self.polling_interval = kwargs.get("polling_interval", 30.0)
        self.task_deadline = kwargs.get("task_deadline", 240.0)
        self.step_in_list_size = kwargs.get("step_in_list_size", 20)
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
            key.lower(): MechConfig.from_dict(value)
            for key, value in mech_to_config_dict.items()
        }
        self.agent_mech_contract_addresses = list(self.mech_to_config.keys())
        try:
            self.agent_mech_contract_address = self.agent_mech_contract_addresses[0]
        except IndexError:
            raise ValueError("No mech contract addresses found!")

        self.mech_marketplace_address: str = self._ensure_get(
            "mech_marketplace_address", kwargs, str
        )
        self.use_mech_marketplace = (
            self.mech_marketplace_address is not None
            and self.mech_marketplace_address != ZERO_ADDRESS
        )
        self.offchain_tx_list: List = list()
        self.default_chain_id: str = self._ensure_get("default_chain_id", kwargs, str)
        # EIP-155 integer chain id for the wildcard data-lake write and the
        # EIP-712 domain. Separate from ``default_chain_id`` (the AEA alias
        # ``"gnosis"``) because the wildcard schema and the EIP-712 domain
        # bind to the integer, and converting at the boundary via a
        # hardcoded alias→int table couples task_execution to a chain
        # registry it has no other reason to know about. Defaults to 0,
        # which fails the wildcard server's per-chain marketplace
        # allowlist on first POST — operator-friendly: the row is rejected
        # at a known boundary rather than silently mis-tagged.
        self.mech_events_chain_id: int = int(kwargs.get("mech_events_chain_id", 0) or 0)
        # Feature flag for the wildcard analytics write path, mirrored from
        # task_submission_abci so the build cost at finalize is also gated:
        # when False (Phase 1 default) ``_finalize_done_task`` skips the
        # ``_build_wildcard_event`` call entirely so nothing rides Tendermint
        # consensus replication. Must agree with the task_submission_abci
        # value across the deployment.
        self.mech_events_enabled: bool = bool(kwargs.get("mech_events_enabled", False))
        self.gnosis_ledger_rpc: str = kwargs.get("gnosis_ledger_rpc", "")
        self.polygon_ledger_rpc: str = kwargs.get("polygon_ledger_rpc", "")
        self.base_ledger_rpc: str = kwargs.get("base_ledger_rpc", "")
        # Lower-case the payment_type keys at load so a checksummed-vs-lowercase
        # hex mismatch at lookup time can't silently fall back to the zero
        # address (which would signal a native-asset deposit for what is really
        # an ERC-20 payment model).
        self.payment_type_to_asset_address: Dict[str, str] = {
            key.lower(): value
            for key, value in kwargs.get("payment_type_to_asset_address", {}).items()
        }
        # Phase 1 ships dark: the offchain HTTP path is disabled by default and
        # enabled per deployment in the Phase 2 rollout. False = today's
        # on-chain + IPFS behaviour, unchanged.
        self.use_offchain: bool = kwargs.get("use_offchain", False)
        # Off-chain preimage retention. Ships dark like use_offchain: False keeps
        # today's behaviour (no durable preimage buffer). When enabled, each
        # off-chain (request, response) pair is mirrored into the kv_store and a
        # background sweeper prunes entries older than the retention window. Only
        # meaningful alongside use_offchain (on-chain deliveries are already
        # public on IPFS).
        self.preimage_retention_enabled: bool = kwargs.get(
            "preimage_retention_enabled", False
        )
        # Retention window for buffered preimages, in seconds (default 24h).
        # NOTE: this is a storage bound, not a cryptographic-erasure
        # guarantee. After expiry the sweeper DELETEs the row so it stops
        # appearing in kv_store queries — the on-disk footprint plateaus
        # rather than growing without bound. The plaintext bytes may
        # remain recoverable from freed SQLite pages / WAL frames until
        # those slots are reused. See preimage.py module docstring.
        self.preimage_retention_seconds: int = kwargs.get(
            "preimage_retention_seconds", 86400
        )
        # How often the sweeper LISTs the namespace to prune expired entries.
        self.preimage_sweep_interval: float = kwargs.get(
            "preimage_sweep_interval", 3600.0
        )
        # kv_store key namespace for preimages; the sweeper LISTs by this prefix.
        self.preimage_key_prefix: str = kwargs.get(
            "preimage_key_prefix", "mech_preimage/"
        )
        # Page size for the sweep LIST_REQUEST. Server clamps at 1000; the
        # default matches the server's own default so unmodified deployments
        # behave identically before/after the pagination wiring.
        self.preimage_list_page_size: int = kwargs.get("preimage_list_page_size", 100)
        # Max kv_store write attempts per request_id before the buffer drops
        # the record + WARNs. Bounds a hot-loop retry against a persistently
        # unhealthy kv_store. 5 is generous for transient glitches; >5 is
        # almost certainly persistent and the record is best-effort audit.
        self.preimage_max_write_attempts: int = kwargs.get(
            "preimage_max_write_attempts", 5
        )
        # Max consecutive LIST ERRORs before the sweep gives up the current
        # walk: cursor cleared, LAST_SWEEP stamped, WARN emitted. The next
        # sweep window (preimage_sweep_interval) is the natural backoff,
        # bounding what would otherwise be a per-tick LIST hot-loop against
        # a persistently failing kv_store.
        self.preimage_max_list_attempts: int = kwargs.get(
            "preimage_max_list_attempts", 5
        )
        self.tools_to_pricing: Dict[str, int] = kwargs.get("tools_to_pricing", {})
        if self.tools_to_pricing:
            self._ensure_same_keys(
                kwargs, self.tools_to_package_hash, self.tools_to_pricing
            )

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

    @classmethod
    def _ensure_same_keys(
        cls, kwargs: Dict, hash_dict: Dict, pricing_dict: Dict
    ) -> None:
        """Ensure that the same keys are available inside two dicts"""
        enforce("skill_context" in kwargs, "Only use on models!")
        hash_keys = set(hash_dict)
        pricing_keys = set(pricing_dict)
        extra_keys_in_hash_dict = sorted(hash_keys - pricing_keys)
        extra_keys_in_pricing_dict = sorted(pricing_keys - hash_keys)

        if extra_keys_in_hash_dict or extra_keys_in_pricing_dict:
            errors = []
            if extra_keys_in_hash_dict:
                errors.append(
                    f"Extra keys in tools_to_packages_hash dictionary: {', '.join(extra_keys_in_hash_dict)}"
                )
            if extra_keys_in_pricing_dict:
                errors.append(
                    f"Extra keys in tools_to_pricing dictionary: {', '.join(extra_keys_in_pricing_dict)}"
                )
            enforce(
                False,
                "; ".join(errors),
            )
