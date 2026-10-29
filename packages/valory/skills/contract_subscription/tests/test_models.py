# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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
"""Tests for the contract-subscription Params."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from packages.valory.skills.contract_subscription.models import Params

SAMPLE_CONFIG = {
    "websocket_provider": "wss://sample.rpc",
    "subscription_id": "sample-subscription",
    "use_polling": True,
    "contract_address": "0x0000000000000000000000000000000000000001",
}


def test_params_consume_their_config_keys() -> None:
    """Keys read by this Params and its websocket parent never reach the base ``Model``."""
    logger = MagicMock()
    skill_context = SimpleNamespace(
        skill_id="valory/contract_subscription:0.1.0", logger=logger
    )
    params = Params(name="params", skill_context=skill_context, **SAMPLE_CONFIG)

    logger.warning.assert_not_called()
    for key, value in SAMPLE_CONFIG.items():
        assert getattr(params, key) == value, key
