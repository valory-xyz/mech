# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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

"""
This module implements a tool which prepares a transaction for the transaction settlement skill.
Please note that the gnosis safe parameters are missing from the payload, e.g., `safe_tx_hash`, `safe_tx_gas`, etc.
"""

import json
from typing import Any, Dict, Optional, Tuple


# a test value
VALUE = 1
# a test address
TO_ADDRESS = "0xFDECa8497223DFa5aE2200D759f769e95dAadE01"


def prepare_tx(
    prompt: str,
) -> Tuple[Optional[str], Any, Optional[Dict[str, Any]], Any]:
    """Perform native transfer."""
    transaction = json.dumps(
        {
            "value": VALUE,
            "to_address": TO_ADDRESS,
        }
    )
    return transaction, prompt, None, None


AVAILABLE_TOOLS = {"prepare_tx": prepare_tx}


def error_response(msg: str) -> Tuple[str, None, None, None]:
    """Return an error mech response."""
    return msg, None, None, None


def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the task"""
    tool = kwargs.get("tool", None)
    if tool is None:
        return error_response("No tool has been specified.")

    prompt = kwargs.get("prompt", None)
    if prompt is None:
        return error_response("No prompt has been given.")

    transaction_builder = AVAILABLE_TOOLS.get(tool, None)
    if transaction_builder is None:
        return error_response(
            f"Tool {tool!r} is not in supported tools: {tuple(AVAILABLE_TOOLS.keys())}."
        )

    return transaction_builder(prompt)
