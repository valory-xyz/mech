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

"""Tests for the EIP-1271 signature check and marketplace/mech read helpers.

Mocks the ledger boundary — the wrapper's contract is purely about mapping a
view return to a Python value, so an in-process mock at the eth_contract
level is the right cut.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests
from web3.exceptions import (
    BadFunctionCallOutput,
    ContractLogicError,
    Web3RPCError,
)

from packages.valory.skills.task_execution.utils.eip1271 import (
    EIP1271_MAGIC_VALUE,
    check_eip1271_signature,
    get_marketplace_domain_separator,
)

_CONTRACT_ADDRESS = "0x1111111111111111111111111111111111111111"
_MESSAGE_HASH = b"\x22" * 32
_SIGNATURE = b"\x33" * 65


def _make_ledger_api(fn_name: str, call_return: Any) -> SimpleNamespace:
    """Fake EthereumApi whose named function's ``call`` returns / raises as configured."""
    call_mock = MagicMock()
    if isinstance(call_return, BaseException):
        call_mock.side_effect = call_return
    else:
        call_mock.return_value = call_return

    def _fn(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(call=call_mock)

    functions_ns = SimpleNamespace(**{fn_name: _fn})
    contract_ns = SimpleNamespace(functions=functions_ns)

    inner_eth = SimpleNamespace(contract=MagicMock(return_value=contract_ns))
    api_ns = SimpleNamespace(eth=inner_eth, to_checksum_address=lambda a: a)
    return SimpleNamespace(api=api_ns)


# --- check_eip1271_signature -------------------------------------------------


def test_check_eip1271_signature_returns_true_on_magic_value() -> None:
    """Return True when the contract returns the EIP-1271 magic value."""
    ledger_api = _make_ledger_api("isValidSignature", EIP1271_MAGIC_VALUE)

    result = check_eip1271_signature(
        ledger_api=ledger_api,
        contract_address=_CONTRACT_ADDRESS,
        message_hash=_MESSAGE_HASH,
        signature=_SIGNATURE,
    )
    assert result is True


@pytest.mark.parametrize(
    "returned",
    [
        pytest.param(b"\x00\x00\x00\x00", id="zero_bytes4"),
        pytest.param(b"\xff\xff\xff\xff", id="all_ones_bytes4"),
        pytest.param(b"\xde\xad\xbe\xef", id="unrelated_bytes4"),
    ],
)
def test_check_eip1271_signature_returns_false_on_non_magic_value(
    returned: bytes,
) -> None:
    """Return False when the contract returns any bytes4 other than the magic value."""
    ledger_api = _make_ledger_api("isValidSignature", returned)

    result = check_eip1271_signature(
        ledger_api=ledger_api,
        contract_address=_CONTRACT_ADDRESS,
        message_hash=_MESSAGE_HASH,
        signature=_SIGNATURE,
    )
    assert result is False


@pytest.mark.parametrize(
    "raised",
    [
        pytest.param(
            ContractLogicError("execution reverted"), id="contract_logic_error"
        ),
        pytest.param(
            BadFunctionCallOutput("decode failure"), id="bad_function_call_output"
        ),
        pytest.param(Web3RPCError("rpc error"), id="web3_rpc_error"),
        pytest.param(ValueError("abi decode failure"), id="value_error"),
        pytest.param(
            requests.exceptions.ConnectionError("boom"), id="connection_error"
        ),
        pytest.param(requests.exceptions.ReadTimeout("slow"), id="read_timeout"),
        pytest.param(requests.exceptions.HTTPError("bad status"), id="http_error"),
    ],
)
def test_check_eip1271_signature_returns_false_on_boundary_exception(
    raised: BaseException,
) -> None:
    """Any recognised failure at the ledger boundary maps to False.

    Web3 raises ``BadFunctionCallOutput`` when ``eth_call`` targets a
    codeless address (the common case for a Safe pointed at the wrong
    proxy) and ``Web3RPCError`` on JSON-RPC failures; ``requests``
    transport failures (connection error, read timeout, HTTP error)
    surface as siblings under ``RequestException``. All must map to
    False so the accept path's fallback treats the outcome as a
    signature rejection instead of crashing the framework's default
    ``propagate`` handler and stopping the agent.

    :param raised: the boundary exception injected by the parametrise.
    """
    ledger_api = _make_ledger_api("isValidSignature", raised)

    result = check_eip1271_signature(
        ledger_api=ledger_api,
        contract_address=_CONTRACT_ADDRESS,
        message_hash=_MESSAGE_HASH,
        signature=_SIGNATURE,
    )
    assert result is False


# --- get_marketplace_domain_separator ---------------------------------------


def test_get_marketplace_domain_separator_returns_bytes32() -> None:
    """Return the 32-byte marketplace domain separator."""
    expected = b"\xab" * 32
    ledger_api = _make_ledger_api("getDomainSeparator", expected)

    result = get_marketplace_domain_separator(
        ledger_api=ledger_api,
        marketplace_address=_CONTRACT_ADDRESS,
    )
    assert result == expected


def test_get_marketplace_domain_separator_rejects_wrong_length() -> None:
    """A ``getDomainSeparator`` return that is not 32 bytes raises."""
    ledger_api = _make_ledger_api("getDomainSeparator", b"\x00" * 31)

    with pytest.raises(ValueError, match="domain_separator length"):
        get_marketplace_domain_separator(
            ledger_api=ledger_api,
            marketplace_address=_CONTRACT_ADDRESS,
        )


# ``get_mech_payment_type`` was removed in favour of the packaged
# ``OlasMechContract.get_mech_type`` wrapper; both call the same
# ``paymentType()`` selector and the wrapper carries its own tests
# alongside the contract package.
