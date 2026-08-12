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
    BadResponseFormat,
    ContractLogicError,
    Web3RPCError,
    Web3ValidationError,
)

from packages.valory.skills.task_execution.utils.eip1271 import (
    EIP1271_MAGIC_VALUE,
    Eip1271Verdict,
    check_eip1271_signature,
    get_marketplace_domain_separator,
)

_CONTRACT_ADDRESS = "0x1111111111111111111111111111111111111111"
_MESSAGE_HASH = b"\x22" * 32
_SIGNATURE = b"\x33" * 65
_TEST_GAS_CAP = 500_000


def _make_ledger_api(fn_name: str, call_return: Any) -> SimpleNamespace:
    """Fake EthereumApi whose named function's ``call`` returns / raises as configured."""  # noqa: E501
    # The returned ``call`` is a real ``MagicMock`` so tests that need
    # to assert on the transaction kwargs (``gas`` cap) can inspect
    # ``call_args`` after the exchange.
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
    ledger_api = SimpleNamespace(api=api_ns)
    # Expose the call mock so tests can assert on the kwargs the caller
    # actually threaded through to ``.call(...)``.
    ledger_api.call_mock = call_mock  # type: ignore[attr-defined]
    return ledger_api


# --- check_eip1271_signature -------------------------------------------------


def test_check_eip1271_signature_returns_valid_on_magic_value() -> None:
    """Return VALID when the contract returns the EIP-1271 magic value."""
    ledger_api = _make_ledger_api("isValidSignature", EIP1271_MAGIC_VALUE)

    result = check_eip1271_signature(
        ledger_api=ledger_api,
        contract_address=_CONTRACT_ADDRESS,
        message_hash=_MESSAGE_HASH,
        signature=_SIGNATURE,
        gas=_TEST_GAS_CAP,
    )
    assert result is Eip1271Verdict.VALID


@pytest.mark.parametrize(
    "returned",
    [
        pytest.param(b"\x00\x00\x00\x00", id="zero_bytes4"),
        pytest.param(b"\xff\xff\xff\xff", id="all_ones_bytes4"),
        pytest.param(b"\xde\xad\xbe\xef", id="unrelated_bytes4"),
    ],
)
def test_check_eip1271_signature_returns_declined_on_non_magic_value(
    returned: bytes,
) -> None:
    """Return DECLINED when the contract returns any bytes4 other than the magic value."""
    ledger_api = _make_ledger_api("isValidSignature", returned)

    result = check_eip1271_signature(
        ledger_api=ledger_api,
        contract_address=_CONTRACT_ADDRESS,
        message_hash=_MESSAGE_HASH,
        signature=_SIGNATURE,
        gas=_TEST_GAS_CAP,
    )
    assert result is Eip1271Verdict.DECLINED


@pytest.mark.parametrize(
    "raised",
    [
        pytest.param(
            ContractLogicError("execution reverted"), id="contract_logic_error"
        ),
        pytest.param(
            BadFunctionCallOutput(
                "Could not transact with/call contract function, "
                "is contract deployed correctly and chain synced?"
            ),
            id="bad_function_call_output_missing_code",
        ),
    ],
)
def test_check_eip1271_signature_returns_declined_on_credential_side_failure(
    raised: BaseException,
) -> None:
    """A revert or codeless target maps to DECLINED (the contract said no).

    ``ContractLogicError`` fires on an explicit ``revert`` (including
    the out-of-gas revert triggered by the gas cap).
    ``BadFunctionCallOutput`` with the missing-code message shape
    fires when the ``eth_call`` target has no code at all — a plain
    EOA whose ``ecrecover`` did not match on the primary path. Both
    are credential-side outcomes; the caller routes ``DECLINED`` to
    401 so the client learns its signature was not accepted.

    :param raised: the boundary exception injected by the parametrise.
    """
    ledger_api = _make_ledger_api("isValidSignature", raised)

    result = check_eip1271_signature(
        ledger_api=ledger_api,
        contract_address=_CONTRACT_ADDRESS,
        message_hash=_MESSAGE_HASH,
        signature=_SIGNATURE,
        gas=_TEST_GAS_CAP,
    )
    assert result is Eip1271Verdict.DECLINED


@pytest.mark.parametrize(
    "raised",
    [
        pytest.param(Web3RPCError("rpc error"), id="web3_rpc_error"),
        pytest.param(
            BadResponseFormat("non-conforming json-rpc envelope"),
            id="bad_response_format",
        ),
        pytest.param(
            Web3ValidationError("schema mismatch"), id="web3_validation_error"
        ),
        pytest.param(ValueError("abi decode failure"), id="value_error"),
        pytest.param(
            BadFunctionCallOutput(
                "Could not decode contract function call to isValidSignature "
                "with return data: b'', output_types: ['bytes4']"
            ),
            id="bad_function_call_output_decode_failure_from_pruned_node",
        ),
        pytest.param(
            requests.exceptions.ConnectionError("boom"), id="connection_error"
        ),
        pytest.param(requests.exceptions.HTTPError("bad status"), id="http_error"),
    ],
)
def test_check_eip1271_signature_returns_call_failed_on_infra_error(
    raised: BaseException,
) -> None:
    """Infra-side failures at the ledger boundary map to CALL_FAILED.

    Distinguishing CALL_FAILED from DECLINED is what stops an RPC
    outage from surfacing to a legitimate Safe requester as 401
    "unauthorized" — the caller routes CALL_FAILED to 503 (retry)
    while DECLINED stays 401 (bad signature). ``BadResponseFormat``
    and ``Web3ValidationError`` are the two ``Web3Exception`` siblings
    raised outside the ``RotatingHTTPProvider.make_request`` boundary
    (in ``web3.manager.formatted_response``), so a narrower ``except``
    tuple would let them propagate out through the accept path and
    crash the framework's default ``propagate`` handler.

    :param raised: the boundary exception injected by the parametrise.
    """
    ledger_api = _make_ledger_api("isValidSignature", raised)

    result = check_eip1271_signature(
        ledger_api=ledger_api,
        contract_address=_CONTRACT_ADDRESS,
        message_hash=_MESSAGE_HASH,
        signature=_SIGNATURE,
        gas=_TEST_GAS_CAP,
    )
    assert result is Eip1271Verdict.CALL_FAILED


def test_check_eip1271_signature_logger_records_branch_for_operator_triage() -> None:
    """The optional ``logger`` records the exception type on CALL_FAILED.

    Pins that a caller supplying its own logger sees the exception type
    in the log line so operator triage on a 503 has the underlying
    fault visible. Without this, every infra-side branch produced the
    same "signature verification failed" line and operators had no
    way to distinguish which of the six causes fired.
    """
    import logging

    ledger_api = _make_ledger_api("isValidSignature", Web3RPCError("rpc down"))
    logger = logging.getLogger("test_check_eip1271_signature_logger_records_branch")
    captured: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    handler = _Handler(level=logging.WARNING)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        result = check_eip1271_signature(
            ledger_api=ledger_api,
            contract_address=_CONTRACT_ADDRESS,
            message_hash=_MESSAGE_HASH,
            signature=_SIGNATURE,
            gas=_TEST_GAS_CAP,
            logger=logger,
        )
    finally:
        logger.removeHandler(handler)
    assert result is Eip1271Verdict.CALL_FAILED
    assert any("CALL_FAILED" in msg for msg in captured), captured
    assert any("Web3RPCError" in msg for msg in captured), captured


@pytest.mark.parametrize(
    "raised",
    [
        pytest.param(requests.exceptions.ReadTimeout("slow"), id="read_timeout"),
        pytest.param(
            requests.exceptions.ConnectTimeout("no route"), id="connect_timeout"
        ),
    ],
)
def test_check_eip1271_signature_raises_timeout_error_on_transport_timeout(
    raised: BaseException,
) -> None:
    """A read/connect timeout propagates as ``TimeoutError`` instead of False.

    The caller distinguishes a slow / hostile ``isValidSignature`` view
    from a plain signature rejection so operators can surface the
    former as a dedicated rejection reason.

    :param raised: the timeout exception injected by the parametrise.
    """
    ledger_api = _make_ledger_api("isValidSignature", raised)

    with pytest.raises(TimeoutError):
        check_eip1271_signature(
            ledger_api=ledger_api,
            contract_address=_CONTRACT_ADDRESS,
            message_hash=_MESSAGE_HASH,
            signature=_SIGNATURE,
            gas=_TEST_GAS_CAP,
        )


def test_check_eip1271_signature_passes_gas_cap_to_eth_call() -> None:
    """The provided ``gas`` cap is threaded to the underlying ``.call(...)``.

    Assert on the exact transaction kwargs the wrapper forwards to the
    web3 contract call so a future refactor that drops or mistypes the
    kwarg surfaces in the test rather than silently letting a caller's
    ``isValidSignature`` run to the block gas limit.
    """
    ledger_api = _make_ledger_api("isValidSignature", EIP1271_MAGIC_VALUE)

    check_eip1271_signature(
        ledger_api=ledger_api,
        contract_address=_CONTRACT_ADDRESS,
        message_hash=_MESSAGE_HASH,
        signature=_SIGNATURE,
        gas=123_456,
    )

    ledger_api.call_mock.assert_called_once_with({"gas": 123_456})


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
