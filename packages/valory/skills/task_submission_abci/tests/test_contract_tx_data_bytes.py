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
"""Regression tests: tx-building contract helpers must return bytes, not str.

Background: open-autonomy v0.21.18 refactored MultiSendContract.encode_data to
drop its HexBytes(data) wrapper. That wrapper used to silently coerce hex-string
`data` inputs into bytes. With the wrapper gone, any helper that returns
`{"data": <encode_abi output>}` as a str now crashes `_to_multisend` with
`TypeError: can't concat str to bytes`. See incident report
docs/incident_report_single_mech_mm_predict_multisend_2026-04-22.docx.

These tests pin the post-fix invariant: every tx-building helper whose output
can feed into `_to_multisend` returns `data` as bytes. The assertions would
fail if a future change reverted either helper back to returning the raw
encode_abi str.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from aea_ledger_ethereum import EthereumApi

from packages.valory.contracts.complementary_service_metadata.contract import (
    ComplementaryServiceMetadata,
)
from packages.valory.contracts.hash_checkpoint.contract import HashCheckpointContract
from packages.valory.contracts.multisend.contract import (
    MultiSendOperation,
    encode_data,
)

ENCODE_ABI_SAMPLE_STR = "0x5b34eba0" + "ab" * 32


def _fake_ledger_api() -> MagicMock:
    """Mock that passes the isinstance(ledger_api, EthereumApi) guard used in some helpers."""
    return MagicMock(spec=EthereumApi)


def _patch_get_instance(
    contract_cls: Any, monkeypatch: pytest.MonkeyPatch
) -> MagicMock:
    """Replace `contract_cls.get_instance` so it returns a mock with a str-returning encode_abi."""
    fake_instance = MagicMock()
    fake_instance.encode_abi.return_value = ENCODE_ABI_SAMPLE_STR
    monkeypatch.setattr(
        contract_cls, "get_instance", classmethod(lambda cls, *_, **__: fake_instance)
    )
    return fake_instance


class TestHashCheckpointGetCheckpointData:
    """HashCheckpointContract.get_checkpoint_data must return bytes in the data field."""

    def test_data_field_is_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The helper converts the encode_abi str into bytes before returning."""
        _patch_get_instance(HashCheckpointContract, monkeypatch)
        result = HashCheckpointContract.get_checkpoint_data(
            ledger_api=_fake_ledger_api(),
            contract_address="0x0000000000000000000000000000000000000001",
            data=b"\xab" * 32,
        )
        assert isinstance(result, dict) and "data" in result
        assert isinstance(
            result["data"], bytes
        ), f"data must be bytes, got {type(result['data']).__name__}"
        # And the bytes round-trip matches the hex payload (less the 0x prefix).
        assert result["data"].hex() == ENCODE_ABI_SAMPLE_STR[2:]

    def test_result_feeds_through_multisend_encode_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: the returned data concatenates cleanly inside multisend.encode_data.

        This is the exact path the regression broke on. With the old str-returning
        helper, this call raised TypeError inside encode_data.
        """
        _patch_get_instance(HashCheckpointContract, monkeypatch)
        result = HashCheckpointContract.get_checkpoint_data(
            ledger_api=_fake_ledger_api(),
            contract_address="0x0000000000000000000000000000000000000001",
            data=b"\xab" * 32,
        )
        tx = {
            "operation": MultiSendOperation.CALL,
            "to": "0x0000000000000000000000000000000000000042",
            "value": 0,
            "data": result["data"],
        }
        encoded = encode_data(tx)
        # Fixed layout: 1 op + 20 to + 32 value + 32 length + len(data).
        expected_len = 1 + 20 + 32 + 32 + len(result["data"])
        assert len(encoded) == expected_len


class TestComplementaryServiceMetadataGetUpdateHashTxData:
    """ComplementaryServiceMetadata.get_update_hash_tx_data must return bytes in the data field."""

    def test_data_field_is_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The helper converts the encode_abi str into bytes before returning."""
        _patch_get_instance(ComplementaryServiceMetadata, monkeypatch)
        result = ComplementaryServiceMetadata.get_update_hash_tx_data(
            ledger_api=_fake_ledger_api(),
            contract_address="0x0000000000000000000000000000000000000002",
            service_id=42,
            metadata_hash=b"\xcd" * 32,
        )
        assert isinstance(result, dict) and "data" in result
        assert isinstance(
            result["data"], bytes
        ), f"data must be bytes, got {type(result['data']).__name__}"
        assert result["data"].hex() == ENCODE_ABI_SAMPLE_STR[2:]

    def test_result_feeds_through_multisend_encode_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: the returned data concatenates cleanly inside multisend.encode_data."""
        _patch_get_instance(ComplementaryServiceMetadata, monkeypatch)
        result = ComplementaryServiceMetadata.get_update_hash_tx_data(
            ledger_api=_fake_ledger_api(),
            contract_address="0x0000000000000000000000000000000000000002",
            service_id=42,
            metadata_hash=b"\xcd" * 32,
        )
        tx = {
            "operation": MultiSendOperation.CALL,
            "to": "0x0000000000000000000000000000000000000042",
            "value": 0,
            "data": result["data"],
        }
        encoded = encode_data(tx)
        expected_len = 1 + 20 + 32 + 32 + len(result["data"])
        assert len(encoded) == expected_len


class TestMultisendEncodeDataRejectsStr:
    """Pin the post-refactor multisend behaviour that motivates the fix.

    Documents that encode_data no longer tolerates a str in the `data` field.
    If this ever starts accepting str again (e.g. the framework re-adds a
    HexBytes wrapper), the fixtures above become defensive rather than
    necessary, but that is an acceptable outcome — the bytes invariant
    still holds.
    """

    def test_str_data_raises_type_error(self) -> None:
        """A str `data` value must raise TypeError inside encode_data."""
        tx = {
            "operation": MultiSendOperation.CALL,
            "to": "0x0000000000000000000000000000000000000042",
            "value": 0,
            "data": ENCODE_ABI_SAMPLE_STR,  # hex str like encode_abi used to return unconverted
        }
        with pytest.raises(TypeError):
            encode_data(tx)
