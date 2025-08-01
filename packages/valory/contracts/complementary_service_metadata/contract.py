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

"""This module contains the class to connect to the ComplementaryServiceMetadata contract."""

from typing import Any, Optional, Dict

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea.crypto.base import LedgerApi


PUBLIC_ID = PublicId.from_str("valory/complementary_service_metadata:0.1.0")


class ComplementaryServiceMetadata(Contract):
    """The Complementary Service Metadata contract."""

    contract_id = PUBLIC_ID

    @classmethod
    def get_raw_transaction(
        cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any
    ) -> Optional[JSONLike]:
        """Get the Safe transaction."""
        raise NotImplementedError  # pragma: nocover

    @classmethod
    def get_raw_message(
        cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any
    ) -> Optional[bytes]:
        """Get raw message."""
        raise NotImplementedError  # pragma: nocover

    @classmethod
    def get_state(
        cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any
    ) -> Optional[JSONLike]:
        """Get state."""
        raise NotImplementedError  # pragma: nocover

    @classmethod
    def get_update_hash_events(  # pragma: nocover
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        receipt: JSONLike,
    ) -> Optional[int]:
        """Returns `ComplementaryMetadataUpdated` event filter."""
        contract_interface = cls.get_instance(
            ledger_api=ledger_api,
            contract_address=contract_address,
        )
        return contract_interface.events.ComplementaryMetadataUpdated().process_receipt(
            receipt
        )

    @classmethod
    def get_token_uri(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        service_id: int,
    ) -> Dict[str, str]:
        """Returns the token URI for a service id."""
        contract_interface = cls.get_instance(
            ledger_api=ledger_api,
            contract_address=contract_address,
        )
        uri = ledger_api.contract_method_call(
            contract_interface, "tokenURI", serviceId=service_id
        )
        return dict(uri=uri)

    @classmethod
    def get_token_cid_hash(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        service_id: int,
    ) -> Dict[str, str]:
        """Returns the CID prefix and metadata hash for a service id."""
        contract_interface = cls.get_instance(
            ledger_api=ledger_api,
            contract_address=contract_address,
        )
        cid_prefix = contract_interface.functions.CID_PREFIX().call()
        latest_hash = contract_interface.functions.mapServiceHashes(service_id).call()

        return dict(hash=cid_prefix + latest_hash.hex())

    @classmethod
    def get_token_hash(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        service_id: int,
    ) -> JSONLike:
        """Returns the metadata hash for a service id."""
        contract_interface = cls.get_instance(
            ledger_api=ledger_api,
            contract_address=contract_address,
        )
        latest_hash = contract_interface.functions.mapServiceHashes(service_id).call()
        return dict(data=latest_hash.hex())

    @classmethod
    def get_update_hash_tx_data(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        service_id: int,
        metadata_hash: bytes,
    ) -> JSONLike:
        """Returns the transaction to update the metadata hash."""
        contract_interface = cls.get_instance(
            ledger_api=ledger_api,
            contract_address=contract_address,
        )
        data = contract_interface.encodeABI(
            fn_name="changeHash", args=[service_id, metadata_hash]
        )
        return dict(data=data)
