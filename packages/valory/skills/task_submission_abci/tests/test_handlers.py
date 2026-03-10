# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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
"""Tests for task_submission_abci.handlers."""

from packages.valory.skills.abstract_round_abci.handlers import (
    ABCIRoundHandler as BaseABCIRoundHandler,
    ContractApiHandler as BaseContractApiHandler,
    HttpHandler as BaseHttpHandler,
    IpfsHandler as BaseIpfsHandler,
    LedgerApiHandler as BaseLedgerApiHandler,
    SigningHandler as BaseSigningHandler,
    TendermintHandler as BaseTendermintHandler,
)
from packages.valory.skills.task_submission_abci import handlers as task_handlers


class TestHandlerAliases:
    """Verify all handler type aliases in task_submission_abci.handlers are correct re-exports."""

    def test_abci_handler(self):
        assert task_handlers.ABCIHandler is BaseABCIRoundHandler

    def test_http_handler(self):
        assert task_handlers.HttpHandler is BaseHttpHandler

    def test_signing_handler(self):
        assert task_handlers.SigningHandler is BaseSigningHandler

    def test_ledger_api_handler(self):
        assert task_handlers.LedgerApiHandler is BaseLedgerApiHandler

    def test_contract_api_handler(self):
        assert task_handlers.ContractApiHandler is BaseContractApiHandler

    def test_tendermint_handler(self):
        assert task_handlers.TendermintHandler is BaseTendermintHandler

    def test_ipfs_handler(self):
        assert task_handlers.IpfsHandler is BaseIpfsHandler
