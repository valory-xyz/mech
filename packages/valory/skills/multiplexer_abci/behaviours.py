# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 Valory AG
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

"""This package contains round behaviours of MultiplexerAbciApp."""

from abc import ABC
from typing import Dict, Generator, List, Set, Type, cast

from packages.valory.contracts.agent_mech.contract import AgentMechContract
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour, BaseBehaviour)
from packages.valory.skills.multiplexer_abci.models import Params, SharedState
from packages.valory.skills.multiplexer_abci.rounds import (MultiplexerAbciApp,
                                                            MultiplexerPayload,
                                                            MultiplexerRound,
                                                            SynchronizedData)


class MultiplexerBaseBehaviour(BaseBehaviour, ABC):
    """Base behaviour for the multiplexer_abci skill."""

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return cast(SynchronizedData, super().synchronized_data)

    @property
    def params(self) -> Params:
        """Return the params."""
        return cast(Params, super().params)

    @property
    def shared_state(self) -> SharedState:
        """Return the params."""
        return cast(SharedState, self.context.state)

    def _get_requests(self, from_block: int = 0) -> Generator[None, None, List[Dict]]:
        """Get the contract requests."""
        response = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_id=str(AgentMechContract.contract_id),
            contract_callable="get_request_events",
            contract_address=self.params.agent_mech_contract_address,
            from_block=from_block,
        )
        if response.performative != ContractApiMessage.Performative.STATE:
            self.context.logger.error(
                f"Couldn't get the latest `Request` event. "
                f"Expected response performative {ContractApiMessage.Performative.STATE.value}, "  # type: ignore
                f"received {response.performative.value}."
            )
            return []

        requests = cast(List[Dict], response.state.body.get("data"))
        return requests

    def _get_deliver(self, from_block: int = 0) -> Generator[None, None, List[Dict]]:
        """Get the contract delivers."""
        response = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_id=str(AgentMechContract.contract_id),
            contract_callable="get_deliver_events",
            contract_address=self.params.agent_mech_contract_address,
            from_block=from_block,
        )
        if response.performative != ContractApiMessage.Performative.STATE:
            self.context.logger.error(
                f"Couldn't get the latest `Deliver` event. "
                f"Expected response performative {ContractApiMessage.Performative.STATE.value}, "  # type: ignore
                f"received {response.performative.value}."
            )
            return []

        delivers = cast(List[Dict], response.state.body.get("data"))
        return delivers

    def extend_pending_tasks(self) -> Generator[None, None, List[Dict]]:
        """Get the requests to send to the mech service."""
        from_block = self.shared_state.last_processed_request_block_number
        requests = yield from self._get_requests(from_block)
        delivers = yield from self._get_deliver(from_block)
        pending_tasks = self.context.shared_state.get("pending_tasks", [])
        for request in requests:
            if (
                request["block_number"]
                > self.shared_state.last_processed_request_block_number
            ):
                self.shared_state.last_processed_request_block_number = request[
                    "block_number"
                ]

            if request["requestId"] not in [
                deliver["requestId"] for deliver in delivers
            ] and request["requestId"] not in [
                pending_req["requestId"] for pending_req in pending_tasks
            ]:
                # store each requests in the pending_tasks list, make sure each req is stored once
                pending_tasks.append(request)
        pending_tasks.sort(key=lambda x: x["block_number"])
        return pending_tasks


class MultiplexerBehaviour(MultiplexerBaseBehaviour):
    """MultiplexerBehaviour"""

    matching_round: Type[AbstractRound] = MultiplexerRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():

            payload_content = MultiplexerRound.WAIT_PAYLOAD

            period_counter = self.synchronized_data.period_counter
            do_reset = period_counter % self.params.reset_period_count == 0
            should_poll_events = self.params.use_polling and (
                period_counter % self.params.polling_interval == 0
            )
            self.context.logger.info(
                f"Period counter: {period_counter}/{self.params.reset_period_count}. Do reset? {do_reset}"
            )
            self.context.logger.info(
                f"Pending tasks: {self.context.shared_state.get('pending_tasks', [])}"
            )
            if should_poll_events:
                pending_tasks = yield from self.extend_pending_tasks()
                self.context.shared_state["pending_tasks"] = pending_tasks

            if self.context.shared_state.get("pending_tasks", []):
                payload_content = MultiplexerRound.EXECUTE_PAYLOAD
            elif do_reset:
                payload_content = MultiplexerRound.RESET_PAYLOAD

            sender = self.context.agent_address
            payload = MultiplexerPayload(sender=sender, content=payload_content)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()


class MultiplexerRoundBehaviour(AbstractRoundBehaviour):
    """MultiplexerRoundBehaviour"""

    initial_behaviour_cls = MultiplexerBehaviour
    abci_app_cls = MultiplexerAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = {MultiplexerBehaviour}
