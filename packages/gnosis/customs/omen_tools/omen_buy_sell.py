# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
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
import functools
import traceback
from typing import Any, Callable, Dict, Optional, Tuple

import openai
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.omen.data_models import (
    OMEN_FALSE_OUTCOME,
    OMEN_TRUE_OUTCOME,
)
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    OmenCollateralTokenContract,
    OmenConditionalTokenContract,
    OmenFixedProductMarketMakerContract,
)
from prediction_market_agent_tooling.tools.web3_utils import add_fraction, prepare_tx
from pydantic import BaseModel
from web3 import Web3
from web3.types import TxParams


MechResponseWithKeys = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]

ENGINE = "gpt-3.5-turbo"
MAX_TOKENS = 500
TEMPERATURE = 0.7

"""NOTE: An LLM is used for generating a dict containing interpreted parameters from the response, such as "recipient_address", "market_address", etc. This could also be done if we could somehow publish the parameters needed by the run method and make it discoverable by the caller."""

BUY_OR_SELL_TOKENS_PROMPT = """You are an LLM inside a multi-agent system that takes in a prompt from a user requesting you to produce transaction parameters which
will later be part of an Ethereum transaction.
Interpret the USER_PROMPT and extract the required information.
Do not use any functions.

[USER_PROMPT]
{user_prompt}

Follow the formatting instructions below for producing an output in the correct format.
{format_instructions}
"""

client: Optional[OpenAI] = None


class BuyOrSell(BaseModel):
    """Buy or Sell Model"""

    sender: str
    market_id: str
    outcome: bool
    amount_to_buy: float


def build_params_from_prompt(user_prompt: str) -> BuyOrSell:
    """Build the params from the prompt"""
    if client is None or client.api_key is None:
        raise ValueError("Client or client.api_key must not be None")

    model = ChatOpenAI(temperature=0, api_key=client.api_key)
    parser = PydanticOutputParser(pydantic_object=BuyOrSell)
    prompt = PromptTemplate(
        template=BUY_OR_SELL_TOKENS_PROMPT,
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser
    return chain.invoke({"user_prompt": user_prompt})


class OpenAIClientManager:
    """Client context manager for OpenAI."""

    def __init__(self, api_key: str):
        """Initializes with API keys, model, and embedding provider. Sets the LLM provider based on the model."""
        self.api_key = api_key

    def __enter__(self) -> OpenAI:
        """Initializes and returns LLM and embedding clients."""
        global client
        if client is None:
            client = OpenAI(api_key=self.api_key)
        return client

    def __exit__(self, exc_type: Any, exc_value: Any, tb: Any) -> None:
        """Closes the LLM client"""
        global client
        if client is not None:
            client.close()
            client = None


def build_approval_for_all_tx_params(
    buy_or_sell: BuyOrSell, market: AgentMarket, w3: Web3
) -> TxParams:
    """# Approve the market maker to withdraw our collateral token."""
    from_address_checksummed = Web3.to_checksum_address(buy_or_sell.sender)
    market_contract: OmenFixedProductMarketMakerContract = market.get_contract()
    conditional_tokens_contract = OmenConditionalTokenContract()

    tx_params_approve_all = prepare_tx(
        web3=w3,
        contract_address=conditional_tokens_contract.address,
        contract_abi=conditional_tokens_contract.abi,
        from_address=from_address_checksummed,
        function_name="setApprovalForAll",
        function_params=[
            market_contract.address,
            True,  # approve=True
        ],
    )
    return tx_params_approve_all


def build_approval_tx_params(
    buy_or_sell: BuyOrSell, market: AgentMarket, w3: Web3
) -> TxParams:
    """# Approve the market maker to withdraw our collateral token."""
    from_address_checksummed = Web3.to_checksum_address(buy_or_sell.sender)
    amount_wei = Web3.to_wei(buy_or_sell.amount_to_buy, "ether")

    market_contract: OmenFixedProductMarketMakerContract = market.get_contract()
    collateral_token_contract = OmenCollateralTokenContract()

    tx_params_approve = prepare_tx(
        web3=w3,
        contract_address=collateral_token_contract.address,
        contract_abi=collateral_token_contract.abi,
        from_address=from_address_checksummed,
        function_name="approve",
        function_params=[
            market_contract.address,
            amount_wei,
        ],
    )
    return tx_params_approve


def build_buy_tokens_tx_params(
    buy_or_sell: BuyOrSell, market: AgentMarket, w3: Web3
) -> TxParams:
    """# Build buy token tx."""
    from_address_checksummed = Web3.to_checksum_address(buy_or_sell.sender)
    amount_wei = Web3.to_wei(buy_or_sell.amount_to_buy, "ether")

    market_contract: OmenFixedProductMarketMakerContract = market.get_contract()

    # Get the index of the outcome we want to buy.
    outcome_str = OMEN_TRUE_OUTCOME if buy_or_sell.outcome else OMEN_FALSE_OUTCOME
    outcome_index: int = market.get_outcome_index(outcome_str)

    # Allow 1% slippage.
    expected_shares = market_contract.calcBuyAmount(amount_wei, outcome_index, web3=w3)

    # Buy shares using the deposited xDai in the collateral token.
    tx_params_buy = prepare_tx(
        web3=w3,
        contract_address=Web3.to_checksum_address(market_contract.address),
        contract_abi=market_contract.abi,
        from_address=from_address_checksummed,
        function_name="buy",
        function_params=[
            amount_wei,
            outcome_index,
            expected_shares,
        ],
        tx_params={"gas": "21000"},
    )
    return tx_params_buy


def build_sell_tokens_tx_params(
    buy_or_sell: BuyOrSell, market: AgentMarket, w3: Web3
) -> TxParams:
    """# Build sell token tx."""
    from_address_checksummed = Web3.to_checksum_address(buy_or_sell.sender)
    amount_wei = Web3.to_wei(buy_or_sell.amount_to_buy, "ether")
    market_contract: OmenFixedProductMarketMakerContract = market.get_contract()
    conditional_token_contract = OmenConditionalTokenContract()

    # Verify, that markets uses conditional tokens that we expect.
    if market_contract.conditionalTokens(web3=w3) != conditional_token_contract.address:
        raise ValueError(
            f"Market {market.id} uses conditional token that we didn't expect, {market_contract.conditionalTokens()} != {conditional_token_contract.address=}"
        )

    # Get the index of the outcome we want to sell.
    outcome_str = OMEN_TRUE_OUTCOME if buy_or_sell.outcome else OMEN_FALSE_OUTCOME
    outcome_index: int = market.get_outcome_index(outcome_str)

    # Calculate the amount of shares we will sell for the given selling amount of xdai.
    max_outcome_tokens_to_sell = market_contract.calcSellAmount(
        amount_wei, outcome_index, web3=w3
    )
    # Allow 1% slippage.
    max_outcome_tokens_to_sell = add_fraction(max_outcome_tokens_to_sell, 0.01)

    # Sell the shares.
    tx_params_sell = prepare_tx(
        web3=w3,
        contract_address=market_contract.address,
        contract_abi=market_contract.abi,
        from_address=from_address_checksummed,
        function_name="sell",
        function_params=[
            amount_wei,
            outcome_index,
            max_outcome_tokens_to_sell,
        ],
        tx_params={"gas": 210000},
    )

    return tx_params_sell


def fetch_params_from_prompt(prompt: str) -> Tuple[BuyOrSell, AgentMarket]:
    """# Fetch the params from the prompt."""
    buy_params = build_params_from_prompt(user_prompt=prompt)
    # Calculate the amount of shares we will get for the given investment amount.
    market: AgentMarket = OmenAgentMarket.get_binary_market(buy_params.market_id)
    return buy_params, market


def build_buy_tx(
    prompt: str, rpc_url: str
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Builds buy transaction request."""

    try:
        buy_params, market = fetch_params_from_prompt(prompt)
        w3 = get_web3(rpc_url)

        tx_params_approve = build_approval_tx_params(
            buy_or_sell=buy_params, market=market, w3=w3
        )
        tx_params_buy = build_buy_tokens_tx_params(
            buy_or_sell=buy_params, market=market, w3=w3
        )

        return build_return_from_tx_params([tx_params_approve, tx_params_buy], prompt)

    except Exception as e:
        traceback.print_exception(e)
        return f"exception occurred - {e}", "", None, None


def build_return_from_tx_params(
    tx_params: list[TxParams], prompt: str
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """# Build return tx."""
    # We return the transactions_dict below in order to be able to return multiple transactions for later execution instead of just one.
    transaction_dict = {}
    for i, tx_dict in enumerate(tx_params):
        transaction_dict[str(i)] = tx_dict

    return "", prompt, transaction_dict, None


def get_web3(gnosis_rpc_url: str) -> Web3:
    """Returns the web3 object"""
    return Web3(Web3.HTTPProvider(gnosis_rpc_url))


def build_sell_tx(
    prompt: str, rpc_url: str
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Builds sell transaction request."""

    try:
        sell_params, market = fetch_params_from_prompt(prompt)
        w3 = get_web3(rpc_url)

        tx_params_approve_all = build_approval_for_all_tx_params(
            buy_or_sell=sell_params, market=market, w3=w3
        )

        tx_params_sell = build_sell_tokens_tx_params(
            buy_or_sell=sell_params, market=market, w3=w3
        )

        return build_return_from_tx_params(
            [tx_params_approve_all, tx_params_sell], prompt
        )

    except Exception as e:
        traceback.print_exception(e)
        return f"exception occurred - {e}", "", None, None


def with_key_rotation(func: Callable) -> Callable:
    """
    Decorator that retries a function with API key rotation on failure.

    :param func: The function to be decorated.
    :type func: Callable
    :returns: Callable -- the wrapped function that handles retries with key rotation.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MechResponseWithKeys:
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponseWithKeys:
            """Retry the function with a new key."""
            try:
                result: MechResponse = func(*args, **kwargs)
                return result + (api_keys,)
            except openai.RateLimitError as e:
                # try with a new key again
                if retries_left["openai"] <= 0 and retries_left["openrouter"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                retries_left["openrouter"] -= 1
                api_keys.rotate("openai")
                api_keys.rotate("openrouter")
                return execute()
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


def error_response(msg: str) -> Tuple[str, None, None, None]:
    """Return an error mech response."""
    return msg, None, None, None


LLM_SETTINGS = {
    "gpt-4-0125-preview": {
        "default_max_tokens": 500,
        "limit_max_tokens": 8192,
        "temperature": 0,
    },
    "gpt-4o-2024-08-06": {
        "default_max_tokens": 500,
        "limit_max_tokens": 4096,
        "temperature": 0,
    },
}

ALLOWED_MODELS = list(LLM_SETTINGS.keys())

ALLOWED_TOOLS = {
    "buy_omen": build_buy_tx,  # buyYesTokens, buyNoTokens
    "sell_omen": build_sell_tx,  # sellYesTokens, sellNoTokens
}


@with_key_rotation
def run(**kwargs: Any) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the task"""
    tool: str | None = kwargs.get("tool", None)

    if tool is None:
        return error_response("No tool has been specified.")

    prompt: str | None = kwargs.get("prompt", None)
    if prompt is None:
        return error_response("No prompt has been given.")

    transaction_builder = ALLOWED_TOOLS.get(tool)
    if transaction_builder is None:
        return error_response(
            f"Tool {tool!r} is not in supported tools: {tuple(ALLOWED_TOOLS.keys())}."
        )

    api_key: str | None = kwargs.get("api_keys", {}).get("openai", None)
    if api_key is None:
        return error_response("No api key has been given.")

    gnosis_rpc_url: str | None = kwargs.get("api_keys", {}).get("gnosis_rpc_url", None)
    if gnosis_rpc_url is None:
        return error_response("No gnosis rpc url has been given.")

    with OpenAIClientManager(api_key):
        return transaction_builder(prompt, gnosis_rpc_url)
