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

"""
python native_transfer_request.py “transfer 0.0001 ETH to 0x4253cB6Fbf9Cb7CD6cF58FF9Ed7FC3BDbd8312fe"
"""

import ast
from typing import Any, Dict, Optional, Tuple, cast

import openai


ENGINE = "gpt-3.5-turbo"
MAX_TOKENS = 500
TEMPERATURE = 0.7
TOOL_PREFIX = "transfer-"

"""NOTE: In the case of native token transfers on evm chains we do not need any contract address or ABI. The only unknowns are the "recipient address" and the "value" to send for evm native transfers."""

NATIVE_TRANSFER_PROMPT = """You are an LLM inside a multi-agent system that takes in a prompt from a user requesting you to execute a native gas token (ETH) transfer to another public address on Ethereum. 
The agent process you are sending your response to requires the unknown transaction parameters in the exact format below, written by you in your response as an input to sign/execute the transaction in the 
agent process.The agent does not know the receiving address, “recipient_address", the value to send, “value”, or the denomination of the "value" given in wei "wei_value" which is converted by you without 
use of any functions, the user prompt indicates to send. The unknown transaction parameters not known beforehand must be constructed by you from the user's prompt information. 

User Prompt: {user_prompt}

only respond with the format below using curly brackets to encapsulate the variables within a json dictionary object and no other text:

"to": recipient_address, 
"value": value, 
"wei_value": wei_value

Do not respond with anything else other than the transaction object you constructed with the correct known variables the agent had before the request and the correct unknown values found in the user request prompt as input to the web3.py signing method.
"""


def make_request_openai_request(
    prompt: str,
    api_key: str,
    engine: str = ENGINE,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Make openai request."""
    openai.api_key = api_key
    max_tokens = max_tokens or MAX_TOKENS
    temperature = temperature or TEMPERATURE
    moderation_result = openai.Moderation.create(prompt)
    if moderation_result["results"][0]["flagged"]:
        return "Moderation flagged the prompt as in violation of terms."

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        timeout=120,
        stop=None,
    )
    return response.choices[0].message.content


def native_transfer(
    prompt: str,
    api_key: str,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """Perform native transfer."""
    tool_prompt = NATIVE_TRANSFER_PROMPT.format(user_prompt=prompt)
    response = make_request_openai_request(prompt=tool_prompt, api_key=api_key)

    try:
        # parse the response to get the transaction object string itself
        parsed_txs = ast.literal_eval(response)
    except SyntaxError:
        return response, None, None

    # build the transaction object, unknowns are referenced from parsed_txs
    transaction = {
        "to": str(parsed_txs["to"]),
        "value": int(parsed_txs["wei_value"]),
    }

    return response, prompt, transaction


AVAILABLE_TOOLS = {
    "native": native_transfer,
}


def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """Run the task"""
    prompt = kwargs["prompt"]
    api_key = kwargs["api_keys"]["openai"]
    tool = cast(str, kwargs["tool"]).replace(TOOL_PREFIX, "")

    if tool not in AVAILABLE_TOOLS:
        return f"No tool named `{kwargs['tool']}`", None, None

    transaction_builder = AVAILABLE_TOOLS[tool]
    return transaction_builder(prompt, api_key)
