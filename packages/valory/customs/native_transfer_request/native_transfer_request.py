# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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
import functools
from typing import Any, Callable, Dict, Optional, Tuple, cast

import anthropic
import googleapiclient
import openai
from openai import OpenAI
from tiktoken import encoding_for_model


client: Optional[OpenAI] = None

MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]


def with_key_rotation(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except anthropic.RateLimitError as e:
                # try with a new key again
                service = "anthropic"
                if retries_left[service] <= 0:
                    raise e
                retries_left[service] -= 1
                api_keys.rotate(service)
                return execute()
            except openai.RateLimitError as e:
                # try with a new key again
                if retries_left["openai"] <= 0 and retries_left["openrouter"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                retries_left["openrouter"] -= 1
                api_keys.rotate("openai")
                api_keys.rotate("openrouter")
                return execute()
            except googleapiclient.errors.HttpError as e:
                # try with a new key again
                rate_limit_exceeded_code = 429
                if e.status_code != rate_limit_exceeded_code:
                    raise e
                service = "google_api_key"
                if retries_left[service] <= 0:
                    raise e
                retries_left[service] -= 1
                api_keys.rotate(service)
                return execute()
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


class OpenAIClientManager:
    """Client context manager for OpenAI."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def __enter__(self) -> OpenAI:
        global client
        if client is None:
            client = OpenAI(api_key=self.api_key)
        return client

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global client
        if client is not None:
            client.close()
            client = None


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


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
    engine: str = ENGINE,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Make openai request."""
    max_tokens = max_tokens or MAX_TOKENS
    temperature = temperature or TEMPERATURE
    moderation_result = client.moderations.create(input=prompt)
    if moderation_result.results[0].flagged:
        return "Moderation flagged the prompt as in violation of terms."

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
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
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Perform native transfer."""
    tool_prompt = NATIVE_TRANSFER_PROMPT.format(user_prompt=prompt)
    response = make_request_openai_request(prompt=tool_prompt)

    try:
        # parse the response to get the transaction object string itself
        parsed_txs = ast.literal_eval(response)
    except SyntaxError:
        return response, None, None, None

    # build the transaction object, unknowns are referenced from parsed_txs
    transaction = {
        "to": str(parsed_txs["to"]),
        "value": int(parsed_txs["wei_value"]),
    }
    return response, prompt, transaction, None


AVAILABLE_TOOLS = {
    "native": native_transfer,
}


@with_key_rotation
def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the task"""
    with OpenAIClientManager(kwargs["api_keys"]["openai"]):
        prompt = kwargs["prompt"]
        api_key = kwargs["api_keys"]["openai"]
        tool = cast(str, kwargs["tool"]).replace(TOOL_PREFIX, "")

        if tool not in AVAILABLE_TOOLS:
            return f"No tool named `{kwargs['tool']}`", None, None, None

        transaction_builder = AVAILABLE_TOOLS[tool]
        response = transaction_builder(prompt, api_key)
        return response
