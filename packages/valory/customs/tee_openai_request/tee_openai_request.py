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
"""Contains the job definitions"""
import functools
from typing import Any, Callable, Dict, Optional, Tuple

import anthropic
import googleapiclient
import openai
import requests
from tiktoken import encoding_for_model


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


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


PREFIX = "tee-openai-"
ENGINES = {
    "chat": ["gpt-3.5-turbo", "gpt-4o-2024-08-06"],
    "completion": ["gpt-3.5-turbo-instruct"],
}
ALLOWED_TOOLS = [PREFIX + value for values in ENGINES.values() for value in values]
AGENT_URL = "https://wapo-testnet.phala.network/ipfs/QmeUiNKgsHiAK3WM57XYd7ssqMwVNbcGwtm8gKLD2pVXiP"


@with_key_rotation
def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""
    api_key = kwargs["api_keys"]["openai"]
    prompt = kwargs["prompt"]
    tool = kwargs["tool"]
    counter_callback = kwargs.get("counter_callback", None)
    if tool not in ALLOWED_TOOLS:
        return (
            f"Tool {tool} is not in the list of supported tools.",
            None,
            None,
            None,
        )

    engine = tool.replace(PREFIX, "")

    params = {"openaiApiKey": api_key, "chatQuery": prompt, "openAiModel": engine}

    # Request to agent contract in TEE
    response = requests.get(AGENT_URL, params=params)

    if response.status_code == 200:
        json_response = response.json()
        if "message" in json_response:
            return str(json_response["message"]), prompt, None, counter_callback
        else:
            return (
                "The 'message' field is not present in the response.",
                None,
                None,
                None,
            )
    else:
        return (
            f"Failed to retrieve data: {response.status_code}, {response.text}.",
            None,
            None,
            None,
        )
