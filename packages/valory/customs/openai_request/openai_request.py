# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2025 Valory AG
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
from typing import Any, Callable, Dict, Optional, Tuple, Union

import anthropic
import googleapiclient
import openai
from openai import OpenAI
from tiktoken import encoding_for_model, get_encoding


client: Optional[OpenAI] = None


MechResponseWithKeys = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]


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
        """Initializes with API keys"""
        self.api_key = api_key

    def __enter__(self) -> OpenAI:
        """Initializes and returns LLM client."""
        global client
        if client is None:
            client = OpenAI(api_key=self.api_key)
        return client

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Closes the LLM client"""
        global client
        if client is not None:
            client.close()
            client = None


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    # Workaround since tiktoken does not have support yet for gpt4.1
    # https://github.com/openai/tiktoken/issues/395
    if model == "gpt-4.1-2025-04-14":
        enc = get_encoding("o200k_base")
    else:
        enc = encoding_for_model(model)
    return len(enc.encode(text))


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 4096,
    "temperature": 0.7,
}
PREFIX = "openai-"
ENGINES = {
    "chat": ["gpt-4.1-2025-04-14"],
    "completion": ["gpt-4.1-2025-04-14"],
}
ALLOWED_TOOLS = [PREFIX + value for values in ENGINES.values() for value in values]


@with_key_rotation
def run(
    **kwargs: Any,
) -> Union[float, Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]]:
    """Run the task"""
    tool = kwargs["tool"]
    if tool not in ALLOWED_TOOLS:
        return (
            f"Tool {tool} is not in the list of supported tools.",
            None,
            None,
            None,
        )

    engine = tool.replace(PREFIX, "")
    delivery_rate = int(kwargs.get("delivery_rate", 0))
    counter_callback: Optional[Callable] = kwargs.get("counter_callback", None)

    if delivery_rate == 0:
        if not counter_callback:
            raise ValueError(
                "A delivery rate of `0` was passed, but no counter callback was given to calculate the max cost with."
            )

        max_cost = counter_callback(
            max_cost=True,
            models_calls=(engine,),
        )
        return max_cost

    with OpenAIClientManager(kwargs["api_keys"]["openai"]):
        max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
        prompt = kwargs["prompt"]
        counter_callback = kwargs.get("counter_callback", None)
        if not client:
            raise RuntimeError("Client not initialized")

        moderation_result = client.moderations.create(input=prompt)
        if moderation_result.results[0].flagged:
            return (
                "Moderation flagged the prompt as in violation of terms.",
                None,
                None,
                None,
            )

        if engine in ENGINES["chat"]:
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
            return response.choices[0].message.content, prompt, None, None
        response = client.completions.create(
            model=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            timeout=120,
            presence_penalty=0,
        )
        return response.choices[0].text, prompt, None, counter_callback
