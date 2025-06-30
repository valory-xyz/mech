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
"""Contains the job definitions"""
import functools
from typing import Any, Callable, Dict, Optional, Tuple

import openai
from openai import OpenAI
from tiktoken import encoding_for_model


client: Optional[OpenAI] = None
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]


def with_key_rotation(func: Callable):
    """
    Decorator that retries a function with API key rotation on failure.

    Expects `api_keys` in kwargs, supporting `rotate(service)` and `max_retries()`.
    Retries the function on key-related exceptions until retries are exhausted.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                # Ensure the result is a tuple and has the correct length
                if isinstance(result, tuple) and len(result) == 4:
                    return result + (api_keys,)
                else:
                    raise ValueError(
                        "Function did not return a valid MechResponse tuple."
                    )
            except openai.error.RateLimitError as e:
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

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Closes the LLM client"""
        global client
        if client is not None:
            client.close()
            client = None


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


DEFAULT_DALLE_SETTINGS = {
    "size": "1024x1024",
    "quality": "standard",
    "n": 1,
}
PREFIX = "dall-e"
ENGINES = {
    "text-to-image": ["-2", "-3"],
}
ALLOWED_MODELS = [PREFIX]
ALLOWED_TOOLS = [PREFIX + value for value in ENGINES["text-to-image"]]
ALLOWED_SIZE = ["1024x1024", "1024x1792", "1792x1024"]
ALLOWED_QUALITY = ["standard", "hd"]


@with_key_rotation
def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""
    with OpenAIClientManager(kwargs["api_keys"]["openai"]):
        tool = kwargs["tool"]
        prompt = kwargs["prompt"]
        size = kwargs.get("size", DEFAULT_DALLE_SETTINGS["size"])
        quality = kwargs.get("quality", DEFAULT_DALLE_SETTINGS["quality"])
        n = kwargs.get("n", DEFAULT_DALLE_SETTINGS["n"])
        counter_callback = kwargs.get("counter_callback", None)
        if tool not in ALLOWED_TOOLS:
            return (
                f"Tool {tool} is not in the list of supported tools.",
                None,
                None,
                None,
            )
        if size not in ALLOWED_SIZE:
            return (
                f"Size {size} is not in the list of supported sizes.",
                None,
                None,
                None,
            )
        if quality not in ALLOWED_QUALITY:
            return (
                f"Quality {quality} is not in the list of supported qualities.",
                None,
                None,
                None,
            )

        response = client.images.generate(
            model=tool,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )
        return response.data[0].url, prompt, None, counter_callback
