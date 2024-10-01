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
"""Contains the job definitions"""

from typing import Any, Dict, Optional, Tuple

import requests
import json

DEFAULT_CORCEL_SETTINGS = {
    "temperature": 0.1,
    "max_tokens": 500,
}

CORCEL_URL = "https://api.corcel.io/v1/chat/completions"

ENGINES = {
    "chat": [
        "llama-3-1-8b",
        "llama-3-1-70b",
        "gpt-3.5-turbo",
        "cortext-ultra",
        "cortext",
        "cortext-lite",
        "gpt-4-1106-preview",
        "gpt-4-turbo-2024-04-09",
        "gpt-40",
        "gemini-pro",
        "davinci-002",
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "babbage-002",
        "gpt-4-0613",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-instruct-0914",
        "gpt-3.5-turbo-instruct",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0125",
        "gpt-4-turbo",
        "gpt-3.5-turbo-0613",
        "gpt-4o-2024-05-13",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "gemma-7b-it",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "cohere.command-r-v1:0",
        "meta.llama2-70b-chat-v1",
        "amazon.titan-text-express-v1",
        "mistral.mistral-7b-instruct-v0:2"
    ]
}

ALLOWED_TOOLS = [value for value in ENGINES["chat"]]


def error_response(msg: str) -> Tuple[str, None, None, None]:
    """Return an error mech response."""
    return msg, None, None, None


def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""

    api_key = kwargs["api_keys"]["corcel"]
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]

    if api_key is None:
        return error_response("Corcel API key is not available.")

    if tool is None:
        return error_response("No tool has been specified.")

    if tool not in ALLOWED_TOOLS:
        return (
            f"Model {tool} is not in the list of supported models.",
            None,
            None,
            None,
        )

    if prompt is None:
        return error_response("No prompt has been given.")

    temperature = kwargs.get("temperature", DEFAULT_CORCEL_SETTINGS["temperature"])
    max_tokens = kwargs.get("max_tokens", DEFAULT_CORCEL_SETTINGS["max_tokens"])
    counter_callback = kwargs.get("counter_callback", None)

    try:
        payload = {
            "model": tool,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": api_key,
        }

        response = requests.post(CORCEL_URL, json=payload, headers=headers)

        # Collect the chunks and concatenate the final response
        full_response = ""
        for line in response.iter_lines():
            if line:
                # Parse each line to extract the "delta" content
                try:
                    data = json.loads(line.decode("utf-8").replace("data: ", ""))
                    if "choices" in data and len(data["choices"]) > 0:
                        chunk = data["choices"][0].get("delta", {}).get("content", "")
                        full_response += chunk
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        return f"An error occurred: {str(e)}", None, None, None

    return full_response, prompt, None, counter_callback
