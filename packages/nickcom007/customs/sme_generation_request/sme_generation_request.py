#!/usr/bin/env python3
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

"""This module implements a Mech tool to generate Subject Matter Expert (SME) roles for a given market question"""
import functools
import json
from typing import Any, Callable, Dict, Optional, Tuple

import anthropic
import openai
from googleapiclient import errors
from openai import OpenAI
from tiktoken import encoding_for_model


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
            except errors.HttpError as e:
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
    enc = encoding_for_model(model)
    return len(enc.encode(text))


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0,
}

ALLOWED_TOOLS = [
    "strong-sme-generator",
    "normal-sme-generator",
]

TOOL_TO_ENGINE = {
    "strong-sme-generator": "gpt-4o-2024-08-06",
    "normal-sme-generator": "gpt-3.5-turbo-0125",
}

SME_GENERATION_SYSTEM_PROMPT = """
This task requires answering Yes or No to a specific question related to certain knowledge domains. The final opinion to the question should be determined by one or more subject matter experts (SME) of the related domains. You need to generate one or more SME roles and their role introduction that you believe to be helpful in forming a correct answer to question in the task.

Examples:
task question: "Will Apple release iphone 15 by 1 October 2023?"
[
        {
            "sme": "Technology Analyst",
            "sme_introduction": "You are a seasoned technology analyst AI assistant. Your goal is to do comprehensive research on the news on the tech companies and answer investor's interested questions in a trustful and accurate way."
        }
]
---
task question: "Will the newly elected ceremonial president of Singapore face any political scandals by 13 September 2023?"
[
        {
            "sme":  "Political Commentator",
            "sme_introduction": "You are an experienced political commentator in Asia. Your main objective is to produce comprehensive, insightful and impartial analysis based on the relevant political news and your politic expertise to form an answer to the question releted to a political event or politician."
        }
]
---
task question: "Will the air strike conflict in Sudan be resolved by 13 September 2023?"
[
       {
            "sme:  "Military Expert",
            "sme_introduction": "You are an experienced expert in military operation and industry. Your main goal is to faithfully and accurately answer a military related question based on the provided intelligence and your professional experience"
        },
       {
            "sme:  "Diplomat",
            "sme_introduction": "You are an senior deplomat who engages in diplomacy to foster peaceful relations, negotiate agreements, and navigate complex political, economic, and social landscapes. You need to form an opinion on a question related to international conflicts based on the related information and your understading in geopolitics."
        },
]
"""

SME_GENERATION_MARKET_PROMPT = """
task question: "{question}"
"""


@with_key_rotation
def run(**kwargs: Any) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Generate SME roles for a given market question

    Raises:
    ValueError: _description_

    :param kwargs: The function kwargs.
    :type kwargs: Any
    :returns: A tuple containing the generated SME roles. The first element is a string that can be loaded with `json.loads`
            to get a list of dict. The dict has two keys: "sme" and "sme_introduction". The value of "sme" is the SME role name,
            and the value of "sme_introduction" is the introduction of the SME role.
    :rtype: tuple(str, optional dict[str, any])
    """
    with OpenAIClientManager(kwargs["api_keys"]["openai"]):
        tool = kwargs["tool"]
        # prompt is the actual question
        prompt = kwargs["prompt"]
        max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])

        if not client:
            raise RuntimeError("Client not initialized")

        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"tool must be one of {ALLOWED_TOOLS}")

        engine = kwargs.get("model", TOOL_TO_ENGINE[tool])
        print(f"ENGINE: {engine}")

        market_question = SME_GENERATION_MARKET_PROMPT.format(question=prompt)
        system_prompt = SME_GENERATION_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": market_question},
        ]
        response = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            timeout=150,
            stop=None,
        )

        generated_sme_roles = response.choices[0].message.content
        # check whether the generated_sme_roles is valid json
        try:
            generated_sme_roles = json.loads(generated_sme_roles)
        except json.decoder.JSONDecodeError as e:
            return (
                f"Failed to generate SME roles due to {e}",
                json.dumps(messages),
                None,
                None,
            )

        return response.choices[0].message.content, json.dumps(messages), None, None
