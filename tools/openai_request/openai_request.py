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
"""Contains the job definitions"""

from typing import Any, Dict, Optional, Tuple

import openai


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0.7,
}
PREFIX = "openai-"
ENGINES = {
    "chat": ["gpt-3.5-turbo", "gpt-4"],
    "completion": ["gpt-3.5-turbo-instruct"],
}
ALLOWED_TOOLS = [PREFIX + value for values in ENGINES.values() for value in values]


def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """Run the task"""
    openai.api_key = kwargs["api_keys"]["openai"]
    max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
    temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
    prompt = kwargs["prompt"]
    tool = kwargs["tool"]
    if tool not in ALLOWED_TOOLS:
        return f"Tool {tool} is not in the list of supported tools.", None, None

    engine = tool.replace(PREFIX, "")
    moderation_result = openai.Moderation.create(prompt)
    if moderation_result["results"][0]["flagged"]:
        return "Moderation flagged the prompt as in violation of terms.", None, None

    if engine in ENGINES["chat"]:
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
        return response.choices[0].message.content, prompt, None
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        timeout=120,
        presence_penalty=0,
    )
    return response.choices[0].text, prompt, None
