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

import google.generativeai as genai


DEFAULT_GEMINI_SETTINGS = {
    "candidate_count": 1,
    "stop_sequences": None,
    "max_output_tokens": 500,
    "temperature": 0.7,
}
PREFIX = "gemini-"
ENGINES = {
    "chat": ["pro", "1.0-pro-001", "1.0-pro-latest", "1.5-pro-latest"],
}

ALLOWED_TOOLS = [PREFIX + value for value in ENGINES["chat"]]


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text using the Gemini model's tokenizer."""
    return genai.count_message_tokens(prompt=text)


def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""

    api_key = kwargs["api_keys"]["gemini"]
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]

    if tool not in ALLOWED_TOOLS:
        return (
            f"Model {tool} is not in the list of supported models.",
            None,
            None,
            None,
        )

    candidate_count = kwargs.get(
        "candidate_count", DEFAULT_GEMINI_SETTINGS["candidate_count"]
    )
    stop_sequences = kwargs.get(
        "stop_sequences", DEFAULT_GEMINI_SETTINGS["stop_sequences"]
    )
    max_output_tokens = kwargs.get(
        "max_output_tokens", DEFAULT_GEMINI_SETTINGS["max_output_tokens"]
    )
    temperature = kwargs.get("temperature", DEFAULT_GEMINI_SETTINGS["temperature"])

    counter_callback = kwargs.get("counter_callback", None)

    genai.configure(api_key=api_key)
    engine = genai.GenerativeModel(tool)

    try:
        response = engine.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=candidate_count,
                stop_sequences=stop_sequences,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            ),
        )

        # Ensure response has a .text attribute
        response_text = getattr(response, "text", None)

    except Exception as e:
        return f"An error occurred: {str(e)}", None, None, None

    return response.text, prompt, None, counter_callback
