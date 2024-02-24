# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 @seichris
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

class PerplexityClientManager:
    """Client context manager for Perplexity AI."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "accept": "application/json",
            "content-type": "application/json"
        }

    def __enter__(self):
        # No action needed on enter
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # No action needed on exit
        pass

    def post(self, url: str, payload: Dict[str, Any]):
        response = requests.post(url, json=payload, headers=self.headers)
        return response.json()

PREFIX = "perplexity-"
ENGINES = {
    "chat": ["pplx-7b-chat", "pplx-70b-chat", "llama-2-70b-chat"],
    "online": ["pplx-7b-online", "pplx-70b-online"],
    "instruct": ["codellama-34b-instruct", "codellama-70b-instruct", "mistral-7b-instruct", "mixtral-8x7b-instruct"],
}

ALLOWED_TOOLS = [PREFIX + value for values in ENGINES.values() for value in values]

def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""
    api_key = kwargs["api_keys"]["perplexity"]
    prompt = kwargs["prompt"]
    tool = kwargs["tool"]
    counter_callback = kwargs.get("counter_callback", None)

    # Check if the tool is allowed
    if tool not in ALLOWED_TOOLS:
        return f"Tool {tool} is not supported.", None, None, None

    model = tool.replace(PREFIX, "")

    with PerplexityClientManager(api_key) as client:
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Be precise and concise."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:
            response = client.post(url, payload)
            result_str = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            prompt_used = prompt
            generated_tx = None  # Replace with actual transaction if applicable
            return result_str, prompt_used, generated_tx, counter_callback

        except Exception as e:
            # Handle any exceptions that occur and return an error message
            error_message = str(e)
            return error_message, None, None, counter_callback