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

"""This module accepts a GraphQL endpoint, executes queries based on a given description, and explains the response in natural language"""

from typing import Any, Dict, Optional, Tuple
import os
from openai import OpenAI
import json
import requests

client: Optional[OpenAI] = None


# Analyze data and generate response using OpenAI
def analyse_data_and_generate_response(description, query, data):
    return f"""
    
    You're a GraphQL query response analyzer. You will be provided with context about the data served by the endpoint, as well as the executed query, to give you a better understanding. Based on this, you are expected to return a short bullet point description of the response in natural language.

    Description: 
    
    {json.dumps(description)}
    
    Query: 
    
    {json.dumps(query)}

    Reponse:

    {json.dumps(data)}

    """


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


def fetch_data_from_indexer(endpoint, query):
    response = requests.post(endpoint, json={"query": query})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to fetch schema: {response.status_code}, {response.text}"
        )


def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""
    with OpenAIClientManager(kwargs["api_keys"]["openai"]):
        max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
        endpoint = kwargs.get("endpoint")
        description = kwargs.get("description")
        tool = kwargs["tool"]
        query = kwargs["query"]
        requested_data = fetch_data_from_indexer(endpoint, query)
        engine = tool.replace(PREFIX, "")
        messages = [
            {
                "role": "user",
                "content": analyse_data_and_generate_response(
                    description, query, requested_data
                ),
            },
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
        return response.choices[0].message.content, None, None, None
