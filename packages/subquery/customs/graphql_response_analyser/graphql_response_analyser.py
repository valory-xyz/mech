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
import os
from openai import OpenAI
import json
import requests

client: Optional[OpenAI] = None


def generate_graphql_query(user_request, schema, description, examples):
    return (
        f"""
    You are a GraphQL query generator. Based on the following GraphQL schema and the user's natural language request, generate a valid GraphQL query.

    GraphQL Project Description: "{description}"

    User Request: "{user_request}"

    GraphQL Schema: {json.dumps(schema)}

    Example Queries:

    """
        + examples
        + """

    GraphQL Query:

"""
    )


# Analyze data and generate response using OpenAI
def analyze_data_and_generate_response(data):
    return f"""
    
    Once the query you have given was executed, the following data was fetched:

    JSON Data: {json.dumps(data)}

    Based on the provided context, please generate a bullet-pointed summary in a machine-readable JSON format. The JSON structure should have an array object named 'analysis_result,' with each analytical conclusion represented as a separate string element within the array.
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


# Fetch the GraphQL schema using introspection query
def fetch_graphql_schema(endpoint):
    introspection_query = """
    {
      __schema {
        types {
          name
          fields {
            name
            type {
              kind
              name
            }
          }
        }
      }
    }
    """
    response = requests.post(endpoint, json={"query": introspection_query})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to fetch schema: {response.status_code}, {response.text}"
        )


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
        request = kwargs.get("request")
        examples = kwargs.get("examples")
        tool = kwargs["tool"]
        schema = fetch_graphql_schema(endpoint)
        prompt = generate_graphql_query(request, schema, description, examples)
        if tool not in ALLOWED_TOOLS:
            return (
                f"Tool {tool} is not in the list of supported tools.",
                None,
                None,
                None,
            )
        engine = tool.replace(PREFIX, "")
        messages = [
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
        query_to_be_used = response.choices[0].message.content
        print(query_to_be_used)
        requested_data = fetch_data_from_indexer(endpoint, query_to_be_used)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "user", "content": query_to_be_used},
            {
                "role": "user",
                "content": analyze_data_and_generate_response(requested_data),
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
        return response.choices[0].message.content, prompt, None, None