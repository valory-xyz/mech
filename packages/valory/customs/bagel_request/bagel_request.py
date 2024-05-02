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

import bagelml

PREFIX = "bagel-"
ENGINES = {
    "search": ["bagel-search"],
    "write": ["bagel-write"],
}
ALLOWED_TOOLS = [PREFIX + value for values in ENGINES.values() for value in values]

DEFAULT_BAGEL_SETTINGS = {
    "top_k": 10,
}


def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""
    api_key = kwargs["api_keys"]["bagel"]
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]
    top_k = kwargs.get("top_k", DEFAULT_BAGEL_SETTINGS["top_k"])

    if tool not in ALLOWED_TOOLS:
        return (
            f"Tool {tool} is not in the list of supported tools.",
            None,
            None,
            None,
        )
    
    if not api_key:
        return (
            "Missing Bagel API key.",
            None,
            None,
            None,
        )

    client = bagelml.Client(api_key=api_key)

    engine = tool.replace(PREFIX, "")

    if engine == "search":
        # Get or create a cluster
        cluster = client.get_or_create_cluster(name="my-cluster", embedding_model="bagel-text")

        # Search the cluster for documents related to the prompt
        response = cluster.find(query_texts=[prompt], n_results=top_k)
        documents, distances, metadatas = map(lambda l: list([item for sublist in l for item in sublist]),
                                              (response['documents'], response['distances'], response['metadatas']))

        return {
            "documents": documents,
            "distances": distances,
            "metadatas": metadatas
        }, None, None, None
    elif engine == "write":
        # Get or create a cluster
        cluster = client.get_or_create_cluster(name="my-cluster", embedding_model="bagel-text")

        # Add documents to the cluster
        cluster.add(
            documents=[prompt],
            metadatas=[{"source": "user_input"}],
            ids=[f"doc_{len(cluster)}"]
        )

        return "Document added successfully.", None, None, None
    else:
        return (
            f"Unsupported engine: {engine}",
            None,
            None,
            None,
        )
