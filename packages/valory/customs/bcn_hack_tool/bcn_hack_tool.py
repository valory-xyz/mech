# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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

"""Contains the Barcelona off-site's hack tool."""

from typing import Optional, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

MODEL = "gpt-4"
PROMPT_TEMPLATE = "Tell me a short joke about {topic}"


def run_langchain_example(topic: str, openai_api_key: str) -> Tuple[str, str]:
    """Run the langchain example and return the result."""
    model = ChatOpenAI(model=MODEL, openai_api_key=openai_api_key)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    return chain.invoke({"topic": topic}), str(prompt)


def run(**kwargs) -> Tuple[Optional[str], Optional[str], None, None]:
    """Run the langchain example."""
    topic = kwargs.pop("topic")
    openai_api_key = kwargs.pop("openai_api_key")
    response, prompt = run_langchain_example(topic, openai_api_key)
    # the expected output is: response, prompt, irrelevant, irrelevant
    return response, prompt, None, None
