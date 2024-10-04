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
import functools
import json
import random
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import anthropic
import googleapiclient
import openai
import requests
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from openai import OpenAI
from pydantic import BaseModel
from tiktoken import encoding_for_model


NEWSAPI_TOP_HEADLINES_URL = "https://newsapi.org/v2/top-headlines"
NEWSAPI_DEFAULT_NEWS_SOURCES = [
    "bbc-news",
    "bbc-sport",
    "abc-news",
    "cnn",
    "google-news",
    "reuters",
    "usa-today",
    "breitbart-news",
    "the-verge",
    "techradar",
]

OMEN_SUBGRAPH_URL = "https://gateway-arbitrum.network.thegraph.com/api/{subgraph_api_key}/subgraphs/id/9fUVQpFwzpdWS9bq5WkAnmKbNNcoBwatMR4yZq81pbbz"
HTTP_OK = 200
MAX_ARTICLES = 40
MAX_LATEST_QUESTIONS = 40

FPMM_CREATORS = ["0x89c5cc945dd550bcffb72fe42bff002429f46fec", "0xffc8029154ecd55abed15bd428ba596e7d23f557"]
FPMMS_QUERY = """
    query fpmms_query($creator_in: [Bytes!], $first: Int) {
      fixedProductMarketMakers(
        where: {creator_in: $creator_in}
        orderBy: creationTimestamp
        orderDirection: desc
        first: $first
      ) {
        question {
          title
        }
      }
    }
    """

DEFAULT_TOPICS = [
    "business",
    "cryptocurrency",
    "politics",
    "science",
    "technology",
    "trending",
    "fashion",
    "social",
    "health",
    "sustainability",
    "internet",
    "travel",
    "food",
    "pets",
    "animals",
    "curiosities",
    "music",
    "economy",
    "arts",
    "entertainment",
    "weather",
    "sports",
    "finance",
    "international",
]

PROPOSE_QUESTION_PROMPT = """You are provided a numbered list of recent news article
    snippets under ARTICLES. Your task is to formulate one novel prediction market question
    with clear, objective outcomes. The question must satisfy all the following criteria:
    - It must be of public interest.
    - It must be substantially different from the ones in EXISTING_QUESTIONS.
    - It must be related to an event happening before EVENT_DAY or on EVENT_DAY.
    - It must not encourage unethical behavior or violence.
    - It should follow a structure similar to these: "Will EVENT occur on or before EVENT_DAY?", "Will EVENT occur by EVENT_DAY?", etc.
    - It must not include unmeasurable statements like "significant increase".
    - It must not reference matches, sport events or any other event that do not occur on EVENT_DAY.
    - The answer must be 'yes' or 'no.
    - The answer must be verified using publicly available sources or news media.
    - The answer must not be an opinion.
    - The answer must be known after EVENT_DAY.

    EXISTING_QUESTIONS
    {latest_questions}

    EVENT_DAY
    {event_day}

    INPUT
    {articles}

    TOPICS
    {topics}
    """

client: Optional[OpenAI] = None


MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]


class LLMQuestionProposalSchema(BaseModel):
    question: str
    topic: str
    article_id: int


def with_key_rotation(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
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
            except googleapiclient.errors.HttpError as e:
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


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0.7,
}
DEFAULT_ENGINES = {
    "propose-question": "gpt-4o-2024-08-06"
}
ALLOWED_TOOLS = ["propose-question"]


def format_utc_timestamp(utc_timestamp: int) -> str:
    dt = datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)
    formatted_date = dt.strftime("%d %B %Y")
    return formatted_date


def gather_articles(news_sources: List[str], newsapi_api_key: str) -> Optional[List[Dict[str, Any]]]:
    """Gather news from NewsAPI (top-headlines endpoint)"""
    headers = {"X-Api-Key": newsapi_api_key}
    parameters = {
        "sources": ",".join(news_sources),
        "pageSize": "100",  # TODO: pagination
    }
    url = NEWSAPI_TOP_HEADLINES_URL
    response = requests.get(
        url=url,
        headers=headers,
        params=parameters,
        timeout=60,
    )
    if response.status_code != HTTP_OK:
        print(
            f"Could not retrieve response from {url}."
            f"Received status code {response.status_code}."
            f"{response}"
        )
        return None

    response_data = json.loads(response.content.decode("utf-8"))
    articles = response_data["articles"]
    return articles


def gather_latest_questions(subgraph_api_key: str) -> List[str]:
    transport = RequestsHTTPTransport(url=OMEN_SUBGRAPH_URL.format(subgraph_api_key=subgraph_api_key))
    client = Client(transport=transport, fetch_schema_from_transport=True)
    variables = {
        "creator_in": FPMM_CREATORS,
        "first": MAX_LATEST_QUESTIONS,
    }
    response = client.execute(gql(FPMMS_QUERY), variable_values=variables)
    items = response.get("fixedProductMarketMakers", [])
    output = [q["question"]["title"] for q in items]
    return output



# TODO
#@with_key_rotation
def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""

    # Verify input
    tool = kwargs.get("tool")
    if not tool or tool not in ALLOWED_TOOLS:
        return (
            f"Error: Tool {tool} is not in the list of supported tools.",
            None,
            None,
            None,
        )

    resolution_time = kwargs.get("resolution_time")
    if resolution_time is None:
        return (
            "Error: 'resolution_time' is not defined.",
            None,
            None,
            None,
        )
    
    # Gather latest opened questions from input or from TheGraph
    latest_questions = kwargs.get("latest_questions")
    if latest_questions is None:
        latest_questions = gather_latest_questions(kwargs["api_keys"]["subgraph"])

    latest_questions = random.sample(latest_questions, min(MAX_LATEST_QUESTIONS, len(latest_questions)))
    latest_questions_string = "\n".join(latest_questions)

    # Gather recent news articles from NewsAPI
    news_sources = kwargs.get("news_sources", NEWSAPI_DEFAULT_NEWS_SOURCES)
    articles = gather_articles(news_sources, kwargs["api_keys"]["newsapi"])

    if articles is None:
        return (
            "Error: Failed to retrieve articles from NewsAPI.",
            None,
            None,
            None,
        )

    articles = random.sample(articles, min(MAX_ARTICLES, len(articles)))

    articles_string = ""
    for i, article in enumerate(articles, start=0):
        articles_string += f"{i} - {article['title']} ({article['publishedAt']}): {article['content']}\n"


    # Define topics
    topics = kwargs.get("topics", DEFAULT_TOPICS)
    topics_string = ", ".join(topics)

    # Call LLM
    with OpenAIClientManager(kwargs["api_keys"]["openai"]):
        max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
        counter_callback = kwargs.get("counter_callback", None)
        engine = kwargs.get("engine", DEFAULT_ENGINES.get(tool))

        prompt_values = {
            "articles": articles_string,
            "topics": topics_string,
            "event_day": format_utc_timestamp(resolution_time),
            "latest_questions": latest_questions_string,
        }

        prompt = PROPOSE_QUESTION_PROMPT.format(**prompt_values)

        moderation_result = client.moderations.create(input=prompt)
        if moderation_result.results[0].flagged:
            return (
                "Error: Moderation flagged the prompt as in violation of terms.",
                None,
                None,
                None,
            )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
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
            response_format={
                'type': 'json_schema',
                'json_schema': 
                    {
                        "name":"whocares",
                        "schema": LLMQuestionProposalSchema.model_json_schema()
                    }
                },
        )

        response = json.loads(response.choices[0].message.content)
        article_id = response["article_id"]
        del response["article_id"]
        response["article"] = articles[article_id]
        return response, prompt, None, None


if __name__ == "__main__":
    import os
    from packages.valory.skills.task_execution.utils.apis import KeyChain
    
    tool = "propose-question"
    keys = KeyChain(
        {
            "openai": [os.getenv("OPENAI_API_KEY")],
            "newsapi": [os.getenv("NEWSAPI_API_KEY")],
            "subgraph": [os.getenv("SUBGRAPH_API_KEY")]
        }
    )

    my_kwargs = dict(
        tool=tool,
        api_keys=keys,
        # news_sources=news_sources,  # Use default value
        # topics=topics,              # Use default value
        resolution_time=1728518400
    )

    print("================================")
    print(f"Start request {tool=}")
    tool_output = run(**my_kwargs)
    print("================================")
    print(f"Output of {tool=}")
    print(tool_output[0])
