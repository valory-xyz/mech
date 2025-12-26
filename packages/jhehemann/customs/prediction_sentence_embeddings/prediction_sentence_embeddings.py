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

"""This module implements a Mech tool for binary predictions."""

import functools
import json
import os
import re
import time
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import groupby
from operator import itemgetter
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import anthropic
import googleapiclient
import openai
import requests
import spacy
import spacy.util
import tiktoken
from bs4 import BeautifulSoup, NavigableString, Tag
from dateutil import parser
from googleapiclient.discovery import build
from requests import Session
from spacy.tokens import Doc
from tiktoken import encoding_for_model
from tqdm import tqdm


MechResponseWithKeys = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]
MaxCostResponse = float


ADDITIONAL_INFO_ENGINE = "gpt-3.5-turbo"


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


class LLMClientManager:
    """Client context manager for LLMs."""

    def __init__(self, api_keys: List, model: str, embedding_provider: str):
        """Initializes with API keys, model, and embedding provider. Sets the LLM provider based on the model."""
        self.api_keys = api_keys
        self.embedding_provider = embedding_provider
        if "gpt" in model:
            self.llm_provider = "openai"
        elif "claude" in model:
            self.llm_provider = "anthropic"
        else:
            self.llm_provider = "openrouter"

    def __enter__(self) -> List:
        """Initializes and returns LLM and embedding clients."""
        clients: List = []
        global client
        if self.llm_provider and client is None:
            client = LLMClient(self.api_keys, self.llm_provider)
            clients.append(client)
        global client_embedding
        if self.embedding_provider and client_embedding is None:
            client_embedding = LLMClient(self.api_keys, self.embedding_provider)
            clients.append(client_embedding)
        return clients

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Closes the LLM client"""
        global client
        if client is not None:
            client.client.close()
            client = None


# pylint: disable=too-few-public-methods
class Usage:
    """Usage class."""

    def __init__(
        self,
        prompt_tokens: Optional[Any] = None,
        completion_tokens: Optional[Any] = None,
    ):
        """Initializes with prompt tokens and completion tokens."""
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


# pylint: disable=too-few-public-methods
class LLMResponse:
    """Response class."""

    def __init__(
        self, content: Optional[str] = None, usage: Optional[Usage] = None
    ):  # pylint: disable=unused-argument
        """Initializes with content and usage class."""
        self.content = content
        self.usage = Usage()


class LLMClient:
    """Client for LLMs."""

    def __init__(self, api_keys: List, llm_provider: str):
        """Initializes with API keys, model, and embedding provider. Sets the LLM provider based on the model."""
        self.api_keys = api_keys
        self.llm_provider = llm_provider
        if self.llm_provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=self.api_keys["anthropic"])  # type: ignore
        if self.llm_provider == "openai":
            self.client = openai.OpenAI(api_key=self.api_keys["openai"])  # type: ignore
        if self.llm_provider == "openrouter":
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_keys["openrouter"],  # type: ignore
            )

    def completions(
        self,
        model: str,
        messages: List = [],  # noqa: B006 pylint:disable=dangerous-default-value
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Any = None,
        max_tokens: Optional[float] = None,
    ) -> Optional[LLMResponse]:
        """Generate a completion from the specified LLM provider using the given model and messages."""
        if self.llm_provider == "anthropic":
            # anthropic can't take system prompt in messages
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "system":
                    system_prompt = messages[i]["content"]
                    del messages[i]

            response_provider = self.client.messages.create(  # pylint: disable=no-member
                model=model,
                messages=messages,
                system=system_prompt,  # pylint: disable=possibly-used-before-assignment
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = LLMResponse()
            response.content = response_provider.content[0].text
            response.usage.prompt_tokens = response_provider.usage.input_tokens
            response.usage.completion_tokens = response_provider.usage.output_tokens
            return response

        if self.llm_provider == "openai":
            response_provider = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                timeout=150,
                stop=None,
            )
            response = LLMResponse()
            response.content = response_provider.choices[0].message.content
            response.usage.prompt_tokens = response_provider.usage.prompt_tokens
            response.usage.completion_tokens = response_provider.usage.completion_tokens
            return response

        if self.llm_provider == "openrouter":
            # TODO investigate the transform parameter https://openrouter.ai/docs#transforms
            # transform = [] # to desactivate prompt compression noqa: E800
            response_provider = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                timeout=150,
                stop=None,
            )
            response = LLMResponse()
            response.content = response_provider.choices[0].message.content
            response.usage.prompt_tokens = response_provider.usage.prompt_tokens
            response.usage.completion_tokens = response_provider.usage.completion_tokens
            return response

        return None

    def embeddings(self, model: Any, input_: Any) -> Any:
        """Returns the embeddings response"""
        if self.llm_provider in ("openai", "openrouter"):
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=input_,
            )
            return response

        print("Only OpenAI embeddings supported currently.")
        return None


client: Optional[LLMClient] = None
client_embedding: Optional[LLMClient] = None


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


NUM_URLS_EXTRACT = 5
MAX_TOTAL_TOKENS_CHAT_COMPLETION = 4000  # Set the limit for cost efficiency
WORDS_PER_TOKEN_FACTOR = 0.75
EMBEDDING_MODEL = "text-embedding-3-large"
MAX_EMBEDDING_TOKEN_INPUT = 8192
EMBEDDING_SIZE = 1536

# number of retries and delay for completion
COMPLETION_RETRIES = 3
COMPLETION_DELAY = 2

LLM_SETTINGS = {
    "gpt-3.5-turbo-0125": {
        "default_max_tokens": 500,
        "limit_max_tokens": 4096,
        "temperature": 0,
    },
    "gpt-4-0125-preview": {
        "default_max_tokens": 500,
        "limit_max_tokens": 8192,
        "temperature": 0,
    },
    "gpt-4o-2024-08-06": {
        "default_max_tokens": 500,
        "limit_max_tokens": 4096,
        "temperature": 0,
    },
}
ALLOWED_TOOLS = [
    "prediction-sentence-embedding-conservative",
    "prediction-sentence-embedding-bold",
]
ALLOWED_MODELS = list(LLM_SETTINGS.keys())
TOOL_TO_ENGINE = {
    "prediction-sentence-embedding-conservative": "gpt-3.5-turbo-0125",
    "prediction-sentence-embedding-bold": "gpt-4o-2024-08-06",
}

PREDICTION_PROMPT = """
INTRODUCTION:
You are a Large Language Model (LLM) within a multi-agent system. Your primary task is to accurately estimate the probabilities for the outcome of a 'market question', \
found in 'USER_PROMPT'. The market question is part of a prediction market, where users can place bets on the outcomes of market questions and earn rewards if the selected outcome occurrs. The 'market question' \
in this scenario has only two possible outcomes: `Yes` or `No`. Each market has a closing date at which the outcome is evaluated. This date is typically stated within the market question.  \
The closing date is considered to be 23:59:59 of the date provided in the market question. If the event specified in the market question has not occurred before the closing date, the market question's outcome is `No`. \
If the event has happened before the closing date, the market question's outcome is `Yes`. You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION", which is \
sourced from a Google search engine query performed a few seconds ago and is meant to assist you in your probability estimation. You must adhere to the following 'INSTRUCTIONS'.


INSTRUCTIONS:
* Examine the user's input labeled 'USER_PROMPT'. Focus on the part enclosed in double quotes, which contains the 'market question'.
* If the 'market question' implies more than two outcomes, output the response "Error" and halt further processing.
* When the current time {timestamp} has passed the closing date of the market and the event specified in the market question has not happened, the market question's outcome is `No` and the user who placed a bet on `No` will receive a reward.
* When the current time {timestamp} has passed the closing date of the market and the event has happened before, the market question's final outcome is `Yes` and the user who placed a bet on `yes` will receive a reward.
* Consider the prediction market with the market question, the closing date and the outcomes in an isolated context that has no influence on the protagonists that are involved in the event in the real world, specified in the market question. The closing date is always arbitrarily set by the market creator and has no influence on the real world. So it is likely that the protagonists of the event in the real world are not even aware of the prediction market and do not care about the market's closing date.
* The probability estimations of the market question outcomes must be as accurate as possible, as an inaccurate estimation will lead to financial loss for the user.
* Utilize your training data and the information provided under "ADDITIONAL_INFORMATION" to generate probability estimations for the outcomes of the 'market question'.
* Examine the itemized list under "ADDITIONAL_INFORMATION" thoroughly and use all the relevant information for your probability estimation. This data is sourced from a Google search engine query done a few seconds ago.
* Use any relevant item in "ADDITIONAL_INFORMATION" in addition to your training data to make the probability estimation. You can assume that you have been provided with the most current and relevant information available on the internet. Still pay close attention on the release and modification timestamps provided in parentheses right before each information item. Some information might be outdated and not relevant anymore.
* More recent information indicated by the timestamps provided in parentheses right before each information item overrides older information within ADDITIONAL_INFORMATION and holds more weight for your probability estimation.
* If there exist contradicting information, evaluate the release and modification dates of those information and prioritize the information that is more recent and adjust your confidence in the probability estimation accordingly.
* Even if not all information might not be released today, you can assume that there haven't been publicly available updates in the meantime except for those inside ADDITIONAL_INFORMATION.
* If the information in "ADDITIONAL_INFORMATION" indicate without a doubt that the event has already happened, it is very likely that the outcome of the market question will be `Yes`.
* If the information in "ADDITIONAL_INFORMATION" indicate that the event will happen after the closing date, it is very likely that the outcome of the market question will be `No`.
* The closer the current time `{timestamp}` is to the closing time the higher the likelyhood that the outcome of the market question will be `No`, if recent information do not clearly indicate that the event will occur before the closing date.
* If there exist recent information indicating that the event will happen after the closing date, it is very likely that the outcome of the market question will be `No`.
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.


USER_PROMPT:
```
{user_prompt}
```

ADDITIONAL_INFORMATION:
```
{additional_information}
```

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain four fields: "p_yes", "p_no", "confidence", and "info_utility", each ranging from 0 to 1.
   - "p_yes": Probability that the market question's outcome will be `Yes`.
   - "p_no": Probability that the market questions outcome will be `No`.
   - "confidence": Indicating the confidence in the estimated probabilities you provided ranging from 0 (lowest confidence) to 1 (maximum confidence). Confidence can be calculated based on the quality and quantity of data used for the estimation.
   - "info_utility": Utility of the information provided in "ADDITIONAL_INFORMATION" to help you make the probability estimation ranging from 0 (lowest utility) to 1 (maximum utility).
* The sum of "p_yes" and "p_no" must equal 1.
* Output only the JSON object in your response. Do not include any other contents in your response.
"""

URL_QUERY_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to formulate search engine queries based on \
a user's 'event question', which specifies an event and any accompanying conditions. The 'event question' allows \
only two outcomes: the event will either occur or not, given the conditions. Find the 'event question' under 'USER_PROMPT' \
and adhere to the 'INSTRUCTIONS'.

INSTRUCTIONS:
* Carefully read the 'event question' under 'USER_PROMPT', enclosed by triple backticks.
* If the 'event question' has more than two outcomes, respond with "Error" and ignore further instructions.
* Create a list of 1-3 unique search queries likely to yield relevant and contemporary information for assessing the event's likelihood under the given conditions.
* Each query must be unique, and they should not overlap or yield the same set of results.
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.

USER_PROMPT:
```
{event_question}
```

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain two fields: "queries", and "urls".
   - "queries": A 1-5 item array of the generated search engine queries.
* Include only the JSON object in your output.
"""

# Global constants for possible attribute names for release and update dates
RELEASE_DATE_NAMES = [
    "date",
    "pubdate",
    "publishdate",
    "OriginalPublicationDate",
    "article:published_time",
    "sailthru.date",
    "article.published",
    "published-date",
    "og:published_time",
    "publication_date",
    "publishedDate",
    "dc.date",
    "DC.date",
    "article:published",
    "article_date_original",
    "cXenseParse:recs:publishtime",
    "DATE_PUBLISHED",
    "pub-date",
    "pub_date",
    "datePublished",
    "date_published",
    "time_published",
    "article:published_date",
    "parsely-pub-date",
    "publish-date",
    "pubdatetime",
    "published_time",
    "publishedtime",
    "article_date",
    "created_date",
    "published_at",
    "lastPublishedDate",
    "og:published_time",
    "og:release_date",
    "article:published_time",
    "og:publication_date",
    "og:pubdate",
    "article:publication_date",
    "product:availability_starts",
    "product:release_date",
    "event:start_date",
    "event:release_date",
    "og:time_published",
    "og:start_date",
    "og:created",
    "og:creation_date",
    "og:launch_date",
    "og:first_published",
    "og:original_publication_date",
    "article:published",
    "article:pub_date",
    "news:published_time",
    "news:publication_date",
    "blog:published_time",
    "blog:publication_date",
    "report:published_time",
    "report:publication_date",
    "webpage:published_time",
    "webpage:publication_date",
    "post:published_time",
    "post:publication_date",
    "item:published_time",
    "item:publication_date",
]

UPDATE_DATE_NAMES = [
    "lastmod",
    "lastmodified",
    "last-modified",
    "updated",
    "dateModified",
    "article:modified_time",
    "modified_date",
    "article:modified",
    "og:updated_time",
    "mod_date",
    "modifiedDate",
    "lastModifiedDate",
    "lastUpdate",
    "last_updated",
    "LastUpdated",
    "UpdateDate",
    "updated_date",
    "revision_date",
    "sentry:revision",
    "article:modified_date",
    "date_updated",
    "time_updated",
    "lastUpdatedDate",
    "last-update-date",
    "lastupdate",
    "dateLastModified",
    "article:update_time",
    "modified_time",
    "last_modified_date",
    "date_last_modified",
    "og:updated_time",
    "og:modified_time",
    "article:modified_time",
    "og:modification_date",
    "og:mod_time",
    "article:modification_date",
    "product:availability_ends",
    "product:modified_date",
    "event:end_date",
    "event:updated_date",
    "og:time_modified",
    "og:end_date",
    "og:last_modified",
    "og:modification_date",
    "og:revision_date",
    "og:last_updated",
    "og:most_recent_update",
    "article:updated",
    "article:mod_date",
    "news:updated_time",
    "news:modification_date",
    "blog:updated_time",
    "blog:modification_date",
    "report:updated_time",
    "report:modification_date",
    "webpage:updated_time",
    "webpage:modification_date",
    "post:updated_time",
    "post:modification_date",
    "item:updated_time",
    "item:modification_date",
]

# Global constant for HTML tags to remove
HTML_TAGS_TO_REMOVE = [
    "script",
    "style",
    "header",
    "footer",
    "aside",
    "nav",
    "form",
    "button",
    "iframe",
    "input",
    "textarea",
    "select",
    "option",
    "label",
    "fieldset",
    "legend",
    "img",
    "audio",
    "video",
    "source",
    "track",
    "canvas",
    "svg",
    "object",
    "param",
    "embed",
    "link",
]


def search_google(query: str, api_key: str, engine: str, num: int = 3) -> List[str]:
    """Search Google using a custom search engine."""
    service = build("customsearch", "v1", developerKey=api_key)
    search = (
        service.cse()  # pylint: disable=no-member
        .list(
            q=query,
            cx=engine,
            num=num,
        )
        .execute()
    )
    return [result["link"] for result in search["items"]]


def extract_json_string(text: str) -> str:
    """Extract's the json string"""
    # This regex looks for triple backticks, captures everything in between until it finds another set of triple backticks.
    pattern = r"(\{[^}]*\})"
    matches = re.findall(pattern, text)
    return matches[0].replace("json", "")


def download_spacy_model(model_name: str) -> None:
    """Downloads the specified spaCy language model if it is not already installed."""
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("spacy model_name must be a non-empty string")
    if not spacy.util.is_package(model_name):
        spacy.cli.download(model_name)
    else:
        print(f"{model_name} is already installed.")


def extract_event_date(doc_question: Doc) -> Optional[str]:
    """
    Extracts the event date from the event question if present.

    :param doc_question: Document text as a spaCy Doc object.
    :type doc_question: Doc

    :returns: The event date in year-month-day format if present, otherwise None.
    :rtype: str or None
    """

    event_date_ymd = None

    # Extract the date from the event question if present
    for ent in doc_question.ents:
        if ent.label_ == "DATE":
            event_date_ymd = standardize_date(ent.text)

    # If event date not formatted as YMD or not found, return None
    try:
        if event_date_ymd is not None:
            datetime.strptime(event_date_ymd, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    return event_date_ymd


def get_max_tokens_for_additional_information(
    max_compl_tokens: int,
    prompt: str,
    enc: tiktoken.Encoding,
    safety_factor: float = 1.05,
) -> int:
    """
    Calculates the estimated maximum number of tokens that can be consumed by the additional information string.

    :param max_compl_tokens: The maximum number of chat completion output tokens.
    :type max_compl_tokens: int
    :param prompt: The user prompt containing the event question.
    :type prompt: str
    :param enc: The tiktoken encoding to be used.
    :type enc: tiktoken.Encoding
    :param safety_factor: The safety factor to be used for prompt variations and message headers.
    :type safety_factor: float

    :returns: The estimated number of tokens that can be consumed by the additional information string.
    :rtype: int
    """

    # Encode the strings into tokens
    user_prompt_enc = enc.encode(prompt)
    prediction_prompt_enc = enc.encode(PREDICTION_PROMPT)

    # Calculate token sum of thus far allocated tokens for the final prediction prompt
    token_sum = len(user_prompt_enc) + len(prediction_prompt_enc) + max_compl_tokens
    token_sum_safety = token_sum * safety_factor

    return int(MAX_TOTAL_TOKENS_CHAT_COMPLETION - token_sum_safety)


def truncate_additional_information(
    additional_informations: str,
    max_add_tokens: int,
    enc: tiktoken.Encoding,
) -> str:
    """
    Truncates additional information string to a specified number of tokens using tiktoken encoding.

    :param additional_informations: The additional information string to be truncated.
    :type additional_informations: str
    :param max_add_tokens: The maximum number of tokens allowed for the additional information string.
    :type max_add_tokens: int
    :param enc: The tiktoken encoding to be used.
    :type enc: tiktoken.Encoding

    :returns: The truncated additional information string.
    :rtype: str
    """

    # Encode the string into tokens
    add_enc = enc.encode(additional_informations)
    len_add_enc = len(add_enc)

    # Truncate additional information string if token sum exceeds maximum allowed
    if len_add_enc <= max_add_tokens:
        return additional_informations

    add_trunc_enc = add_enc[: -int(len_add_enc - max_add_tokens)]
    return enc.decode(add_trunc_enc)


def get_urls_from_queries(
    queries: List[str], api_key: str, engine: str, num: int = 3
) -> List[str]:
    """
    Fetch unique URLs from search engine queries, limiting the number of URLs per query.

    :param queries: List of search engine queries.
    :type queries: List[str]
    :param api_key: API key for the search engine.
    :type api_key: str
    :param engine: Custom Google search engine ID.
    :type engine: str
    :param num: Number of returned URLs per query (optional, defaults to 3).
    :type num: int

    :raises ValueError: If the number of URLs per query exceeds the maximum allowed.

    :returns: Unique list of URLs, omitting PDF and download-related URLs.
    :rtype: List[str]
    """

    results = set()
    max_num_fetch = 10

    if num > max_num_fetch:
        raise ValueError(f"The maximum number of URLs per query is {max_num_fetch}.")

    for query in queries:
        fetched_urls = search_google(
            query=query,
            api_key=api_key,
            engine=engine,
            num=max_num_fetch,  # Limit the number of returned URLs per query
        )

        # Add only unique URLs up to 'num' per query, omitting PDF and 'download' URLs
        count = 0
        for url in fetched_urls:
            if url not in results and not url.endswith(".pdf"):
                results.add(url)
                count += 1
                if count >= num:
                    break
    print("get_urls_from_queries result:")
    for url in results:
        print(url)

    return list(results)


def get_urls_from_queries_serper(
    queries: List[str], api_key: str, num: int = 3
) -> List[str]:
    """Get URLs from search engine queries using Serper API."""
    urls: List[str] = []
    for query in queries:
        try:
            url = "https://google.serper.dev/search"
            payload = json.dumps({"q": query})
            headers = {
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            }
            response = requests.request(
                "POST", url, headers=headers, data=payload, timeout=30
            )
            response.raise_for_status()
            data = response.json()
            organic = data.get("organic", [])
            urls.extend(item["link"] for item in organic[:num])
        except Exception as e:
            print(f"Error fetching URLs for query '{query}': {e}")
    
    print("get_urls_from_queries_serper result:")
    for url in urls:
        print(url)
    
    return list(set(urls))


def standardize_date(date_text: str) -> Optional[str]:
    """
    Standardizes a given date string to the format 'YYYY-MM-DD' or 'MM-DD' if possible.

    :param date_text: The date string to be standardized.
    :type date_text: str

    :returns: The standardized date string if possible, otherwise None.
    :rtype: str or None
    """

    try:
        # Compile regex patterns for month and day
        month_regex = re.compile(
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
            re.IGNORECASE,
        )
        day_regex = re.compile(r"\b\d{1,2}\b")

        # Parse date_text using dateutil parser
        parsed_date = parser.parse(date_text)

        # Check if year, month, and day are in the original date_text
        month_exists = month_regex.search(date_text) is not None
        day_exists = day_regex.search(date_text) is not None
        year_exists = str(parsed_date.year) in date_text

        # Format the parsed date accordingly
        if year_exists and month_exists and day_exists:
            return parsed_date.strftime("%Y-%m-%d")
        if month_exists and day_exists:
            return parsed_date.strftime("%m-%d")
        return None
    except Exception:
        return None


def get_context_around_isolated_event_date(
    doc_text: Doc,
    event_date_ymd: str,
    len_sentence_threshold: int,
    max_context: int = 50,
) -> List:
    """
    Extract sentences around isolated dates within the text.

    :param doc_text: Document text as a spaCy Doc object.
    :type doc_text: Doc
    :param event_date_ymd: Event date in year-month-day format.
    :type event_date_ymd: str
    :param len_sentence_threshold: Minimum number of words required for a sentence to be considered contextful.
    :type len_sentence_threshold: int
    :param max_context: Maximum number of words to include in the context (optional, defaults to 50).
    :type max_context: int

    :raises ValueError: If the maximum context is less than the threshold or greater than 100.

    :returns: List of sentences surrounding the target date.
    :rtype: List
    """

    # Check max_context value constraints
    if max_context < len_sentence_threshold:
        raise ValueError(
            f"The maximum number of words must be greater than or equal to the minimum number of words ({len_sentence_threshold}) required for a sentence to be considered contextful."
        )
    if max_context > 100:
        raise ValueError(
            "The maximum number of words must be less than or equal to 300."
        )

    contexts_list = []
    len_doc_text = len(doc_text)

    # Extract the month and day from the event date
    event_date_md = event_date_ymd[5:]

    for ent in doc_text.ents:
        if ent.label_ == "DATE":
            standardized_date = standardize_date(ent.text)
            if standardized_date is None:
                continue

            # Check if the entity matches the target date
            if standardized_date in (event_date_ymd, event_date_md):
                sentence = next(
                    sent
                    for sent in doc_text.sents
                    if sent.start <= ent.start and sent.end >= ent.end
                )

                context_words = len(sentence.text.split())

                # Extend the context if the sentence is too short
                if context_words < len_sentence_threshold:
                    start_token, end_token = sentence.start, sentence.end
                    while context_words < max_context:
                        # Extend the context from the start of the sentence
                        new_start = start_token - 1
                        while (
                            new_start >= 0 and doc_text[new_start].is_sent_start is None
                        ):
                            new_start -= 1
                        if new_start >= 0:
                            context_words += len(
                                doc_text[new_start:start_token].text.split()
                            )
                            start_token = new_start

                        # Break if max_context is reached
                        if context_words >= max_context:
                            break

                        # Extend the context from the end of the sentence
                        new_end = end_token + 1
                        while (
                            new_end < len_doc_text
                            and doc_text[new_end].sent == sentence.sent
                        ):
                            new_end += 1
                        if new_end < len_doc_text:
                            context_words += len(
                                doc_text[end_token:new_end].text.split()
                            )
                            end_token = new_end

                        # Break if max_context is reached
                        if context_words >= max_context:
                            break

                        # Break if max_context cannot be reached
                        if new_end == len_doc_text and start_token <= 0:
                            break

                    context = doc_text[
                        max(0, start_token) : min(len_doc_text, end_token)
                    ].text
                    contexts_list.append(context)

    return contexts_list


def concatenate_short_sentences(sentences: List, len_sentence_threshold: int) -> List:
    """Concatenates consecutive sentences shorter than the given word count threshold."""
    modified_sentences = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        word_count = len(sentence.split())

        # Check if the sentence is shorter than the threshold
        while word_count < len_sentence_threshold:
            i += 1
            # Break the loop if we reach the end of the list
            if i >= len(sentences):
                break
            next_sentence = sentences[i]
            sentence += " " + next_sentence
            word_count += len(next_sentence.split())

        modified_sentences.append(sentence)
        i += 1

    return modified_sentences


def extract_similarity_scores(
    text: str,
    query_emb: Any,
    event_date: str,
    nlp: Any,
    date: str,
) -> List[Tuple[str, float, str]]:
    """
    Extract relevant information from website text based on a given event question.

    :param text: The website text to extract information from.
    :type text: str
    :param query_emb: The query embeddings
    :type query_emb: Any
    :param event_date: Event date in year-day-month format.
    :type event_date: str
    :param nlp: The spaCy NLP model.
    :type nlp: Any
    :param date: The release and modification dates of the website.
    :type date: str

    :returns: List of tuples containing the ex
    :rtype: list of tuple(str, float, str)
    """

    # Constants for sentence length and number thresholds
    len_sentence_threshold = 10
    num_sentences_threshold = 1000
    sentences = []
    # event_date_sentences = []
    seen = set()

    # Truncate text for performance optimization
    text = text[:50000]

    # Apply NLP pipeline to text
    doc_text = nlp(text)

    # Extract unique sentences
    for sent in doc_text.sents:
        sentence_text = sent.text
        if (
            len(sentence_text.split()) >= len_sentence_threshold
            and sentence_text not in seen
        ):
            sentences.append(sentence_text)
            seen.add(sentence_text)

    # flake8: noqa: E800
    ## Temporarily deactivated: News sites with a lot of date occurrences lead to false positives
    ## The embedding model is not advanced enough
    # Extract contextual sentences around event date occurrences within too short sentences
    # if event_date is not None:
    #     event_date_sentences.extend(
    #         get_context_around_isolated_event_date(
    #             doc_text, event_date, len_sentence_threshold, max_context=50
    #         )
    #     )
    # sentences.extend(event_date_sentences)
    # flake8: enable: E800

    if not sentences:
        return []

    # Concatenate short sentences
    sentences = concatenate_short_sentences(sentences, len_sentence_threshold)

    # Limit the number of sentences for performance optimization
    sentences = sentences[:num_sentences_threshold]

    similarities = []

    # Encode sentences using spaCy model
    for _, sentence in enumerate(sentences):
        doc_sentence = nlp(sentence)
        similarity_score = query_emb.similarity(doc_sentence)
        similarities.append(similarity_score)

    # Create tuples and store them in a list
    sentence_similarity_date_tuples = [
        (sentence, similarity, date)
        for sentence, similarity in zip(sentences, similarities)
        if similarity > 0.4
    ]

    return sentence_similarity_date_tuples


def get_date(soup: BeautifulSoup) -> str:
    """
    Retrieves the release and modification dates from the soup object containing the HTML tree.

    :param soup: The BeautifulSoup object for the webpage.
    :type soup: BeautifulSoup

    :returns: A string representing the release and modification dates.
    :rtype: str
    """

    release_date = "unknown"
    modified_date = "unknown"

    # Search for an update or modified date in the meta tags
    for name in UPDATE_DATE_NAMES:
        meta_tag = soup.find("meta", {"name": name}) or soup.find(
            "meta", {"property": name}
        )
        if meta_tag and isinstance(meta_tag, Tag):
            content = meta_tag.get("content", "")
            if isinstance(content, Iterable) and not isinstance(content, (str, bytes)):
                modified_date = " ".join(content)
            else:
                modified_date = str(content)

    # If not found, then look for release or publication date
    for name in RELEASE_DATE_NAMES:
        meta_tag = soup.find("meta", {"name": name}) or soup.find(
            "meta", {"property": name}
        )
        if meta_tag and isinstance(meta_tag, Tag):
            content = meta_tag.get("content", "")
            if isinstance(content, Iterable) and not isinstance(content, (str, bytes)):
                release_date = " ".join(content)
            else:
                release_date = str(content)

    # flake8: noqa: E800
    ## Temporarily deactivated
    # # Fallback to using the first time tag if neither release nor modified dates are found
    # if release_date == "unknown" and modified_date == "unknown":
    #     time_tag = soup.find("time")
    #     if time_tag:
    #         release_date = time_tag.get("datetime", "")
    # flake8: enable: E800

    return f"({release_date}, {modified_date})"


def extract_sentences(
    html: str,
    query_emb: Any,
    event_date: str,
    nlp: Any,
) -> List[Tuple[str, float, str]]:
    """
    Extract relevant information from HTML string.

    :param html: The HTML content to extract text from.
    :type html: str
    :param query_emb: The query embeddings
    :type query_emb: Any
    :param event_date: Event date in year-month-day format.
    :type event_date: str
    :param nlp: The spaCy NLP model.
    :type nlp: Any

    :raises ValueError: If the HTML content is empty.
    :raises ValueError: If the release or update date could not be extracted from the HTML.

    :returns: List of tuples containing the extracted sentences, their similarity scores, and release dates.
    :rtype: list of tuple(str, float, str)
    """

    if not html:
        raise ValueError("HTML is empty.")

    soup = BeautifulSoup(html, "html.parser")

    # Get the date of the website
    date = get_date(soup)
    if date is None:
        raise ValueError("Could not extract release or update date from HTML.")

    # Remove unnecessary tags to clean up text
    for element in soup(HTML_TAGS_TO_REMOVE):
        element.replace_with(NavigableString(" "))

    # Extract and clean text
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ". ".join(chunk for chunk in chunks if chunk)
    text = re.sub(r"\.{2,}", ".", text)

    # Get List of (sentence, similarity, date) tuples
    similarity_scores = extract_similarity_scores(
        text=text,
        query_emb=query_emb,
        event_date=event_date,
        nlp=nlp,
        date=date,
    )

    if not similarity_scores:
        return []

    return similarity_scores


def process_in_batches(
    urls: List[str], batch_size: int = 15, timeout: int = 10
) -> Generator[List[Tuple[Future, str]], None, None]:
    """
    Process URLs in batches using a generator and thread pool executor.

    :param urls: List of URLs to process.
    :type urls: list of str
    :param batch_size: Size of the processing batch (optional, defaults to 5).
    :type batch_size: int
    :param timeout: Timeout for each request in seconds (optional, defaults to 10).
    :type timeout: int

    :raises ValueError: If the batch_size is less than or equal to zero.
    :raises ValueError: If the timeout is less than or equal to zero.

    :yield: List containing Future objects and URLs for each batch.
    :rtype: list of tuple(Future, str)
    """

    if batch_size <= 0:
        raise ValueError("The 'batch_size' size must be greater than zero.")

    if timeout <= 0:
        raise ValueError("The 'timeout' must be greater than zero.")

    session = Session()
    session.max_redirects = 5

    # User-Agent headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/117.0"
    }
    session.headers.update(headers)

    # Using ThreadPoolExecutor to execute requests in parallel
    with ThreadPoolExecutor() as executor:
        # Loop through the URLs in batch_size of size 'batch_size'
        for i in range(0, len(urls), batch_size):
            batch = urls[i : i + batch_size]

            # Submit the batch of URLs for processing
            futures = []
            for url in batch:
                try:
                    # Submit a HEAD request to the url and check Content-Type
                    head_future = executor.submit(
                        session.head,
                        url,
                        headers=headers,
                        timeout=timeout,
                        allow_redirects=True,
                    )
                    head_response = head_future.result()
                    if "text/html" not in head_response.headers.get("Content-Type", ""):
                        continue

                    # Submit a GET request to the url
                    futures.append(
                        (
                            executor.submit(
                                session.get, url, headers=headers, timeout=timeout
                            ),
                            url,
                        )
                    )
                except Exception as e:
                    print(f"An error occurred: {e}")

            yield futures


def extract_and_sort_sentences(
    urls: List[str],
    event_question: str,
    nlp: Any,
) -> List[Tuple[str, float, str]]:
    """
    Extract texts from a list of URLs using Spacy models.

    :param urls: List of URLs to extract text from.
    :type urls: list of str
    :param event_question: Event-related question for text extraction.
    :type event_question: str
    :param nlp: The spaCy NLP model.
    :type nlp: Any

    :returns: List of tuples containing the extracted sentences, their similarity scores, and release dates.
    :rtype: list of tuple(str, float, str)
    """

    # Initialize empty list for storing extracted sentences along with their similarity scores and release dates
    all_sentences = []

    # Process the event question with spacy
    doc_question = nlp(event_question)
    event_date = extract_event_date(doc_question)

    # Create embedding for event question with Spacy embedder model
    query_emb = nlp(event_question)

    if event_date is None:
        print(
            f"Could not extract precise event date from event question: {event_question}"
        )

    # Process URLs in batches
    for batch in process_in_batches(urls=urls):
        for future, url in tqdm(batch, desc="Processing URLs", unit="URL"):
            print(f"Processing {url}")
            try:
                result = future.result()
                if result.status_code != 200:
                    del result
                    continue
                # Extract relevant information for the event question
                extracted_sentences = extract_sentences(
                    html=result.text,
                    query_emb=query_emb,
                    event_date=event_date,  # type: ignore
                    nlp=nlp,
                )

                # Delete the result object to free memory
                del result

                # Append the extracted text if available and increment the count
                if extracted_sentences:
                    all_sentences.extend(extracted_sentences)

            except requests.exceptions.Timeout:
                print(f"Request for {url} timed out.")

            except Exception as e:
                print(f"An error occurred: {e}")

    all_sentences.sort(
        key=lambda x: x[1], reverse=True
    )  # Assuming the second element is the similarity score

    # print similarity scores along with the sentences
    for sentence, similarity, _ in all_sentences:
        if similarity > 0.4:
            print()
            print(f"{similarity}: {sentence}")

    return all_sentences


def join_and_group_sentences(
    sentences: List[Tuple[str, float, str]], max_words: int
) -> str:
    """
    Join the sentences and group them by date.

    :param sentences: List of tuples containing the extracted sentences, their similarity scores, and release dates.
    :type sentences: list of tuple(str, float, str)
    :param max_words: Maximum number of words allowed for the output summary.
    :type max_words: int

    :returns: The joined sentences grouped by date.
    :rtype: str
    """
    # Initialize final output string and word count
    final_output = ""
    current_word_count = 0

    # Initialize a list to hold the sentences that will be included in the final output
    filtered_sentences = []

    # Filter sentences based on word count
    for sentence, _, date in sentences:
        additional_word_count = len(sentence.split())
        if current_word_count + additional_word_count <= max_words:
            filtered_sentences.append((sentence, date))
            current_word_count += additional_word_count
        else:
            break

    # Sort filtered_sentences by date for grouping
    filtered_sentences.sort(key=itemgetter(1))

    # Group by date and iterate
    for date, group in groupby(filtered_sentences, key=itemgetter(1)):
        sentences_group = [sentence for sentence, _ in group]
        concatenated_sentences = " | ".join(sentences_group)

        # Formatting the string as per your requirement
        formatted_string = f"- {date}:{concatenated_sentences}\n\n"

        # Add this formatted string to the final output
        final_output += formatted_string

    return final_output


def generate_prediction_with_retry(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    retries: int = COMPLETION_RETRIES,
    delay: int = COMPLETION_DELAY,
    counter_callback: Optional[Callable] = None,
) -> Tuple[Any, Optional[Callable]]:
    """Attempt to generate a prediction with retries on failure."""
    if not client:
        raise Exception(  # pylint: disable=broad-exception-raised
            "Client not initialized"
        )

    attempt = 0
    while attempt < retries:
        try:
            response = client.completions(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                timeout=90,
                stop=None,
            )

            if response and response.content is not None:
                if counter_callback is not None:
                    counter_callback(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        model=model,
                        token_counter=count_tokens,
                    )

                    extracted_block = extract_json_string(response.content)

                return (
                    extracted_block,  # pylint: disable=possibly-used-before-assignment
                    counter_callback,
                )
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(delay)
            attempt += 1
    raise Exception(  # pylint: disable=broad-exception-raised
        "Failed to generate prediction after retries"
    )


def fetch_additional_information(
    event_question: str,
    max_add_words: int,
    google_api_key: str,
    google_engine: str,
    serper_api_key: Optional[str],
    search_provider: str,
    nlp: Any,
    engine: str = ADDITIONAL_INFO_ENGINE,
    temperature: float = 0.5,
    max_compl_tokens: int = 500,
) -> str:
    """
    Get urls from a web search and extract relevant information based on an event question.

    :param event_question: The question related to the event.
    :type event_question: str
    :param max_add_words: The maximum number of words allowed for additional information.
    :type max_add_words: int
    :param google_api_key: The API key for the Google service.
    :type google_api_key: str
    :param google_engine: The Google engine to be used.
    :type google_engine: str
    :param serper_api_key: The API key for the Serper service.
    :type serper_api_key: Optional[str]
    :param temperature: The temperature parameter for the engine.
    :type temperature: float
    :param nlp: The spaCy NLP model.
    :type nlp: Any
    :param engine: The openai engine. Defaults to "gpt-3.5-turbo".
    :type engine: str
    :param max_compl_tokens: The maximum number of tokens for the engine's response.
    :type max_compl_tokens: int

    :returns: The relevant information fetched from all the URLs concatenated.
    :rtype: str
    """
    if not client:
        return "Client not initialized"
    # Create URL query prompt
    url_query_prompt = URL_QUERY_PROMPT.format(event_question=event_question)

    # Create messages for the OpenAI engine
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": url_query_prompt},
    ]

    # Fetch queries from the OpenAI engine
    response = client.completions(
        model=engine,
        messages=messages,
        temperature=temperature,  # Override the default temperature parameter set for the engine
        max_tokens=max_compl_tokens,  # Override the default max_compl_tokens parameter set for the engine
        n=1,
        timeout=90,
        stop=None,
    )

    if response and response.content is not None:
        # Parse the response content
        extracted_block = extract_json_string(response.content)
        json_data = json.loads(extracted_block)

    # Print queries each on a new line
    print("QUERIES:\n")
    for query in json_data[  # pylint: disable=possibly-used-before-assignment
        "queries"
    ]:
        print(f"query: {query}\n")

    # Get URLs from queries
    # Determine which search provider to use
    if search_provider == "serper":
        if not serper_api_key:
            raise RuntimeError("Serper API key not found")
        urls = get_urls_from_queries_serper(
            queries=json_data["queries"],
            api_key=serper_api_key,
        )
    else:  # default to google
        urls = get_urls_from_queries(
            json_data["queries"],
            api_key=google_api_key,
            engine=google_engine,
        )

    # Extract relevant sentences from URLs
    relevant_sentences_sorted = extract_and_sort_sentences(
        urls=urls,
        event_question=event_question,
        nlp=nlp,
    )

    # Join the sorted sentences and group them by date
    additional_informations = join_and_group_sentences(
        relevant_sentences_sorted, max_add_words
    )

    return additional_informations


@with_key_rotation
def run(**kwargs: Any) -> Union[MaxCostResponse, MechResponse]:
    """
    Run the task with the given arguments.

    :param kwargs: Keyword arguments that specify settings and API keys.
    :type kwargs: dict

    :raises ValueError: If the tool is not supported.
    :raises ValueError: If the event question is not found in the prompt.

    :returns: The generated content and any additional data.
    :rtype: tuple of str and optional dict[str, any]
    """

    tool = kwargs["tool"]
    engine = kwargs.get("model", TOOL_TO_ENGINE[tool])

    delivery_rate = int(kwargs.get("delivery_rate", 0))
    counter_callback: Optional[Callable] = kwargs.get("counter_callback", None)
    if delivery_rate == 0:
        if not counter_callback:
            raise ValueError(
                "A delivery rate of `0` was passed, but no counter callback was given to calculate the max cost with."
            )

        max_cost = counter_callback(
            max_cost=True,
            models_calls=(
                ADDITIONAL_INFO_ENGINE,
                engine,
            ),
        )
        return max_cost

    with LLMClientManager(kwargs["api_keys"], engine, embedding_provider="openai"):
        prompt = kwargs["prompt"]
        max_compl_tokens = kwargs.get(
            "max_tokens", LLM_SETTINGS[engine]["default_max_tokens"]
        )
        temperature = kwargs.get("temperature", LLM_SETTINGS[engine]["temperature"])
        counter_callback = kwargs.get("counter_callback", None)

        openai.api_key = kwargs["api_keys"]["openai"]
        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"TOOL {tool} is not supported.")

        # Print the settings
        print(f"MECH TOOL: {tool}")
        print(f"PROMPT: {prompt}")
        print(f"MAX OPENAI RETURN TOKENS: {max_compl_tokens}")
        print(f"LLM TEMPERATURE: {temperature}")

        # Load the spacy model
        download_spacy_model("en_core_web_md")
        nlp = spacy.load("en_core_web_md")

        # Get the LLM engine to be used
        engine = TOOL_TO_ENGINE[tool]

        # Extract the event question from the prompt
        event_question_match = re.search(r"\"(.+?)\"", prompt)
        if not event_question_match:
            raise ValueError("No event question found in prompt.")
        event_question = event_question_match.group(1)

        print(f"EVENT_QUESTION: {event_question}")
        print()

        # Get the tiktoken base encoding
        enc = tiktoken.get_encoding("cl100k_base")

        # Calculate the maximum number of tokens and words that can be consumed by the additional information string
        max_add_tokens = get_max_tokens_for_additional_information(
            max_compl_tokens=max_compl_tokens,
            prompt=prompt,
            enc=enc,
        )
        max_add_words = int(max_add_tokens * 0.75)

        # Fetch additional information
        search_provider = kwargs.get("search_provider", "google").lower()
        additional_information = fetch_additional_information(
            event_question=event_question,
            engine=ADDITIONAL_INFO_ENGINE,
            temperature=0.5,
            max_compl_tokens=max_compl_tokens,
            nlp=nlp,
            max_add_words=max_add_words,
            google_api_key=kwargs["api_keys"]["google_api_key"],
            google_engine=kwargs["api_keys"]["google_engine_id"],
            serper_api_key=kwargs["api_keys"].get("serperapi", None),
            search_provider=search_provider,
        )

        # Truncate additional information to stay within the chat completion token limit of 4096
        additional_information = truncate_additional_information(
            additional_information,
            max_add_tokens,
            enc=enc,
        )

        # Get the current utc timestamp
        current_time_utc = datetime.now(timezone.utc)
        formatted_time_utc = (
            current_time_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-6] + "Z"
        )

        # Generate the prediction prompt
        prediction_prompt = PREDICTION_PROMPT.format(
            event_question=event_question,
            user_prompt=prompt,
            additional_information=additional_information,
            timestamp=formatted_time_utc,
        )
        print(f"\nPREDICTION PROMPT: {prediction_prompt}\n")

        # Create messages for the OpenAI engine
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prediction_prompt},
        ]

        # Generate the response
        extracted_block, counter_callback = generate_prediction_with_retry(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_compl_tokens,
            retries=COMPLETION_RETRIES,
            delay=COMPLETION_DELAY,
            counter_callback=counter_callback,
        )
        print(f"RESPONSE: {extracted_block}")
        return extracted_block, prediction_prompt, None, counter_callback
