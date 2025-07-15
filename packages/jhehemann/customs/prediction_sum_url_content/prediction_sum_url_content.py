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

"""This module implements a Mech tool for binary predictions."""
import functools
import json
import re
import traceback
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import anthropic
import openai
import requests
import spacy
import tiktoken
from bs4 import BeautifulSoup, NavigableString, Tag
from dateutil import parser
from googleapiclient import errors
from googleapiclient.discovery import build
from openai import OpenAI
from requests import Session
from sentence_transformers import (  # pylint: disable=import-error
    SentenceTransformer,
    util,
)
from spacy.tokens import Doc
from tiktoken import encoding_for_model
from tqdm import tqdm


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
        """Initializes with API keys, model, and embedding provider. Sets the LLM provider based on the model."""
        self.api_key = api_key

    def __enter__(self) -> OpenAI:
        """Initializes and returns LLM and embedding clients."""
        global client
        if client is None:
            client = OpenAI(api_key=self.api_key)
        return client

    def __exit__(self, exc_type: Any, exc_value: Any, tb: Any) -> None:
        """Closes the LLM client"""
        global client
        if client is not None:
            client.close()
            client = None


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


NUM_URLS_EXTRACT = 5
MAX_TOTAL_TOKENS_CHAT_COMPLETION = 4096  # Set the limit for cost efficiency
WORDS_PER_TOKEN_FACTOR = 0.75
DEFAULT_OPENAI_SETTINGS = {
    "max_compl_tokens": 200,
    "temperature": 0,
}

ALLOWED_TOOLS = [
    "prediction-offline-sum-url-content",
    "prediction-online-sum-url-content",
]
TOOL_TO_ENGINE = {
    "prediction-offline-sum-url-content": "gpt-4o-2024-08-06",
    "prediction-online-sum-url-content": "gpt-4o-2024-08-06",
}


PREDICTION_PROMPT = """
INTRODUCTION:
You are a Large Language Model (LLM) within a multi-agent system. Your primary task is to accurately estimate the probability of a specified event occurring by a specified deadline, \
detailed in the 'event question' found in 'USER_PROMPT'. This 'event question' should ideally have only two possible outcomes: the event will either occur or not \
by the deadline specified in the question, which is 23:59:59 of the date provided. It is critical that you incorporate this deadline date in your \
probability estimation. You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION", which is \
sourced from a Google search engine query performed a few seconds ago and is meant to assist you in your estimation. You must adhere to the following 'INSTRUCTIONS'.


INSTRUCTIONS:
* Examine the user's input labeled 'USER_PROMPT'. Focus on the part enclosed in double quotes, which contains the 'event question'.
* If the 'event question' implies more than two outcomes, output the response "Error" and halt further processing.
* Utilize your training data to generate a probability estimation for the event specified in the 'event question' occurring by the given deadline of 23:59:59 on the specified date.
* Also rely on your training data to analyze what information would be relevant to make a probability estimation for the event specified in the 'event question' occurring by the given deadline.
* Examine the itemized list under "ADDITIONAL_INFORMATION". This data is sourced from a Google search engine query done a few seconds ago.
* You can use any item in "ADDITIONAL_INFORMATION" in addition to your training data to make the probability estimation.
* If there exist any information in "ADDITIONAL_INFORMATION" that is related to the 'event question' you can assume that you have been provided with the most current and relevant information available on the internet, regardless of its age. Still pay close attention on the release and modification timestamps for each information item provided in parentheses right before it as well as the current time {timestamp}. Even if "ADDITIONAL_INFORMATION" contains the most recent and relevant information about the topic in the event question, it does not imply that it is relevant to make a prediction about the event question.
* If there exist any information in "ADDITIONAL_INFORMATION" that is related to the 'event question', but does not clearly state that the event has already happened, you can assume that the event has not happened by now `{timestamp}`.
* Given the importance of the deadline for the 'event question,' recent information generally holds more weight for your probability estimation. However, do not disregard older information that provides foundational or contextual relevance to the event in question. Use your judgment to weigh the importance of recency against the relevance of older data.
* Factor the deadline into your probability estimation. It determines the timeframe by which the event must occur for the 'event question' to be answered affirmatively. For reference, the current time is `{timestamp}`.
* Decrease the probability estimation of the event occurring drastically the closer the current time `{timestamp}` is to the deadline, if you have not found information clearly indicating that the event will happen within the remaining time.
* If the event question is formulated too vaguely or if the information under "ADDITIONAL_INFORMATION" contradict each other, decrease the confidence value in your probability estimation accordingly.
* If the information in "ADDITIONAL_INFORMATION" indicate without a doubt that the event has already happened, set the probability estimation to a very high score. If not, make a probability estimation based on the information provided as well as your training data for context and background information.
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
   - "p_yes": Estimated probability that the event occurs within the deadline.
   - "p_no": Estimated probability that the 'event question' does not occur within the deadline.
   - "confidence": Indicating the confidence in the estimated probabilities you provided ranging from 0 (lowest confidence) to 1 (maximum confidence). Confidence can be calculated based on the quality and quantity of data used for the estimation.
   - "info_utility": Utility of the information provided in "ADDITIONAL_INFORMATION" to help you make the prediction ranging from 0 (lowest utility) to 1 (maximum utility).
* The sum of "p_yes" and "p_no" must equal 1.
* Output only the JSON object in your response. Do not include any other contents in your response.
* Never use Markdown syntax highlighting, such as ```json``` to surround the output. Only output the raw json string.
* This is incorrect:"```json{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}```"
* This is incorrect:```json"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"```
* This is correct:"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"
"""

URL_QUERY_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to formulate search engine queries based on \
a user's 'event question', which specifies an event and any accompanying conditions. The 'event question' allows \
only two outcomes: the event will either occur or not, given the conditions. Find the 'event question' under 'USER_PROMPT' \
and adhere to the 'INSTRUCTIONS'.

INSTRUCTIONS:
* Carefully read the 'event question' under 'USER_PROMPT', enclosed by triple backticks.
* If the 'event question' has more than two outcomes, respond with "Error" and ignore further instructions.
* Create a list of 1-4 unique search queries likely to yield relevant and contemporary information for assessing the event's likelihood under the given conditions.
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
* This is incorrect: "```json{{"queries": []}}```"
* This is incorrect: "```json"{{"queries": []}}"```"
* This is correct: "{{"queries": []}}"
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
    if event_date_ymd is None or not datetime.strptime(event_date_ymd, "%Y-%m-%d"):
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


def extract_relevant_information(
    text: str, query_emb: Any, event_date: str, model: Any, nlp: Any, max_words: int
) -> str:
    """
    Extract relevant information from website text based on a given event question.

    :param text: The website text to extract information from.
    :type text: str
    :param query_emb: The query embeddings.
    :type query_emb: Any
    :param event_date: Event date in year-day-month format.
    :type event_date: str
    :param model: The BERT model for text embeddings.
    :type model: Any
    :param nlp: The spaCy NLP model.
    :type nlp: Any
    :param max_words: Maximum number of words allowed for output.
    :type max_words: int

    :returns: The relevant sentences extracted from the website text.
    :rtype: str
    """

    # Constants for sentence length and number thresholds
    len_sentence_threshold = 5
    num_sentences_threshold = 1000
    sentences = []
    event_date_sentences: List = []
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
    sentences.extend(event_date_sentences)

    # Extract contextual sentences around event date occurences within too short sentences
    event_date_sentences.extend(
        get_context_around_isolated_event_date(
            doc_text, event_date, len_sentence_threshold, max_context=50
        )
    )

    if not sentences:
        return ""

    # Limit the number of sentences for performance optimization
    sentences = sentences[:num_sentences_threshold]

    # Encode event question calculate similarity scores
    sent_emb = model.encode(sentences)
    similarities = util.dot_score(query_emb, sent_emb)[0].cpu().tolist()

    # Extract top relevant sentences
    relevant_sentences = [
        sent
        for sent, sim in sorted(
            zip(sentences, similarities), key=lambda x: x[1], reverse=True
        )
        if sim > 0.4
    ]

    if not relevant_sentences:
        return ""

    # Truncate text to fit max_words limit
    output = " ".join(relevant_sentences[:20])
    output_words = output.split(" ")
    if len(output_words) > max_words:
        output = " ".join(output_words[:max_words])

    return output


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

    # Fallback to using the first time tag if neither release nor modified dates are found
    if release_date == "unknown" and modified_date == "unknown":
        time_tag = soup.find("time")
        if time_tag and isinstance(time_tag, Tag):
            datetime_content = time_tag.get("datetime", "")
            if isinstance(datetime_content, Iterable) and not isinstance(
                datetime_content, (str, bytes)
            ):
                release_date = " ".join(datetime_content)
            else:
                release_date = str(datetime_content)

    return f"({release_date}, {modified_date})"


def extract_text(
    html: str,
    query_emb: Any,
    event_date: str,
    model: Any,
    nlp: Any,
    max_words: int,
) -> str:
    """
    Extract relevant information from HTML string.

    :param html: The HTML content to extract text from.
    :type html: str
    :param query_emb: The query embeddings.
    :type query_emb: Any
    :param event_date: Event date in year-month-day format.
    :type event_date: str
    :param model: Pre-trained model for sentence transformer.
    :type model: Any
    :param nlp: NLP object for additional text processing.
    :type nlp: Any
    :param max_words: Maximum number of words for the output summary.
    :type max_words: int

    :raises ValueError: If the HTML content is empty.
    :raises ValueError: If the release or update date could not be extracted from the HTML.

    :returns: Relevant website information with release date.
    :rtype: str
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

    # Get summarized text
    relevant_text = extract_relevant_information(
        text=text,
        query_emb=query_emb,
        event_date=event_date,
        model=model,
        nlp=nlp,
        max_words=max_words,
    )

    if not relevant_text:
        return ""

    return f"{date}: {relevant_text}"


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


def extract_texts(
    urls: List[str],
    event_question: str,
    max_words_per_url: int,
    nlp: Any,
) -> List[str]:
    """
    Extract texts from a list of URLs using BERT and Spacy.

    :param urls: List of URLs to extract text from.
    :type urls: list of str
    :param event_question: Event-related question for text extraction.
    :type event_question: str
    :param max_words_per_url: Maximum number of words allowed to extract for each URL.
    :type max_words_per_url: int
    :param nlp: Maximum number of words allowed to extract for each URL.
    :type nlp: Any

    :returns: List of extracted texts.
    :rtype: list of str
    """

    # Maximum number of allowed extractions
    max_allowed = 25

    # Initialize empty list for storing extracted texts
    extracted_texts = []

    # Initialize count and stop flag
    count = 0
    stop = False

    # Process the event question with spacy
    doc_question = nlp(event_question)
    event_date = extract_event_date(doc_question)

    # Initialize Sentence Transformer model
    model = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1")

    # Create sentence embeddings for event question with Sentence Transformer
    query_emb = model.encode(event_question)

    if event_date is None:
        raise ValueError(
            f"Could not extract precise event date from event question: {event_question}"
        )

    # Process URLs in batches
    for batch in process_in_batches(urls=urls):
        for future, url in tqdm(batch, desc="Processing URLs"):
            print(f"Processing {url}")
            try:
                result = future.result()
                if result.status_code != 200:
                    del result
                    continue
                # Extract relevant information for the event question
                extracted_text = extract_text(
                    html=result.text,
                    query_emb=query_emb,
                    event_date=event_date,
                    model=model,
                    nlp=nlp,
                    max_words=max_words_per_url,
                )

                # Delete the result object to free memory
                del result

                # Append the extracted text if available and increment the count
                if extracted_text:
                    # extracted_texts.append(f"{url}\n{extracted_text}") noqa: E800
                    extracted_texts.append(extracted_text)
                count += 1

                # Break if the maximum number of extractions is reached
                if count >= max_allowed:
                    stop = True
                    break

            except requests.exceptions.Timeout:
                print(f"Request for {url} timed out.")

            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()  # Print stack trace for debugging

        # Break if the maximum number of extractions is reached
        if stop:
            break

    return extracted_texts


def fetch_additional_information(
    event_question: str,
    max_add_words: int,
    google_api_key: str,
    google_engine: str,
    nlp: Any,
    engine: str = "gpt-4o-2024-08-06",
    temperature: float = 1.0,
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
    :param nlp: The nlp object
    :type nlp: Any
    :param engine: The openai engine. Defaults to "gpt-3.5-turbo".
    :type engine: str
    :param temperature: The temperature parameter for the engine.
    :type temperature: float
    :param max_compl_tokens: The maximum number of tokens for the engine's response.
    :type max_compl_tokens: int

    :returns: The relevant information fetched from all the URLs concatenated.
    :rtype: str
    """
    if not client:
        return "Client not initialized"
    # Create URL query prompt
    url_query_prompt = URL_QUERY_PROMPT.format(event_question=event_question)

    # Perform moderation check
    moderation_result = client.moderations.create(input=url_query_prompt)
    if moderation_result.results[0].flagged:
        return "Moderation flagged the prompt as in violation of terms."

    # Create messages for the OpenAI engine
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": url_query_prompt},
    ]

    # Fetch queries from the OpenAI engine
    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,  # Override the default temperature parameter set for the engine
        max_tokens=max_compl_tokens,  # Override the default max_compl_tokens parameter set for the engine
        n=1,
        timeout=90,
        stop=None,
    )

    # Parse the response content
    print(f"RESPONSE: {response}")
    json_data = json.loads(response.choices[0].message.content)
    # Print queries each on a new line
    print("QUERIES:\n")
    for query in json_data["queries"]:
        print(f"query: {query}\n")

    # Get URLs from queries
    urls = get_urls_from_queries(
        json_data["queries"],
        api_key=google_api_key,
        engine=google_engine,
    )

    # Get max number of words per URL
    max_words_per_url = max_add_words // len(urls) if len(urls) > 0 else 0

    # Extract texts from URLs
    texts = extract_texts(
        urls=urls,
        event_question=event_question,
        max_words_per_url=max_words_per_url,
        nlp=nlp,
    )

    # Join the texts and return
    additional_informations = "\n\n".join(["- " + text for text in texts])

    return additional_informations


@with_key_rotation
def run(**kwargs: Any) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """
    Run the task with the given arguments.

    :param kwargs: Keyword arguments that specify settings and API keys.
    :type kwargs: dict

    :raises ValueError: If the tool is not supported.
    :raises ValueError: If the event question is not found in the prompt.

    :returns: The generated content and any additional data.
    :rtype: tuple of str and optional dict[str, any]
    """
    with OpenAIClientManager(kwargs["api_keys"]["openai"]):
        if not client:
            raise RuntimeError("Client not initialized")

        tool = kwargs["tool"]
        prompt = kwargs["prompt"]
        max_compl_tokens = kwargs.get(
            "max_tokens", DEFAULT_OPENAI_SETTINGS["max_compl_tokens"]
        )
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])

        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"TOOL {tool} is not supported.")

        # Print the settings
        print(f"MECH TOOL: {tool}")
        print(f"PROMPT: {prompt}")
        print(f"MAX OPENAI RETURN TOKENS: {max_compl_tokens}")
        print(f"LLM TEMPERATURE: {temperature}")

        # Load the spacy model
        nlp = spacy.load("en_core_web_sm")

        # Get the LLM engine to be used
        engine = kwargs.get("model", TOOL_TO_ENGINE[tool])
        print(f"ENGINE: {engine}")

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
        additional_information = (
            fetch_additional_information(
                event_question=event_question,
                engine=engine,
                temperature=temperature,
                max_compl_tokens=max_compl_tokens,
                nlp=nlp,
                max_add_words=max_add_words,
                google_api_key=kwargs["api_keys"]["google_api_key"],
                google_engine=kwargs["api_keys"]["google_engine_id"],
            )
            if tool == "prediction-online-sum-url-content"
            else ""
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

        # Extract event date and format it to ISO 8601 with UTC timezone and 23:59:59 time
        doc_question = nlp(event_question)
        raw_event_date = extract_event_date(doc_question)
        if raw_event_date is None:
            raise ValueError("Could not extract a valid event date.")
        parsed_event_date = datetime.strptime(raw_event_date, "%Y-%m-%d")
        final_event_date = parsed_event_date.replace(
            hour=23, minute=59, second=59, microsecond=0, tzinfo=timezone.utc
        )
        formatted_event_date = (
            final_event_date.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-6] + "Z"
        )

        # Generate the prediction prompt
        prediction_prompt = PREDICTION_PROMPT.format(
            event_question=event_question,
            user_prompt=prompt,
            timepoint=formatted_event_date,
            additional_information=additional_information,
            timestamp=formatted_time_utc,
        )
        print(f"\nPREDICTION PROMPT: {prediction_prompt}\n")

        # Perform moderation
        moderation_result = client.moderations.create(input=prediction_prompt)
        if moderation_result.results[0].flagged:
            return (
                "Moderation flagged the prompt as in violation of terms.",
                None,
                None,
                None,
            )

        # Create messages for the OpenAI engine
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prediction_prompt},
        ]

        # Generate the response
        response = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_compl_tokens,
            n=1,
            timeout=150,
            stop=None,
        )
        print(f"RESPONSE: {response}")
        return response.choices[0].message.content, None, None, None
