# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
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

# pylint: disable=too-many-arguments,too-many-locals

import functools
import json
import re
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from io import BytesIO
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import PyPDF2
import anthropic
import faiss
import googleapiclient
import numpy as np
import openai
import requests
from docstring_parser import parse
from googleapiclient.discovery import build
from markdownify import markdownify as md
from openai import OpenAI
from pydantic import BaseModel, Field
from readability import Document as ReadabilityDocument
from tiktoken import Encoding, encoding_for_model, get_encoding


TOKENS_DISTANCE_TO_LIMIT = 200
DOC_TOKEN_LIMIT = 7000  # Maximum tokens per document for embeddings
BUFFER = 500  # Buffer for the total tokens in the embeddings batch
MAX_EMBEDDING_TOKENS = 300000 - BUFFER  # Maximum tokens for the embeddings batch
N_MODEL_CALLS = 6
GOOGLE_RATE_LIMIT_EXCEEDED_CODE = 429
DEFAULT_DELIVERY_RATE = 100


client: Optional[OpenAI] = None

MechResponseWithKeys = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]
MaxCostResponse = float


def get_model_encoding(model: str) -> Encoding:
    """Get the appropriate encoding for a model."""
    # Workaround since tiktoken does not have support yet for gpt4.1
    # https://github.com/openai/tiktoken/issues/395
    if model == "gpt-4.1-2025-04-14":
        return get_encoding("o200k_base")
    return encoding_for_model(model)


# Clean text by removing emojis and non-printable characters.
def clean_text(text: str) -> str:
    """Remove emojis and non-printable characters, collapse whitespace."""
    emoji_pattern = re.compile(
        "[\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    # Decode using UTF-8, replacing invalid bytes
    text = text.encode("utf-8", "replace").decode("utf-8", "replace")
    text = "".join(ch for ch in text if ch.isprintable())
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Utility: truncate text to a maximum number of tokens.
def truncate_text(text: str, model: str, max_tokens: int) -> str:
    """Truncate text to the first max_tokens tokens based on model encoding."""
    enc = get_model_encoding(model)
    token_ids = enc.encode(text)
    if len(token_ids) <= max_tokens:
        return text
    return enc.decode(token_ids[:max_tokens])


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
                if e.status_code != GOOGLE_RATE_LIMIT_EXCEEDED_CODE:
                    raise e
                service = "google_api_key"
                if retries_left[service] <= 0:
                    raise e
                retries_left[service] -= 1
                api_keys.rotate(service)
                return execute()
            except Exception as e:  # pylint: disable=broad-except
                print(f"Unexpected error: {e}")
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


def count_tokens_for_openai_api_first_check(text: str) -> int:
    """Estimate the number of tokens for OpenAI embeddings API's initial check."""
    # OpenAI's embeddings endpoint uses an approximate calculation of 0.25 tokens per UTF-8 byte
    # (i.e., 1 token per 4 bytes) for the 300k token limit. This is not the same as tiktoken's
    # actual token count, but is used for the API's first validation.
    # Reference: https://community.openai.com/t/max-total-embeddings-tokens-per-request/1254699/6
    return len(text.encode("utf-8")) // 4


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = get_model_encoding(model)
    return len(enc.encode(text))


class OpenAIClientManager:
    """Client context manager for OpenAI."""

    def __init__(self, api_key: str):
        """Initializes with API keys"""
        self.api_key = api_key

    def __enter__(self) -> OpenAI:
        """Initializes and returns LLM client."""
        global client
        if client is None:
            client = OpenAI(api_key=self.api_key)
        return client

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Closes the LLM client"""
        global client
        if client is not None:
            client.close()
            client = None


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0,
}
OPEN_AI_SETTINGS = {
    "gpt-4.1-2025-04-14": {
        """
        Error code: 400 - {'error': {'message': 'max_tokens is too large: 1047576. This model supports at most 32768 completion tokens, whereas you provided 1047576.', 'type': 'invalid_request_error', 'param': 'max_tokens', 'code': 'invalid_value'}}
        """
        "max_tokens": 32_768,
        "temperature": 0,
    },
}
MAX_TOKENS = {
    "gpt-4o-2024-08-06": 4096,
    "gpt-4.1-2025-04-14": 4096,
}
ALLOWED_TOOLS = [
    "resolve-market-reasoning-gpt-4.1",
]
TOOL_TO_ENGINE = {
    "resolve-market-reasoning-gpt-4.1": "gpt-4.1-2025-04-14",
}
DEFAULT_NUM_WORDS: Dict[str, Optional[int]] = defaultdict(lambda: 300)
NUM_QUERIES = 3
NUM_URLS_PER_QUERY = 3
SPLITTER_CHUNK_SIZE = 1800
SPLITTER_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_BATCH_SIZE = 1000
EMBEDDING_SIZE = 3072
NUM_NEIGHBORS = 4
BUFFER_TOKENS = 250


class OpenAISchema(BaseModel):  # type: ignore[misc]
    """OpenAISchema"""

    @classmethod  # type: ignore[misc]
    @property
    def openai_schema(cls) -> Dict[str, Any]:
        """
        Return the schema in the format of OpenAI's schema as jsonschema.

        Note:
            It's important to add a docstring to describe how to best use this class,
            it will be included in the description attribute and be part of the prompt.

        :returns: A dictionary in the format of OpenAI's schema as jsonschema.
        :rtype: dict
        """
        schema = cls.model_json_schema()
        docstring = parse(cls.__doc__ or "")
        parameters = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        for param in docstring.params:
            if (name := param.arg_name) in parameters["properties"] and (
                description := param.description
            ):
                if "description" not in parameters["properties"][name]:
                    parameters["properties"][name]["description"] = description

        parameters["required"] = sorted(
            k for k, v in parameters["properties"].items() if "default" not in v
        )

        if "description" not in schema:
            if docstring.short_description:
                schema["description"] = docstring.short_description
            else:
                schema["description"] = (
                    f"Correctly extracted `{cls.__name__}` with all "
                    f"the required parameters with correct types"
                )

        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }

    @classmethod
    def from_response(cls, completion: Dict[str, Any]) -> "OpenAISchema":
        """
        Convert the response from OpenAI into the class instance.

        :param completion: The response from OpenAI
        :type completion: dict

        :returns: The instance of the class
        :rtype: OpenAISchema
        """
        message = completion.choices[0].message  # type: ignore

        return cls.model_validate_json(
            message.function_call.arguments,
        )


class Queries(OpenAISchema):
    """Queries schema"""

    queries: List[str]


class Date(OpenAISchema):
    """Date schema"""

    date_available: bool = Field(..., description="Whether the date is available")
    year: Optional[int] = Field(..., description="The year the article was published")
    month: Optional[str] = Field(..., description="The month the article was published")
    day: Optional[int] = Field(..., description="The day the article was published")


class Results(OpenAISchema):
    """Results schema"""

    has_occurred: bool = Field(..., description="Whether the event has occurred.")


class Valid(OpenAISchema):
    """Question validity schema."""

    is_valid: bool = Field(..., description="Whether the question is valid.")
    reason: Optional[str] = Field(
        ..., description="Reason that the question is invalid."
    )


class Determinable(OpenAISchema):
    """Question determinability schema."""

    is_determinable: bool = Field(
        ...,
        description="Whether it is possible to answer the question based on the information provided and reasoning.",
    )


class Document(BaseModel):
    """Document model"""

    text: str
    date: str
    url: str
    embedding: Optional[List[float]] = None
    tokens: int = 0


URL_QUERY_PROMPT = """
 You are an expert fact checker in a team tasked with determining whether an event happened before a given date in the past.
* Your role in the team to come up with search queries to be used to find relevant news articles that may help in determining whether the event occured.
* You are provided with the input question about the event under the label "USER_PROMPT".
* You must follow the instructions under the label "INSTRUCTIONS".

INSTRUCTIONS
* Read the input under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" is a question about whether an event happened.
* The "USER_PROMPT" will contain a date which in the past.
* The event will only have has two possible outcomes: either the event has happened or the event has not happened.
* If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
* You should come up with {num_queries} diverse queries to search for relevant news articles that may help in determining whether the event occured.
* Focus on capturing different aspects and interpretations of the question to ensure comprehensive coverage of the topic.
* Make sure the queries are in past tense and are in the form of a question.
* ONLY function calls are allowed in the response.

USER_PROMPT:
```
{user_prompt}
```
"""

GET_DATE_PROMPT = """
INSTRUCTIONS
* You are an expert data analyst that takes in extracted text from a web search result.
* You are provided with text extracted from a relevant web page under the label "EXTRACTED_TEXT" delimited by three backticks.
* Your task is to extract the date that the web page was published.
* If there is no date information available, you should not try to guess. Instead indicate that it is not available.
* Your response should only be a function call with the extracted date information as arguments.

EXTRACTED_TEXT:
```
{extracted_text}
```
"""

PREDICTION_PROMPT = """
INSTRUCTIONS
* You are an expert data analyst.
* You are provided with the input question about the event under the label "USER_PROMPT".
* You are provided with a colleague's reasoning as to whether the event occurred based on online research under the label "REASONING" delimited by three backticks.
* Your task is to parse the decision on whether an event occurred.
* The answer that you give should match the answer that you come to in the reasoning field
* ONLY function calls are allowed in the response.

USER_PROMPT:
```
{user_prompt}
```

REASONING:
```
{reasoning}
```
"""

REASONING_PROMPT = """
You are an expert fact checker that takes in a question asking whether an event will happen before a given date.
That date has now passed and your role is to determine whether the event actually happened before the date.
You are provided with the input question about the event under the label "USER_PROMPT". You must follow the instructions
under the label "INSTRUCTIONS".

INSTRUCTIONS
* Read the input question under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" specifies a question about whether an event happened before a certain date.
* The date will has already passed, so you need to determine whether the event did or did not happen. There are only two
possible answers: either the event did happen or it did not happen.
* If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
* You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION" delimited by three backticks.
* The items in "ADDITIONAL_INFORMATION" "ARTICLE (N), DATE: (MONTH/YEAR), URL: (URL), CONTENT: (CONTENT)"
* You can use any item in "ADDITIONAL_INFORMATION" in addition to your training data.
* If an item in "ADDITIONAL_INFORMATION" is not relevant, you must ignore that item for the estimation.
* Ideally, these will be news articles about the event in question.
* Pay special attention to the date of the article if it is available.
* You should show your process of thinking through the problem step by step, taking the date and information of the various articles into consideration, and explain your reasoning for your decision as to whether an event occurred by the specified date.
* The articles will not always explicitly contain all the information needed to determine the answer. In this case, you may need to make an educated guess based on certain assumptions. If you need to do this, please provide your assumptions in your explanation.

Here are some examples of how you can figure out whether an event occurred by the date:
* If an article says that the event did happen and the date of the article is before the question date, then it is likely that the event did occur before the question date.
* If an article is talking about whether an event will happen and the date of the article is after the question date, then it is likely that the event did not happen before the question date.

USER_PROMPT:
```
{user_prompt}
```

ADDITIONAL_INFORMATION:
```
{formatted_docs}
```
"""

VALID_PROMPT = """
* You are an expert data analyst.
* You are provided with a question about an event (submitted to a prediction market) under "USER_PROMPT" delimited by three backticks.
* Your task is to determine whether the question is valid.
* You are provided with rules that determine whether a question is invalid (as well as examples) under the label "RULES".
* Your response should only be a function call with the information about whether a question is valid and reason as arguments.

RULES
* Questions with relative dates should be marked as invalid. E.g. Invalid: Who will be the president of the United States in 6 months? (“in 6 months depends on the current time”).
* Questions about moral values and not facts should be marked as invalid. E.g. Invalid: “Is it ethical to eat meat?”.
* Questions in which none of the answers are valid should be marked as invalid. E.g. Invalid: “What is the result of 1+1?” with the outcomes “0” and “1”.
* Questions in which multiple answers are valid should be marked as invalid. E.g. Invalid: “Who will be the Time person of the year 1937?” with answers “Chiang Kai-shek” and “Soong Mei-ling” (they got the prize jointly).

USER_PROMPT:
```
{user_prompt}
```
"""

DETERMINABLE_PROMPT = """
* You are an expert data analyst.
* You are provided with a question about an event (submitted to a prediction market) under "USER_PROMPT" delimited by three backticks.
* You are provided with a colleague's reasoning as to whether the event occurred based on online research under the label "REASONING" delimited by three backticks.
* Your task is to determine whether it is possible to answer whether the event actually happened before the date based on the content of this reasoning.
* The answer that you give should reflect the opinion in the reasoning field.
* Your response should only be a function call with the information about whether a question is valid and reason as arguments.

USER_PROMPT:
```
{user_prompt}
```

REASONING:
```
{reasoning}
```

"""

SYSTEM_PROMPT = """You are a world class algorithm for generating structured output from a given input."""


def multi_queries(
    client_: OpenAI,
    prompt: str,
    engine: str,
    num_queries: int,
    counter_callback: Optional[Callable] = None,
) -> Tuple[List[str], Optional[Callable]]:
    """Generate multiple queries for fetching information from the web."""

    url_query_prompt = URL_QUERY_PROMPT.format(
        user_prompt=prompt, num_queries=num_queries
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url_query_prompt},
    ]

    response = client_.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=DEFAULT_OPENAI_SETTINGS["temperature"],
        max_tokens=DEFAULT_OPENAI_SETTINGS["max_tokens"],
        n=1,
        timeout=150,
        stop=None,
        functions=[Queries.openai_schema],
        function_call={"name": "Queries"},
    )
    queries = Queries.from_response(response)

    # append the user's question to the list of queries
    queries.queries.append(prompt)

    if counter_callback:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=engine,
            token_counter=count_tokens,
        )
    return queries.queries, counter_callback


def search_google(query: str, api_key: str, engine: str, num: int) -> List[str]:
    """Performs a Google Custom Search and returns a list of result links."""
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


def get_urls_from_queries(
    queries: List[str], api_key: str, engine: str, num: int
) -> List[str]:
    """Get URLs from search engine queries"""
    results = []
    for query in queries:
        try:
            for url in search_google(
                query=query,
                api_key=api_key,
                engine=engine,
                num=num,
            ):
                results.append(url)
        except googleapiclient.errors.HttpError as e:
            if e.resp.status == GOOGLE_RATE_LIMIT_EXCEEDED_CODE:
                print(
                    f"Rate limit exceeded for query: {query}. Trying to rotate API key."
                )
                raise e
            print(f"HTTP error for query {query}: {e}")

    unique_results = list(set(results))
    return unique_results


def get_urls_from_queries_serper(
    queries: List[str], api_key: str, num: int
) -> List[str]:
    """Get URLs from search engine queries using Serper."""
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
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error fetching URLs for query '{query}': {e}")

    return list(set(urls))


def get_dates(
    client_: OpenAI,
    text: str,
    counter_callback: Optional[Callable] = None,
) -> Tuple[str, Optional[Callable]]:
    """Get the date from the extracted text"""
    adjusted_text = adjust_additional_information(
        prompt=GET_DATE_PROMPT, additional_information=text, model="gpt-4.1-2025-04-14"
    )
    get_date_prompt = GET_DATE_PROMPT.format(extracted_text=adjusted_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_date_prompt},
    ]
    response = client_.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=messages,
        temperature=0,
        n=1,
        timeout=90,
        stop=None,
        functions=[Date.openai_schema],
        function_call={"name": "Date"},
    )
    date = Date.from_response(response)
    if date.date_available:
        if counter_callback:
            counter_callback(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model="gpt-4.1-2025-04-14",
                token_counter=count_tokens,
            )
            return f"{date.year}-{date.month}-{date.day}", counter_callback
        return f"{date.year}-{date.month}-{date.day}", None
    return "Date not available", None


def extract_text_from_pdf(
    url: str, num_words: Optional[int] = None
) -> Optional[Document]:
    """Extract text from a PDF document at the given URL."""
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        if "application/pdf" not in response.headers.get("Content-Type", ""):
            raise ValueError("URL does not point to a PDF document")

        with BytesIO(response.content) as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

        doc = Document(text=text[:num_words] if num_words else text, date="", url=url)

        return doc
    except Exception as e:  # pylint: disable=broad-except
        print(f"An error occurred: {e}")
        return None


def extract_text(
    client_: OpenAI,
    html: str,
    num_words: Optional[int] = None,
    counter_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[Document, Optional[Callable]]:
    """Extract text from a single HTML document"""
    text = ReadabilityDocument(html).summary()
    text = text = md(text, heading_style="ATX")
    date, counter_callback = get_dates(
        client_=client_, text=text, counter_callback=counter_callback
    )
    doc = Document(text=text[:num_words] if num_words else text, date=date, url="")
    return doc, counter_callback


def extract_texts(
    urls: List[str],
    client_: OpenAI,
    counter_callback: Optional[Callable] = None,
) -> Tuple[List[Document], Optional[Callable]]:
    """Extract texts from URLs"""
    extracted_texts = []
    for batch in process_in_batches(urls=urls) or []:
        for future, url in batch:
            try:
                if url.lower().endswith(".pdf"):
                    result = extract_text_from_pdf(url)
                    if result:
                        extracted_texts.append(result)
                    continue
                result = future.result()
                if not result:
                    print(f"No result returned for {url}")
                    continue
                if not isinstance(result, requests.Response):
                    continue
                if result.status_code != 200:
                    continue
                # first 4 bytes is pdf
                if result.content[:4] == b"%PDF":
                    result = extract_text_from_pdf(url)
                    if result:
                        extracted_texts.append(result)
                    continue
                doc, counter_callback = extract_text(
                    html=result.text,
                    client_=client_,
                    counter_callback=counter_callback,
                )
                doc.url = url
                extracted_texts.append(doc)
            except requests.exceptions.ReadTimeout:
                print(f"Request timed out: {url}.")
            except Exception as e:  # pylint: disable=broad-except
                print(f"An error occurred: {e}")
    return extracted_texts, counter_callback


def process_in_batches(
    urls: List[str], window: int = 5, timeout: int = 50
) -> Generator[List[Tuple[Future, str]], None, None]:
    """Iter URLs in batches."""
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(urls), window):
            batch = urls[i : i + window]
            futures = [
                (executor.submit(requests.get, url, timeout=timeout), url)
                for url in batch
            ]
            yield futures


def recursive_character_text_splitter(
    text: str, max_tokens: int, overlap: int
) -> List[str]:
    """Splits the input text into chunks of size `max_tokens`, with an overlap between chunks."""
    if len(text) <= max_tokens:
        return [text]
    return [text[i : i + max_tokens] for i in range(0, len(text), max_tokens - overlap)]


def get_embeddings(split_docs: List[Document]) -> List[Document]:
    """Get embeddings for the split documents: clean, truncate, then batch by token count."""
    # Preprocess each document: clean and truncate to DOC_TOKEN_LIMIT
    # Filter out any documents that exceed the maximum token limit individually
    filtered_docs = []
    total_tokens_count = 0
    for doc in split_docs:
        # if we are very close to the limit then break the loop
        if MAX_EMBEDDING_TOKENS - total_tokens_count < TOKENS_DISTANCE_TO_LIMIT:
            break
        cleaned = clean_text(doc.text)
        # TODO we could summarize instead of truncating
        doc.text = truncate_text(cleaned, EMBEDDING_MODEL, DOC_TOKEN_LIMIT)
        # filter empty strings
        doc.text = doc.text.strip()
        doc.tokens = count_tokens(doc.text, EMBEDDING_MODEL)
        if total_tokens_count + doc.tokens > MAX_EMBEDDING_TOKENS:
            continue
        if doc.text:
            filtered_docs.append(doc)
            total_tokens_count += doc.tokens

    # Process documents in batches that respect the total token limit
    processed_docs = []
    i = 0
    while i < len(filtered_docs):
        current_batch_docs = []
        current_batch_tokens = 0

        while i < len(filtered_docs):
            doc = filtered_docs[i]
            doc.tokens = count_tokens_for_openai_api_first_check(doc.text)

            if current_batch_tokens + doc.tokens > MAX_EMBEDDING_TOKENS:
                break

            current_batch_docs.append(doc)
            current_batch_tokens += doc.tokens
            i += 1

        if not current_batch_docs:
            # This should not happen after filtering, but just in case
            i += 1
            continue
        batch_texts = [doc.text for doc in current_batch_docs]

        if not client:
            raise RuntimeError("Embeddings not intialized")
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_texts,
        )

        for j, emb in enumerate(response.data):
            assert j == emb.index, "Embeddings response out-of-order"
            current_batch_docs[j].embedding = emb.embedding
            processed_docs.append(current_batch_docs[j])

    return processed_docs


def find_similar_chunks(
    query: str, docs_with_embeddings: List[Document], k: int = 4
) -> List:
    """Similarity search to find similar chunks to a query"""
    if not client:
        raise RuntimeError("Client not initialized")

    query_embedding = (
        client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        .data[0]
        .embedding
    )

    index = faiss.IndexFlatIP(EMBEDDING_SIZE)  # pylint: disable=no-value-for-parameter
    index.add(  # pylint: disable=no-value-for-parameter
        np.array([doc.embedding for doc in docs_with_embeddings])
    )
    _, indices = index.search(  # pylint: disable=no-value-for-parameter
        np.array([query_embedding]), k
    )

    return [docs_with_embeddings[i] for i in indices[0]]


def fetch_additional_information(
    client_: OpenAI,
    prompt: str,
    engine: str,
    google_api_key: Optional[str],
    google_engine_id: Optional[str],
    serper_api_key: Optional[str],
    counter_callback: Optional[Callable] = None,
) -> Tuple:
    """Fetch additional information from the web."""
    if not google_api_key:
        raise RuntimeError("Google API key not found")
    if not google_engine_id:
        raise RuntimeError("Google Engine Id not found")
    if not serper_api_key:
        raise RuntimeError("Serper API key not found")

    # generate multiple queries for fetching information from the web
    queries, counter_callback = multi_queries(
        client_=client,
        prompt=prompt,
        engine=engine,
        num_queries=NUM_QUERIES,
        counter_callback=counter_callback,
    )
    print(f"Queries: {queries}")

    # get the top URLs for the queries
    urls = []
    try:
        urls = get_urls_from_queries(
            queries=queries,
            api_key=google_api_key,
            engine=google_engine_id,
            num=NUM_URLS_PER_QUERY,
        )
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error with Google Search API: {e}.")

    if not urls:
        print("Falling back to Serper API.")
        urls = get_urls_from_queries_serper(
            queries=queries,
            api_key=serper_api_key,
            num=NUM_URLS_PER_QUERY,
        )
    print(f"URLs: {urls}")

    # Extract text and dates from the URLs
    docs, counter_callback = extract_texts(
        urls=urls, client_=client_, counter_callback=counter_callback
    )

    # Remove None values from the list
    docs = [doc for doc in docs if doc]

    # remove doc with ""
    docs = [doc for doc in docs if hasattr(doc, "text") and doc.text != ""]

    # Chunk the documents
    split_docs = []
    for doc in docs:
        t = recursive_character_text_splitter(
            doc.text, SPLITTER_CHUNK_SIZE, SPLITTER_OVERLAP
        )
        split_docs.extend(
            [Document(text=chunk, date=doc.date, url=doc.url) for chunk in t]
        )
    print(f"Split Docs: {len(split_docs)}")

    if len(split_docs) == 0:
        raise ValueError("No valid documents found from the provided URLs")

    # Remove None values from the list
    split_docs = [doc for doc in split_docs if doc]

    # Embed the documents
    docs_with_embeddings = get_embeddings(split_docs)
    print(f"Docs with embeddings: {len(docs_with_embeddings)}")

    # Find similar chunks
    similar_chunks = find_similar_chunks(
        query=prompt,
        docs_with_embeddings=docs_with_embeddings,
        k=NUM_NEIGHBORS,
    )
    print(f"Similar Chunks: {len(similar_chunks)}")

    # Format the additional information
    additional_information = "\n".join(
        [
            f"ARTICLE {i}, URL: {doc.url}, DATE: {doc.date}, CONTENT: {doc.text}\n"
            for i, doc in enumerate(similar_chunks)
        ]
    )

    return additional_information, queries, counter_callback


def adjust_additional_information(
    prompt: str, additional_information: str, model: str
) -> str:
    """Adjust the additional_information to fit within the token budget"""

    # Initialize tiktoken encoder for the specified model
    enc = get_model_encoding(model)

    # Encode the user prompt to calculate its token count
    prompt_tokens = len(enc.encode(prompt))

    # Calculate available tokens for additional_information
    max_tokens_model = MAX_TOKENS.get(model)
    if max_tokens_model is None:
        raise ValueError(f"Max tokens for model {model} not defined in MAX_TOKENS.")
    MAX_PREDICTION_PROMPT_TOKENS = (
        max_tokens_model - DEFAULT_OPENAI_SETTINGS["max_tokens"]
    )
    available_tokens = MAX_PREDICTION_PROMPT_TOKENS - prompt_tokens - BUFFER_TOKENS

    # Encode the additional_information
    additional_info_tokens = enc.encode(additional_information)

    # If additional_information exceeds available tokens, truncate it
    if len(additional_info_tokens) > available_tokens:
        truncated_info_tokens = additional_info_tokens[:available_tokens]
        # Decode tokens back to text, ensuring the output fits within the budget
        additional_information = enc.decode(truncated_info_tokens)

    return additional_information


@with_key_rotation
def run(**kwargs: Any) -> Union[MaxCostResponse, MechResponse]:
    """Run the task"""
    tool = kwargs["tool"]
    engine = kwargs.get("model", TOOL_TO_ENGINE[tool])
    delivery_rate = int(kwargs.get("delivery_rate", DEFAULT_DELIVERY_RATE))
    counter_callback: Optional[Callable] = kwargs.get("counter_callback", None)
    if delivery_rate == 0:
        if not counter_callback:
            raise ValueError(
                "A delivery rate of `0` was passed, but no counter callback was given to calculate the max cost with."
            )

        max_cost = counter_callback(
            max_cost=True,
            models_calls=(engine,) * N_MODEL_CALLS,
        )
        return max_cost

    with OpenAIClientManager(kwargs["api_keys"]["openai"]):
        prompt = kwargs["prompt"]
        counter_callback = kwargs.get("counter_callback", None)
        api_keys = kwargs.get("api_keys", {})
        google_api_key = api_keys.get("google_api_key", None)
        google_engine_id = api_keys.get("google_engine_id", None)
        serper_api_key = api_keys.get("serperapi", None)

        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"Tool {tool} is not supported.")
        if not client:
            raise RuntimeError("Client not initialized")

        max_tokens = OPEN_AI_SETTINGS.get(engine, {}).get(
            "max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"]
        )

        # Check if question is valid
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": VALID_PROMPT.format(user_prompt=prompt),
            },
        ]

        response_valid = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=DEFAULT_OPENAI_SETTINGS["temperature"],
            max_tokens=max_tokens,
            n=1,
            timeout=150,
            stop=None,
            functions=[Valid.openai_schema],
            function_call={"name": "Valid"},
        )

        valid_results = Valid.from_response(response_valid)
        print(f"Valid: {valid_results}")

        if not valid_results.is_valid:
            return valid_results.json(), None, None, None

        (
            additional_information,
            _,
            counter_callback,
        ) = fetch_additional_information(
            client_=client,
            prompt=prompt,
            engine=engine,
            google_api_key=google_api_key,
            google_engine_id=google_engine_id,
            serper_api_key=serper_api_key,
            counter_callback=counter_callback,
        )

        # Adjust the additional_information to fit within the token budget
        adjusted_info = adjust_additional_information(
            prompt=PREDICTION_PROMPT,
            additional_information=additional_information,
            model=engine,
        )

        # Do reasoning
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": REASONING_PROMPT.format(
                    user_prompt=prompt, formatted_docs=adjusted_info
                ),
            },
        ]

        response_reasoning = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=DEFAULT_OPENAI_SETTINGS["temperature"],
            max_tokens=max_tokens,
            n=1,
            timeout=150,
            stop=None,
        )

        reasoning = response_reasoning.choices[0].message.content

        # Check if question is determinable
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": DETERMINABLE_PROMPT.format(
                    user_prompt=prompt, reasoning=reasoning
                ),
            },
        ]

        response_determinable = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=DEFAULT_OPENAI_SETTINGS["temperature"],
            max_tokens=max_tokens,
            n=1,
            timeout=150,
            stop=None,
            functions=[Determinable.openai_schema],
            function_call={"name": "Determinable"},
        )

        determinable_results = Determinable.from_response(response_determinable)
        print(f"Determinable: {determinable_results}")

        if not determinable_results.is_determinable:
            return determinable_results.json(), reasoning, None, None

        # Make the prediction
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PREDICTION_PROMPT.format(
                    user_prompt=prompt, reasoning=reasoning
                ),
            },
        ]

        response_prediction = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=DEFAULT_OPENAI_SETTINGS["temperature"],
            max_tokens=max_tokens,
            n=1,
            timeout=150,
            stop=None,
            functions=[Results.openai_schema],
            function_call={"name": "Results"},
        )

        results = Results.from_response(response_prediction)
        print(f"Results: {results}")

        # Make the prediction
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PREDICTION_PROMPT.format(
                    user_prompt=prompt, reasoning=reasoning
                ),
            },
        ]

        response_prediction = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=DEFAULT_OPENAI_SETTINGS["temperature"],
            max_tokens=max_tokens,
            n=1,
            timeout=150,
            stop=None,
            functions=[Results.openai_schema],
        )

        results = Results.from_response(response_prediction)
        print(f"Results: {results}")

        if counter_callback is not None:
            counter_callback(
                input_tokens=response_reasoning.usage.prompt_tokens
                + response_prediction.usage.prompt_tokens,
                output_tokens=response_reasoning.usage.completion_tokens
                + response_prediction.usage.completion_tokens,
                model=engine,
                token_counter=count_tokens,
            )
        return results.json(), reasoning, None, counter_callback
