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
import copy
import functools
import json
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from io import BytesIO
from itertools import islice
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import PyPDF2
import anthropic
import faiss
import googleapiclient
import numpy as np
import openai
import requests
from googleapiclient.discovery import build
from markdownify import markdownify as md
from pydantic import BaseModel, PositiveInt
from readability import Document as ReadabilityDocument
from requests.exceptions import RequestException, TooManyRedirects
from tiktoken import Encoding, encoding_for_model, get_encoding


MechResponseWithKeys = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]
MaxCostResponse = float


# Regular expression patterns
IMG_TAG_PATTERN = r"<img[^>]*>"
MARKDOWN_IMG_PATTERN = r"!\[.*?\]\((?:data:image/[^;]*;base64,[^)]*|.*?)\)"
DATA_URI_IMG_PATTERN = r'data:image/[^;]*;base64,[^"]*'
MARKDOWN_LINK_PATTERN = r"\[.*?\]\(.*?\)"
IMAGE_PATTERNS = [DATA_URI_IMG_PATTERN, MARKDOWN_IMG_PATTERN, IMG_TAG_PATTERN]
PHOTO_CREDIT_PATTERN = r"Photo:.*?\n"
IMAGE_CREDIT_PATTERN = r"Image:.*?\n"
IMAGE_RELATED_PATTERNS = [
    MARKDOWN_IMG_PATTERN,
    MARKDOWN_LINK_PATTERN,
    PHOTO_CREDIT_PATTERN,
    IMAGE_CREDIT_PATTERN,
]
N_MODEL_CALLS = 3

USER_AGENT_HEADER = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
GOOGLE_RATE_LIMIT_EXCEEDED_CODE = 429
DEFAULT_DELIVERY_RATE = 100


def get_model_encoding(model: str) -> Encoding:
    """Get the appropriate encoding for a model."""
    # Workaround since tiktoken does not have support yet for gpt4.1
    # https://github.com/openai/tiktoken/issues/395
    if model == "gpt-4.1-2025-04-14":
        return get_encoding("o200k_base")

    return encoding_for_model(model)


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
            except Exception as e:
                print(f"Unexpected error: {e}")
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


class LLMClientManager:
    """Client context manager for LLMs."""

    def __init__(self, api_keys: Dict, model: str, embedding_provider: str):
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
        clients = []
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

    def __init__(self, content: Optional[str] = None, usage: Optional[Usage] = None):
        """Initializes with content and usage class."""
        self.content = content
        self.usage = Usage()


class LLMClient:
    """Client for LLMs."""

    def __init__(self, api_keys: Dict, llm_provider: str):
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
        messages: List = [],  # noqa: B006
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
            messages_copy = copy.deepcopy(messages)
            for i in range(len(messages_copy) - 1, -1, -1):
                if messages_copy[i]["role"] == "system":
                    system_prompt = messages_copy[i]["content"]
                    del messages_copy[i]

            response_provider = self.client.messages.create(  # pylint: disable=no-member
                model=model,
                messages=messages_copy,
                system=system_prompt,  # pylint: disable=possibly-used-before-assignment
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = LLMResponse()
            response.content = response_provider.content[0].text
            response.usage.prompt_tokens = response_provider.usage.input_tokens
            response.usage.completion_tokens = response_provider.usage.output_tokens
            return response

        if self.llm_provider in ["openai", "openrouter"]:
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
        if self.llm_provider not in ["openai", "openrouter"]:
            print("Only OpenAI embeddings supported currently.")
            return None

        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=input_,
            )
            return response
        except Exception as e:
            raise ValueError(
                f"Error while generating the embeddings for the docs {e}"
            ) from e


client: Optional[LLMClient] = None
client_embedding: Optional[LLMClient] = None

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
    "gpt-4.1-2025-04-14": {
        "default_max_tokens": 4096,
        "limit_max_tokens": 1_047_576,
        "temperature": 0,
    },
    "claude-3-haiku-20240307": {
        "default_max_tokens": 1000,
        "limit_max_tokens": 200_000,
        "temperature": 0,
    },
    "claude-3-5-sonnet-20240620": {
        "default_max_tokens": 1000,
        "limit_max_tokens": 200_000,
        "temperature": 0,
    },
    "claude-4-sonnet-20250514": {
        "default_max_tokens": 4096,
        "limit_max_tokens": 200_000,
        "temperature": 0,
    },
    "claude-3-opus-20240229": {
        "default_max_tokens": 1000,
        "limit_max_tokens": 200_000,
        "temperature": 0,
    },
    "databricks/dbrx-instruct:nitro": {
        "default_max_tokens": 500,
        "limit_max_tokens": 32_768,
        "temperature": 0,
    },
    "nousresearch/nous-hermes-2-mixtral-8x7b-sft": {
        "default_max_tokens": 1000,
        "limit_max_tokens": 32_000,
        "temperature": 0,
    },
}
ALLOWED_TOOLS = [
    "prediction-request-reasoning",
    # LEGACY
    "prediction-request-reasoning-claude",
]
ALLOWED_MODELS = list(LLM_SETTINGS.keys())
DEFAULT_NUM_URLS = 3
DEFAULT_NUM_QUERIES = 2
SPLITTER_CHUNK_SIZE = 300
SPLITTER_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_BATCH_SIZE = 1000
EMBEDDING_SIZE = 3072
EMBEDDING_CTX_LENGTH = 8191  # for this embedding model
NUM_NEIGHBORS = 3
BUFFER_TOKENS = 250
HTTP_TIMEOUT = 20
HTTP_MAX_REDIRECTS = 5
HTTP_MAX_RETIES = 2
DEFAULT_RETRIES = 3
DEFAULT_DELAY = 2

DOC_TOKEN_LIMIT = 7000  # Maximum tokens per document for embeddings
# Reasoning prompt has around 250 tokens so we adjust the max tokens accordingly
# to avoid exceeding the limit when adding reasoning
PREDICTION_PROMPT_LENGTH = 500
BUFFER = 15000  # Buffer to avoid exceeding the limit when adding reasoning
# This is a rough estimate, actual token count may vary based on the model and text
DEFAULT_MAX_EMBEDDING_TOKENS = 300000
MAX_NR_DOCS = 1000
TOKENS_DISTANCE_TO_LIMIT = 200
if DEFAULT_MAX_EMBEDDING_TOKENS - PREDICTION_PROMPT_LENGTH - BUFFER <= 0:
    raise ValueError("Wrong MAX_EMBEDDING_TOKENS configuration")


class ExtendedDocument(BaseModel):
    """Document model"""

    text: str
    url: str
    tokens: PositiveInt = 0
    embedding: Optional[List[float]] = None


URL_QUERY_PROMPT = """
Here is the user prompt: {USER_PROMPT}

Please read the prompt carefully and identify the key pieces of information that need to be searched for in order to comprehensively address the topic.

Brainstorm a list of {NUM_QUERIES} different search queries that cover various aspects of the user prompt. Each query should be focused on a specific sub-topic or question related to the overarching prompt.

Please write each search query inside its own tags, like this: <query>example search query here</query>

The queries should be concise while still containing enough information to return relevant search results. Focus the queries on gathering factual information to address the prompt rather than opinions.

After you have written all {NUM_QUERIES} search queries, please submit your final response.

<queries></queries>
"""


PREDICTION_PROMPT = """
You will be evaluating the likelihood of an event based on a user's question and reasoning provided by another AI.
The user's question is: <user_input> {USER_INPUT} </user_input>

The reasoning from the other AI is: {REASONING}

Carefully consider the user's question and the provided reasoning. Then, think through the following:
 - The probability that the event specified in the user's question will happen (p_yes)
 - The probability that the event will not happen (p_no)
 - Your confidence level in your prediction
 - How useful the reasoning was in helping you make your prediction (info_utility)

Provide your final scores in the following format: <p_yes>probability between 0 and 1</p_yes> <p_no>probability between 0 and 1</p_no>
your confidence level between 0 and 1 <info_utility>utility of the reasoning between 0 and 1</info_utility>

Remember, p_yes and p_no should add up to 1. Provide your reasoning for each score in the scratchpad before giving your final scores.

Your response should be structured as follows:
<p_yes></p_yes>
<p_no></p_no>
<info_utility></info_utility>
<confidence></confidence>
<analysis></analysis>
"""


REASONING_PROMPT = """
Here is the user's question: {USER_PROMPT}
Here is some additional information that may be relevant to answering the question: <additional_information> {ADDITIONAL_INFOMATION} </additional_information>

Please carefully read the user's question and the additional information provided. Think through the problem step-by-step, taking into account:

- The key details from the user's question, such as the specific event they are asking about and the date by which they want to know if it will occur
- Any relevant facts or context provided in the additional information that could help inform your reasoning
- Your own knowledge and analytical capabilities to reason through the likelihood of the event happening by the specified date

Explain your thought process and show your reasoning for why you believe the event either will or will not occur by the given date. Provide your response inside tags.
<reasoning></reasoning>
"""


MULTI_QUESTIONS_PROMPT = """
Your task is to take the following user question:
{USER_INPUT}

Generate 3 different versions of this question that capture different aspects, perspectives or phrasings of the original question. The goal is to help retrieve a broader set of relevant documents from a vector database by providing multiple ways of expressing the user's information need.

Please provide your output in the following XML tags:

<multiple_questions>
[First question version]
[Second question version]
[Third question version]
</multiple_questions>

Each question version should aim to surface different keywords, concepts or aspects of the original question while still being relevant to answering the user's core information need. Vary the phrasing and terminology used in each question. However, do not introduce any information that is not implied by the original question.

"""


SYSTEM_PROMPT = """You are a world class algorithm for generating structured output from a given input."""


def get_max_embeddings_tokens(model: str) -> int:
    """Get the maximum number of tokens for embeddings based on the model."""

    if model in LLM_SETTINGS:
        # Maximum tokens for the embeddings batch
        # there are models with values under 300000
        limit_max_tokens = min(
            LLM_SETTINGS[model]["limit_max_tokens"], DEFAULT_MAX_EMBEDDING_TOKENS
        )
        max_embeddings_tokens = limit_max_tokens - PREDICTION_PROMPT_LENGTH - BUFFER
        if max_embeddings_tokens <= 0:
            raise ValueError(
                f"Model {model} has a limit_max_tokens that is too low for embeddings."
            )
        return max_embeddings_tokens

    return DEFAULT_MAX_EMBEDDING_TOKENS - PREDICTION_PROMPT_LENGTH - BUFFER


def create_messages(
    user_content: str, system_content: str = SYSTEM_PROMPT
) -> List[Dict]:
    """Create standard message structure for LLM requests."""
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def parser_query_response(response: str, num_queries: int = 5) -> List[str]:
    """Parse the response from the query generation model with optional enhancements."""
    queries = response.split("<queries>")[1].split("</queries>")[0].split("\n")
    parsed_queries = [query.strip() for query in queries if query.strip()]
    enhanced_queries = []

    for query in parsed_queries:
        if query[0].isdigit():
            query = ". ".join(query.split(". ")[1:])
        query = query.replace('"', "")
        enhanced_queries.append(query)

    if len(enhanced_queries) == num_queries * 2:
        enhanced_queries = enhanced_queries[::2]

    # Remove doubel quotes from the queries
    final_queries = [query.replace('"', "") for query in enhanced_queries]

    # if there are any xml tags in the queries, remove them
    final_queries = [re.sub(r"<[^>]*>", "", query) for query in final_queries]

    return final_queries


def parser_multi_questions_response(response: str) -> List[str]:
    """Parse the response from the multi questions generation model."""
    questions = (
        response.split("<multiple_questions>")[1]
        .split("</multiple_questions>")[0]
        .split("\n")
    )
    return [question.strip() for question in questions if question.strip()]


def parser_reasoning_response(response: str) -> str:
    """Parse the response from the reasoning model."""
    if "<reasoning>" not in response or "</reasoning>" not in response:
        print("Invalid response format: missing reasoning tags")
        return ""
    reasoning = response.split("<reasoning>")[1].split("</reasoning>")[0]
    return reasoning.strip()


def parser_prediction_response(response: str) -> str:
    """Parse the response from the prediction model."""
    tags = ["p_yes", "p_no", "info_utility", "confidence"]
    results = {}

    for key in tags:
        try:
            value_str = response.split(f"<{key}>")[1].split(f"</{key}>")[0].strip()
            value = float(value_str)
            results[key] = value
        except Exception as e:
            print("Not a valid answer from the model")
            print(f"response = {response}")
            raise ValueError(f"Error for {key}: {value}") from e

    return json.dumps(results)


def multi_queries(
    prompt: str,
    model: str,
    num_queries: int,
    counter_callback: Optional[Callable] = None,
    temperature: Optional[float] = LLM_SETTINGS["gpt-4.1-2025-04-14"]["temperature"],
    max_tokens: Optional[int] = LLM_SETTINGS["gpt-4.1-2025-04-14"][
        "default_max_tokens"
    ],
) -> Tuple[List[str], Optional[Callable]]:
    """Generate multiple queries for fetching information from the web."""
    if not client:
        raise RuntimeError("Client not initialized")

    url_query_prompt = URL_QUERY_PROMPT.format(
        USER_PROMPT=prompt, NUM_QUERIES=num_queries
    )
    messages = create_messages(user_content=url_query_prompt)

    response = client.completions(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not response or response.content is None:
        raise RuntimeError("Response not found")
    if counter_callback:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=model,
            token_counter=count_tokens,
        )
    queries = parser_query_response(response.content, num_queries=num_queries)
    # remove empty queries, including ""
    queries = [query for query in queries if query.strip() != ""]
    if len(queries) > DEFAULT_NUM_QUERIES:
        queries = queries[:DEFAULT_NUM_QUERIES]
    queries.append(prompt)

    return queries, counter_callback


def search_google(query: str, api_key: str, engine: str, num: int) -> List[str]:
    """Search Google for the given query."""
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
    return list(set(urls))


def extract_text_from_pdf(
    url: str, num_words: Optional[int] = None
) -> Optional[ExtendedDocument]:
    """Extract text from a PDF document at the given URL."""
    try:
        response = requests.get(
            url,
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": USER_AGENT_HEADER},
        )
        response.raise_for_status()

        if "application/pdf" not in response.headers.get("Content-Type", ""):
            raise ValueError("URL does not point to a PDF document")

        with BytesIO(response.content) as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

        doc = ExtendedDocument(
            text=text[:num_words] if num_words else text, date="", url=url
        )
        print(f"Using PDF: {url}: {doc.text[:300]}...")
        return doc

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def extract_text(
    html: str, num_words: Optional[int] = None
) -> Optional[ExtendedDocument]:
    """Extract text from a single HTML document"""
    # Remove image patterns
    for pattern in IMAGE_PATTERNS:
        html = re.sub(pattern, "", html)

    text = ReadabilityDocument(html).summary()
    text = md(text, heading_style="ATX")
    if text is None:
        return None

    # Remove any remaining image-related content
    for pattern in IMAGE_RELATED_PATTERNS:
        text = re.sub(pattern, "", text)

    words = text.split()
    text = " ".join(words[:num_words]) if num_words else " ".join(words)
    # final cleaning
    doc = ExtendedDocument(text=text, url="")
    return doc


def extract_texts(
    urls: List[str], num_words: Optional[int] = None
) -> List[ExtendedDocument]:
    """Extract texts from URLs with improved error handling, excluding failed URLs."""
    extracted_texts = []
    for batch in process_in_batches(urls=urls) or []:
        for future, url in batch:
            if future is None:
                continue
            try:
                result = future.result()
                if not result:
                    print(f"No result returned for {url}")
                    continue
                if isinstance(result, requests.Response) and result.status_code == 200:
                    # Check if URL ends with .pdf or content starts with %PDF
                    is_pdf = url.endswith(".pdf") or result.content[:4] == b"%PDF"
                    doc = (
                        extract_text_from_pdf(url, num_words=num_words)
                        if is_pdf
                        else extract_text(html=result.text, num_words=num_words)
                    )

                    if doc:
                        doc.url = url
                        extracted_texts.append(doc)
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue
    return extracted_texts


def process_in_batches(
    urls: List[str],
    window: int = 5,
    timeout: int = HTTP_TIMEOUT,
    max_redirects: int = HTTP_MAX_REDIRECTS,
    retries: int = HTTP_MAX_RETIES,
) -> Generator[List[Tuple[Optional[Future], str]], None, None]:
    """Iter URLs in batches with improved error handling and retry mechanism."""
    session: requests.Session
    with ThreadPoolExecutor() as executor, requests.Session() as session:
        session.max_redirects = max_redirects
        for i in range(0, len(urls), window):
            batch = urls[i : i + window]
            futures = []
            for url in batch:
                future = None
                attempt = 0
                while attempt < retries:
                    try:
                        future = executor.submit(
                            session.get,
                            url,
                            timeout=timeout,
                            headers={"User-Agent": USER_AGENT_HEADER},
                        )
                        break
                    except (TooManyRedirects, RequestException) as e:
                        print(f"Attempt {attempt + 1} failed for {url}: {e}")
                        attempt += 1
                        if attempt == retries:
                            print(f"Max retries reached for {url}. Moving to next URL.")
                futures.append((future, url))
            yield futures


def recursive_character_text_splitter(
    text: str, max_tokens: int, overlap: int
) -> List[str]:
    """Splits the input text into chunks of size `max_tokens`, with an overlap between chunks."""
    if len(text) <= max_tokens:
        return [text]
    return [text[i : i + max_tokens] for i in range(0, len(text), max_tokens - overlap)]


def clean_text(text: str) -> str:
    """Remove emojis and non-printable characters, collapse whitespace."""
    emoji_pattern = re.compile(
        "["
        "\U0001f300-\U0001f5ff"
        "\U0001f600-\U0001f64f"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    # Decode using UTF-8, replacing invalid bytes
    text = text.encode("utf-8", "replace").decode("utf-8", "replace")
    text = "".join(ch for ch in text if ch.isprintable())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_text(text: str, model: str, max_tokens: int) -> str:
    """Truncate text to the first max_tokens tokens based on model encoding."""
    enc = encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def get_embeddings(
    split_docs: List[ExtendedDocument], model: str
) -> List[ExtendedDocument]:
    """Get embeddings for the split documents: clean, truncate, then batch by token count."""
    if not client_embedding:
        raise RuntimeError("Embeddings not intialized")
    # Preprocess each document: clean and truncate to DOC_TOKEN_LIMIT
    # Filter out any documents that exceed the maximum token limit individually
    filtered_docs = []
    total_tokens_count = 0
    max_embeddings_tokens = get_max_embeddings_tokens(model)
    for doc in split_docs:
        # if we are very close to the limit then break the loop
        if max_embeddings_tokens - total_tokens_count < TOKENS_DISTANCE_TO_LIMIT:
            break
        cleaned = clean_text(doc.text)
        # TODO we could summarize instead of truncating
        doc.text = truncate_text(cleaned, EMBEDDING_MODEL, DOC_TOKEN_LIMIT)
        # filter empty strings
        doc.text = doc.text.strip()
        doc.tokens = count_tokens(doc.text, EMBEDDING_MODEL)
        if total_tokens_count + doc.tokens > max_embeddings_tokens:
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
            if doc.tokens == 0:
                doc.tokens = count_tokens(doc.text, EMBEDDING_MODEL)

            if current_batch_tokens + doc.tokens > max_embeddings_tokens:
                break

            current_batch_docs.append(doc)
            current_batch_tokens += doc.tokens
            i += 1

        if not current_batch_docs:
            # This should not happen after filtering, but just in case
            i += 1
            continue
        batch_texts = [doc.text for doc in current_batch_docs]
        response = client_embedding.embeddings(
            model=EMBEDDING_MODEL,
            input_=batch_texts,
        )

        for j, emb in enumerate(response.data):
            assert j == emb.index, "Embeddings response out-of-order"
            current_batch_docs[j].embedding = emb.embedding
            processed_docs.append(current_batch_docs[j])

    return processed_docs


def find_similar_chunks(
    query: str, docs_with_embeddings: List[ExtendedDocument], k: int = 4
) -> List:
    """Similarity search to find similar chunks to a query"""
    if not client_embedding:
        raise RuntimeError("Embeddings not intialized")

    query_embedding = (
        client_embedding.embeddings(
            model=EMBEDDING_MODEL,
            input_=query,
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


def multi_questions_response(
    prompt: str,
    model: str,
    temperature: float = LLM_SETTINGS["gpt-4.1-2025-04-14"]["temperature"],
    max_tokens: int = LLM_SETTINGS["gpt-4.1-2025-04-14"]["default_max_tokens"],
    counter_callback: Optional[Callable] = None,
) -> Tuple[List[str], Optional[Callable]]:
    """Generate multiple questions for fetching information from the web."""
    if not client:
        raise RuntimeError("Client not initialized")
    try:
        multi_questions_prompt = MULTI_QUESTIONS_PROMPT.format(USER_INPUT=prompt)
        messages = create_messages(user_content=multi_questions_prompt)

        response = client.completions(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not response or response.content is None:
            raise RuntimeError("Response not found")

        if counter_callback:
            counter_callback(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=model,
                token_counter=count_tokens,
            )

        # append the user's question to the list of questions
        multi_questions = parser_multi_questions_response(response.content)
        multi_questions.append(prompt)

        return multi_questions, counter_callback

    except Exception as e:
        print(f"Error generating multiple questions: {e}")
        return [prompt], counter_callback


def reciprocal_rank_refusion(
    similar_chunks: List[ExtendedDocument], k: int
) -> List[ExtendedDocument]:
    """Reciprocal rank refusion to re-rank the similar chunks based on the text."""
    fused_chunks = {}
    for rank, doc in enumerate(similar_chunks):
        doc_text = doc.text
        if doc_text not in fused_chunks:
            fused_chunks[doc_text] = (doc, 0.0)
        fused_chunks[doc_text] = (doc, fused_chunks[doc_text][1] + 1 / (rank + 60))

    sorted_fused_chunks = sorted(
        fused_chunks.values(), key=lambda x: x[1], reverse=True
    )

    return [doc for doc, _ in sorted_fused_chunks[:k]]


def do_reasoning_with_retry(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    retries: int = DEFAULT_RETRIES,
    delay: int = DEFAULT_DELAY,
    counter_callback: Optional[Callable] = None,
) -> Tuple[str, Optional[Callable]]:
    """Attempt to do reasoning with retries on failure."""
    if not client:
        raise RuntimeError("Client not intialized")

    attempt = 0
    tool_errors = []
    while attempt < retries:
        try:
            response_reasoning = client.completions(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                timeout=90,
                stop=None,
            )
            if not response_reasoning or response_reasoning.content is None:
                return "", counter_callback

            if counter_callback is not None:
                counter_callback(
                    input_tokens=response_reasoning.usage.prompt_tokens,
                    output_tokens=response_reasoning.usage.completion_tokens,
                    model=model,
                    token_counter=count_tokens,
                )
            reasoning = parser_reasoning_response(response_reasoning.content)

            return reasoning, counter_callback
        except Exception as e:
            error = f"Attempt {attempt + 1} failed with error: {e}"
            time.sleep(delay)
            # join the tool errors with the exception message
            tool_errors.append(error)
            attempt += 1
    error_message = (
        f"Failed to generate prediction after retries:\n{chr(10).join(tool_errors)}"
    )
    raise Exception(error_message)  # pylint: disable=broad-exception-raised


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    # Check if we're using a Claude model and we have an active client
    if "claude" in model.lower() and client and client.llm_provider == "anthropic":
        try:
            # Use Anthropic's tokenizer when available
            response = client.client.messages.count_tokens(
                model=model, messages=[{"role": "user", "content": text}]
            )
            return response.input_tokens
        except (AttributeError, Exception):
            # Fallback if the method doesn't exist or fails
            print("Using fallback enconding for Claude models")
            enc = get_encoding("cl100k_base")
            return len(enc.encode(text))
    enc = get_model_encoding(model)
    return len(enc.encode(text))


def fetch_additional_information(  # pylint: disable=too-many-statements
    prompt: str,
    model: str,
    google_api_key: Optional[str],
    google_engine_id: Optional[str],
    serper_api_key: Optional[str],
    search_provider: str,
    counter_callback: Optional[Callable[[int, int, str], None]] = None,
    source_links: Optional[Dict] = None,
    num_urls: int = DEFAULT_NUM_URLS,
    num_queries: int = DEFAULT_NUM_QUERIES,
    temperature: float = LLM_SETTINGS["gpt-4.1-2025-04-14"]["temperature"],
    max_tokens: int = LLM_SETTINGS["gpt-4.1-2025-04-14"]["default_max_tokens"],
) -> Tuple[str, List[str], Optional[Callable[[int, int, str], None]]]:
    """Fetch additional information from the web."""
    # generate multiple queries for fetching information from the web
    try:
        queries, counter_callback = multi_queries(
            prompt=prompt,
            model=model,
            num_queries=num_queries,
            counter_callback=counter_callback,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print(f"Queries: {queries}")
    except Exception as e:
        print(f"Error generating queries: {e}")
        queries = [prompt]

    # get the top URLs for the queries
    if not source_links:
        # Determine which search provider to use
        if search_provider == "serper":
            if not serper_api_key:
                raise RuntimeError("Serper API key not found")
            urls = get_urls_from_queries_serper(
                queries=queries,
                api_key=serper_api_key,
                num=num_urls,
            )
        else:  # default to google
            if not google_api_key:
                raise RuntimeError("Google API key not found")
            if not google_engine_id:
                raise RuntimeError("Google Engine Id not found")
            urls = get_urls_from_queries(
                queries=queries,
                api_key=google_api_key,
                engine=google_engine_id,
                num=num_urls,
            )
        print(f"URLs: {urls}")

        urls = list(set(urls))

        # Extract text and dates from the URLs
        docs = extract_texts(
            urls=urls,
        )
    else:
        docs = []
        for url, content in islice(source_links.items(), num_urls or len(source_links)):
            doc = extract_text(html=content)
            if doc:
                doc.url = url
                docs.append(doc)

    # Remove None values from the list
    docs = [doc for doc in docs if doc]

    # remove empty documents with ""
    filtered_docs = [doc for doc in docs if hasattr(doc, "text") and doc.text != ""]

    # Chunk the documents
    split_docs = []
    for doc in filtered_docs:
        try:
            t = recursive_character_text_splitter(
                doc.text, SPLITTER_CHUNK_SIZE, SPLITTER_OVERLAP
            )
            split_docs.extend(
                [ExtendedDocument(text=chunk, url=doc.url) for chunk in t]
            )
        except Exception as e:
            print(f"Error splitting document: {e}")
            continue

    # Remove None values from the list
    split_docs = [doc for doc in split_docs if doc]

    print(f"Split Docs: {len(split_docs)}")

    if len(split_docs) == 0:
        raise ValueError("No valid documents found from the provided URLs")

    if len(split_docs) > MAX_NR_DOCS:
        # truncate the split_docs to the first MAX_NR_DOCS documents
        split_docs = split_docs[:MAX_NR_DOCS]
    # Embed the documents
    docs_with_embeddings = get_embeddings(split_docs, model)

    # multi questions prompt
    questions, counter_callback = multi_questions_response(
        prompt=prompt,
        model=model,
        counter_callback=counter_callback,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    print(f"Questions: {questions}")

    similar_chunks = []
    for question in questions:
        similar_chunks.extend(
            find_similar_chunks(
                query=question,
                docs_with_embeddings=docs_with_embeddings,
                k=NUM_NEIGHBORS,
            )
        )
    print(f"Similar Chunks before refusion: {len(similar_chunks)}")

    # Reciprocal rank refusion
    similar_chunks = reciprocal_rank_refusion(similar_chunks, NUM_NEIGHBORS)
    print(f"Similar Chunks after refusion: {len(similar_chunks)}")

    # Format the additional information
    additional_information = "\n".join(
        [
            f"ARTICLE {i}, URL: {doc.url}, CONTENT: {doc.text}\n"
            for i, doc in enumerate(similar_chunks)
        ]
    )

    return additional_information, queries, counter_callback


def extract_question(prompt: str) -> str:
    """Uses regexp to extract question from the prompt"""
    # Match from 'question "' to '" and the `yes`' to handle nested quotes
    pattern = r'question\s+"(.+?)"\s+and\s+the\s+`yes`'
    try:
        question = re.findall(pattern, prompt, re.DOTALL)[0]
    except Exception as e:
        print(f"Error extracting question: {e}")
        question = prompt

    return question


@with_key_rotation
def run(**kwargs: Any) -> Union[MaxCostResponse, MechResponse]:
    """Run the task"""
    tool = kwargs["tool"]
    model = kwargs.get("model")
    if model is None:
        raise ValueError("Model must be specified in kwargs")

    delivery_rate = int(kwargs.get("delivery_rate", DEFAULT_DELIVERY_RATE))
    counter_callback: Optional[Callable] = kwargs.get("counter_callback", None)
    if delivery_rate == 0:
        if not counter_callback:
            raise ValueError(
                "A delivery rate of `0` was passed, but no counter callback was given to calculate the max cost with."
            )

        max_cost = counter_callback(
            max_cost=True,
            models_calls=(model,) * N_MODEL_CALLS,
        )
        return max_cost

    if "claude" in tool:  # maintain backwards compatibility
        model = "claude-4-sonnet-20250514"
    print(f"MODEL for prediction request reasoning: {model}")
    with LLMClientManager(kwargs["api_keys"], model, embedding_provider="openai"):
        prompt = extract_question(kwargs["prompt"])
        max_tokens = kwargs.get("max_tokens", LLM_SETTINGS[model]["default_max_tokens"])
        temperature = kwargs.get("temperature", LLM_SETTINGS[model]["temperature"])
        num_urls = kwargs.get("num_urls", DEFAULT_NUM_URLS)
        num_queries = kwargs.get("num_queries", DEFAULT_NUM_QUERIES)
        counter_callback = kwargs.get("counter_callback", None)

        api_keys = kwargs.get("api_keys", {})
        google_api_key = api_keys.get("google_api_key", None)
        google_engine_id = api_keys.get("google_engine_id", None)
        serper_api_key = api_keys.get("serperapi", None)
        search_provider = api_keys.get("search_provider", "google")

        if not client:
            raise RuntimeError("Client not initialized")

        # Make sure the model is supported
        if model not in ALLOWED_MODELS:
            raise ValueError(f"Model {model} not supported.")

        # make sure the tool is supported
        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"Tool {tool} not supported.")

        (
            additional_information,
            _,
            counter_callback,
        ) = fetch_additional_information(
            prompt=prompt,
            model=model,
            google_api_key=google_api_key,
            google_engine_id=google_engine_id,
            serper_api_key=serper_api_key,
            search_provider=search_provider,
            counter_callback=counter_callback,
            source_links=kwargs.get("source_links", None),
            num_urls=num_urls,
            num_queries=num_queries,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Reasoning prompt
        reasoning_prompt = REASONING_PROMPT.format(
            USER_PROMPT=prompt, ADDITIONAL_INFOMATION=additional_information
        )

        # Do reasoning
        messages = create_messages(user_content=reasoning_prompt)

        reasoning, counter_callback = do_reasoning_with_retry(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=DEFAULT_RETRIES,
            delay=DEFAULT_DELAY,
            counter_callback=counter_callback,
        )

        # Prediction prompt
        prediction_prompt = PREDICTION_PROMPT.format(
            USER_INPUT=prompt, REASONING=reasoning
        )

        # Make the prediction
        messages = create_messages(user_content=prediction_prompt)

        response_prediction = client.completions(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not response_prediction or response_prediction.content is None:
            return (
                "Response Prediction Not Valid",
                prediction_prompt,
                None,
                counter_callback,
            )

        prediction = parser_prediction_response(response_prediction.content)
        if not prediction:
            return "Prediction Not Valid", prediction_prompt, None, counter_callback

        if counter_callback:
            counter_callback(
                input_tokens=response_prediction.usage.prompt_tokens,
                output_tokens=response_prediction.usage.completion_tokens,
                model=model,
                token_counter=count_tokens,
            )

        return (
            prediction,
            reasoning_prompt + "////" + prediction_prompt,
            None,
            counter_callback,
        )
