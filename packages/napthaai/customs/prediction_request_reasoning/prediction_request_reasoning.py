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

"""This module implements a Mech tool for binary predictions."""

import re
import json
import faiss
import PyPDF2
import requests
import time
import numpy as np
from io import BytesIO
from itertools import islice
from pydantic import BaseModel
from collections import defaultdict
from tiktoken import encoding_for_model
from markdownify import markdownify as md
from googleapiclient.discovery import build
from readability import Document as ReadabilityDocument
from concurrent.futures import Future, ThreadPoolExecutor
from requests.exceptions import RequestException, TooManyRedirects
from typing import Any, Dict, Generator, List, Optional, Tuple, Callable, Union


class LLMClientManager:
    """Client context manager for LLMs."""

    def __init__(
        self, api_keys: List, llm_provider: str = None, embedding_provider: str = None
    ):
        self.api_keys = api_keys
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider

    def __enter__(self):
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

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global client
        if client is not None:
            client.client.close()
            client = None


class Usage:
    """Usage class."""

    def __init__(self, prompt_tokens=None, completion_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class LLMResponse:
    """Response class."""

    def __init__(self, content: Optional[str] = None, usage: Optional[Usage] = None):
        self.content = content
        self.usage = Usage()


class LLMClient:
    """Client for LLMs."""

    def __init__(self, api_keys: Dict, llm_provider: str):
        self.api_keys = api_keys
        self.llm_provider = llm_provider
        if self.llm_provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_keys["anthropic"])
        if self.llm_provider == "openai":
            import openai

            self.client = openai.OpenAI(api_key=self.api_keys["openai"])
        if self.llm_provider == "openrouter":
            import openai

            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_keys["openrouter"],
            )

    def completions(
        self,
        model: str,
        messages: List = [],
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop=None,
        max_tokens: Optional[float] = None,
    ):
        if self.llm_provider == "anthropic":
            # anthropic can't take system prompt in messages
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "system":
                    system_prompt = messages[i]["content"]
                    del messages[i]

            response_provider = self.client.messages.create(
                model=model,
                messages=messages,
                system=system_prompt,
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
            # transform = [] # to desactivate prompt compression
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

    def embeddings(self, model, input):
        if self.llm_provider == "openai" or self.llm_provider == "openrouter":
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=input,
            )
            return response
        else:
            print("Only OpenAI embeddings supported currently.")
            return None


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
    "claude-3-haiku-20240307": {
        "default_max_tokens": 1000,
        "limit_max_tokens": 200_000,
        "temperature": 0,
    },
    "claude-3-sonnet-20240229": {
        "default_max_tokens": 1000,
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
]
ALLOWED_MODELS = list(LLM_SETTINGS.keys())
DEFAULT_MODEL = "claude-3-haiku-20240307"
TOOL_TO_ENGINE = {tool: DEFAULT_MODEL for tool in ALLOWED_TOOLS}
DEFAULT_NUM_URLS = defaultdict(lambda: 3)
DEFAULT_NUM_QUERIES = defaultdict(lambda: 3)
SPLITTER_CHUNK_SIZE = 300
SPLITTER_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_BATCH_SIZE = 1000
EMBEDDING_SIZE = 3072
NUM_NEIGHBORS = 3
BUFFER_TOKENS = 250
HTTP_TIMEOUT = 20
HTTP_MAX_REDIRECTS = 5
HTTP_MAX_RETIES = 2
DEFAULT_RETRIES = 3
DEFAULT_DELAY = 2

class Document(BaseModel):
    text: str
    url: str
    embedding: Optional[List[float]] = None


URL_QUERY_PROMPT = """
Here is the user prompt: {USER_PROMPT}

Please read the prompt carefully and identify the key pieces of information that need to be searched for in order to comprehensively address the topic.

Brainstorm a list of {NUM_QUERIES} different search queries that cover various aspects of the user prompt. Each query should be focused on a specific sub-topic or question related to the overarching prompt.

Please write each search query inside its own tags, like this: example search query here

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
    reasoning = response.split("<reasoning>")[1].split("</reasoning>")[0]
    return reasoning.strip()


def parser_prediction_response(response: str) -> str:
    """Parse the response from the prediction model."""
    results = {}
    if "p_yes" not in response:
        print("Not a valid answer from the model")
        print(f"response = {response}")
        return results

    for key in ["p_yes", "p_no", "info_utility", "confidence"]:
        try:
            value = response.split(f"<{key}>")[1].split(f"</{key}>")[0].strip()
            if key in ["p_yes", "p_no", "info_utility", "confidence"]:
                value = float(value)
            results[key] = value
        except Exception:
            raise ValueError(f"Error parsing {key}")

    results = json.dumps(results)
    return results


def multi_queries(
    prompt: str,
    engine: str,
    num_queries: int,
    counter_callback: Optional[Callable[[int, int, str], None]] = None,
    temperature: Optional[float] = LLM_SETTINGS[DEFAULT_MODEL]["temperature"],
    max_tokens: Optional[int] = LLM_SETTINGS[DEFAULT_MODEL]["default_max_tokens"],
) -> List[str]:
    """Generate multiple queries for fetching information from the web."""
    url_query_prompt = URL_QUERY_PROMPT.format(
        USER_PROMPT=prompt, NUM_QUERIES=num_queries
    )

    messages = [
        {"role": "user", "content": url_query_prompt},
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    response = client.completions(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if counter_callback:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=engine,
            token_counter=count_tokens,
        )
    queries = parser_query_response(response.content, num_queries=num_queries)
    queries.append(prompt)

    return queries, counter_callback


def search_google(query: str, api_key: str, engine: str, num: int) -> List[str]:
    """Search Google for the given query."""
    service = build("customsearch", "v1", developerKey=api_key)
    search = (
        service.cse()
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
        except Exception:
            pass
    unique_results = list(set(results))
    return unique_results


def extract_text_from_pdf(url: str, num_words: Optional[int] = None) -> str:
    """Extract text from a PDF document at the given URL."""
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()

        if "application/pdf" not in response.headers.get("Content-Type", ""):
            return ValueError("URL does not point to a PDF document")

        with BytesIO(response.content) as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

        doc = Document(text=text[:num_words] if num_words else text, date="", url=url)
        print(f"Using PDF: {url}: {doc.text[:300]}...")
        return doc

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def extract_text(
    html: str,
    num_words: Optional[int] = None,
) -> str:
    """Extract text from a single HTML document"""
    text = ReadabilityDocument(html).summary()

    # use html2text to convert HTML to markdown
    text = md(text, heading_style="ATX")

    if text is None:
        return None

    if num_words:
        text = " ".join(text.split()[:num_words])
    else:
        text = " ".join(text.split())

    doc = Document(text=text, url="")
    return doc


def extract_texts(urls: List[str], num_words: Optional[int] = None) -> List[Document]:
    """Extract texts from URLs with improved error handling, excluding failed URLs."""
    extracted_texts = []
    for batch in process_in_batches(urls=urls):
        for future, url in batch:
            if future is None:
                continue
            try:
                result = future.result()
                if result.status_code == 200:
                    # Check if URL ends with .pdf or content starts with %PDF
                    if url.endswith(".pdf") or result.content[:4] == b"%PDF":
                        doc = extract_text_from_pdf(url, num_words=num_words)
                    else:
                        doc = extract_text(html=result.text, num_words=num_words)
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
) -> Generator[None, None, List[Tuple[Optional[Future], str]]]:
    """Iter URLs in batches with improved error handling and retry mechanism."""
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
                        future = executor.submit(session.get, url, timeout=timeout)
                        break
                    except (TooManyRedirects, RequestException) as e:
                        print(f"Attempt {attempt + 1} failed for {url}: {e}")
                        attempt += 1
                        if attempt == retries:
                            print(f"Max retries reached for {url}. Moving to next URL.")
                futures.append((future, url))
            yield futures


def recursive_character_text_splitter(text, max_tokens, overlap):
    if len(text) <= max_tokens:
        return [text]
    else:
        return [
            text[i : i + max_tokens] for i in range(0, len(text), max_tokens - overlap)
        ]


def get_embeddings(split_docs: List[Document]) -> List[Document]:
    """Get embeddings for the split documents."""
    for batch_start in range(0, len(split_docs), EMBEDDING_BATCH_SIZE):
        batch_end = batch_start + EMBEDDING_BATCH_SIZE
        batch = [doc.text for doc in split_docs[batch_start:batch_end]]
        response = client_embedding.embeddings(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        for i, be in enumerate(response.data):
            assert i == be.index
        batch_embeddings = [e.embedding for e in response.data]
        for i, doc in enumerate(split_docs[batch_start:batch_end]):
            doc.embedding = batch_embeddings[i]
    return split_docs


def find_similar_chunks(
    query: str, docs_with_embeddings: List[Document], k: int = 4
) -> List:
    """Similarity search to find similar chunks to a query"""
    query_embedding = (
        client_embedding.embeddings(
            model=EMBEDDING_MODEL,
            input=query,
        )
        .data[0]
        .embedding
    )

    index = faiss.IndexFlatIP(EMBEDDING_SIZE)
    index.add(np.array([doc.embedding for doc in docs_with_embeddings]))
    D, I = index.search(np.array([query_embedding]), k)

    return [docs_with_embeddings[i] for i in I[0]]


def multi_questions_response(
    prompt: str,
    engine: str,
    temperature: float = LLM_SETTINGS[DEFAULT_MODEL]["temperature"],
    max_tokens: int = LLM_SETTINGS[DEFAULT_MODEL]["default_max_tokens"],
    counter_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[str]:
    """Generate multiple questions for fetching information from the web."""
    try:
        multi_questions_prompt = MULTI_QUESTIONS_PROMPT.format(USER_INPUT=prompt)
        messages = [
            {"role": "user", "content": multi_questions_prompt},
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        response = client.completions(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if counter_callback:
            counter_callback(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=engine,
                token_counter=count_tokens,
            )

        # append the user's question to the list of questions
        multi_questions = parser_multi_questions_response(response.content)
        multi_questions.append(prompt)

        return multi_questions, counter_callback

    except Exception as e:
        print(f"Error generating multiple questions: {e}")
        return [prompt], counter_callback


def reciprocal_rank_refusion(similar_chunks: List[Document], k: int) -> List[Document]:
    """Reciprocal rank refusion to re-rank the similar chunks based on the text."""
    fused_chunks = {}
    for rank, doc in enumerate(similar_chunks):
        doc_text = doc.text
        if doc_text not in fused_chunks:
            fused_chunks[doc_text] = (doc, 0)
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
):
    """Attempt to do reasoning with retries on failure."""
    attempt = 0
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
            print(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(delay)
            attempt += 1
    raise Exception("Failed to generate prediction after retries")

def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


def fetch_additional_information(
    prompt: str,
    engine: str,
    google_api_key: Optional[str],
    google_engine_id: Optional[str],
    counter_callback: Optional[Callable[[int, int, str], None]] = None,
    source_links: Optional[List[str]] = None,
    num_urls: Optional[int] = DEFAULT_NUM_URLS,
    num_queries: Optional[int] = DEFAULT_NUM_QUERIES,
    temperature: Optional[float] = LLM_SETTINGS[DEFAULT_MODEL]["temperature"],
    max_tokens: Optional[int] = LLM_SETTINGS[DEFAULT_MODEL]["default_max_tokens"],
) -> Tuple[str, List[str], Optional[Callable[[int, int, str], None]]]:
    """Fetch additional information from the web."""

    # generate multiple queries for fetching information from the web
    try:
        queries, counter_callback = multi_queries(
            prompt=prompt,
            engine=engine,
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
            split_docs.extend([Document(text=chunk, url=doc.url) for chunk in t])
        except Exception as e:
            print(f"Error splitting document: {e}")
            continue
    print(f"Split Docs: {len(split_docs)}")

    # Remove None values from the list
    split_docs = [doc for doc in split_docs if doc]

    # Embed the documents
    docs_with_embeddings = get_embeddings(split_docs)
    print(f"Docs with embeddings: {len(docs_with_embeddings)}")

    # multi questions prompt
    questions, counter_callback = multi_questions_response(
        prompt=prompt,
        engine=engine,
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
    pattern = r"\"(.*?)\""
    try:
        question = re.findall(pattern, prompt)[0]
    except Exception as e:
        print(f"Error extracting question: {e}")
        question = prompt

    return question


def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the task"""
    with LLMClientManager(
        kwargs["api_keys"], kwargs["llm_provider"], embedding_provider="openai"
    ):
        tool = kwargs["tool"]
        model = kwargs.get("model", TOOL_TO_ENGINE[tool])
        prompt = extract_question(kwargs["prompt"])
        engine = kwargs.get("model", TOOL_TO_ENGINE[tool])
        print(f"ENGINE: {engine}")
        max_tokens = kwargs.get(
            "max_tokens", LLM_SETTINGS[engine]["default_max_tokens"]
        )
        temperature = kwargs.get("temperature", LLM_SETTINGS[engine]["temperature"])
        num_urls = kwargs.get("num_urls", DEFAULT_NUM_URLS[tool])
        num_queries = kwargs.get("num_queries", DEFAULT_NUM_QUERIES[tool])
        counter_callback = kwargs.get("counter_callback", None)
        api_keys = kwargs.get("api_keys", {})
        google_api_key = api_keys.get("google_api_key", None)
        google_engine_id = api_keys.get("google_engine_id", None)

        # Make sure the model is supported
        if model not in ALLOWED_MODELS:
            raise ValueError(f"Model {model} not supported.")

        # make sure the tool is supported
        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"Tool {tool} not supported.")

        (
            additional_information,
            queries,
            counter_callback,
        ) = fetch_additional_information(
            prompt=prompt,
            engine=engine,
            google_api_key=google_api_key,
            google_engine_id=google_engine_id,
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": reasoning_prompt},
        ]
        reasoning, counter_callback = do_reasoning_with_retry(
            model=engine,
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prediction_prompt},
        ]

        response_prediction = client.completions(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        prediction = parser_prediction_response(response_prediction.content)

        if counter_callback:
            counter_callback(
                input_tokens=response_prediction.usage.prompt_tokens,
                output_tokens=response_prediction.usage.completion_tokens,
                model=engine,
                token_counter=count_tokens,
            )

        return (
            prediction,
            reasoning_prompt + "////" + prediction_prompt,
            None,
            counter_callback,
        )
