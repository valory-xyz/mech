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

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from docstring_parser import parse
import faiss
from googleapiclient.discovery import build
from heapq import nlargest
import html2text
from itertools import islice
import json
import numpy as np
import tiktoken
from openai import OpenAI
from pydantic import BaseModel, Field
from readability import Document
import requests
from string import punctuation
from typing import Any, Dict, Generator, List, Optional, Tuple, Callable

client: Optional[OpenAI] = None


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
    "max_tokens": 300,
    "temperature": 0,
}
MAX_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
}
ALLOWED_TOOLS = [
    "prediction-request-rag",
]
TOOL_TO_ENGINE = {tool: "gpt-3.5-turbo" for tool in ALLOWED_TOOLS}
# the default number of URLs to fetch online information for
DEFAULT_NUM_URLS = defaultdict(lambda: 3)
# the default number of queries to generate for fetching online information
DEFAULT_NUM_QUERIES = defaultdict(lambda: 5)
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_BATCH_SIZE = 1000
SPLITTER_MAX_TOKENS = 1800
SPLITTER_OVERLAP = 50
NUM_NEIGHBOURS = 4


PREDICTION_PROMPT = """
You are an LLM inside a multi-agent system that takes in a prompt of a user requesting a probability estimation
for a given event. You are provided with an input under the label "USER_PROMPT". You are also provided with ADDITIONAL_INFORMATION.
INSTRUCTIONS
* Read the input under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" specifies an event.
* The event will only have two possible outcomes: either the event will happen or the event will not happen.
* If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
* You must provide a probability estimation of the event happening, based on your training data.
* You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION" delimited by three backticks.
* You can use any item in "ADDITIONAL_INFORMATION" in addition to your training data.
* If an item in "ADDITIONAL_INFORMATION" is not relevant, you must ignore that item for the estimation.
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.
USER_QUESTION: 
```{user_prompt}```
ADDITIONAL_INFORMATION: 
```{additional_information}```
"""

URL_QUERY_PROMPT = """
Given the user's question: please generate {num_queries} diverse and relevant search queries that can be used to find information on the internet to answer the initial question. 
Focus on capturing different aspects and interpretations of the question to ensure comprehensive coverage of the topic.
USER's QUESTION: {user_prompt}
"""

SYSTEM_PROMPT = """You are a world class algorithm for generating structured output from a given input. 
You make predictions about the probability of an event happening based on the information provided in the input."""


class OpenAISchema(BaseModel):  # type: ignore[misc]
    @classmethod  # type: ignore[misc]
    @property
    def openai_schema(cls) -> Dict[str, Any]:
        """
        Return the schema in the format of OpenAI's schema as jsonschema
        Note:
            Its important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.
        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
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
        Convert the response from OpenAI into the class instance
        Args:
            completion (dict): The response from OpenAI
        Returns:
            OpenAISchema: The instance of the class
        """

        message = completion.choices[0].message

        return cls.model_validate_json(
            message.function_call.arguments,
        )

class Results(OpenAISchema):
    p_yes: float =  Field(description="Estimated probability that the event in the USER_QUESTION occurs.")
    p_no: float = Field(description="Estimated probability that the event in the USER_QUESTION does not occur.")
    confidence: float = Field(description="A value between 0 and 1 indicating the confidence in the prediction. 0 indicates lowest confidence value; 1 maximum confidence value.")
    info_utility: float = Field(description="Utility of the information provided in ADDITIONAL_INFORMATION to help you make the prediction. 0 indicates lowest utility; 1 maximum utility.")


class Queries(OpenAISchema):
    queries: List[str]


def search_google(query: str, api_key: str, engine: str, num: int) -> List[str]:
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
        for url in search_google(
            query=query,
            api_key=api_key,
            engine=engine,
            num=num,
        ):
            results.append(url)
    unique_results = list(set(results))
    return unique_results


def find_similar_chunks(
    query: str, chunk_to_embedding: Dict, k: int = 4
) -> List:
    """Similarity search to find similar chunks to a query"""


    query_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    ).data[0].embedding

    index = faiss.IndexFlatIP(1536)
    index.add(np.array(list(chunk_to_embedding.values())))
    D, I = index.search(np.array([query_embedding]), k)

    return [list(chunk_to_embedding.keys())[i] for i in I[0]]


def get_embeddings(split_docs):
    # Make chunks to embeddings mapping
    chunk_to_embedding = {}
    for batch_start in range(0, len(split_docs), EMBEDDING_BATCH_SIZE):
        batch_end = batch_start + EMBEDDING_BATCH_SIZE
        batch = split_docs[batch_start:batch_end]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        for i, be in enumerate(response.data):
            assert i == be.index
        batch_embeddings = [e.embedding for e in response.data]
        for chunk, embedding in zip(batch, batch_embeddings):
            chunk_to_embedding[chunk] = embedding
    return chunk_to_embedding


def extract_text(
    html: str,
) -> str:
    """Extract text from a single HTML document"""
    text = Document(html).summary()

    # use html2text to convert HTML to markdown
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = True
    text = h.handle(text)

    # if text is None, return an empty string
    if text is None:
        return ""

    # remove newlines and extra spaces
    text = " ".join(text.split())
    
    return text


def process_in_batches(
    urls: List[str], window: int = 5, timeout: int = 10
) -> Generator[None, None, List[Tuple[Future, str]]]:
    """Iter URLs in batches."""
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(urls), window):
            batch = urls[i : i + window]
            futures = [
                (executor.submit(requests.get, url, timeout=timeout), url)
                for url in batch
            ]
            yield futures


def extract_texts(urls: List[str]) -> List[str]:
    """Extract texts from URLs"""
    max_allowed = 5
    extracted_texts = []
    count = 0
    stop = False
    for batch in process_in_batches(urls=urls):
        for future, url in batch:
            try:
                result = future.result()
                if result.status_code != 200:
                    continue
                extracted_texts.append(
                    extract_text(html=result.text)
                )
                count += 1
                if count >= max_allowed:
                    stop = True
                    break
            except requests.exceptions.ReadTimeout:
                print(f"Request timed out: {url}.")
            except Exception as e:
                print(f"An error occurred: {e}")
        if stop:
            break
    return extracted_texts


def recursive_character_text_splitter(text, max_tokens, overlap):
    if len(text) <= max_tokens:
        return [text]
    else:
        return [text[i:i+max_tokens] for i in range(0, len(text), max_tokens - overlap)]


def fetch_additional_information(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    google_api_key: Optional[str],
    google_engine: Optional[str],
    num_urls: Optional[int],
    num_queries: Optional[int],
    counter_callback: Optional[Callable] = None,
    source_links: Optional[List[str]] = None,
) -> str:
    """Fetch additional information."""
    url_query_prompt = URL_QUERY_PROMPT.format(user_prompt=prompt, num_queries=num_queries)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url_query_prompt},
    ]

    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        timeout=150,
        stop=None,
        functions=[Queries.openai_schema],
    )
    queries = Queries.from_response(response)
    if not source_links:
        urls = get_urls_from_queries(
            queries.queries,
            google_api_key,
            google_engine,
            num_urls,
        )
        texts = extract_texts(urls)
    else:
        texts = []
        for source_link in islice(source_links.values(), num_urls):
            texts.append(extract_text(html=source_link))

    split_texts = []
    for text in texts:
        split_texts += recursive_character_text_splitter(text, SPLITTER_MAX_TOKENS, SPLITTER_OVERLAP)
    chunk_to_embedding = get_embeddings(split_texts)
    retrieved_chunks = find_similar_chunks(prompt, chunk_to_embedding, NUM_NEIGHBOURS)

    if counter_callback:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=engine,
        )
        return "\n".join(["--- " + text for text in retrieved_chunks]), counter_callback
    return "\n".join(["--- " + text for text in retrieved_chunks]), None


def adjust_additional_information(
    prompt: str, 
    prompt_template:str, 
    additional_information: str, 
    model: str
) -> str:
    """Adjust the additional_information to fit within the token budget"""

    # Initialize tiktoken encoder for the specified model
    enc = tiktoken.encoding_for_model(model)
    
    # Encode the user prompt to calculate its token count
    prompt = prompt_template.format(user_prompt=prompt, additional_information="")
    prompt_tokens = len(enc.encode(prompt))
    
    # Calculate available tokens for additional_information
    MAX_PREDICTION_PROMPT_TOKENS = MAX_TOKENS[model] - DEFAULT_OPENAI_SETTINGS["max_tokens"]
    available_tokens = MAX_PREDICTION_PROMPT_TOKENS - prompt_tokens
    
    # Encode the additional_information
    additional_info_tokens = enc.encode(additional_information)
    
    # If additional_information exceeds available tokens, truncate it
    if len(additional_info_tokens) > available_tokens:
        truncated_info_tokens = additional_info_tokens[:available_tokens]
        # Decode tokens back to text, ensuring the output fits within the budget
        additional_information = enc.decode(truncated_info_tokens)
    
    return additional_information


def run(**kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Run the task"""
    with OpenAIClientManager(kwargs["api_keys"]["openai"]):

        tool = kwargs["tool"]
        prompt = kwargs["prompt"]
        max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
        num_urls = kwargs.get("num_urls", DEFAULT_NUM_URLS[tool])
        num_queries = kwargs.get("num_queries", DEFAULT_NUM_QUERIES[tool])
        counter_callback = kwargs.get("counter_callback", None)
        api_keys = kwargs.get("api_keys", {})
        google_api_key = api_keys.get("google_api_key", None)
        google_engine_id = api_keys.get("google_engine_id", None)
        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"Tool {tool} is not supported.")

        engine = TOOL_TO_ENGINE[tool]
        additional_information, counter_callback = fetch_additional_information(
            prompt,
            engine,
            temperature,
            max_tokens,
            google_api_key,
            google_engine_id,
            num_urls,
            num_queries,
            counter_callback=counter_callback,
            source_links=kwargs.get("source_links", None),
        )
        additional_information = adjust_additional_information(
            prompt,
            PREDICTION_PROMPT,
            additional_information,
            engine
        )
        prediction_prompt = PREDICTION_PROMPT.format(
            user_prompt=prompt, additional_information=additional_information
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prediction_prompt},
        ]
        response = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            timeout=150,
            stop=None,
            functions=[Results.openai_schema],
        )
        results = str(Results.from_response(response))

        pairs = str(results).split()
        result_dict = {}
        for pair in pairs:
            key, value = pair.split("=")
            result_dict[key] = float(value)  # Convert value to float
        results = result_dict
        results = json.dumps(results)

        if counter_callback is not None:
            counter_callback(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=engine,
            )
            return results, prediction_prompt, counter_callback
        return results, prediction_prompt, None