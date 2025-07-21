# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2025 Valory AG
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
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from heapq import nlargest
from io import BytesIO
from itertools import islice
from string import punctuation
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, cast

import PyPDF2
import anthropic
import googleapiclient
import openai
import requests
import spacy
from googleapiclient.discovery import build
from markdownify import markdownify as md
from pydantic import BaseModel, PositiveInt
from readability import Document
from spacy import Language
from spacy.cli import download
from spacy.lang.en import STOP_WORDS
from spacy.tokens import Doc, Span
from tiktoken import encoding_for_model, get_encoding


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

    def __init__(self, api_keys: List, model: str):
        """Initializes with API keys and llm provider"""
        self.api_keys = api_keys
        if "gpt" in model:
            self.llm_provider = "openai"
        elif "claude" in model:
            self.llm_provider = "anthropic"
        else:
            self.llm_provider = "openrouter"

    def __enter__(self) -> Any:
        """Initializes and returns LLM client."""
        global client
        if client is None:
            client = LLMClient(self.api_keys, self.llm_provider)
        return client

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Closes the LLM client"""
        global client
        if client is not None:
            client.client.close()
            client = None


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


class LLMResponse:
    """Response class."""

    def __init__(self, content: Optional[str] = None, usage: Optional[Usage] = None):
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
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_keys["anthropic"])  # type: ignore
        if self.llm_provider == "openai":
            import openai

            self.client = openai.OpenAI(api_key=self.api_keys["openai"])  # type: ignore
        if self.llm_provider == "openrouter":
            import openai

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
            # default value if not found
            system_prompt = SYSTEM_PROMPT_FORECASTER
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

        if self.llm_provider in ["openai", "openrouter"]:
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


class ExtendedDocument(BaseModel):
    """Extended Document Model"""

    text: str
    url: str
    tokens: PositiveInt = 0
    embedding: Optional[List[float]] = None


client: Optional[LLMClient] = None


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


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    # Check if we're using a Claude model and we have an active client
    if "claude" in model.lower() and client and client.llm_provider == "anthropic":
        try:
            # Use Anthropic's tokenizer when available
            response = client.messages.count_tokens(  # type: ignore # pylint: disable=no-member
                model=model, messages=[{"role": "user", "content": text}]
            )
            return response.input_tokens
        except (AttributeError, Exception):
            # Fallback if the method doesn't exist or fails
            print("Using fallback enconding for Claude models")
            enc = get_encoding("cl100k_base")
            return len(enc.encode(text))
    # Workaround since tiktoken does not have support yet for gpt4.1
    # https://github.com/openai/tiktoken/issues/395
    if model == "gpt-4.1-2025-04-14":
        enc = get_encoding("o200k_base")
    else:
        enc = encoding_for_model(model)
    return len(enc.encode(text))


FrequenciesType = Dict[str, float]
ScoresType = Dict[Span, float]


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
    "deepseek/deepseek-r1:free": {
        "default_max_tokens": 1000,
        "limit_max_tokens": 65_336,
        "temperature": 0,
    },
}
ALLOWED_TOOLS = [
    "prediction-offline",
    "prediction-online",
    # "prediction-online-summarized-info",
    # LEGACY
    "claude-prediction-offline",
    "claude-prediction-online",
]
ALLOWED_MODELS = list(LLM_SETTINGS.keys())
# the default number of URLs to fetch online information for
DEFAULT_NUM_URLS = defaultdict(lambda: 3)
DEFAULT_NUM_URLS["prediction-online-summarized-info"] = 7
# the default number of words to fetch online information for
DEFAULT_NUM_WORDS: Dict[str, Optional[int]] = defaultdict(lambda: 300)
DEFAULT_NUM_WORDS["prediction-online-summarized-info"] = None
# how much of the initial content will be kept during summarization
DEFAULT_COMPRESSION_FACTOR = 0.05
# the vocabulary to use for the summarization
DEFAULT_VOCAB = "en_core_web_sm"
# number of retries and delay for completion
COMPLETION_RETRIES = 3
COMPLETION_DELAY = 2
MAX_NR_DOCS = 1000
HTTP_TIMEOUT = 20

PREDICTION_PROMPT = """
You are an LLM inside a multi-agent system that takes in a prompt of a user requesting a probability estimation
for a given event. You are provided with an input under the label "USER_PROMPT". You must follow the instructions
under the label "INSTRUCTIONS". You must provide your response in the format specified under "OUTPUT_FORMAT".

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

USER_PROMPT:
```
{user_prompt}
```

ADDITIONAL_INFORMATION:
```
{additional_information}
```

OUTPUT_FORMAT
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain four fields: "p_yes", "p_no", "confidence", and "info_utility".
* Each item in the JSON must have a value between 0 and 1.
   - "p_yes": Estimated probability that the event in the "USER_PROMPT" occurs.
   - "p_no": Estimated probability that the event in the "USER_PROMPT" does not occur.
   - "confidence": A value between 0 and 1 indicating the confidence in the prediction. 0 indicates lowest
     confidence value; 1 maximum confidence value.
   - "info_utility": Utility of the information provided in "ADDITIONAL_INFORMATION" to help you make the prediction.
     0 indicates lowest utility; 1 maximum utility.
* The sum of "p_yes" and "p_no" must equal 1.
* Output only the JSON object. Do not include any other contents in your response.
* This is incorrect:"```json{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}```"
* This is incorrect:```json"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"```
* This is correct:"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"
"""

OFFLINE_PREDICTION_PROMPT = """
You are a superforecasting agent specializing in predicting the probabilities of binary outcomes (yes or no). Your responses MUST be in JSON format as specified below.

**INSTRUCTIONS:**

1. **Analyze the Question:** Carefully understand the question's core meaning and any relevant context.

2. **Gather Information:** Consider historical data, trends, and any information provided under "ADDITIONAL_INFORMATION".  Evaluate the relevance and utility of this information.

3. **Estimate Probabilities:**  Determine the probabilities of "yes" (p_yes) and "no" (p_no) outcomes based on your analysis. These *must* be floating-point numbers between 0 and 1.

4. **Ensure Validity:** The sum of p_yes and p_no *must* equal 1.0.

5. **Assess Confidence:**  Evaluate your confidence in the prediction (confidence). This *must* be a floating-point number between 0 and 1. A value of 0 indicates the lowest confidence, and 1 indicates the highest.

6. **Evaluate Information Utility:** Assess how helpful the "ADDITIONAL_INFORMATION" was in making your prediction (info_utility). This *must* be a floating-point number between 0 and 1. A value of 0 means the information was not useful, and 1 means it was highly relevant.

7. **JSON Output ONLY:** Your ENTIRE response *must* be a valid JSON object conforming to the "OUTPUT_FORMAT" below.  Do *not* include any surrounding text, explanations, or conversational elements.


**USER_PROMPT:**
{user_prompt}

**ADDITIONAL_INFORMATION:**
{additional_information}


**OUTPUT_FORMAT:**
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain four fields: "p_yes", "p_no", "confidence", and "info_utility".
* This is incorrect:"```json{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}```"
* This is incorrect:```json"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"```
* This is correct:"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"
IMPORTANT: Adhere strictly to the JSON format. No extra text or explanations are allowed. The values for p_yes, p_no, confidence, and info_utility MUST be floating-point numbers between 0 and 1, and p_yes + p_no MUST equal 1.0.
"""


URL_QUERY_PROMPT = """
You are an LLM inside a multi-agent system that takes in a prompt of a user requesting a probability estimation
for a given event. You are provided with an input under the label "USER_PROMPT". You must follow the instructions
under the label "INSTRUCTIONS". You must provide your response in the format specified under "OUTPUT_FORMAT".

INSTRUCTIONS
* Read the input under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" specifies an event.
* The event will only have two possible outcomes: either the event will happen or the event will not happen.
* If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.

USER_PROMPT:
```
{user_prompt}
```

OUTPUT_FORMAT
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain two fields: "queries", and "urls".
   - "queries": An array of strings of size between 1 and 5. Each string must be a search engine query that can help obtain relevant information to estimate
     the probability that the event in "USER_PROMPT" occurs. You must provide original information in each query, and they should not overlap
     or lead to obtain the same set of results.
* Output only the JSON object. Do not include any other contents in your response.
* Never use Markdown syntax highlighting, such as ```json``` to surround the output. Only output the raw json string.
* This is incorrect: "```json{{"queries": []}}```"
* This is incorrect: "```json"{{"queries": []}}"```"
* This is correct: "{{"queries": []}}"
"""
SYSTEM_PROMPT_FORECASTER = "You are an expert market forecaster. Your primary function is to generate accurate and insightful predictions in the requested format"


def search_google(query: str, api_key: str, engine: str, num: int) -> List[str]:
    """Search Google using a custom search engine."""
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


def extract_text(
    html: str, num_words: Optional[int] = None
) -> Optional[ExtendedDocument]:
    """Extract text from a single HTML document"""
    # Remove image tags and their content
    html = re.sub(r"<img[^>]*>", "", html)
    # Remove markdown image syntax
    html = re.sub(r"!\[.*?\]\(.*?\)", "", html)
    # Remove data URI images
    html = re.sub(r'data:image/[^;]*;base64,[^"]*', "", html)

    text = Document(html).summary()
    text = md(text, heading_style="ATX")
    if text is None:
        return None

    # Remove any remaining image-related content
    text = re.sub(
        r"!\[.*?\]\(.*?\)", "", text
    )  # Remove markdown images again after conversion
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # Remove markdown links
    text = re.sub(r"Photo:.*?\n", "", text)  # Remove photo credits
    text = re.sub(r"Image:.*?\n", "", text)  # Remove image credits

    words = text.split()
    text = " ".join(words[:num_words]) if num_words else " ".join(words)
    # final cleaning
    doc = ExtendedDocument(text=clean_text(text), url="")
    return doc


def extract_text_from_pdf(
    url: str, num_words: Optional[int] = None
) -> Optional[ExtendedDocument]:
    """Extract text from a PDF document at the given URL."""
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
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


def process_in_batches(
    urls: List[str], window: int = 5, timeout: int = 10
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


def extract_texts(urls: List[str], num_words: Optional[int]) -> List[ExtendedDocument]:
    """Extract texts from URLs"""
    max_allowed = 5
    extracted_docs: List[ExtendedDocument] = []
    count = 0
    stop = False
    for batch in process_in_batches(urls=urls):
        for future, url in batch:
            if future is None:
                continue
            doc = None
            try:
                result = future.result()
                if result.status_code == 200:
                    # Check if URL ends with .pdf or content starts with %PDF
                    if url.endswith(".pdf") or result.content[:4] == b"%PDF":
                        doc = extract_text_from_pdf(url, num_words=num_words)
                    else:
                        doc = extract_text(html=result.text, num_words=num_words)
            except requests.exceptions.ReadTimeout:
                print(f"Request timed out: {url}.")
            except Exception as e:
                print(f"Error processing {url}: {e}")
            if doc and doc.text != "":
                doc.url = url
                extracted_docs.append(doc)
                count += 1
            if count >= max_allowed:
                stop = True
                break
        if stop:
            break
    return extracted_docs


def extract_json_string(text: str) -> str:
    """Extract's the json string"""
    # This regex looks for triple backticks, captures everything in between until it finds another set of triple backticks.
    pattern = r"(\{[^}]*\})"
    matches = re.findall(pattern, text)
    return matches[0].replace("json", "")


def extract_multi_queries(text: str) -> Any:
    """Extract multiple queries from the given text"""
    # strip empty whitespace
    text = text.strip()

    # check if line starts with ````json
    if not text.startswith("```json"):
        text = extract_json_string(text)

    return json.loads(text)


def fetch_multi_queries_with_retry(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    retries: int = COMPLETION_RETRIES,
    delay: int = COMPLETION_DELAY,
    counter_callback: Optional[Callable] = None,
) -> Tuple[dict, Optional[Callable]]:
    """Attempt to fetch multi-queries with retries on failure."""
    if not client:
        raise RuntimeError("Client not initialized")

    """Attempt to fetch multi-queries with retries on failure."""
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
            if not response or response.content is None:
                raise RuntimeError("Response not found")
            # Attempt to extract JSON data from the response
            json_data = extract_multi_queries(response.content)

            if counter_callback:
                counter_callback(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=model,
                    token_counter=count_tokens,
                )
            return json_data, counter_callback
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(delay)
            attempt += 1
    raise Exception("Failed to fetch multi-queries after retries")


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
        raise Exception("Client not initialized")
    attempt = 0
    tool_errors = []
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

            if (
                response
                and response.content is not None
                and counter_callback is not None
            ):
                counter_callback(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=model,
                    token_counter=count_tokens,
                )

                extracted_block = extract_json_string(response.content)

            return extracted_block, counter_callback
        except Exception as e:
            error = f"Attempt {attempt + 1} failed with error: {e}"
            time.sleep(delay)
            # join the tool errors with the exception message
            tool_errors.append(error)
            attempt += 1
    error_message = (
        f"Failed to generate prediction after retries:\n{chr(10).join(tool_errors)}"
    )
    raise Exception(error_message)


def fetch_additional_information(
    user_prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    google_api_key: Optional[str],
    google_engine: Optional[str],
    num_urls: int,
    num_words: int,
    counter_callback: Optional[Callable] = None,
    source_links: Optional[Dict] = None,
) -> Tuple[str, Any]:
    """Fetch additional information."""
    if not google_api_key:
        raise RuntimeError("Google API key not found")
    if not google_engine:
        raise RuntimeError("Google Engine Id not found")
    if not client:
        raise RuntimeError("Client not initialized")

    url_query_prompt = URL_QUERY_PROMPT.format(user_prompt=user_prompt)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": url_query_prompt},
    ]
    try:
        json_data, counter_callback = fetch_multi_queries_with_retry(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=COMPLETION_RETRIES,
            delay=COMPLETION_DELAY,
            counter_callback=counter_callback,
        )
    except Exception as e:
        print(f"Fetch multi queries with retry failed with exception: {e}")
        json_data = {"queries": [user_prompt]}

    if not source_links:
        urls = get_urls_from_queries(
            json_data["queries"],
            google_api_key,
            google_engine,
            num_urls,
        )
        docs = extract_texts(urls, num_words)
    else:
        docs = []
        for url, content in islice(source_links.items(), 3):
            doc = extract_text(html=content, num_words=num_words)
            if doc and doc.text != "":
                doc.url = url
                docs.append(doc)

    if len(docs) > MAX_NR_DOCS:
        # truncate the split_docs to the first MAX_NR_DOCS documents
        docs = docs[:MAX_NR_DOCS]
    # Format the additional information
    additional_information = "\n".join(
        [
            f"ARTICLE {i}, URL: {doc.url}, CONTENT: {doc.text}\n"
            for i, doc in enumerate(docs)
        ]
    )
    return additional_information, counter_callback


def load_model(vocab: str) -> Language:
    """Utilize spaCy to load the model and download it if it is not already available."""
    try:
        return spacy.load(vocab)
    except OSError:
        print("Downloading language model...")
        download(vocab)
        return spacy.load(vocab)


def calc_word_frequencies(doc: Doc) -> FrequenciesType:
    """Get the frequency of each word in the given text, excluding stop words and punctuations."""
    word_frequencies: Dict = defaultdict(lambda: 0)
    for token in doc:
        word = token.text
        lower = word.lower()
        if lower not in STOP_WORDS.union(punctuation):
            word_frequencies[lower] += 1

    max_frequency = max(word_frequencies.values())
    normalized_frequencies = defaultdict(
        lambda: 0,
        {
            word: frequency / max_frequency
            for word, frequency in word_frequencies.items()
        },
    )
    return normalized_frequencies


def calc_sentence_scores(
    sentence_tokens: List[Span], word_frequencies: FrequenciesType
) -> ScoresType:
    """Calculate the sentence scores."""
    sentence_scores: Dict = defaultdict(lambda: 0)
    for sentence in sentence_tokens:
        for token in sentence:
            sentence_scores[sentence] += word_frequencies[token.text.lower()]

    return sentence_scores


def summarize(text: str, compression_factor: float, vocab: str) -> str:
    """Summarize the given text, retaining the given compression factor."""
    if not text:
        raise ValueError("Cannot summarize empty text!")

    nlp = load_model(vocab)
    doc = nlp(text)
    word_frequencies = calc_word_frequencies(doc)
    sentence_tokens = list(doc.sents)
    sentence_scores = calc_sentence_scores(sentence_tokens, word_frequencies)
    n = int(len(sentence_tokens) * compression_factor)
    summary = nlargest(n, sentence_scores, key=lambda k: sentence_scores[k])
    summary_words = [word.text for word in summary]
    summary_text = "".join(summary_words)
    return summary_text


def adjust_additional_information(
    user_prompt: str, prompt_template: str, additional_information: str, model: str
) -> str:
    """Adjust the additional_information to fit within the token budget"""

    # Encode the user prompt to calculate its token count
    user_prompt = prompt_template.format(
        user_prompt=user_prompt, additional_information=additional_information
    )
    prompt_tokens = count_tokens(text=user_prompt, model=model)

    # Calculate available tokens for additional_information
    MAX_PREDICTION_PROMPT_TOKENS = (
        LLM_SETTINGS[model]["limit_max_tokens"]
        - LLM_SETTINGS[model]["default_max_tokens"]
    )
    available_tokens = cast(int, MAX_PREDICTION_PROMPT_TOKENS) - prompt_tokens

    # Encode the additional_information
    additional_info_tokens = count_tokens(text=additional_information, model=model)

    # If additional_information exceeds available tokens, truncate it
    if additional_info_tokens > available_tokens:
        return additional_information[:available_tokens]

    return additional_information


@with_key_rotation
def run(**kwargs: Any) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the task"""
    tool = kwargs["tool"]
    engine = kwargs.get("model")
    if engine is None:
        raise ValueError("Model must be specified in kwargs")

    if "claude" in tool:  # maintain backwards compatibility
        engine = "claude-3-5-sonnet-20240620"
    print(f"ENGINE used for {tool}: {engine}")
    with LLMClientManager(kwargs["api_keys"], engine):
        user_prompt = kwargs["prompt"]  # question
        max_tokens = kwargs.get(
            "max_tokens", LLM_SETTINGS[engine]["default_max_tokens"]
        )
        temperature = kwargs.get("temperature", LLM_SETTINGS[engine]["temperature"])
        num_urls = kwargs.get("num_urls", DEFAULT_NUM_URLS[tool])
        num_words = kwargs.get("num_words", DEFAULT_NUM_WORDS[tool])
        compression_factor = kwargs.get(
            "compression_factor", DEFAULT_COMPRESSION_FACTOR
        )
        vocab = kwargs.get("vocab", DEFAULT_VOCAB)
        counter_callback = kwargs.get("counter_callback", None)
        api_keys = kwargs.get("api_keys", {})
        google_api_key = api_keys.get("google_api_key", None)
        google_engine_id = api_keys.get("google_engine_id", None)

        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"Tool {tool} is not supported.")

        active_prompt = PREDICTION_PROMPT
        additional_information = ""
        if tool in ["prediction-online", "claude-prediction-online"]:
            additional_information, counter_callback = fetch_additional_information(
                user_prompt,
                engine,
                temperature,
                max_tokens,
                google_api_key,
                google_engine_id,
                num_urls,
                num_words,
                counter_callback=counter_callback,
                source_links=kwargs.get("source_links", None),
            )
        elif "claude" not in engine:
            # used improved prompt in all models except the Claude ones
            active_prompt = OFFLINE_PREDICTION_PROMPT

        if additional_information and tool == "prediction-online-summarized-info":
            additional_information = summarize(
                additional_information, compression_factor, vocab
            )
        if additional_information:
            # check the limit of tokens
            additional_information = adjust_additional_information(
                user_prompt, active_prompt, additional_information, engine
            )
        prediction_prompt = active_prompt.format(
            user_prompt=user_prompt, additional_information=additional_information
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_FORECASTER},
            {"role": "user", "content": prediction_prompt},
        ]

        extracted_block, counter_callback = generate_prediction_with_retry(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=COMPLETION_RETRIES,
            delay=COMPLETION_DELAY,
            counter_callback=counter_callback,
        )

        return extracted_block, prediction_prompt, None, counter_callback
