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
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import islice
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, cast

import anthropic
import googleapiclient
import openai
import requests
import tiktoken
from googleapiclient.discovery import build
from markdownify import markdownify as md
from openai import OpenAI
from readability import Document
from tiktoken import encoding_for_model


client: Optional[OpenAI] = None


N_MODEL_CALLS = 2
USER_AGENT_HEADER = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
GOOGLE_RATE_LIMIT_EXCEEDED_CODE = 429
DEFAULT_DELIVERY_RATE = 100


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


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    # Workaround since tiktoken does not have support yet for gpt4.1
    # https://github.com/openai/tiktoken/issues/395
    if model == "gpt-4.1-2025-04-14":
        enc = tiktoken.get_encoding("o200k_base")
    else:
        enc = encoding_for_model(model)
    return len(enc.encode(text))


NUM_URLS_EXTRACT = 5
DEFAULT_NUM_WORDS: Dict[str, Optional[int]] = defaultdict(lambda: 300)
DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0.0,
}
MAX_TOKENS = {
    "gpt-3.5-turbo-0125": 4096,
    "gpt-4-0125-preview": 8192,
    "gpt-4o-2024-08-06": 4096,
    "gpt-4.1-2025-04-14": 4096,
}
ALLOWED_TOOLS = [
    "prediction-offline-sme",
    "prediction-online-sme",
]
TOOL_TO_ENGINE = {
    "prediction-offline-sme": "gpt-4o-2024-08-06",
    "prediction-online-sme": "gpt-4o-2024-08-06",
}

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
* Never use Markdown syntax highlighting, such as ```json``` to surround the output. Only output the raw json string.
* This is incorrect:"```json{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}```"
* This is incorrect:```json"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"```
* This is correct:"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"
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

SME_GENERATION_MARKET_PROMPT = """
task question: "{question}"
"""

SME_GENERATION_SYSTEM_PROMPT = """
This task requires answering Yes or No to a specific question related to certain knowledge domains. The final opinion to the question should be determined by one or more subject matter experts (SME) of the related domains. You need to generate one or more SME roles and their role introduction that you believe to be helpful in forming a correct answer to question in the task.

Examples:
task question: "Will Apple release iphone 15 by 1 October 2023?"
[
        {
            "sme": "Technology Analyst",
            "sme_introduction": "You are a seasoned technology analyst AI assistant. Your goal is to do comprehensive research on the news on the tech companies and answer investor's interested questions in a trustful and accurate way."
        }
]
---
task question: "Will the newly elected ceremonial president of Singapore face any political scandals by 13 September 2023?"
[
        {
            "sme":  "Political Commentator",
            "sme_introduction": "You are an experienced political commentator in Asia. Your main objective is to produce comprehensive, insightful and impartial analysis based on the relevant political news and your politic expertise to form an answer to the question releted to a political event or politician."
        }
]
---
task question: "Will the air strike conflict in Sudan be resolved by 13 September 2023?"
[
       {
            "sme:  "Military Expert",
            "sme_introduction": "You are an experienced expert in military operation and industry. Your main goal is to faithfully and accurately answer a military related question based on the provided intelligence and your professional experience"
        },
       {
            "sme:  "Diplomat",
            "sme_introduction": "You are an senior deplomat who engages in diplomacy to foster peaceful relations, negotiate agreements, and navigate complex political, economic, and social landscapes. You need to form an opinion on a question related to international conflicts based on the related information and your understading in geopolitics."
        },
]
"""


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
                if e.status_code != GOOGLE_RATE_LIMIT_EXCEEDED_CODE:
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


def search_google(query: str, api_key: str, engine: str, num: int = 3) -> List[str]:
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
    items = search.get("items")
    if items is not None:
        return [result["link"] for result in items]
    return []


def get_urls_from_queries(
    queries: List[str], api_key: str, engine: str, num: int = 3
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


def extract_text(
    html: str,
    num_words: Optional[int] = None,
) -> str:
    """Extract text from a single HTML document"""
    text = Document(html).summary()
    text = md(text, heading_style="ATX")
    if text is None:
        return ""

    if num_words:
        return " ".join(text.split()[:num_words])

    # remove newlines and extra spaces
    text = " ".join(text.split())

    return text


def process_in_batches(
    urls: List[str], window: int = 5, timeout: int = 10
) -> Generator[List[Tuple[Future, str]], None, None]:
    """Iter URLs in batches."""
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(urls), window):
            batch = urls[i : i + window]
            futures = [
                (
                    executor.submit(
                        requests.get,
                        url,
                        timeout=timeout,
                        headers={"User-Agent": USER_AGENT_HEADER},
                    ),
                    url,
                )
                for url in batch
            ]
            yield futures


def extract_texts(urls: List[str], num_words: int = 300) -> List[Dict]:
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
                doc: Dict = {}
                text = extract_text(html=result.text, num_words=num_words)
                doc["text"] = text
                doc["url"] = url
                extracted_texts.append(doc)
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


def fetch_additional_information(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    google_api_key: Optional[str],
    google_engine: Optional[str],
    num_urls: int,
    num_words: int,
    counter_callback: Optional[Callable] = None,
    source_links: Optional[Dict] = None,
) -> Tuple[str, Optional[Callable[[int, int, str], None]]]:
    """Fetch additional information."""
    if not google_api_key:
        raise RuntimeError("Google API key not found")
    if not google_engine:
        raise RuntimeError("Google Engine Id not found")
    if not client:
        raise RuntimeError("Client not initialized")

    url_query_prompt = URL_QUERY_PROMPT.format(user_prompt=prompt)
    moderation_result = client.moderations.create(input=url_query_prompt)
    if moderation_result.results[0].flagged:
        return "", counter_callback
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": url_query_prompt},
    ]
    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        timeout=90,
        stop=None,
    )
    json_data = json.loads(response.choices[0].message.content)

    if not source_links:
        urls = get_urls_from_queries(
            json_data["queries"],
            api_key=google_api_key,
            engine=google_engine,
        )
        texts = extract_texts(urls, num_words)
    else:
        texts = []
        for url, content in islice(source_links.items(), 3):
            doc: dict = {}
            text = (
                extract_text(html=content, num_words=num_words),
                url,
            )
            doc["text"] = text
            doc["url"] = url
            texts.append(doc)
    # Format the additional information
    additional_information = "\n".join(
        [
            f"ARTICLE {i}, URL: {doc['url']}, CONTENT: {doc['text']}\n"
            for i, doc in enumerate(texts)
        ]
    )
    if counter_callback:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=engine,
            token_counter=count_tokens,
        )

    return additional_information, counter_callback


def get_sme_role(
    engine: str,
    temperature: float,
    max_tokens: int,
    prompt: str,
    counter_callback: Optional[Callable] = None,
) -> Tuple[str, str, Optional[Callable]]:
    """Get SME title and introduction"""
    if not client:
        raise RuntimeError("Client not initialized")

    market_question = SME_GENERATION_MARKET_PROMPT.format(question=prompt)
    system_prompt = SME_GENERATION_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": market_question},
    ]
    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        timeout=150,
        stop=None,
    )
    generated_sme_roles = response.choices[0].message.content
    sme = json.loads(generated_sme_roles)[0]
    if counter_callback is not None:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            model=engine,
            token_counter=count_tokens,
        )
        return sme["sme"], sme["sme_introduction"], counter_callback
    return sme["sme"], sme["sme_introduction"], None


def adjust_additional_information(
    prompt: str, prompt_template: str, additional_information: str, model: str
) -> str:
    """Adjust the additional_information to fit within the token budget"""

    # Initialize tiktoken encoder for the specified model
    enc = tiktoken.encoding_for_model(model)

    # Encode the user prompt to calculate its token count
    prompt = prompt_template.format(user_prompt=prompt, additional_information="")
    prompt_tokens = len(enc.encode(prompt))

    # Calculate available tokens for additional_information
    MAX_PREDICTION_PROMPT_TOKENS = (
        MAX_TOKENS[model] - DEFAULT_OPENAI_SETTINGS["max_tokens"]
    )
    available_tokens = cast(int, MAX_PREDICTION_PROMPT_TOKENS) - prompt_tokens

    # Encode the additional_information
    additional_info_tokens = enc.encode(additional_information)

    # If additional_information exceeds available tokens, truncate it
    if len(additional_info_tokens) > available_tokens:
        truncated_info_tokens = additional_info_tokens[:available_tokens]
        # Decode tokens back to text, ensuring the output fits within the budget
        additional_information = enc.decode(truncated_info_tokens)

    return additional_information


@with_key_rotation
def run(
    **kwargs: Any,
) -> Union[float, Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]]:
    """Run the task"""
    tool = kwargs["tool"]
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")

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
        max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
        source_links = kwargs.get("source_links", None)
        num_urls = kwargs.get("num_urls", NUM_URLS_EXTRACT)
        num_words = kwargs.get("num_words", DEFAULT_NUM_WORDS)
        api_keys = kwargs.get("api_keys", {})
        google_api_key = api_keys.get("google_api_key", None)
        google_engine_id = api_keys.get("google_engine_id", None)

        if not client:
            raise RuntimeError("Client not initialized")

        try:
            _, sme_introduction, counter_callback = get_sme_role(
                engine,
                temperature,
                max_tokens,
                prompt,
                counter_callback=counter_callback,
            )
        except Exception as e:
            print(f"An error occurred during SME role creation: {e}")
            print("Using default SME introduction.")
            sme_introduction = "You are a helpful assistant."

        if tool.startswith("prediction-online"):
            additional_information, counter_callback = fetch_additional_information(
                prompt=prompt,
                engine=engine,
                temperature=temperature,
                max_tokens=max_tokens,
                google_api_key=google_api_key,
                google_engine=google_engine_id,
                num_urls=num_urls,
                num_words=num_words,
                counter_callback=counter_callback,
                source_links=source_links,
            )
        else:
            additional_information = None
        if additional_information:
            additional_information = adjust_additional_information(
                prompt=prompt,
                prompt_template=PREDICTION_PROMPT,
                additional_information=additional_information,
                model=engine,
            )
        prediction_prompt = PREDICTION_PROMPT.format(
            user_prompt=prompt, additional_information=additional_information
        )
        moderation_result = client.moderations.create(input=prediction_prompt)
        if moderation_result.results[0].flagged:
            return (
                "Moderation flagged the prompt as in violation of terms.",
                prediction_prompt,
                None,
                counter_callback,
            )
        messages = [
            {"role": "system", "content": sme_introduction},
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
        )
        if counter_callback is not None:
            counter_callback(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=engine,
                token_counter=count_tokens,
            )

        return (
            response.choices[0].message.content,
            prediction_prompt,
            None,
            counter_callback,
        )
