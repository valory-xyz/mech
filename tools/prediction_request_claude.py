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

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT, Stream
import json
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import openai
import requests
from anthropic.types import Completion
from bs4 import BeautifulSoup
from googleapiclient.discovery import build


NUM_URLS_EXTRACT = 5
DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0.7,
}
ALLOWED_TOOLS = [
    "claude-prediction-offline",
    "claude-prediction-online",
]
TOOL_TO_ENGINE = {
    "claude-prediction-offline": "claude-2",
    "claude-prediction-online": "claude-2",
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
* Output only the JSON object to be parsed by Python's "json.loads()". Do not include any other contents in your response.
"""


def search_google(query: str, api_key: str, engine: str, num: int = 3) -> List[str]:
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


def get_urls_from_queries(queries: List[str], api_key: str, engine: str) -> List[str]:
    """Get URLs from search engine queries"""
    results = []
    for query in queries:
        for url in search_google(
            query=query,
            api_key=api_key,
            engine=engine,
            num=3,  # Number of returned results
        ):
            results.append(url)
    unique_results = list(set(results))
    return unique_results


def extract_text(
    html: str,
    num_words: int = 300,  # TODO: summarise using LLM instead of limit
) -> str:
    """Extract text from a single HTML document"""
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return text[:num_words]


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


def extract_texts(urls: List[str], num_words: int = 300) -> List[str]:
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
                    extract_text(html=result.text, num_words=num_words)
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


def fetch_additional_information(
    prompt: str,
    engine: str,
    google_api_key: str,
    google_engine: str,
    anthropic: Anthropic,
) -> str:
    """Fetch additional information."""
    url_query_prompt = URL_QUERY_PROMPT.format(user_prompt=prompt)
    url_query_prompt = f"{HUMAN_PROMPT}{url_query_prompt}{AI_PROMPT}"
    completion = anthropic.completions.create(
        model=engine,
        max_tokens_to_sample=300,
        prompt=url_query_prompt,
    )
    json_data = json.loads(completion.completion)
    urls = get_urls_from_queries(
        json_data["queries"],
        api_key=google_api_key,
        engine=google_engine,
    )
    texts = extract_texts(urls)
    return "\n".join(["- " + text for text in texts])


def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """Run the task"""
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]
    anthropic = Anthropic(api_key=kwargs["api_keys"]["anthropic"])

    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")

    engine = TOOL_TO_ENGINE[tool]
    additional_information = (
        fetch_additional_information(
            prompt=prompt,
            engine=engine,
            google_api_key=kwargs["api_keys"]["google_api_key"],
            google_engine=kwargs["api_keys"]["google_engine_id"],
            anthropic=anthropic,
        )
        if tool == "claude-prediction-online"
        else ""
    )
    prediction_prompt = PREDICTION_PROMPT.format(
        user_prompt=prompt, additional_information=additional_information
    )
    prediction_prompt = f"{HUMAN_PROMPT}{prediction_prompt}{AI_PROMPT}"

    completion = anthropic.completions.create(
        model=engine,
        max_tokens_to_sample=300,
        prompt=prediction_prompt,
    )
    return completion.completion, prediction_prompt, None
