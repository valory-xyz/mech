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

import json
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from heapq import nlargest
from itertools import islice
from string import punctuation
from typing import Any, Dict, Generator, List, Optional, Tuple, Callable

import tiktoken
from openai import OpenAI

import requests
import  html2text
from readability import Document
from googleapiclient.discovery import build

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
    "max_tokens": 500,
    "temperature": 0,
}
MAX_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
}
ALLOWED_TOOLS = [
    "resolve-market-reasoning",
]
TOOL_TO_ENGINE = {tool: "gpt-3.5-turbo" for tool in ALLOWED_TOOLS}
# the default number of URLs to fetch online information for
DEFAULT_NUM_URLS = defaultdict(lambda: 3)
# the default number of words to fetch online information for
DEFAULT_NUM_WORDS: Dict[str, Optional[int]] = defaultdict(lambda: 300)

PREDICTION_PROMPT = """
You are an expert fact checker that takes in a question asking whether an event will happen before a given date. 
That date has now passed and your role is to determine whether the event actually happened before the date.
You are provided with the input question about the event under the label "USER_PROMPT". You must follow the instructions
under the label "INSTRUCTIONS". You must provide your response in the format specified under "OUTPUT_FORMAT".

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
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.

Here are some examples of how you can figure out whether an event occurred by the date:
* If an article says that the event did happen and the date of the article is before the question date, then it is likely that the event did occur before the question date.
* If an article is talking about whether an event will happen and the date of the article is after the question date, then it is likely that the event did not happen before the question date.

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
* The JSON must contain two fields: "reasoning", and "has_occurred".
   - "reasoning": A string that shows you thinking through the problem step by step, taking the date and information of the various articles into consideration. It should help to explain your reasoning for your decision as to whether an event occurred by the specified date. 
   - "has_occurred": When the event in "USER_PROMPT" has occurred, the value of "has_occurred" must be True if it has occurred, and False if it has not. The answer that you give for this field should match the answer that you come to in the reasoning field. 
* Output only the JSON object. Do not include any other contents in your response.
"""

URL_QUERY_PROMPT = """
* You are an expert fact checker in a team tasked with determining whether an event happened before a given date in the past. 
* Your role in the team to come up with search queries to be used to find relevant news articles that may help in determining whether the event occured. 
* You are provided with the input question about the event under the label "USER_PROMPT". 
* You must follow the instructions under the label "INSTRUCTIONS". 
* You must provide your response in the format specified under "OUTPUT_FORMAT".

INSTRUCTIONS
* Read the input under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" is a question about whether an event happened.
* The "USER_PROMPT" will contain a date which in the past.
* The event will only have has two possible outcomes: either the event has happened or the event has not happened.
* If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
* You should come up with queries to search for relevant news articles that may help in determining whether the event occured. 
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.

USER_PROMPT:
```
{user_prompt}
```

OUTPUT_FORMAT
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain two fields: "queries", and "urls".
   - "queries": An array of strings of size between 1 and 4. Each string must be a search engine query that can help
     obtain relevant information to check that the event in "USER_PROMPT" occurs.
     You must provide original information in each query, and they should not overlap.
     or lead to obtain the same set of results.
* Output only the JSON object. Do not include any other contents in your response.
"""

GET_DATE_PROMPT = """
INSTRUCTIONS
* You are an expert data analyst that takes in extracted text from a web search result. 
* You are provided with text extracted from a relevant web page under the label "EXTRACTED_TEXT" delimited by three backticks.
* Your task is to extract the date that the web page was published. 
* If there is no date information available, you should not try to guess. Instead indicate that it is not available.
* You must provide your response in the format specified under "OUTPUT_FORMAT".

EXTRACTED_TEXT:
```
{extracted_text}
```

OUTPUT_FORMAT
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain two fields: "year" and "month". The month field must be the name of the month e.g. January, not a number. If there is no information for a field within the extracted text, the value should be none. 
* Output only the JSON object. Do not include any other contents in your response.
"""


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


def extract_text(
    html: str,
    num_words: int = 300,  # TODO: summerise using GPT instead of limit
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

    date = get_dates(text)

    if not num_words:
        return text
    return text[:num_words], date


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

def get_dates(text):
    get_date_prompt = GET_DATE_PROMPT.format(extracted_text=text)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": get_date_prompt},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        # max_tokens=4096,
        n=1,
        timeout=90,
        stop=None,
    )
    json_data = json.loads(response.choices[0].message.content)
    year = json_data['year']
    month = json_data['month']
    date = f"{month}/{year}"

    return date

def extract_texts(urls: List[str], num_words: Optional[int]) -> List[str]:
    """Extract texts from URLs"""
    max_allowed = 5
    extracted_texts = []
    dates = {}
    count = 0
    stop = False
    for batch in process_in_batches(urls=urls):
        for future, url in batch:
            try:
                result = future.result()
                if result.status_code != 200:
                    continue
                extracted_text, extracted_date = extract_text(html=result.text, num_words=num_words)
                extracted_texts.append(
                    f"ARTICLE {count}, DATE: {extracted_date}, URL: {url}, CONTENT: {extracted_text}"
                )
                dates[url] = extracted_date  
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
    return extracted_texts, dates


def fetch_additional_information(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    google_api_key: Optional[str],
    google_engine: Optional[str],
    num_urls: Optional[int],
    num_words: Optional[int],
    counter_callback: Optional[Callable] = None,
    source_links: Optional[List[str]] = None,
) -> str:
    """Fetch additional information."""
    url_query_prompt = URL_QUERY_PROMPT.format(user_prompt=prompt)
    moderation_result = client.moderations.create(input=url_query_prompt)
    if moderation_result.results[0].flagged:
        return ""
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
            google_api_key,
            google_engine,
            num_urls,
        )
        texts, dates = extract_texts(urls, num_words)
    else:
        texts = []
        for source_link in islice(source_links.values(), num_urls):
            texts.append(extract_text(html=source_link, num_words=num_words))

        return "\n".join(["- " + text for text in texts]), json_data["queries"], dates, counter_callback
    return "\n".join(["- " + text for text in texts]), json_data["queries"], dates, None

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


def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the task"""
    with OpenAIClientManager(kwargs["api_keys"]["openai"]):
        tool = kwargs["tool"]
        prompt = kwargs["question"]
        max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
        num_urls = kwargs.get("num_urls", DEFAULT_NUM_URLS[tool])
        num_words = kwargs.get("num_words", DEFAULT_NUM_WORDS[tool])
        counter_callback = kwargs.get("counter_callback", None)
        api_keys = kwargs.get("api_keys", {})
        google_api_key = api_keys.get("google_api_key", None)
        google_engine_id = api_keys.get("google_engine_id", None)

        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"Tool {tool} is not supported.")

        engine = TOOL_TO_ENGINE[tool]
        additional_information, queries, dates, counter_callback = fetch_additional_information(
            prompt,
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
            return "Moderation flagged the prompt as in violation of terms.", None, None
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
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
            )
            return response.choices[0].message.content, additional_information, queries, dates, counter_callback
        return response.choices[0].message.content, additional_information, queries, dates, None
