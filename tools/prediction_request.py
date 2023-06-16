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
import random
from typing import Any, Dict, List, Optional, Tuple

import googlesearch
import openai
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options


NUM_URLS_EXTRACT = 5

DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0.7,
}

ALLOWED_TOOLS = ["prediction_offline", "prediction_online"]

TOOL_TO_ENGINE = {
    "prediction_offline": "gpt-3.5-turbo",
    "prediction_online": "gpt-3.5-turbo",
}

PREDICTION_PROMPT = """
You are an LLM inside a multi-agent system that takes in a prompt of a user requesting a probability estimation
for a specific event. You must follow these instructions:
* The user will provide the event under the label "EVENT" delimited by three backticks.
* The event will only have two possible outcomes: either the event will happen or the event will not happen.
* You need to provide a probability estimation of the event happening, based on your training data.
* You are provided an array of information items under the label "ADITIONAL_INFORMATION" delimited by three backticks.
* You can use any items "ADITIONAL_INFORMATION" in addition to your training data.
* If an item in "ADITIONAL_INFORMATION" is not relevant, you must ignore it for the prediction.
* You need to provide an output in JSON format with the following format:
   - "p_yes": a value between 0 and 1 indicating the probability that the event occurs.
   - "p_no": a value between 0 and 1 indicating the probability that the event does not occur.
   - "confidence": a value between 0 and 1 indicating the confidence in the prediction, where 0 indicates lowest
     confidence value, and 1 maximum confidence value.
   - "info_utility": Utility of the information provided under ADITIONAL_INFORMATION to help you making the prediction.
     A value where 0 indicates lowest utility, and 1 maximum utility.
* The probability distribution must be well defined: the sum of p_yes and p_no must equal 1.

USER_PROMPT:
```
{user_prompt}
```

ADITIONAL_INFORMATION:
```
{additional_information}
```

Do not answer anything else than the JSON containing the probability estimation.
Your response must be a JSON object parseable by Python's json.loads().
"""

URL_QUERY_PROMPT = """
You are an LLM inside a multi-agent system that takes in a prompt of a user requesting a probability estimation
for a specific event.
The event is provided under the label "EVENT" quoted by three backticks.
You must output a JSON containing the following fields:
- "queries": An array of strings of size between 1 and 5. Each string must be a search engine query that can help the user to obtain relevant information to estimate
  the probability that the event occurs. You must provide original information in each query, and they should not overlap
  or lead the user to obtain the same set of results.
- "urls": An array of strings of size between 1 and 5. Each string must be a relevant URL where the user can find information
  about EVENT. The provided URLs must not repeat and must not be search engines.

USER_PROMPT:```{user_prompt}```

Do not answer anything else than the JSON containing the probability estimation.
Your response must be a JSON object parseable by Python's json.loads().
"""


def run(**kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Run the task"""
    openai.api_key = kwargs["api_keys"]["openai"]
    max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
    temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
    prompt = kwargs["prompt"]
    tool = kwargs["tool"]

    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")

    engine = TOOL_TO_ENGINE[tool]

    additional_information = ""

    if tool == "prediction_online":
        url_query_prompt = URL_QUERY_PROMPT.format(user_prompt=prompt)

        moderation_result = openai.Moderation.create(url_query_prompt)

        if moderation_result["results"][0]["flagged"]:
            return "Moderation flagged the prompt as in violation of terms."

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": url_query_prompt},
        ]
        response = openai.ChatCompletion.create(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            timeout=120,
            stop=None,
        )

        json_data = json.loads(response.choices[0].message.content)

        queries = json_data["queries"]
        urls = json_data["urls"]
        urls_from_queries = get_urls_from_queries(queries)
        urls.extend(urls_from_queries)
        url_texts = extract_texts(urls)
        additional_information = "\n".join(["- " + text for text in url_texts])

    prediction_prompt = PREDICTION_PROMPT.format(
        user_prompt=prompt, additional_information=additional_information
    )

    moderation_result = openai.Moderation.create(prediction_prompt)

    if moderation_result["results"][0]["flagged"]:
        return "Moderation flagged the prompt as in violation of terms."

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prediction_prompt},
    ]

    response = openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        timeout=120,
        stop=None,
    )

    return response.choices[0].message.content, None


def get_urls_from_queries(queries: List[str]) -> List[str]:
    """Get URLs from search engine queries"""
    results = []

    for query in queries:
        print(f"query: {query}")
        for url in googlesearch.search(query, num_results=3):
            results.append(url)

    unique_results = list(set(results))
    return unique_results


def extract_texts(urls: List[str], num_words: int = 300) -> List[str]:
    """Extract texts from URLs"""
    selected_urls = random.sample(urls, NUM_URLS_EXTRACT)
    extracted_texts = []

    options = Options()
    options.add_argument("-headless")
    browser = webdriver.Firefox(options=options)

    def get_html(url: str) -> str:
        """Get HTML content of a given URL"""
        browser.get(url)
        html = browser.find_element(By.TAG_NAME, "body").get_attribute("innerHTML")
        return html

    def extract_text(html: str) -> str:
        """Extract text from a single HTML document"""
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        return text

    for url in selected_urls:
        html = get_html(url)
        text = extract_text(html)
        words = text.split()[:num_words]
        extracted_texts.append(" ".join(words))

    browser.quit()
    return extracted_texts
