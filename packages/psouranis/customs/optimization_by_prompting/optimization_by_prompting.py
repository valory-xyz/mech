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

"""A script that implements the optimization by prompting methodology."""
import functools
import json
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from io import StringIO
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, cast

import anthropic
import googleapiclient
import openai
import pandas as pd
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from langchain.chains import LLMChain
from langchain.llms import OpenAI as OpenAILLM
from langchain.prompts import PromptTemplate
from openai import OpenAI
from pandas import DataFrame
from sklearn.metrics import roc_auc_score
from tiktoken import encoding_for_model


client: Optional[OpenAI] = None

MechResponseWithKeys = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]


N_MODEL_CALLS = 2
USER_AGENT_HEADER = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"


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
    enc = encoding_for_model(model)
    return len(enc.encode(text))


# Provide several examples in order to backtest the resulted prompt
EXAMPLES = """query;event
"Will Apple release iphone 15 by 1 October 2023?";1
"Will the newly elected ceremonial president of Singapore face any political scandals by 13 September 2023?";0
"Will Russia Invade Ukraine in 2022";1
"Will Finland and Sweden apply to join NATO in 2023?";1
"Will Charles become King in 2022?";1
"""

NUM_URLS_EXTRACT = 5
DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0.8,
}
ALLOWED_TOOLS = [
    "deepmind-optimization-strong",
    "deepmind-optimization",
]
TOOL_TO_ENGINE = {
    "deepmind-optimization-strong": "gpt-4o-2024-08-06",
    "deepmind-optimization": "gpt-3.5-turbo-0125",
}

PREDICTION_PROMPT_INSTRUCTIONS = """
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
"""


PREDICTION_PROMPT_FORMAT = """
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
* Output only the JSON object. Do not include any other contents in your response
* This is incorrect: "```json{{"queries": []}}```"
* This is incorrect: "```json"{{"queries": []}}"```"
* This is correct: "{{"queries": []}}".
"""

TEMPLATE_INSTRUCTOR = """You are an advanced reasoning agent that suggest to a bot ways to predict world events very accurately.
You are given the following:
(1) The previous instructions.
(2) A metric score that evaluates the previous instructions given to the bot. Best metric score is 1.

You are asked to refine the instructions in order to reach the best score.
Try to think the steps one by one.

Example format:
INSTRUCTIONS: previous instructions here
METRIC SCORE: score between 0 and 1 here

INSTRUCTIONS: {instructions}
METRIC SCORE: {score}
NEW INSTRUCTIONS:"""

PROMPT_INSTRUCTOR = PromptTemplate(
    input_variables=["instructions", "score"], template=TEMPLATE_INSTRUCTOR
)
OUTPUT_FORMAT = """
Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
The JSON must contain a field "p_yes" which marks the probability of the event happening.
A valid example is: {{"p_yes": 0.5}}
"""


def evaluate_prompt(prompt: str, df: DataFrame, llm: OpenAILLM) -> List:
    """Evaluates a prompt on each row of the provided DataFrame and returns probabilities."""
    prompt += OUTPUT_FORMAT
    chain = LLMChain(llm=llm, prompt=prompt)
    probas = []

    for row in df.itertuples():
        pred_chain = chain.run(
            {"user_prompt": row.query, "additional_information": OUTPUT_FORMAT}
        )
        try:
            dictionary_match = float(
                eval(pred_chain)["p_yes"]  # pylint: disable=eval-used
            )
        except BaseException as e:  # noqa: B036
            print(f"Error occurred while running evaluate_prompt: {e}")
            match = re.search(r"\{.*\}", pred_chain)
            if match:
                dictionary_match = float(
                    eval(match.group(0))["p_yes"]  # pylint: disable=eval-used
                )
            else:
                print("No match found in prediction chain")
                dictionary_match = 0.0

        probas.append(dictionary_match)

    return probas


def calculate_score(
    df: DataFrame, answer_key: str = "event", prob_key: str = "probability"
) -> float:
    """Calculates the ROC AUC score between the true labels and predicted probabilities."""
    return roc_auc_score(df[answer_key], df[prob_key])


def create_new_instructions(llm: OpenAILLM, instructions: Any, score: float) -> Any:
    """Generates new instructions based on the provided score and existing instructions."""
    chain = LLMChain(llm=llm, prompt=PROMPT_INSTRUCTOR)
    evaluations = chain.run({"instructions": instructions, "score": score})
    return evaluations


def prompt_engineer(
    openai_api_key: str,
    init_instructions: str,
    instructions_format: str,
    iterations: int = 3,
    model_name: str = "gpt-4o-2024-08-06",
) -> Any:
    """Iteratively refines a prompt template using a large language model to maximize performance score."""
    llm = OpenAILLM(model_name=model_name, openai_api_key=openai_api_key)
    score_template = {"template": init_instructions, "score": 0.0}

    df = pd.read_csv(StringIO(EXAMPLES), sep=";")
    template = init_instructions

    for _ in range(iterations):
        generated_template = template + instructions_format
        try:
            prompt = PromptTemplate(
                input_variables=["user_prompt", "additional_information"],
                template=generated_template,
            )
        except Exception as e:
            # it may happen that the generated prompt is not valid
            # in that case, we just skip it
            print(f"Failed to parse template {generated_template}: {e}")
            # regenerate the template
            template = create_new_instructions(
                llm=llm,
                instructions=score_template["template"],
                score=cast(float, score_template["score"]),
            )
            continue

        df["probability"] = evaluate_prompt(prompt=prompt, llm=llm, df=df)

        score = calculate_score(df)
        print(f"Score: {score}\n")
        if score > cast(float, score_template["score"]):
            print(f"Best template score: {score} \nTemplate: {template}\n")
            score_template["template"] = template
            score_template["score"] = score
        template = create_new_instructions(
            llm=llm,
            instructions=score_template["template"],
            score=cast(float, score_template["score"]),
        )

    return score_template["template"]


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


def get_urls_from_queries_serper(queries: List[str], api_key: str, num: int = 3) -> List[str]:
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


def extract_text(
    html: str,
    num_words: int = 300,  # TODO: summerise using GPT instead of limit
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


def extract_texts(urls: List[str], num_words: int = 300) -> List[str]:
    """Extract texts from URLs"""
    max_allowed = 5
    extracted_texts = []
    count = 0
    stop = False
    for batch in process_in_batches(urls=urls) or []:
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
    temperature: float,
    max_tokens: int,
    google_api_key: str,
    google_engine: str,
    serper_api_key: Optional[str],
    search_provider: str,
) -> str:
    """Fetch additional information."""
    if not client:
        raise RuntimeError("Client not initialized")

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
    
    # Determine which search provider to use
    if search_provider == "serper":
        if not serper_api_key:
            raise RuntimeError("Serper API key not found")
        urls = get_urls_from_queries_serper(
            queries=json_data["queries"],
            api_key=serper_api_key,
        )
    else:  # default to google
        urls = get_urls_from_queries(
            json_data["queries"],
            api_key=google_api_key,
            engine=google_engine,
        )
    texts = extract_texts(urls)
    return "\n".join(["- " + text for text in texts])


@with_key_rotation
def run(
    **kwargs: Any,
) -> Union[float, Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]]:
    """Run the task"""
    tool = kwargs["tool"]
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")

    engine = kwargs.get("model", TOOL_TO_ENGINE[tool])
    delivery_rate = int(kwargs.get("delivery_rate", 0))
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

        if not client:
            raise RuntimeError("Client not initialized")

        openai_key = kwargs["api_keys"]["openai"]
        serper_api_key = kwargs["api_keys"].get("serperapi", None)
        search_provider = kwargs.get("search_provider", "google").lower()

        additional_information = fetch_additional_information(
            prompt=prompt,
            engine=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            google_api_key=kwargs["api_keys"]["google_api_key"],
            google_engine=kwargs["api_keys"]["google_engine_id"],
            serper_api_key=serper_api_key,
            search_provider=search_provider,
        )

        instructions = prompt_engineer(
            openai_key, PREDICTION_PROMPT_INSTRUCTIONS, PREDICTION_PROMPT_FORMAT
        )
        instructions += PREDICTION_PROMPT_FORMAT
        prediction_prompt = instructions.format(
            user_prompt=prompt, additional_information=additional_information
        )

        moderation_result = client.moderations.create(input=prediction_prompt)
        if moderation_result.results[0].flagged:
            return (
                "Moderation flagged the prompt as in violation of terms.",
                None,
                None,
                None,
            )
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
        return response.choices[0].message.content, prediction_prompt, None, None
