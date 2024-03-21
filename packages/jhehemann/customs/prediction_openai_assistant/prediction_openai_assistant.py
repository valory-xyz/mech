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

import json
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from heapq import nlargest
from itertools import islice
from string import punctuation
from typing import Any, Dict, Generator, List, Optional, Tuple, Callable
from packages.jhehemann.customs.prediction_openai_assistant.infer_rules import get_market_rules
from packages.jhehemann.customs.prediction_openai_assistant.research_additional_information import research_additional_information

from openai import OpenAI

import requests
import spacy
import tiktoken
from markdownify import markdownify as md
from readability import Document
from googleapiclient.discovery import build
from spacy import Language
from spacy.cli import download
from spacy.lang.en import STOP_WORDS
from spacy.tokens import Doc, Span
from tiktoken import encoding_for_model

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

def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


FrequenciesType = Dict[str, float]
ScoresType = Dict[Span, float]


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0.7,
}
ALLOWED_TOOLS = [
    "prediction-openai-assistant"
]
MAX_TOKENS = {
    "gpt-3.5-turbo-1106": 4096,
    "gpt-4": 8192,
}
TOOL_TO_ENGINE = {tool: "gpt-3.5-turbo-1106" for tool in ALLOWED_TOOLS}
# the default number of URLs to fetch online information for
DEFAULT_NUM_URLS = defaultdict(lambda: 3)
DEFAULT_NUM_URLS["prediction-openai-assistant"] = 3
# the default number of words to fetch online information for
DEFAULT_NUM_WORDS: Dict[str, Optional[int]] = defaultdict(lambda: 300)

# how much of the initial content will be kept during summarization
DEFAULT_COMPRESSION_FACTOR = 0.05
# the vocabulary to use for the summarization
DEFAULT_VOCAB = "en_core_web_sm"
RUN_TERMINATED_STATES = ["expired", "completed", "failed", "cancelled"]
RUN_RUNNING_STATES = ["queued", "in_progress"]
RUN_ACTION_REQUIRED_STATES = ["requires_action"]

# * Use the current date and time ({timestamp}) as a reference to understand the context of the market question, but focus primarily on the market question's specified date to guide your answer.


ASSISTANT_INSTRUCTIONS = """
You are an autonomous AI agent that gathers highly reliable and valid information from different sources. You are provided with \
a market question that is currently open to bet on on a prediction market. Your task is to gather current and highly reliable \
information and make a probability estimation for the market question. You have access to an agent ecosystem that provides you with \
a vast amount of tools and information to help you make the estimation.

INSTRUCTIONS:
* Examine the market question labeled 'MARKET_QUESTION'.
* The probability estimations of the market question outcomes must be as accurate as possible, as an inaccurate estimation will lead to financial loss for the user.
* Utilize your training data and the tools provided to make your estimations
* The specified date in the user question is the key determinant for your response
* Carefully evaluate when events or information are reported to occur in relation to the market question's specified date.
* If a search generates relevant information, indicating towards one of the two outcomes, but does not seem to guarantee the outcome by the specified date, you should estimate the probability of the outcome happening within the time frame from the current date to the specified date
* You must provide a response in the format specified under "OUTPUT_FORMAT"
* Do not include any other contents in your response

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()"
* The JSON must contain four fields: "p_yes", "p_no", "confidence"
* Each item in the JSON must have a value between 0 and 1
    - "p_yes": Probability that the user question's outcome will be `Yes`
    - "p_no": Probability that the user question's outcome will be `No`
    - "confidence": Your confidence (value between 0 and 1) in your estimated probabilities
    - "info_utility": Utility of the information provided by the agents that helped you make the estimation. 0 indicates lowest utility; 1 maximum utility.
Do not include any other contents except for the JSON object in your outputs.
"""


# PREDICTION_PROMPT = """
# You are an LLM inside a multi-agent system that takes in a prompt of a user requesting a probability estimation
# for a given event. You are provided with an input under the label "USER_PROMPT". You must follow the instructions
# under the label "INSTRUCTIONS". You must provide your response in the format specified under "OUTPUT_FORMAT".

# INSTRUCTIONS
# * Read the input under the label "USER_PROMPT" delimited by three backticks.
# * The "USER_PROMPT" specifies an event.
# * The event will only have two possible outcomes: either the event will happen or the event will not happen.
# * If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
# * You must provide a probability estimation of the event happening, based on your training data.
# * You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION" delimited by three backticks.
# * You can use any item in "ADDITIONAL_INFORMATION" in addition to your training data.
# * If an item in "ADDITIONAL_INFORMATION" is not relevant, you must ignore that item for the estimation.
# * You must provide your response in the format specified under "OUTPUT_FORMAT".
# * Do not include any other contents in your response.

# USER_PROMPT:
# ```
# {user_prompt}
# ```

# ADDITIONAL_INFORMATION:
# ```
# {additional_information}
# ```

# OUTPUT_FORMAT
# * Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
# * The JSON must contain four fields: "p_yes", "p_no", "confidence", and "info_utility".
# * Each item in the JSON must have a value between 0 and 1.
#    - "p_yes": Estimated probability that the event in the "USER_PROMPT" occurs.
#    - "p_no": Estimated probability that the event in the "USER_PROMPT" does not occur.
#    - "confidence": A value between 0 and 1 indicating the confidence in the prediction. 0 indicates lowest
#      confidence value; 1 maximum confidence value.
#    - "info_utility": Utility of the information provided in "ADDITIONAL_INFORMATION" to help you make the prediction.
#      0 indicates lowest utility; 1 maximum utility.
# * The sum of "p_yes" and "p_no" must equal 1.
# * Output only the JSON object. Do not include any other contents in your response.
# * This is incorrect:"```json{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}```"
# * This is incorrect:```json"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"```
# * This is correct:"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"
# """

# URL_QUERY_PROMPT = """
# You are an LLM inside a multi-agent system that takes in a prompt of a user requesting a probability estimation
# for a given event. You are provided with an input under the label "USER_PROMPT". You must follow the instructions
# under the label "INSTRUCTIONS". You must provide your response in the format specified under "OUTPUT_FORMAT".

# INSTRUCTIONS
# * Read the input under the label "USER_PROMPT" delimited by three backticks.
# * The "USER_PROMPT" specifies an event.
# * The event will only have two possible outcomes: either the event will happen or the event will not happen.
# * If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
# * You must provide your response in the format specified under "OUTPUT_FORMAT".
# * Do not include any other contents in your response.

# USER_PROMPT:
# ```
# {user_prompt}
# ```

# OUTPUT_FORMAT
# * Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
# * The JSON must contain two fields: "queries", and "urls".
#    - "queries": An array of strings of size between 1 and 5. Each string must be a search engine query that can help obtain relevant information to estimate
#      the probability that the event in "USER_PROMPT" occurs. You must provide original information in each query, and they should not overlap
#      or lead to obtain the same set of results.
# * Output only the JSON object. Do not include any other contents in your response.
# * Never use Markdown syntax highlighting, such as ```json``` to surround the output. Only output the raw json string.
# * This is incorrect: "```json{{"queries": []}}```"
# * This is incorrect: "```json"{{"queries": []}}"```"
# * This is correct: "{{"queries": []}}"
# """

PREDICTION_ASSISTANT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_market_rules",
            "description": "Get the rules for a prediction market question that defines under which conditions the market question will be resolved as 'Yes' and 'No'",
            "parameters": {
                "type": "object",
                "properties": {
                    "market_question": {"type": "string", "description": "The market question to infer the rules for"},
                },
                "required": ["market_question"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "research_additional_information",
            "description": "A search engine optimized for comprehensive, accurate, and trusted information to help you make a probability estimation for a market question",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_query": {"type": "string", "description": "The exactly phrased market question to infer the rules for"},
                },
                "required": ["input_query"],
            }
        }
    }
]


JSON_ASSISTANT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "format_prediction_values",
            "description": "Format the prediction values and return them in JSON format. You must NOT use this tool before you have made the prediction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "p_yes": {"type": "string", "description": "The estimated probability that the market resolves as 'Yes'"},
                    "p_no": {"type": "string", "description": "The estimated probability that the market resolves as 'No'"},
                    "confidence": {"type": "string", "description": "Confidence in the prediction"},
                    "info_utility": {"type": "string", "description": "Utility of the information provided"},
                },
                "required": ["p_yes", "p_no", "confidence", "info_utility"],
            }
        }
    }
]


### Functions that can be called by an assistant


def format_prediction_values(p_yes, p_no, confidence, info_utility):
    """Format the prediction values and return them in JSON format"""

    # Construct a dictionary with the prediction values
    prediction_values = {
        "p_yes": float(p_yes),
        "p_no": float(p_no),
        "confidence": float(confidence),
        "info_utility": float(info_utility)
    }

    # Convert the dictionary to a JSON-formatted string
    json_output = json.dumps(prediction_values, indent=4)

    return json_output


OPENAI_TOOLS_FUNCTIONS = {
    "get_market_rules": get_market_rules,
    "research_additional_information": research_additional_information,
    "format_prediction_values": format_prediction_values,
}


### Regular functions


def is_valid_json_with_fields_and_values(json_string):
    """
    Check if the input string is valid JSON, contains the required fields,
    and adheres to the value constraints for each field.

    Parameters:
    - json_string (str): The string to be checked.

    Returns:
    - bool: True if the string meets all criteria, False otherwise.
    """
    required_fields = ["p_yes", "p_no", "confidence", "info_utility"]

    try:
        # Attempt to parse the JSON string
        data = json.loads(json_string)

        # Check if all required fields are present
        if not all(field in data for field in required_fields):
            return False

        # Check if 'p_yes' and 'p_no' are floats within [0, 1] and their sum equals 1
        if not all(isinstance(data[field], float) and 0 <= data[field] <= 1 for field in ["p_yes", "p_no"]):
            return False
        if data["p_yes"] + data["p_no"] != 1:
            return False

        # Check if 'confidence' and 'info_utility' are floats within [0, 1]
        if not all(isinstance(data[field], float) and 0 <= data[field] <= 1 for field in ["confidence", "info_utility"]):
            return False

        return True

    except json.JSONDecodeError:
        # If json_string is not valid JSON, return False
        return False




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


def extract_texts(urls: List[str], num_words: Optional[int]) -> List[str]:
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
                doc = {}
                doc['text'] = extract_text(html=result.text, num_words=num_words)
                doc['url'] = url
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


# def fetch_additional_information(
#     prompt: str,
#     engine: str,
#     temperature: float,
#     max_tokens: int,
#     google_api_key: Optional[str],
#     google_engine: Optional[str],
#     num_urls: Optional[int],
#     num_words: Optional[int],
#     source_links: Optional[List[str]] = None,
# ) -> Tuple[str, Any]:
#     """Fetch additional information."""
#     url_query_prompt = URL_QUERY_PROMPT.format(user_prompt=prompt)
#     moderation_result = client.moderations.create(input=url_query_prompt)
#     if moderation_result.results[0].flagged:
#         return ""
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": url_query_prompt},
#     ]
#     response = client.chat.completions.create(
#         model=engine,
#         messages=messages,
#         temperature=temperature,
#         max_tokens=max_tokens,
#         n=1,
#         timeout=90,
#         stop=None,
#     )
#     json_data = json.loads(response.choices[0].message.content)
#     if not source_links:
#         urls = get_urls_from_queries(
#             json_data["queries"],
#             google_api_key,
#             google_engine,
#             num_urls,
#         )
#         texts = extract_texts(urls, num_words)
#     else:
#         texts = []
#         for url, content in islice(source_links.items(), 3):
#             doc = {}
#             doc['text'], doc['url'] = extract_text(html=content, num_words=num_words), url
#             texts.append(doc)
#     # Format the additional information
#     additional_information = "\n".join(
#         [
#             f"ARTICLE {i}, URL: {doc['url']}, CONTENT: {doc['text']}\n"
#             for i, doc in enumerate(texts)
#         ]
#     )
#     return additional_information


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
    word_frequencies = defaultdict(lambda: 0)
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
    sentence_scores = defaultdict(lambda: 0)
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
    summary = nlargest(n, sentence_scores, key=sentence_scores.get)
    summary_words = [word.text for word in summary]
    summary_text = "".join(summary_words)
    return summary_text


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


def execute_function(tool_call, client, google_api_key=None, google_engine_id=None, engine=None):
    function_name = tool_call.function.name
    function_to_call = OPENAI_TOOLS_FUNCTIONS[function_name]
    function_args = json.loads(tool_call.function.arguments)

    if function_name == "get_market_rules":
        return function_to_call(**function_args, client=client)
    elif function_name == "research_additional_information":
        return function_to_call(
            **function_args,
            client=client,
            google_api_key=google_api_key,
            google_engine_id=google_engine_id,
            engine=engine,
        )
    else:
        return function_to_call(**function_args)


def is_terminated(run):
    """Check if the run is terminated"""
    return run.status in RUN_TERMINATED_STATES


def wait_for_run(
    client: OpenAI,
    thread_id: str,
    run_id: str,
    timeout=60,
):
    timeout = timeout
    start_time = time.time()
    thread_id = thread_id
    run_id = run_id

    # print(f"Thread ID: {thread_id}\n")
    # print(f"Run ID: {run_id}\n")

    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    # print(f"Run retrieved:\n{run}\n")
    # print(f"Run status: {run.status}\n")

    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        print(f"Run status: {run.status}")

        if run.status not in RUN_RUNNING_STATES:
            print(f"Run status changed to: {run.status}\n")
            # print(f"Run new:\n{run}\n")
            return run

        if time.time() - start_time > timeout:
            print(f"Run timed out. Run status:\n{run.status}\n")
            return None

        time.sleep(0.1)


def call_tools(
    run,
    google_api_key=None,
    google_engine_id=None,
    engine=None,
):
    print(f"Required action:\n {run.required_action}\n")

    # List to store futures
    futures_dict = {}

    # Outer ThreadPoolExecutor to manage parallel execution of all functions
    with ThreadPoolExecutor(max_workers=len(run.required_action.submit_tool_outputs.tool_calls)) as executor:
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            # Submit each function execution as a separate task to the executor
            future = executor.submit(execute_function, tool_call, client, google_api_key, google_engine_id, engine)
            futures_dict[future] = tool_call.id

        tool_outputs = []
        # Wait for all futures to complete and print their results
        for future in as_completed(futures_dict):
            result = future.result()
            tool_call_id = futures_dict[future]
            tool_outputs.append({
                "tool_call_id": tool_call_id,
                "output": result
            })
    return tool_outputs


def wait_for_run_termination(
    client: OpenAI,
    thread_id,
    run_id,
    google_api_key=None,
    google_engine_id=None,
    engine=None,
    return_tool_outputs_only=False,
    current_iteration=0,
    max_iterations=3,
):
    # Termination condition for recursion
    if current_iteration >= max_iterations:
        print(f"Reached maximum number of iterations: {max_iterations}")
        return None # Stop the recursion
    
    run = wait_for_run(client, thread_id, run_id)

    if run and run.status in RUN_ACTION_REQUIRED_STATES:
        tool_outputs = call_tools(run, google_api_key, google_engine_id, engine)
        for tool_output in tool_outputs:
            print(f"TOOL OUTPUT:\n{tool_output['output']}\n\n")
        
        if return_tool_outputs_only:
            client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
            return tool_outputs
        
        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run.id,
            tool_outputs=tool_outputs,
        )

        run = wait_for_run_termination(client, thread_id, run.id)
        # print(f"Run status: {run.status}\n")

        thread_messages = client.beta.threads.messages.list(thread_id)
        response = thread_messages.data[0].content[0].text.value
        print(f"Assistant message added to thread:\n{thread_id}: {response}\n")

    return run


# def count_tokens(runs) -> Tuple[int, int, int]:
#     """Count the number of tokens in the runs"""
#     prompt_tokens = 0
#     completion_tokens = 0
#     total_tokens = 0
#     for run in runs:
#         prompt_tokens += run.usage.prompt_tokens
#         completion_tokens += run.usage.completion_tokens
#         total_tokens += run.usage.total_tokens
#     return prompt_tokens, completion_tokens, total_tokens


def run(**kwargs) -> Tuple[Optional[str], Any, Optional[Dict[str, Any]], Any]:
    """Run the task"""
    with OpenAIClientManager(kwargs["api_keys"]["openai"]):
        tool = kwargs["tool"]
        prompt = kwargs["prompt"]
        max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
        num_urls = kwargs.get("num_urls", DEFAULT_NUM_URLS[tool])
        num_words = kwargs.get("num_words", DEFAULT_NUM_WORDS[tool])
        compression_factor = kwargs.get("compression_factor", DEFAULT_COMPRESSION_FACTOR)
        vocab = kwargs.get("vocab", DEFAULT_VOCAB)
        counter_callback = kwargs.get("counter_callback", None)
        api_keys = kwargs.get("api_keys", {})
        google_api_key = api_keys.get("google_api_key", None)
        google_engine_id = api_keys.get("google_engine_id", None)

        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"Tool {tool} is not supported.")

        engine = TOOL_TO_ENGINE[tool]
        run_ids = []

        try:
            # Create an openai assistant
            assistant = client.beta.assistants.create(
                name="Prediction Agent",
                instructions=ASSISTANT_INSTRUCTIONS,
                tools=PREDICTION_ASSISTANT_TOOLS,
                model=engine,
            )
            
            # Create a thread
            thread = client.beta.threads.create(
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )

            # Apply the prediction assistant to the thread
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
            )
            run_ids.append(run.id)

            # Wait until run is in one of the terminal states
            run = wait_for_run_termination(
                client,
                thread.id,
                run.id,
                google_api_key=google_api_key,
                google_engine_id=google_engine_id,
                engine=engine,
            )
            response = client.beta.threads.messages.list(thread.id).data[0].content[0].text.value
            
            if not is_valid_json_with_fields_and_values(response):
                client.beta.threads.messages.create(
                    thread.id,
                    role="user",
                    content="Output your answer in JSON format.",
                )

                # update assistant and replace the tools with the JSON assistant tool
                assistant = client.beta.assistants.update(
                    assistant.id,
                    tools=JSON_ASSISTANT_TOOLS,
                )

                run = client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant.id,
                )
                run_ids.append(run.id)

                tool_outputs = wait_for_run_termination(
                    client,
                    thread.id,
                    run.id,
                    google_api_key=google_api_key,
                    google_engine_id=google_engine_id,
                    engine=engine,
                    return_tool_outputs_only=True,
                )

                if isinstance(tool_outputs, list):
                    prediction = tool_outputs[0]["output"]
                    # Print the type of the prediction variable. 
                    print(f"Type of prediction: {type(prediction)}")
                    # Convert the prediction variable to string 
                    prediction = str(prediction)
                    # Print the prediction variable.
                    print(f"Type of prediction: {type(prediction)}")
                else:
                    prediction = None

            # print()
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # print()
            
            # runs = client.beta.threads.runs.list(thread_id=thread.id)
            # for r in runs:
            #     print(f"RUN: {r.id}")
            #     steps = client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=r.id)
            #     for s in steps:
            #         print(f"STEP: {s.id}")
            #         print(s)
            #         print("\n----------------------------------------------------\n")
            
            # print()
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # print()

            # Print all the thread messages
            thread_messages = client.beta.threads.messages.list(thread.id)
            # for message in thread_messages:
            #     print(f"{message.role}: {message.content[0].text.value}")
            #     print("\n----------------------------------------------------\n")
            
            # # Create an openai assistant
            # assistant = client.beta.assistants.create(
            #     name="JSON Agent",
            #     instructions=JSON_AGENT_INSTRUCTIONS,
            #     model=engine,
            # )
            
            # Add the response to the thread as a message. 



            print(f"FINAL OUTPUT:\n{prediction}")
            print(f"\nIS VALID RESPONSE: {is_valid_json_with_fields_and_values(prediction)}\n")

            return prediction, prompt, None, counter_callback
        
        finally:
            # For later:
            # Update assistant and replace JSON tool with the prediction assistant tools for next run
            client.beta.assistants.update(
                assistant.id,
                tools=PREDICTION_ASSISTANT_TOOLS,
            )
            # Delete run, thread and assistant
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run_ids[0])
            if run and not is_terminated(run):
                print(f"Run found with status: {run.status}")
                client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
                print(f"Run cancelled: {run.id}")
                if thread:
                    client.beta.threads.delete(thread.id)
                    print(f"Thread deleted: {thread.id}")
                    if assistant:
                        client.beta.assistants.delete(assistant.id)
                        print(f"Assistant deleted: {assistant.id}")
                    else:
                        print("Assistant not found.")
                else:
                    print("Thread not found.")
            elif is_terminated(run):
                print(f"Run successfully terminated: {run.id}")
            else:
                print("Run not found.")
                




