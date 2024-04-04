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

from datetime import datetime
import json
import re
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from packages.jhehemann.customs.infer_market_rules.infer_market_rules import get_market_rules
from packages.jhehemann.customs.research.research import research

from openai import OpenAI

import spacy
from spacy import Language
from spacy.cli import download
from spacy.tokens import Span
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
    "temperature": 0.0,
}
ALLOWED_TOOLS = [
    "prediction-openai-assistant"
]
MAX_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
}
TOOL_TO_ENGINE = {tool: "gpt-3.5-turbo" for tool in ALLOWED_TOOLS}
# the default number of URLs to fetch online information for
DEFAULT_NUM_URLS = defaultdict(lambda: 3)
DEFAULT_NUM_URLS["prediction-openai-assistant"] = 3
# the default number of words to fetch online information for
DEFAULT_NUM_WORDS: Dict[str, Optional[int]] = defaultdict(lambda: 300)

# how much of the initial content will be kept during summarization
DEFAULT_COMPRESSION_FACTOR = 0.05
# the vocabulary to use for the summarization
DEFAULT_VOCAB = "en_core_web_sm"

ASSISTANT_INSTRUCTIONS_PREDICTION = """
You are a highly advanced data scientist and reasoning expert. Your task is to provide accurate and robust probability estimations for the outcome of a prediction market question. \
You source all your knowledge from training and available information to perform this task.
"""

REPORT_PROMPT = """
Imagine today is {current_date} and someone on the street asks you the market question specified under MARKET_QUESTION. Prepare a concise but informative evaluation report that discusses the potential outcome of the market question based \
on the search output and the market rules.

Structure your report in the following sections and sub-sections:

* Introduction and Background
* Findings (in search output) and Analysis
    - Will the event in the market question happen? (Do NOT mention any date here)
    - When will it happen? (Name the date in the search output here)
    - Calculate the difference between the actual event date and the date specified in the market question.
* Outcome Evaluation (Event and Date) with common knowledge
* Conclusion
* Caveats

MARKET_QUESTION:
```
{market_question}
```

MARKET_RULES:
```
{market_rules}
```

SEARCH_OUTPUT:
```
{additional_information}
```

Output only the raw report without any additional information or formatting.
"""

PREDICTION_PROMPT_TEMPLATE = """
Use the research report from the previous answer and the market rules below to provide an estimated probability how the market question will resolve along with your confidence \
and the utility of the information provided. Output your answer in a single JSON object that contains four fields: "p_yes", "p_no", "confidence", "info_utility". \
Each item in the JSON must have a value between 0 and 1. Do not include any other contents in your response. Do not use formatting characters in your response.

Use the market rules as the basis for your probability estimation.
MARKET_RULES:
{market_rules_part}
"""


def trim_json_formatting(output_string):
    # Define a regular expression pattern that matches the start and end markers
    # with optional newline characters
    pattern = r'^```json\n?\s*({.*?})\n?```$'
    
    # Use re.DOTALL to make '.' match newlines as well
    match = re.match(pattern, output_string, re.DOTALL)
    
    if match:
        # Extract the JSON part from the matched pattern
        print("JSON formatting characters found and removed")
        formatted_json = match.group(1)
        return formatted_json
    else:
        # Return the original string if no match is found
        return output_string


def is_valid_json_with_fields_and_values(json_string) -> bool:
    """
    Check if a string is valid JSON, contains the required fields, 
    and adheres to the value constraints for each field
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
        return False


def load_model(vocab: str) -> Language:
    """Utilize spaCy to load the model and download it if it is not already available."""
    try:
        return spacy.load(vocab)
    except OSError:
        print("Downloading language model...")
        download(vocab)
        return spacy.load(vocab)


def extract_question(text):
    # Pattern to match a question enclosed in escaped quotation marks
    pattern = r'\"(.*?)\"'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    # If a match is found, return the first group (the content within the quotation marks)
    if match:
        return match.group(1)
    else:
        # If no match is found, return an informative message or handle it as needed
        return None


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
 
        # Extract the market question from the prompt delimited by escaped quotation marks
        market_question = extract_question(prompt)
        if market_question is None:
            return None, None, "Market question not found in prompt", None
        print(f"MARKET QUESTION:\n{market_question}\n")

        # Get the market rules from the Infer Rules tool
        market_rules, counter_callback = get_market_rules(market_question, client, counter_callback)
        print(f"MARKET RULES:\n{market_rules}\n")
        
        # Get additional information from the Research tool
        additional_inforamtion, counter_callback = research(market_question, client, google_api_key, google_engine_id, engine, market_rules, counter_callback)

        # Generate a report prompt based on the market question, market rules, additional information and the current date
        current_date = datetime.now().strftime('%B %d, %Y')
        report_prompt = REPORT_PROMPT.format(
            market_question=market_question,
            market_rules=market_rules,
            additional_information=additional_inforamtion,
            current_date=current_date
        )
        print(f"REPORT PROMPT:\n{report_prompt}\n")
        
        messages_report = [
            {"role": "system", "content": "You are a professional journalist"},
            {"role": "user", "content": report_prompt},
        ]
        # Generate a report based on the messages
        response = client.chat.completions.create(
            model=engine,
            messages=messages_report,
            temperature=temperature,
        )
        if counter_callback is not None:
            counter_callback(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=engine,
                token_counter=count_tokens,
            )
        output = response.choices[0].message.content
        print(f"OUTPUT:\n{output}\n")

        # Split the "Rules" part from the "Status" part of the market rules
        market_rules_part = market_rules.split("Rules:", 1)[1]
        market_rules_part = "Rules:" + market_rules_part
        
        # Generate a prediction prompt based on the market question and the "Rules" part of the market rules
        prediction_prompt = PREDICTION_PROMPT_TEMPLATE.format(market_question=market_question, market_rules_part=market_rules_part)
        print(f"PREDICTION PROMPT:{prediction_prompt}")
        
        messages_prediction = [
            {"role": "system", "content": ASSISTANT_INSTRUCTIONS_PREDICTION},
            # {"role": "user", "content": report_prompt}, # Uncomment this line to include the report prompt for more context
            {"role": "assistant", "content": output},
            {"role": "user", "content": prediction_prompt},
        ]

        thread_history = [
            {"role": "user", "content": report_prompt},
            {"role": "assistant", "content": output},
            {"role": "user", "content": prediction_prompt},
        ]
        thread_history_string = json.dumps(thread_history, indent=4)

        # Generate a prediction based on the messages
        response = client.chat.completions.create(
            model=engine,
            messages=messages_prediction,
            temperature=temperature,
        )
        if counter_callback is not None:
            counter_callback(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=engine,
                token_counter=count_tokens,
            )
        output = response.choices[0].message.content
        output = trim_json_formatting(output)
        print(f"OUTPUT:\n{output}\n")

        return response, thread_history_string, None, counter_callback
              

                




