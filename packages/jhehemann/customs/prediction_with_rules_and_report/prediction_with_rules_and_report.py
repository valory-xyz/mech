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
from typing import Any, Dict, List, Optional, Tuple, Callable
from openai import OpenAI
from spacy.tokens import Span
from tiktoken import encoding_for_model
from packages.jhehemann.customs.infer_market_rules.infer_market_rules import get_market_rules
from packages.jhehemann.customs.research.research import research


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
    "prediction-with-rules-and-report-gpt-3.5-turbo",
    "prediction-with-rules-and-report-gpt-4-turbo",
]
TOOL_TO_ENGINE = {
    "prediction-with-rules-and-report-gpt-3.5-turbo": "gpt-3.5-turbo",
    "prediction-with-rules-and-report-gpt-4-turbo": "gpt-4-turbo",
}

MAX_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "gpt-4-turbo": 8192,
}

# the default number of URLs to fetch online information for
DEFAULT_NUM_URLS = defaultdict(lambda: 3)
DEFAULT_NUM_URLS["prediction-with-rules-and-report"] = 3
# the default number of words to fetch online information for
DEFAULT_NUM_WORDS: Dict[str, Optional[int]] = defaultdict(lambda: 300)

# how much of the initial content will be kept during summarization
DEFAULT_COMPRESSION_FACTOR = 0.05
# the vocabulary to use for the summarization
DEFAULT_VOCAB = "en_core_web_sm"


REPORT_PROMPT = """
Your task is to write a concise evaluation report that discusses the potential outcome of the QUESTION found below. Your evaluation must be based \
on the SEARCH_OUTPUT and your domain expertise.

INSTRUCTIONS:
* Carefully read the QUESTION
* Examine the definitions in QUESTION_STATUS
* Analyze the SEARCH_OUTPUT and evaluate the date when the event will happen
* For reference, today's date is {current_date}. Use this information to determine timelines.
* Source your domain expertise and write a concise evaluation report that discusses the potential outcome of the QUESTION
* Give your response in the format specified under "OUTPUT_FORMAT". Aim for a response of about 200 words.

OUTPUT_FORMAT:
* Introduction and Context (including definitions from QUESTION_STATUS)
* QUESTION
* Findings and Analysis (Use domain expertise to justify your answers)
    * Event (Will the exact event specified in the QUESTION happen? Has it already happened?)
    * Date (On what date will the event specified in the QUESTION happen? You must provide a specific date on what you believe the event will happen. If you are uncertain, provide a range of dates.)

QUESTION:
```
{market_question}
```

QUESTION_STATUS:
```
{question_status}
```

SEARCH_OUTPUT:
```
{additional_information}
```

Output only the report without any additional information or formatting.
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


PREDICTION_PROMPT = """
You are a detective and an expert in solving complicated problems with logical conclusions. Your task is to provide an evaluation and make probability estimations for the outcomes 'Yes' and 'No' of a market question.

INSTRUCTIONS:
* You are provided with the MARKET_QUESTION under the label "MARKET_QUESTION".
* This MARKET_QUESTION consists of an event and a specific date. It is yet uncertain whether the event will happen aligning with the questioned date in the MARKET_QUESTION.
* There are only two outcomes possible for the MARKET_QUESTION: 'Yes' and 'No'.
* Under which conditions the MARKET_QUESTION's outcome will be 'Yes' and under which it will be 'No' are defined in the OUTCOME_RULES.
* You are also provided with a colleague's REASONING, under the label "REASONING", as to whether the event specified in the question will occur based on online research and also importantly when it will occur.
* You are provided with the OUTCOME_RULES that define conditions to help you evaluate the likelihood of the outcomes 'Yes' and 'No' under the label "OUTCOME_RULES".
* Your task splits into the following parts:
    - outcome evaluation
        - Start with the definitions provided in the OUTCOME_RULES.
        - Summarize the REASONING provided by your colleague under the label "REASONING"
        - Evaluate whether the event will happen and when it will happen.
        - Calculate the difference between the date in the REASONING and the date in the OUTCOME_RULES and determine if the event will happen before, on, or after the MARKET_QUESTION's date or if it has already happened.
        - Compare the REASONING with the OUTCOME_RULES to evaluate the likelihood of the MARKET_QUESTION's outcomes 'Yes' and 'No'. Use terms like 'almost certain', 'likely', 'moderate', 'unlikely', 'almost impossible'. 
    - Make probability estimations for the market's outcomes 'Yes' and 'No' taking the OUTCOME_RULES and the REASONING into account.
* Provide your evaluation process step by step and conclude with your likelihood estimation.
* For reference, today's date is {current_date}. Use this information to determine timelines.

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()"
* The JSON must contain five fields: "outcome_evaluation", "p_yes", "p_no", "confidence", "info_utility" each ranging from 0 to 1, except "outcome_evaluation" which is a string
    - "outcome_evaluation": Your output of the first parts of your task executed in order (aim for a response of about 200 words)
    - "p_yes": Probability of the MARKET_QUESTION's outcome being 'Yes' according to the OUTCOME_RULES
    - "p_no": Probability of the MARKET_QUESTION's outcome being 'No' according to the OUTCOME_RULES
    - "confidence": Your confidence in the estimation
    - "info_utility": Utility of the information in the REASONING
* Include only the JSON object in your output

REASONING:
```
{report}
```

OUTCOME_RULES:
```
{market_rules}
```

MARKET_QUESTION:
```
{market_question}
```
"""

def trim_json_formatting(text) -> str:
    """Trim the JSON formatting characters from string."""
    # Regex pattern that matches the start and end markers with optional newline characters
    pattern = r'^\s*```\s*json\s*({.*?})\s*```\s*$'

    # Use re.DOTALL to make '.' match newlines as well
    match = re.match(pattern, text, re.DOTALL)
    if match:
        formatted_json = match.group(1)
        return formatted_json
    else:
        return text


def remove_unwanted_fields(json_str) -> str:
    """Remove all fields from a JSON string except 'p_yes', 'p_no', 'confidence', and 'info_utility'."""
    # Load the JSON string into a Python dictionary
    data = json.loads(json_str)
    
    # Define the keys that you want to keep
    keys_to_keep = {'p_yes', 'p_no', 'confidence', 'info_utility'}
    
    # Use dictionary comprehension to keep only the desired keys
    filtered_data = {k: v for k, v in data.items() if k in keys_to_keep}
    
    # Convert the filtered dictionary back into a JSON string
    modified_json_str = json.dumps(filtered_data, indent=4)
    
    return modified_json_str


def extract_question(text) -> str:
    """Extract the question from prompt enclosed in escaped quotation marks."""
    pattern = r'\"(.*?)\"'
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def remove_date_from_query(query: str) -> str:
    """Remove time-related information from query"""
    date_pattern = r"\b(?:on or by |on or before |before |by |on )?(?:(\d{1,2})(st|nd|rd|th)? (January|February|March|April|May|June|July|August|September|October|November|December)|(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})(st|nd|rd|th)?,?) \d{4}\b"
    new_query = re.sub(date_pattern, "", query)
    return new_query


def get_sme_role(
    engine, temperature, max_tokens, prompt, counter_callback=None
) -> Tuple[str, str, Optional[Callable]]:
    """Get SME title and introduction"""
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
        if not market_question:
            return "Market question not found in prompt", None, None, None
        print(f"MARKET QUESTION:\n{market_question}\n")

        # Get the market rules from the Infer Rules tool
        market_status, market_rules, counter_callback = get_market_rules(market_question, client, counter_callback)
        print(f"MARKET STATUS: {market_status}\n")
        print(f"MARKET RULES:\n{market_rules}\n")
        
        # Get additional information from the Research tool
        additional_inforamtion, counter_callback = research(market_question, client, google_api_key, google_engine_id, engine, market_status, market_rules, counter_callback)

        # Remove date-related information from the market question
        market_question_no_date = remove_date_from_query(market_question)

        # Generate a report prompt based on the market question, market rules, additional information and the current date
        current_date = datetime.now().strftime('%B %d, %Y')
        report_prompt = REPORT_PROMPT.format(
            market_question=market_question_no_date,
            market_rules=market_rules,
            additional_information=additional_inforamtion,
            current_date=current_date,
            question_status=market_status
        )
        
        # Get the subject matter expert role and introduction
        sme = ""
        sme_introduction = ""
        try:
            sme, sme_introduction, counter_callback = get_sme_role(
                engine,
                temperature,
                max_tokens,
                market_question,
                counter_callback=counter_callback,
            )
        except Exception as e:
            print(f"An error occurred during SME role creation: {e}")
            print("Using default SME introduction.")
            sme_introduction = "You are a professional journalist."
        
        if sme:
            print(f"SME ROLE: {sme}")
        else:
            print("SME role not found.")
        print(f"SME INTRODUCTION: {sme_introduction}\n")

        messages_report = [
            {"role": "system", "content": sme_introduction},
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
        print(f"RESEARCH OUTPUT:\n########################\n{output}\n########################\n")

        prediction_prompt = PREDICTION_PROMPT.format(market_question=market_question, market_rules=market_rules, current_date=current_date, report=output)

        system_prediction_prompt = "You are a seasoned prediction market analyst with a deep understanding of how prediction markets work and how to assess the likelihood of different market resolutions. Your goal is to provide a well-reasoned analysis and probability estimations for the resolution of the prediction market based on your expertise in prediction markets and relevant domain knowledge. Carefully consider the market rules to make your evaluation."

        messages_prediction = [
            {"role": "system", "content": system_prediction_prompt},
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
            response_format={ "type": "json_object" },
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
        print(f"PREDICTION OUTPUT:\n########################\n{output}\n########################\n")

        # Remove conclusion field from the JSON string
        output = remove_unwanted_fields(output)
        
        return output, thread_history_string, None, counter_callback
              

                




