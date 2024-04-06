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
from typing import Any, Dict, List, Optional, Tuple, Callable
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
    "prediction-with-rules-and-report"
]
MAX_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
}
TOOL_TO_ENGINE = {tool: "gpt-3.5-turbo" for tool in ALLOWED_TOOLS}
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
Your task is to prepare a concise and informative evaluation report that discusses the potential outcome of the QUESTION found below. Your evaluation must be based \
on the SEARCH_OUTPUT and your domain expertise.
Adhere to the following instructions:

INSTRUCTIONS:
* Carefully read the QUESTION
* Separate the QUESTION into its components
* Carefully read the search output provided.
* Analyze the search output and evaluate the date when the event will actually happen
* Source your domain expertise to provide caveats
* Give your response in the format specified under "OUTPUT_FORMAT"

SEARCH_OUTPUT:
```
{additional_information}
```

QUESTION:
```
{market_question}
```

TODAYS_DATE: {current_date}

OUTPUT_FORMAT:
* Introduction and Context
* Findings and Analysis
    - Will the event specified in the question happen?
    - On what date will the event actually happen? Has the event already happened? You must provide a specific date. If you are uncertain, provide a range of dates.
* Conclusion with common sense reasoning
* Caveats
Output only the raw report without any additional information or formatting.
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


PREDICTION_PROMPT_TEMPLATE = """
Given your previous answer your task is to provide a probability estimation how the market question will eventually resolve. You are provided with the MARKET_RULES for the resolution of the MARKET_QUESTION.
You must adhere to the following instructions:

INSTRUCTIONS:
* Carefully read the MARKET_RULES
* Analyze the report from your previous answer
* Note: Today's date is {current_date}
* Write a detailed conclusion of the report from your previous answer and evaluate the likelihood of the MARKET_QUESTION resolving as 'Yes' or 'No' by referring to the conditions in the MARKET_RULES.
* Is there a discrepancy between the actual event date and the date specified in the MARKET_QUESTION? If yes, consider this in your evaluation.
* Make a probability estimation based on the conclusion
* Provide your confidence in the estimation and the utility of the information in the report
* Give your response in the format specified under "OUTPUT_FORMAT"

MARKET_QUESTION:
{market_question}

MARKET_RULES:
{market_rules_part}

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()"
* The JSON must contain five fields: "conclusion", "p_yes", "p_no", "confidence", "info_utility" each ranging from 0 to 1
    - "conclusion": A detailed analysis of the report from the previous answer evaluating the likelihood of the MARKET_QUESTION resolving as 'Yes' or 'No'
    - "p_yes": The estimated probability that the market question will resolve as 'Yes'
    - "p_no": The estimated probability that the market question will resolve as 'No'
    - "confidence": Your confidence in the probability estimation
    - "info_utility": Your assessment of the utility of the information provided in the report
* Include only the JSON object in your output
"""


def trim_json_formatting(text) -> str:
    """Trim the JSON formatting characters from string."""
    # Regex pattern that matches the start and end markers with optional newline characters
    pattern = r'^```json\n?\s*({.*?})\n?```$'

    # Use re.DOTALL to make '.' match newlines as well
    match = re.match(pattern, text, re.DOTALL)
    if match:
        formatted_json = match.group(1)
        return formatted_json
    else:
        return text


def remove_conclusion_field(json_str) -> str:
    """Remove the 'conclusion' field from a JSON string."""
    data = json.loads(json_str)
    if 'conclusion' in data:
        del data['conclusion']
    modified_json_str = json.dumps(data, indent=4)    
    return modified_json_str


def extract_question(text) -> str:
    """Extract the question from prompt enclosed in escaped quotation marks."""
    pattern = r'\"(.*?)\"'
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def remove_date_from_query(query: str) -> str:
    """Remove time-related information from query"""
    date_pattern = r"\b( on or before | by | on )?\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b"
    new_query = re.sub(date_pattern, "", query)
    return new_query


def split_before_evaluation(text) -> Tuple[str, str]:
    """Split string at last occurrence of 'Evaluation'"""
    eval_index = text.rfind("Evaluation")
    if eval_index == -1:
        return text, ""
    
    # Find the last newline character before "Evaluation"
    newline_index = text.rfind("\n", 0, eval_index)
    if newline_index == -1:
        return text, ""
    
    # Split the string at the found newline index
    part1 = text[:newline_index]
    part2 = text[newline_index + 1:]
    
    return part1, part2


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
        market_rules, counter_callback = get_market_rules(market_question, client, counter_callback)
        print(f"MARKET RULES:\n{market_rules}\n")
        
        # Get additional information from the Research tool
        additional_inforamtion, counter_callback = research(market_question, client, google_api_key, google_engine_id, engine, market_rules, counter_callback)

        question_status = market_rules.split("\nRules:", 1)[0]

        market_question_no_date = remove_date_from_query(market_question)
        market_question_when = f"When {market_question_no_date}"

        # Generate a report prompt based on the market question, market rules, additional information and the current date
        current_date = datetime.now().strftime('%B %d, %Y')
        report_prompt = REPORT_PROMPT.format(
            market_question=market_question_when,
            market_rules=market_rules,
            additional_information=additional_inforamtion,
            current_date=current_date,
            question_status=question_status
        )
        print(f"REPORT PROMPT:\n{report_prompt}\n")
        
        # Get the subject matter expert role and introduction
        try:
            sme, sme_introduction, counter_callback = get_sme_role(
                engine,
                temperature,
                max_tokens,
                prompt,
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
        print(f"SME INTRODUCTION: {sme_introduction}")
        print()

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
        print(f"OUTPUT:\n{output}\n")

        # Split the "Rules" part from the "Status" part of the market rules
        market_rules_part = market_rules.split("Rules:", 1)[1]
        market_rules_part = "Rules:" + market_rules_part
        
        # Generate a prediction prompt based on the market question and the "Rules" part of the market rules
        prediction_prompt = PREDICTION_PROMPT_TEMPLATE.format(market_question=market_question, market_rules_part=market_rules_part, current_date=current_date)
        print(f"PREDICTION PROMPT:{prediction_prompt}")
        
        messages_prediction = [
            {"role": "system", "content": sme_introduction},
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

        # Remove conclusion field from the JSON string
        output = remove_conclusion_field(output)
        
        return output, thread_history_string, None, counter_callback
              

                




