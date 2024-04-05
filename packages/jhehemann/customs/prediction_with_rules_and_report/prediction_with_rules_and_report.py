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
You are a data scientist and reasoning expert. Your task is to provide accurate and robust probability estimations for the outcome of a prediction market question. \
You source all your knowledge from training and available information to perform this task.
"""

REPORT_PROMPT = """
Your task is to prepare a detailed and informative evaluation report that discusses the potential outcome of the QUESTION found below. Your evaluation must be based \
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
    - Will the event happen?
    - On what date will the event actually happen? Has the event already happened? You must provide a specific date. If you are uncertain, provide a range of dates.
* Conclusion
* Caveats
Output only the raw report without any additional information or formatting.
"""


REPORT_PROMPT_EXPERIMENT = """
Your task is to prepare a detailed and informative evaluation report that discusses the potential outcome of the QUESTION found below. The conditions for the outcome \
are specified in the RESOLUTION_RULES. Your evaluation must be based on the SEARCH_OUTPUT and your domain expertise.
Adhere to the following instructions:

INSTRUCTIONS:
* Carefully read the QUESTION
* Separate the QUESTION into its components
* Carefully read the search output provided.
* Analyze the search output and evaluate the likelihood of the event specified in the QUESTION happening or not happening
* Analyze the search output and evaluate the date when the event will actually happen
* Calculate and explain the difference between the actual event date and the date specified in the QUESTION. Use common sense reasoning and the publication dates of the articles as a reference.
* Evaluate and explain the final outcome adhering to the RESOLUTION_RULES. Only if the event and the time frame are met as per the market question, the outcome will be 'Yes'.
* Source your domain expertise to provide caveats
* Give your response in the format specified under "OUTPUT_FORMAT"

QUESTION:
```
{market_question}
```

SEARCH_OUTPUT:
```
{additional_information}
```

RESOLUTION_RULES:
```
{market_rules}
```

OUTPUT_FORMAT:
* Introduction and Context (QUESTION and RESOLUTION_RULES)
* Findings and Analysis (SEARCH_OUTPUT)
    - Will the event happen? (Do NOT mention any date here)
    - When will the event actually happen?
    - Difference between the actual event date and the date specified in the QUESTION
* Evaluation
* Caveats
Output only the raw report without any additional information or formatting.
"""


REPORT_PROMPT_BEST = """
Imagine some unexperienced person asks you the question specified under QUESTION.
Your task is to prepare a detailed and informative evaluation report that discusses the potential outcome of the question based \
on the search output and your domain expertise. 

Structure your report in the following sections and sub-sections:

* Introduction and Context
* Findings (in search output)
    - Will the event in the question happen? (Do NOT mention any date here)
    - On what date will the event actually happen? May the event have already happened? (Analyze the search output)
    - Calculate the difference between the actual event date and the date specified in the question. Use common sense reasoning and the publication dates of the articles as a reference.
* Outcome Evaluation (Event and Date) with common knowledge and domain expertise
* Given the market rules, conclude the likelihood of the event happening or not happening
* Caveats

MARKET_RULES:
```
{market_rules}
```

QUESTION:
```
{market_question}
```

SEARCH_OUTPUT:
```
{additional_information}
```

Output only the raw report without any additional information or formatting.
"""
# Note it may be possible that the event in the question has already happened or will not happen.
# - Will the event in the question happen? (Do NOT mention any date here)
# - On what date will it happen? (Be unbiased and only use the information provided in the search output)


# QUESTION_STATUS:
# ```
# {question_status}
# ```

# MARKET_RULES:
# ```
# {market_rules}
# ```

# REPORT_PROMPT_PART_2 = """
# Imagine today is {current_date} and someone on the street asks you the market question specified under MARKET_QUESTION. Prepare a concise but informative evaluation report that discusses the potential outcome of the market question based \
# on the additional information and the market rules. Adhere to the following instructions:
# * Carefully read the market question and the market rules.
# * Analyze the additional information provided.
# * It is very important to check the additional information for the event date as it is crucial for the outcome evaluation.

# Structure your report in the following sections and sub-sections:

# * Outcome Evaluation (Event and Date) with common knowledge
# * Conclusion
# * Caveats

# MARKET_QUESTION:
# ```
# {market_question}
# ```

# MARKET_RULES:
# ```
# {market_rules}
# ```

# ADDITONAL_INFORMATION:
# ```
# {additional_information}
# ```

# Output only the raw report without any additional information or formatting.
# """


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


def remove_date_from_query(query: str) -> str:
    # Define a regex pattern to match dates
    date_pattern = r"\b( on or before | by | on )?\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b"
    new_query = re.sub(date_pattern, "", query)
    return new_query


def split_before_evaluation(text):
    # Find the last occurrence of "Evaluation"
    eval_index = text.rfind("Evaluation")
    
    if eval_index == -1:
        # "Evaluation" not found in the text
        return text, ""
    
    # Find the last newline character before "Evaluation"
    newline_index = text.rfind("\n", 0, eval_index)
    
    if newline_index == -1:
        # Newline not found before "Evaluation", no splitting
        return text, ""
    
    # Split the string at the found newline index
    part1 = text[:newline_index]  # Part before the newline
    part2 = text[newline_index + 1:]  # Part after the newline, including "Evaluation"
    
    return part1, part2


def get_sme_role(
    engine, temperature, max_tokens, prompt, counter_callback=None
) -> Tuple[str, str]:
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
        if market_question is None:
            return None, None, "Market question not found in prompt", None
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


        # # Split the first part of the output at the last newline character before "Evaluation"
        # part_1, part_2 = split_before_evaluation(output)

        # prompt_part_2 = REPORT_PROMPT_PART_2.format(
        #     market_question=market_question,
        #     market_rules=market_rules,
        #     additional_information=part_1,
        #     current_date=current_date
        # )
        # print(f"REPORT PROMPT PART 2:\n{prompt_part_2}\n")

        # messages_report_part_2 = [
        #     {"role": "system", "content": "You are a professional journalist"},
        #     {"role": "user", "content": output},
        # ]
        # # Generate a report based on the messages
        # response_part_2 = client.chat.completions.create(
        #     model=engine,
        #     messages=messages_report_part_2,
        #     temperature=temperature,
        # )
        # if counter_callback is not None:
        #     counter_callback(
        #         input_tokens=response_part_2.usage.prompt_tokens,
        #         output_tokens=response_part_2.usage.completion_tokens,
        #         model=engine,
        #         token_counter=count_tokens,
        #     )
        # output_part_2 = response_part_2.choices[0].message.content
        # print(f"OUTPUT PART 2:\n{output_part_2}\n")



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

        return response, thread_history_string, None, counter_callback
              

                




