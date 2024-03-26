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
import re
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from heapq import nlargest
from itertools import islice
from string import punctuation
from typing import Any, Dict, Generator, List, Optional, Tuple, Callable
from packages.jhehemann.customs.tool_get_market_rules.infer_rules import get_market_rules
from packages.jhehemann.customs.tool_research.research_additional_information import research
from packages.jhehemann.customs.tool_format_prediction_values.format_prediction_values import format_prediction_values

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

ASSISTANT_INSTRUCTIONS_REPORT = """You are a professional researcher and expert for prediction markets."""

ASSISTANT_INSTRUCTIONS_REPORT_LONG = """
You are an autonomous AI agent that gathers highly reliable and valid information from different sources. You are provided with a prediction market question. \
Your task is to gather current and highly reliable information and write a comprehensive report that provides relevant information to make an accurate and robust \
probability estimation for the outcome of the market question. You have access to two tools that can help you gather information and structure your report. \
You cannot call each tool more than once.
"""

ASSISTANT_INSTRUCTIONS_PREDICTION = """
You are a highly advanced data scientist and reasoning expert. Your task is to provide accurate and robust probability estimations for the outcome of a prediction market question. \
You source all your knowledge from training and objectively analyze all information that is provided.

Your response must be structured in the following format:
* Your response must be only a single JSON object to be parsed by Python's "json.loads()"
* The JSON object must contain four fields: "p_yes", "p_no", "confidence", "info_utility"
* Each item in the JSON must have a value between 0 and 1
    - "p_yes": Probability that the market will resolve as `Yes`
    - "p_no": Probability that the market will resolve as `No`
    - "confidence": Your confidence (value between 0 and 1) in your estimated probabilities
    - "info_utility": Utility of the information provided by the agents that helped you make the estimation. 0 indicates lowest utility; 1 maximum utility.
Do not include any other contents except for the JSON object in your outputs.
"""

REPORT_PROMPT_TEMPLATE = """
Take only the market question as a search query and conduct a search using the research tool. Prepare a detailed evaluation report that discusses the outcome of the market question '{market_question}' based on the market rules and the search output. Structure your report in the following sections:

- Search output
- Market rules
- Evaluation
- Caveats
"""

REPORT_PROMPT_TEMPLATE_ONLY = """
Take only the selected relevant search queries and conduct separate researches using the research tool. Prepare a detailed evaluation report that discusses the outcome of the market question '{market_question}' based on the market rules and the search output.
"""


REPORT_PROMPT_TEMPLATE_THREAD_FLOW_DETAILED = """
Take only the selected relevant search queries and conduct separate researches using the research tool. Prepare a detailed report on the market question: '{market_question}', if unable, explain why.
The report will be used as a basis for placing a bet on a prediction market, whose rules you can find in our conversation history. It is important to provide a report that specifically addresses all aspects of the market question.
If the report does not consider all aspects of the market question, it probably will result in financial loss. your report with markdown into:

- Introduction & Background
- Findings and Analysis
- Conclusion and evaluation of the market question's outcome
- Caveats

Don't limit yourself to just stating each finding; provide a thorough, full and comprehensive analysis of each finding.
Use markdown syntax. Include as much relevant information as possible and try not to summarize.
"""


REPORT_PROMPT_TEMPLATE_OK= """
Prepare a detailed report on the market question: '{market_question}', if unable, explain why. 
Use all available tools for research; Think of additional information that could be relevant to the market question and search for it.

Instructions:
- Analyze the market question and information that you need to provide a reliable probability estimation.
- Use the tools provided to gather relevant information. You can use the research tool up to three times.
- Use the information gathered to structure a comprehensive report that provides relevant information to answer the market question.

Organize your report with markdown into:

    Introduction: Summarize the market question and its significance.
    Background: Provide essential context, including historical data and relevant trends.
    Findings and Analysis: Analyze each finding in relation to the market question's date, current date, and search output dates. Discuss the event's probability occurring by the specified date in the market question.
    Rules and Conclusion: Highlight the market rules and key findings and their implications for the outcome of the market question.
    Caveats: Note any limitations or assumptions and their potential effects.

Your report should not predict but analyze the event's timing and rationale. Cite all sources for credibility. If the tools provide insufficient or irrelevant information, state why.
"""

REPORT_PROMPT_TEMPLATE_WITH_EXAMPLES= """
Your goal is to provide a relevant information report that offers crucial insights in order to make an informed prediction for the MARKET QUESTION: '{market_question}'.

Prepare a full comprehensive report that provides relevant information to answer the aforementioned question.
If that is not possible, state why.
You will structure your report in the following sections:

- Introduction and Background
- Findings and Analysis
- Conclusion
- Caveats

Don't limit yourself to just stating each finding; provide a thorough, full and comprehensive analysis of each finding.
Use markdown syntax. Include as much relevant information as possible and try not to summarize.

Here you can find much simplified examples of a report conclusion:
<Starting part of Report 1>
Conclusion: The research output strongly suggest that the new Tesla Cyber Truck will be released by the end of November 2023. This is supported by the announcement from Tesla CEO Elon Musk stating "the first deliveries will be made by the end of November".
However, the market question asks if the release will happen by 30 October 2023. This is one month earlier than the expected release date. Thus, it is highly unlikely that the market question will resolve as 'Yes'.
</Ending part of Report 1>

<Starting part of Report 2>
Conclustion: The research output suggests that Rafael Nadal has made significant progress in his recovery from injury. This is supported by a TennisPoint article stating that he has started training again. However, the market question asks if he will return to competitive tennis by 21 March 2024. As the 
current date is 19 March 2024, and the search results do not provide a specific return date or timeline, it is unlikely that he will return to competitive tennis by the specified date.
</Ending part of Report 2>
"""

REPORT_PROMPT_TEMPLATE_OLD= """
Your goal is to provide a relevant information report that offers crucial insights in order to make an informed prediction for the MARKET QUESTION: '{market_question}'. \
It is essential that the chronological aspect is emphasized, with the date of the market question serving as a pivotal factor in shaping the analysis and conclusions. \
For reference, use the current date {timestamp} to understand the timeline of the market question.

Prepare a full comprehensive report that provides relevant information to answer the aforementioned question. Consider publication dates and timelines stated in gathered inforamtion for the outcome of the market question.
If that is not possible, state why.
You will structure your report in the following sections:

- Introduction
- Background
- Findings and Analysis
- Conclusion
- Caveats

Don't limit yourself to just stating each finding; provide a thorough, full and comprehensive analysis of each finding.
Use markdown syntax. Include as much relevant information as possible and try not to summarize.
"""

PREDICTION_PROMPT_TEMPLATE = """
Given the market question, its rules and the research report your task is to make a probability estimation for the outcome of the market question. \
Examine the market question and the research report and provide the estimated probability that the market resolves as 'Yes' and 'No' along with your confidence \
and the utility of the information provided. Output your answer in a single JSON object that contains four fields: "p_yes", "p_no", "confidence", "info_utility". \
Each item in the JSON must have a value between 0 and 1. Do not include any other contents in your response. Do not use formatting characters in your response.

MARKET_QUESTION: {market_question}
"""

RESEARCH_PLAN_PROMPT_TEMPLATE = """
Your goal is to prepare a research plan for {query}.

The plan must consist of {search_limit} search engine queries separated by commas.
Return ONLY the queries, separated by commas and without quotes.
The queries must be phrased as questions.
"""

RESEARCH_PLAN_PROMPT_TEMPLATE_COMPLEX = """
Your only task is to generate standard search engine queries. Analyze the prediction market question to identify the core topic and contextual factors. Craft search queries that not only focus on the main subject but also explore its \
geographical, temporal, or domain-specific contexts. Include perspectives from various stakeholders and investigate related subjects for broader insights. \
Ensure to cover the latest developments and alternative viewpoints in the search queries that might provide a search output that gives a comprehensive understanding of the market question.

PREDICTION_MARKET_QUESTION: {query}

OUTPUT FORMAT:
* Return ONLY the queries, separated by commas and without quotes or search engine syntax.
* Limit the number of queries to {search_limit}.
"""


QUERY_RERANKING_PROMPT_TEMPLATE = """
Evaluate the queries and decide which ones will provide the best and wholesome data to answer the question. Do not modify the queries.

OUTPUT FORMAT:
Return only the best three queries, in order of relevance as a comma separated list of strings with no quotes.
"""


GET_MARKET_RULES_PROMPT = """
Get the current state and rules for the prediction market question: '{query}'.
"""


RESEARCH_ASSISTANT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_market_rules",
            "description": "Get current state and rules for a prediction market question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "market_question": {"type": "string", "description": "The MARKET QUESTION to infer the rules for"},
                },
                "required": ["market_question"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "research",
            "description": "A search engine tool. You can only call this tool once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "market_question": {"type": "string", "description": "The exactly phrased prediction market question."},
                },
                "required": ["market_question"],
            }
        }
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "create_search_plan",
    #         "description": "Get a research plan for a prediction market question containing the most relevant search queries.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "market_question": {"type": "string", "description": "The exact prediction market question."},
    #             },
    #             "required": ["market_question"],
    #         }
    #     }
    # },
]


PREDICTION_ASSISTANT_TOOLS = [
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


OPENAI_TOOLS_FUNCTIONS = {
    "get_market_rules": get_market_rules,
    "research": research,
    "format_prediction_values": format_prediction_values,
}


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


def load_model(vocab: str) -> Language:
    """Utilize spaCy to load the model and download it if it is not already available."""
    try:
        return spacy.load(vocab)
    except OSError:
        print("Downloading language model...")
        download(vocab)
        return spacy.load(vocab)


def execute_function(tool_call, client, google_api_key=None, google_engine_id=None, engine=None, market_rules=None):
    function_name = tool_call.function.name
    function_to_call = OPENAI_TOOLS_FUNCTIONS[function_name]
    function_args = json.loads(tool_call.function.arguments)

    if function_name == "get_market_rules":
        return function_to_call(**function_args, client=client)
    elif function_name == "research":
        return function_to_call(
            **function_args,
            client=client,
            google_api_key=google_api_key,
            google_engine_id=google_engine_id,
            engine=engine,
            market_rules=market_rules
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
    timeout=100,
):
    timeout = timeout
    start_time = time.time()
    thread_id = thread_id
    run_id = run_id

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
    market_rules=None,
):
    # print(f"Required action:\n {run.required_action}\n")
    tool_calls = run.required_action.submit_tool_outputs.tool_calls
    print(f"Number of tools to call: {len(tool_calls)}")
    for tool_call in tool_calls:
        tool_call_type = tool_call.type
        function_name = tool_call.function.name
        try:
            function_arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            print(f"Error parsing function arguments: {e}")
            function_arguments = {}  # Use an empty dict or handle the error as needed
        print(f"Tool Call Type: {tool_call_type}, Function Name: {function_name}, Arguments: {function_arguments}")
    

    tool_outputs = []
    if len(tool_calls) > 2:
        dropped_tool_calls = tool_calls[3:]
        tool_calls = tool_calls[:3]
        for tool_call in dropped_tool_calls:
            print(f"ID of tool_call that was dropped: {tool_call.id}")
            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "output": "Too many tools to call. Limiting to 3."
            })
    
    futures_dict = {}
    # Outer ThreadPoolExecutor to manage parallel execution of all functions
    with ThreadPoolExecutor(max_workers=len(run.required_action.submit_tool_outputs.tool_calls)) as executor:
        for tool_call in tool_calls:
            # Submit each function execution as a separate task to the executor
            future = executor.submit(execute_function, tool_call, client, google_api_key, google_engine_id, engine, market_rules)
            futures_dict[future] = tool_call.id

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
    market_rules=None,
):
    # Termination condition for recursion
    if current_iteration >= max_iterations:
        print(f"Reached maximum number of iterations: {max_iterations}")
        return None # Stop the recursion
    
    run = wait_for_run(client, thread_id, run_id)

    if run and run.status in RUN_ACTION_REQUIRED_STATES:
        tool_outputs = call_tools(run, google_api_key, google_engine_id, engine, market_rules)

        if return_tool_outputs_only:
            client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
            return tool_outputs
        
        for tool_output in tool_outputs:
            print(f"TOOL OUTPUT:\n{tool_output['output']}\n\n")
        

        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run.id,
            tool_outputs=tool_outputs,
        )

        run = wait_for_run_termination(
            client,
            thread_id,
            run.id,
            google_api_key=google_api_key,
            google_engine_id=google_engine_id,
            engine=engine,
            market_rules=market_rules,
        )

    return run


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
        run_ids = []
        assistant_ids = []
        thread_ids = []

        try:
            # Create an openai assistant
            assistant_report = client.beta.assistants.create(
                name="Report Agent",
                instructions=ASSISTANT_INSTRUCTIONS_REPORT,
                tools=RESEARCH_ASSISTANT_TOOLS,
                model=engine,
            )
            assistant_ids.append(assistant_report.id)

            # Create an openai assistant
            assistant_prediction = client.beta.assistants.create(
                name="Prediction Agent",
                instructions=ASSISTANT_INSTRUCTIONS_PREDICTION,
                model=engine,
            )
            assistant_ids.append(assistant_prediction.id)

            # Extract the market question from the prompt delimited by escaped quotation marks
            market_question = extract_question(prompt)
            if market_question is None:
                return None, None, "Market question not found in prompt", None
            

            ### Create a thread with the initial prompt (get the market rules)
            get_market_rules_prompt = GET_MARKET_RULES_PROMPT.format(query=market_question)
            thread = client.beta.threads.create(
                messages=[
                    {"role": "user", "content": get_market_rules_prompt},
                ]
            )
            thread_ids.append(thread.id)

            # Apply the research assistant to the thread
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_report.id,
            )
            # Wait until run is in one of the terminal states
            run = wait_for_run_termination(
                client,
                thread.id,
                run.id,
                google_api_key=google_api_key,
                google_engine_id=google_engine_id,
                engine=engine,
            )
            thread_messages = client.beta.threads.messages.list(thread.id)
            response = thread_messages.data[0].content[0].text.value
            print(f"Assistant message added to thread {thread.id}:\n{response}\n")
            market_rules = response

            # ### Generate search queries based on the market rules
            # search_plan_prompt = RESEARCH_PLAN_PROMPT_TEMPLATE.format(query=market_question, search_limit="twelve")
            # client.beta.threads.messages.create(
            #         thread.id,
            #         role="user",
            #         content=search_plan_prompt,
            # )
            # # Apply the research assistant to the thread
            # run = client.beta.threads.runs.create(
            #     thread_id=thread.id,
            #     assistant_id=assistant_report.id,
            # )
            # # Wait until run is in one of the terminal states
            # run = wait_for_run_termination(
            #     client,
            #     thread.id,
            #     run.id,
            #     google_api_key=google_api_key,
            #     google_engine_id=google_engine_id,
            #     engine=engine,
            # )
            # thread_messages = client.beta.threads.messages.list(thread.id)
            # response = thread_messages.data[0].content[0].text.value
            # print(f"Assistant message added to thread {thread.id}:\n{response}\n")

            # ### Sort the search queries based on relevance
            # # Add prompt for query reranking to the thread
            # client.beta.threads.messages.create(
            #         thread.id,
            #         role="user",
            #         content=QUERY_RERANKING_PROMPT_TEMPLATE,
            # )
            # # Apply the research assistant to the thread
            # run = client.beta.threads.runs.create(
            #     thread_id=thread.id,
            #     assistant_id=assistant_report.id,
            # )
            # # Wait until run is in one of the terminal states
            # run = wait_for_run_termination(
            #     client,
            #     thread.id,
            #     run.id,
            #     google_api_key=google_api_key,
            #     google_engine_id=google_engine_id,
            #     engine=engine,
            # )
            # thread_messages = client.beta.threads.messages.list(thread.id)
            # response = thread_messages.data[0].content[0].text.value
            # print(f"Assistant message added to thread {thread.id}:\n{response}\n")


            # for message in thread_messages.data:
            #     print(f"Message content:\n{message}\n")


            # assistant_report_tools = client.beta.assistants.update(
            #     assistant_report.id,
            #     tools=RESEARCH_ASSISTANT_TOOLS,
            # )

            ### Generate research report with the search queries as tool inputs
            report_prompt = REPORT_PROMPT_TEMPLATE.format(market_question=market_question)
            client.beta.threads.messages.create(
                    thread.id,
                    role="user",
                    content=report_prompt,
            )
            # Apply the research assistant to the thread
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_report.id,
            )
            # Wait until run is in one of the terminal states
            run = wait_for_run_termination(
                client,
                thread.id,
                run.id,
                google_api_key=google_api_key,
                google_engine_id=google_engine_id,
                engine=engine,
                market_rules=market_rules,
            )
            thread_messages = client.beta.threads.messages.list(thread.id)
            response = thread_messages.data[0].content[0].text.value
            print(f"Assistant message added to thread {thread.id}:\n{response}\n")

            






            
            
            
            
            # # Get the current date and time
            # timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            # # report_prompt = REPORT_PROMPT_TEMPLATE.format(market_question=market_question, timestamp=timestamp)
            
            
            # client.beta.threads.messages.create(
            #         thread.id,
            #         role="user",
            #         content=prediction_prompt,
            # )

            # # Apply the prediction assistant to the thread
            # run = client.beta.threads.runs.create(
            #     thread_id=thread.id,
            #     assistant_id=assistant_report.id,
            # )
            # run_ids.append(run.id)

            # # Wait until run is in one of the terminal states
            # run = wait_for_run_termination(
            #     client,
            #     thread.id,
            #     run.id,
            #     google_api_key=google_api_key,
            #     google_engine_id=google_engine_id,
            #     engine=engine,
            # )
            # thread_messages = client.beta.threads.messages.list(thread.id)
            # response = thread_messages.data[0].content[0].text.value
            # print(f"Assistant message added to thread {thread.id}:\n{response}\n")
            
            if not is_valid_json_with_fields_and_values(response):
                # client.beta.threads.messages.create(
                #     thread.id,
                #     role="user",
                #     content="Output your answer in JSON format.",
                # )
                prediction_prompt = PREDICTION_PROMPT_TEMPLATE.format(market_question=market_question)

                # Append the prediction prompt as a message to the thread
                client.beta.threads.messages.create(
                    thread.id,
                    role="user",
                    content=prediction_prompt,
                )

                # Apply the prediction assistant to the thread with the new message
                run = client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant_prediction.id,
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

                # Get the final message from the thread
                thread_messages = client.beta.threads.messages.list(thread.id)
                response = thread_messages.data[0].content[0].text.value
                print(f"Assistant message added to thread {thread.id}:\n{response}\n")
            
            # Trim JSON formatting characters if present
            response = trim_json_formatting(response)

            print(f"FINAL OUTPUT:\n{response}")
            print(f"\nIS VALID RESPONSE: {is_valid_json_with_fields_and_values(response)}\n")

            return response, prompt, None, counter_callback
        
        finally:
            # Clean the client: Cancel runs and delete threads and assistants
            if thread_ids:
                for id in thread_ids:
                    runs = client.beta.threads.runs.list(id)
                    for run in runs.data:
                        if not is_terminated(run):
                            print(f"Cancelling {run.id}")
                            client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
                            run = wait_for_run_termination(client, id, run.id)
                        else:
                            print(f"Run already terminated: {run.id}")
                    client.beta.threads.delete(id)

            if assistant_ids:
                for id in assistant_ids:
                    client.beta.assistants.delete(id)
                    print(f"Assistant deleted: {id}")
              

                




