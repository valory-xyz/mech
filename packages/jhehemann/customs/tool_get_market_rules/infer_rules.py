"""This module implements a research agent for extracting relevant information from URLs."""

# from typing import Any, Dict, Generator, List, Optional, Tuple
# from datetime import datetime, timezone
import json
from typing import Dict
# import re
#import os
# from concurrent.futures import Future, ThreadPoolExecutor
# from itertools import groupby
# from operator import itemgetter

# from bs4 import BeautifulSoup, NavigableString
# from googleapiclient.discovery import build
# from langchain.pydantic_v1 import BaseModel, Field
# from langchain.tools import BaseTool
# # from langchain_core.callbacks import (
# #     AsyncCallbackManagerForToolRun,
# #     CallbackManagerForToolRun,
# # )
# # from langchain_openai import OpenAI, ChatOpenAI
# from langchain.schema import HumanMessage, SystemMessage

# # from urllib.parse import urlparse
# from typing import Optional, Type
# import requests
# from requests import Session
# import spacy
# import spacy.util
# import spacy_universal_sentence_encoder
# import tiktoken

# from dateutil import parser
# from tqdm import tqdm

from openai import OpenAI

SYSTEM_PROMPT = """You are a world class algorithm for generating structured output from a given input."""

#It is important to differentiate between the implied facts of the market question and the question itself. Taking the previous example, while the question may be misleading in implying that there is a planned test flight on that specific date, the rules can and must be formulated as definitive and


INFER_RULES_PROMPT = """

You are a Large Language Model in a multi-agent system. Your task is to infer the current status and rules for a prediction market question. \
Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. The rules that \
you define should be based on information that can be found on the internet. You are provided with some examples below. You must adhere to the instructions.

INSTRUCTIONS:
* Carefully read the market question
* Pay detailled attention on the phrasing of the market question
* Analyze what the phrasing implies about the current status, who the involved parties are and what the conditions are
* Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'


EXAMPLES:
```
Question: "Will the new climate bill be passed by both the Senate and the House before the end of the fiscal year on 30 September 2024?"
Answer:
    Status: The question implies that a new climate bill is under consideration by the United States Congress, which consists of two chambers: the Senate and the House of Representatives. If there exists such a bill, the question suggests that it has not yet been passed by both chambers.
    Rules:
    'Yes': The question resolves as 'Yes' if there exists a climate bill and it is passed by both the Senate, which is the upper chamber, and the House, which is the lower chamber of the United States Congress on or before 30 September 2024.
    'No': The question resolves as 'No' if there exists no bill or it isn't passed by both chambers by 30 September 2024, or only one chamber passes it by this date. The market also resolves as 'No' if both chambers pass a different bill and not a new climate bill by this date.

Question: "Will Tesla successfully launch its new electric vehicle, Model Z, on 30 June 2024?"
Answer:
    Status: The question implies that Tesla, an influential electric vehicle manufacturer, is planning to release a new electric vehicle, which is called Model Z, on 30 June 2024. Furthermore, the question suggests that up to today, the release has not yet occurred.
    Rules:
    'Yes': The question resolves as 'Yes' if Tesla, an influential electric vehicle manufacturer, officially releases a Model Z on 30 June 2024. This must be evidenced by a public event, press release, or substantial media coverage that confirms the release will occur on this specific date.
    'No': The question resolves as 'No' if Tesla, an influential electric vehicle manufacturer, does not release a Model Z on 30 June 2024. This includes any release occurring before or after this date, or the absence of any official announcement, public event, press release, or significant media coverage confirming a release on 30 June 2024. The market also resolves as 'No' if Company Y releases a different vehicle Model W on this date.

Market Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by the end of 2024?"
Answer:
    Status: The question implies that FIFA, the international governing body of football, is considering funding the construction of new football stadiums specifically for local clubs in England. The question suggests that FIFA has not yet allocated funds for this purpose.
    Rules:
    'Yes': The question resolves as 'Yes' if FIFA, the international governing body of football, has directly allocated funds for the construction of new football stadiums specifically for local clubs in England by 31 December 2024. This allocation must be evidenced by an official announcement, press release, or documented agreement confirming the funding by this specific date.
    'No': The question resolves as 'No' if FIFA, the international governing body of football, has not allocated funds for the construction of new football stadiums for local clubs in England by 31 December 2024. This includes any funding allocated after this date, the absence of any official announcement, press release, or documented agreement confirming the allocation by this date. The market also resolves as 'No' if FIFA allocates funds for stadium construction in a different country or for a different purpose by the end of 2024.
```

Question: "{market_question}"
Answer:
"""



INFER_RULES_PROMPT_NICE = """
Assume the role of a Logical Analyst tasked with distilling the essence of the 'MARKET_QUESTION' into two clear, overarching rules. Your objective is to formulate a single, encompassing rule that, if met, decisively indicates a 'Yes' outcome, and another that, if met, indicates a 'No' outcome. These rules should capture the entirety of the market question's conditions and implications.

Instructions:
* Analyze the 'MARKET_QUESTION' thoroughly, identifying its core components and implications.
* Formulate Two Key Rules:
    - Craft one rule that fully encapsulates the scenario leading to a 'Yes' resolution. This rule should consider all necessary conditions described in the market question.
    - Develop one rule that fully represents the conditions under which the question would resolve as 'No', capturing all critical aspects that would lead to this outcome.

MARKET_QUESTION:
{market_question}

OUTPUT_FORMAT:
* Status: Briefly summarize the market question's context and implications.
* Comprehensive Rules:
    - 'Yes' Rule: [Your single, all-encompassing rule for a 'Yes' outcome.]
    - 'No' Rule: [Your single, all-encompassing rule for a 'No' outcome.]

Ensure each rule is self-contained and thoroughly addresses the market question, enabling clear and decisive resolution based on its fulfillment.
"""


INFER_RULES_PROMPT_BREAK_DOWN = """
Your role is to act as a Logical Analyst, tasked with formulating rules for a given prediction market question. Each rule should be comprehensive, directly responding to the market question in its entirety.

Key Instructions:
First, identify the key components of the 'MARKET_QUESTION'. Next, for each component, create a rule that would indicate a 'Yes' outcome and a rule for a 'No' outcome. Finally, combine these rules into comprehensive guidelines for resolving the entire market question.

MARKET_QUESTION:
{market_question}

OUTPUT_FORMAT:
* Status: Summarize the essence of the market question.
* Rules: Comprehensive guidelines.
"""


INFER_RULES_PROMPT_LONG = """
Assume the role of a Logical Analyst within a sophisticated multi-agent system, tasked with formulating definitive \
rules for a prediction market question. Your objective is to apply logical reasoning to establish clear, standalone \
conditions that individually suffice to resolve the market question as 'Yes' or 'No'

Focus on the following guidelines:

* Logical Analysis: Conduct a thorough examination of the 'MARKET_QUESTION'. Apply logical reasoning to deconstruct \
and understand every aspect and implied condition of the question.
* Critical Interpretation:  Scrutinize the phrasing of the market question with an analytical mindset. Identify and \
interpret the explicit and implicit logical premises suggested by the question.
* Independent Rule Formulation: Construct rules using logical principles, where each bullet point under 'Yes' and 'No' \
represents a complete, independently sufficient condition for resolving the market question accordingly. Think of \
each bullet point as an individual logical statement that, if true, conclusively determines the outcome.

MARKET_QUESTION: {market_question}

OUTPUT_FORMAT:
* Begin with a 'Status' section that provides a concise overview of the situation and the market question's implications.
* Proceed to 'Rules', segmented into:
    - 'Yes': List bullet points, each representing a complete and independent criterion. The fulfillment of any one of \
these criteria is sufficient for the market question to resolve as 'Yes'.
    - 'No':  List bullet points, each outlining a standalone condition. Meeting any one of these conditions alone is \
enough for the market question to resolve as 'No'.

Ensure that each rule is self-contained, reflecting all necessary information to support decisive action in the prediction market.
"""


INFER_RULES_PROMPT_OLD = """
You are a Large Language Model in a multi-agent system. Your task is to infer the rules for a prediction market question. \
Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. The rules that \
you define should be based on information that can be found on the internet. Find the market question under 'MARKET_QUESTION' and \
adhere to the following 'INSTRUCTIONS'.

INSTRUCTIONS:
* Carefully read the market question
* Pay detailled attention on the phrasing of the market question
* Analyze what the phrasing implies about the current status
* Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'
* You must provide your response in the format specified under "OUTPUT_FORMAT"
* Do not include any other contents in your response.


MARKET_QUESTION:
```
{market_question}
```

OUTPUT_FORMAT:
* Output two paragraphs, the status paragraph describing the current status and facts that the market question's phrasing implies, and the rules paragraph describing when the market question will be resolved as 'Yes' and 'No'.
* The second paragraph must contain two sub-paragraphs that contain bulletpoints with the rules for when the market question will be resolved as 'Yes' and 'No'.
* Do not include any formatting characters in your response!
* Do not include any other contents or explanations in your response!
"""


def get_market_rules(
    market_question: str,
    client: OpenAI,
):
    """Infer market rules for a prediction market question."""
    
    temperature=0

    # Remove double quotes from the input query to avoid issues with react agent execution
    market_question = market_question.replace('"', "'") 

    # Create prompt
    infer_rules_prompt = INFER_RULES_PROMPT.format(market_question=market_question)
    
    # Create messages for the model engine
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": infer_rules_prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
    )
    response_message = response.choices[0].message.content 

    return response_message