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


INFER_RULES_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to infer the rules for a prediction market question. \
You are assisting an agent that generates predictions for the outcome of the market question. For this, the agent needs reliable \
and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. To predict the outcome \
the agent can use a search engine to find relevant information. The rules that you create should be based on information that can be found \
on the internet. Find the market question under 'MARKET_QUESTION' and adhere to the following 'INSTRUCTIONS'.

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
    print(response_message)


    

    return response