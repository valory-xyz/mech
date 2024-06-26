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

from datetime import datetime, timedelta
import json
import re
import requests
from requests import Session
from collections import defaultdict
from typing import Any, Dict, List, Generator, Optional, Tuple, Callable
from openai import OpenAI
import tiktoken
from tiktoken import encoding_for_model
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import faiss
from googleapiclient.discovery import build
import numpy as np
from pydantic import BaseModel, Field
from readability import Document
from markdownify import markdownify as md
from dateutil import parser

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


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0.0,
}

LLM_SETTINGS = {
    "gpt-3.5-turbo": DEFAULT_OPENAI_SETTINGS,
    "gpt-4-turbo-preview": DEFAULT_OPENAI_SETTINGS,
}

ALLOWED_TOOLS = [
    "prediction-with-rules-and-report-gpt-3.5-turbo",
    "prediction-with-rules-and-report-gpt-4-turbo-preview",
]
TOOL_TO_ENGINE = {
    "prediction-with-rules-and-report-gpt-3.5-turbo": "gpt-3.5-turbo",
    "prediction-with-rules-and-report-gpt-4-turbo-preview": "gpt-4-turbo-preview",
}

ALLOWED_MODELS = list(LLM_SETTINGS.keys())

MAX_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "gpt-4-turbo-preview": 8192,
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

NUM_URLS_PER_QUERY = 4
TEXT_CHUNK_LENGTH = 300
TEXT_CHUNK_OVERLAP = 50
MAX_CHUNKS_TOKENS_TO_SUMMARIZE = 500
MAX_TEXT_CHUNKS_TOTAL = 30
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_EMBEDDING_TOKEN_INPUT = 8192
EMBEDDING_SIZE = 1536
WEEKS_TO_SCRAPE_NEWS = 4

# Global constants for possible attribute names for release and update dates
RELEASE_DATE_NAMES = [
    'date',
    'pubdate',
    'publishdate',
    'OriginalPublicationDate',
    'dateCreated',
    'article:published_time',
    'sailthru.date',
    'article.published',
    'published-date',
    'og:published_time',
    'publication_date',
    'publishedDate',
    'dc.date',
    'DC.date',
    'article:published',
    'article_date_original',
    'cXenseParse:recs:publishtime',
    'DATE_PUBLISHED',
    'pub-date',
    'datePublished',
    'date_published',
    'ArticleDate',
    'time_published',
    'article:published_date',
    'parsely-pub-date',
    'publish-date',
    'pubdatetime',
    'published_time',
    'publishedtime',
    'article_date',
    'created_date',
    'published_at',
    'lastPublishedDate',
    'og:published_time',
    'og:release_date',
    'article:published_time',
    'og:publication_date',
    'og:pubdate',
    'article:publication_date',
    'product:availability_starts',
    'product:release_date',
    'event:start_date',
    'event:release_date',
    'og:time_published',
    'og:start_date',
    'og:created',
    'og:creation_date',
    'og:launch_date',
    'og:first_published',
    'og:original_publication_date',
    'article:published',
    'article:pub_date',
    'news:published_time',
    'news:publication_date',
    'blog:published_time',
    'blog:publication_date',
    'report:published_time',
    'report:publication_date',
    'webpage:published_time',
    'webpage:publication_date',
    'post:published_time',
    'post:publication_date',
    'item:published_time',
    'item:publication_date'
]

UPDATE_DATE_NAMES = [
    'lastmod',
    'lastmodified',
    'last-modified',
    'updated',
    'dateModified',
    'article:modified_time',
    'modified_date',
    'article:modified',
    'og:updated_time',
    'mod_date',
    'modifiedDate',
    'lastModifiedDate',
    'lastUpdate',
    'last_updated',
    'LastUpdated',
    'UpdateDate',
    'updated_date',
    'revision_date',
    'sentry:revision',
    'article:modified_date',
    'date_updated',
    'time_updated',
    'lastUpdatedDate',
    'last-update-date',
    'lastupdate',
    'dateLastModified',
    'article:update_time',
    'modified_time',
    'last_modified_date',
    'date_last_modified',
    'og:updated_time',
    'og:modified_time',
    'article:modified_time',
    'og:modification_date',
    'og:mod_time',
    'article:modification_date',
    'product:availability_ends',
    'product:modified_date',
    'event:end_date',
    'event:updated_date',
    'og:time_modified',
    'og:end_date',
    'og:last_modified',
    'og:modification_date',
    'og:revision_date',
    'og:last_updated',
    'og:most_recent_update',
    'article:updated',
    'article:mod_date',
    'news:updated_time',
    'news:modification_date',
    'blog:updated_time',
    'blog:modification_date',
    'report:updated_time',
    'report:modification_date',
    'webpage:updated_time',
    'webpage:modification_date',
    'post:updated_time',
    'post:modification_date',
    'item:updated_time',
    'item:modification_date'
]

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

RESEARCH_PLAN_PROMPT_TEMPLATE = """
Your goal is to prepare a research plan for {query}.

The plan must consist of {search_limit} search engine queries separated by commas.
Return ONLY the queries, separated by commas and without quotes.
The queries must be phrased as concise, but descriptive questions that will help you find relevant information about the event and its date.
"""

QUERY_RERANKING_PROMPT_TEMPLATE = """
Evaluate the queries and decide which ones will provide the best data to answer the question. Do not modify the queries.
Select only the best seven queries, in order of relevance.

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()"
* The JSON must contain one field: "queries"
    - "queries": A 7 item array of the generated search engine queries
* Include only the JSON object in your output
"""

SUMMARIZE_PROMPT = """
Your task is to summarize relevant information from 'SEARCH_OUTPUT'. The SEARCH_OUTPUT contains bulletpoints that you must \
combine and summarize into a single paragraph. You must adhere to the following 'INSTRUCTIONS'.

INSTRUCTIONS:
* Carefully read the 'QUESTION'
* Select a combination of only the most relevant information from 'SEARCH_OUTPUT' that may help answering the QUESTION
* An information can be considered relevant if it might support or refute the QUESTION
* Summarize the selected information in a way that is concise and informative and include the dates mentioned
* Do not try to answer or reference the QUESTION, but only summarize the relevant information from 'SEARCH_OUTPUT' in an unbiased way.
* You must not infer or add any new information.
* Do not modify the information in 'SEARCH_OUTPUT' with respect to the QUESTION.
* If there are dates or timeframes mentioned along with the relevant information, you must include them.

QUESTION: {input_query}

SEARCH_OUTPUT:
```
{chunks}
```

OUTPUT_FORMAT:
* Only output the summary containing the combination of the most relevant information from 'SEARCH_OUTPUT'.
* The summary must be a single very concise and informative paragraph.
* Solely respond with "Error", if there is no relevant information in 'SEARCH_OUTPUT'.
* Do not include any other contents in your response!
"""

FINAL_SUMMARY_PROMPT = """
You are provided with search outputs from multiple sources. These search outputs were received in response to the search \
query found below. Your task is to select a collection of unique and most relevant bulletpoints that may help to answer the search query and reveal \
if the event happens and on which date the event is expected to occur.

INSTRUCTIONS:
* Carefully read the search query.
* Select only the relevant bulletpoints from the search outputs that are useful and relevant and could help answering the search query.
* Each bullet includes its article number in parentheses at the end. This number is crucial for identifying the information and must be included at the end of each bullet.
* An information can be considered relevant if it might support or refute the search query.
* An information must also be considered relevant if it reveals a specific date or time frame when the event is expected to occur
* If there are redundant bulletpoints, you must drop all exept for one. Select the most relevant by two criteria:
    - Firstly: Select the one that mentiones specific dates over the ones that mention relative dates or week days
    - Secondly: Select the one that is listed more to the bottom of the search output
* If there are conflicting information, you must include both sides of the argument in the selected bulletpoints
* Give your response in the format specified under "OUTPUT_FORMAT"

SEARCH_OUTPUT:
```
{chunks}
```

SEARCH_QUERY: {input_query}
SUB_QUERIES:
- Will the event happen?
- On what date will the event happen? (DD/MM/YYYY)
- Has the event happened already?

OUTPUT_FORMAT:
* Only output the collection of five selected unique and relevant bulletpoints with the corresponding article numbers in parentheses.
* The bulletpoints should be useful and relevant and help answering the search query and sub-queries.
"""


SYSTEM_PROMPT_INFER_RULES = """You are a world class algorithm for generating structured output from a given input."""

# Prompt template for inferring rules for a question that asks for an event to happen "by" a specific date
INFER_RULES_PROMPT_BY = """
You are a Large Language Model in a multi-agent system. Your task is to infer the rules for a prediction market question. \
Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. \
You are provided with some examples below. You must adhere to the instructions.

INSTRUCTIONS:
* Carefully read the market question
* Examine the EXAMPLES provided to understand the structure of the rules you need to define
* Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'

EXAMPLES:
Question: "Will Beyoncé release a full album for her 'country era' on or before 9th January 2025?"
Answer:
The outcome will be 'Yes' if:
    - Beyoncé releases a full music album for her 'country era' on or before 9 January 2025.
    - Beyoncé has already released such an album recently.
The outcome will be 'No' if:
    - Beyoncé does not release a full album for her 'country era' by 9 January 2025.
    - Beyoncé releases a 'country era' album after 9 January 2025.

Question: "Will there be another case of H5N1 bird flu in Texas by 11 November 2024?"
Answer:
The outcome will be 'Yes' if:
    - A new case of H5N1 bird flu occurs in the state of Texas on or before 11 November 2024.
    - A new case of H5N1 bird flu has recently occurred in Texas.
The outcome will be 'No' if:
    - No new case of H5N1 bird flu is confirmed in Texas on or before 11 November 2024.
    - The next new case of H5N1 occurs after 11 November 2024.

Question: "Will an arrest be made in the alleged arson at Sen. Bernie Sanders' Vermont office on or before May 7, 2024?"
Answer:
The outcome will be 'Yes' if:
    - An arrest is made in connection with the alleged arson at Senator Bernie Sanders' Vermont office on or before 7 May 2026.
    - An arrest has already been made recently in relation to the arson case.
The outcome will be 'No' if:
    - No arrest is made in connection with the alleged arson at Senator Bernie Sanders' Vermont office by 7 May 2026.
    - The arrest related to the arson case occurs after 7 May 2026.
  
Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by 31 December 2024?"
Answer:
The outcome will be 'Yes' if:
    - FIFA funds or begins disbursing funds for the construction of new football stadiums for local clubs in England on or before 31 December 2024.
    - FIFA has already begun funding these projects recently.
The outcome will be 'No' if:
    - FIFA does not fund nor begin disbursing funds for the construction of new football stadiums for local clubs in England on or before 31 December 2024.
    - FIFA allocates funds after 31 December 2024.
   
Question: "Will a new climate bill be passed by both the Senate and the House on or by 22 October 2024?"
Answer:
The outcome will be 'Yes' if:
    - Both the Senate and the House of Representatives pass a new climate bill on or by before 22 October 2024.
    - Both chambers have already passed a new climate bill recently.
The outcome will be 'No' if:
    - A new climate bill does not exist, or one of the two chambers does not pass it on or before 22 October 2024.
    - The Senate and the House pass the bill after 22 October 2024.
    - Congress does not convene a session to vote on the bill on or before 22 October 2024.
   
Question: "Will Microsoft announce a significant AI-related takeover by 16 April 2024?"
Answer:
The outcome will be 'Yes' if:
    - Microsoft announces a significant acquisition or takeover related to artificial intelligence (AI) on or before 16 April 2024.
    - Microsoft has recently announced a significant AI-related takeover.
The outcome will be 'No' if:
    - Microsoft does not announce any significant AI-related takeover by 16 April 2024.
    - Microsoft announces an AI-related takeover after 16 April 2024.

Question: "Will Samsung replace its voice assistant Bixby by 7 April 2024?"
Answer:
The outcome will be 'Yes' if:
    - Samsung replaces its voice assistant Bixby with a new voice assistant on or before 7 April 2024.
    - Samsung has recently replaced Bixby with a new voice assistant.
The outcome will be 'No' if:
    - Samsung does not replace its voice assistant Bixby by 7 April 2024.
    - Samsumg replaces Bixby after 7 April 2024.

Question: "Will Google destroy all browsing data collected in Incognito mode by 2 October 2022?"
Answer:
The outcome will be 'Yes' if:
    - Google destroys all browsing data collected in Incognito mode on or before 2 October 2022.
    - Google has already completed the destruction of all such data recently.
The outcome will be 'No' if:
    - Google does not destroy all browsing data collected in Incognito mode on or before 2 October 2022.
    - Google completes the destruction of the data after 2 October 2022.
    
Question: "Will President Joe Biden make another visit to Baltimore over the Francis Scott Key Bridge collapse by 16 June 2024?
Answer:
The outcome will be 'Yes' if:
    - Joe Biden makes another official visit to Baltimore in relation to the Francis Scott Key Bridge collapse on or before 16 June 2024.
    - Joe Biden has recently visited Baltimore over the bridge collapse incident another time.
The outcome will be 'No' if:
    - Joe Biden does not make another visit to Baltimore regarding the Francis Scott Key Bridge collapse by 16 June 2024.
    - Joe Bidens next visit related to the bridge collapse occurs after 16 June 2024.

Question: "Will Gilberto Ramirez defend his WBA (Super) cruiserweight title successfully by 19 February 2021?"
Answer:
The outcome will be 'Yes' if:
    - Gilberto Ramirez successfully defends his WBA (Super) cruiserweight title in a bout on or before 19 February 2021.
    - Gilberto Ramirez has already successfully defended his title recently.
The outcome will be 'No' if:
    - Gilberto Ramirez does not successfully defend his title on or before 19 February 2021.
    - The defense match occurs after 19 February 2021.
    
Question: "Will Disney Plus implement its password-sharing crackdown by August 25, 2024?"
Answer:
The outcome will be 'Yes' if:
    - Disney Plus implements its password-sharing crackdown policy on or before 25 August 2024.
    - Disney Plus has already implemented the password-sharing crackdown recently.
The outcome will be 'No' if:
    - Disney Plus does not implement its password-sharing crackdown policy on or before 25 August 2024.
    - Disney Plus implements a password-sharing crackdown policy after 25 August 2024.
        
Question: "{market_question}"
Answer:
"""

# Prompt template for inferring rules for a question that asks for an event to happen "on" a specific date
INFER_RULES_PROMPT_ON = """
You are a Large Language Model in a multi-agent system. Your task is to infer the rules for a prediction market question. \
Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. \
You are provided with some examples below. You must adhere to the instructions.

INSTRUCTIONS:
* Carefully read the market question
* Examine the EXAMPLES provided to understand the structure of the rules you need to define
* Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'

EXAMPLES:
Question: "Will the Powerball jackpot reach $1 billion by the drawing on 9th January 2024?"
Answer:
The outcome will be 'Yes' if:
    - The Powerball jackpot amount reaches or exceeds $1 billion on or before 9 January 2024, and a drawing takes place on the specific day of 9 January 2024.
    - The Powerball jackpot has already exceeded $1 billion and maintains this amount until a drawing on 9 January 2024.
The outcome will be 'No' if:
    - No Powerball drawing takes place on 9 January 2024.
    - The Powerball drawing takes place before or after 9 January 2024.
    - The Powerball jackpot amount is less than $1 billion on 9 January 2024.
    - The Powerball jackpot reaches $1 billion or more after the drawing on 9 January 2024.

Question: "Will Tesla successfully launch its electric vehicle, Model Z, on 14 June 2024?"
Answer:
The outcome will be 'Yes' if:
    - Tesla releases an electric vehicle named Model Z on the specific day of 14 June 2024.
The outcome will be 'No' if:
    - Tesla does not release an electric vehicle named Model Z on 14 June 2024.
    - Tesla releases a Model Z before or after 14 June 2024.

Question: "Will Prince William and Prince Harry appear separately at the event honoring Princess Diana on 16 September 2025?"
Answer:
The outcome will be 'Yes' if:
    - Both Prince William and Prince Harry make individual appearances at an event honoring Princess Diana on the specific day of 16 September 2025 without appearing together.
The outcome will be 'No' if:
    - There is no event scheduled honoring Princess Diana on 16 September 2025.
    - The event honoring Princess Diana takes place before or after 16 September 2025.
    - Prince William and Prince Harry do not appear separately at an event honoring Princess Diana on 16 September 2025.
    - Either Prince William or Prince Harry does not attend such an event on 16 September 2025.

Question: "Will Xiaomi's SU7 electric vehicle still have a waiting list in China on April 18, 2024?"
Answer:
The outcome will be 'Yes' if:
    - Xiaomi's SU7 electric vehicle continues to have an active waiting list in China on the specific day of 18 April 2024.
The outcome will be 'No' if:
    - Xiaomi's SU7 electric vehicle does not have a waiting list in China on 18 April 2024.
    - Xiaomi cleares or discontinues the waiting list for the SU7 electric vehicle before 18 April 2024.

Question: "Will Gilberto Ramirez defend his WBA (Super) cruiserweight title successfully on 26 November 2020?"
Answer:
The outcome will be 'Yes' if:
    - Gilberto Ramirez successfully defends his WBA (Super) cruiserweight title in a bout on the specific day of 26 November 2020.
The outcome will be 'No' if:
    - There is no defense match scheduled for Gilberto Ramirez on 26 November 2020.
    - The defense match occurs before or after 26 November 2020.
    - Gilberto Ramirez participates in a title defense bout on 26 November 2020 but does not emerge victorious, thereby losing his WBA (Super) cruiserweight title.
    - Gilberto Ramirez is not the reigning WBA (Super) cruiserweight titleholder on 26 November 2020.

Question: "Will there be another case of H5N1 bird flu in Texas on 11 November 2024?"
Answer:
The outcome will be 'Yes' if:
    - A new case of H5N1 bird flu occurs in the state of Texas on the specific day of 11 November 2024.
The outcome will be 'No' if:
    - No new case of H5N1 bird flu occurs in Texas on 11 November 2024.
    - The next case of H5N1 bird flu in Texas occurs before or after 11 November 2024.

Question: "Will a new climate bill be passed by both the Senate and the House on 30 September 2024?"
Answer:
The outcome will be 'Yes' if:
    - Both the Senate and the House of Representatives officially pass a new climate bill on the specific day of 30 September 2024.
The outcome will be 'No' if:
    - Congress does not convene a session to vote on the bill on 30 September 2024.
    - A new climate bill does not exist, or one of the two chambers does not pass it on 30 September 2024.
    - Only one chamber passes the bill on 30 September 2024.
    - The Senate and the House pass the bill before or after 30 September 2024.

Question: "Will Google destroy all browsing data collected in Incognito mode on 2nd October 2022?"
Answer:
The outcome will be 'Yes' if:
    - Google destroys all browsing data collected in Incognito mode on the specific day of 2 October 2022.
The outcome will be 'No' if:
    - Google does not destroy all browsing data collected in Incognito mode on 2 October 2022.
    - Google destroys the Incognito browsing data before or after 2 October 2022.

Question: "Will Samsung replace its voice assistant Bixby on 7 April 2024?"
Answer:
The outcome will be 'Yes' if:
    - Samsung replaces its voice assistant Bixby with a new voice assistant on the specific day of 7 April 2024.
The outcome will be 'No' if:
    - Samsung does not replace its voice assistant Bixby on 7 April 2024.
    - Samsung replaces Bixby before or after 7 April 2024.

Question: "Will Beyoncé release a full album for her 'country era' on 12 July 2025?"
Answer:
The outcome will be 'Yes' if:
    - Beyoncé releases a full music album for her 'country era' on the specific day of 12 July 2025.
The outcome will be 'No' if:
    - Beyoncé does not release a full album for her 'country era' on 12 July 2025.
    - Beyoncé releases a 'country era' album before or after 12 July 2025.

Question: "Will Disney Plus implement its password-sharing crackdown on 7 February 2025?"
Answer:
The outcome will be 'Yes' if:
    - Disney Plus implements a password-sharing crackdown policy on the specific day of 7 February 2025.
The outcome will be 'No' if:
    - Disney Plus does not implement a password-sharing crackdown policy on 7 February 2025.
    - Disney Plus implements a password-sharing crackdown policy before or after 7 February 2025.

Question: "Will the Royals and Chiefs relocate from Kansas City following the rejection of the stadium tax on 24 June 2024?"
Answer:
The outcome will be 'Yes' if:
    - Both the Royals and Chiefs relocate from Kansas City following the rejection of the stadium tax on the specific day of 24 June 2024.
The outcome will be 'No' if:
    - Either the Royals or the Chiefs do not complete the relocation from Kansas City following the rejection of the stadium tax on 24 June 2024.
    - The rejection of the stadium tax does not lead to the relocation of both teams from Kansas City on 24 June 2024.
    - The Royals and the Chiefs complete relocation before or after 24 June 2024.

Question: "Will Joe Rogan face significant backlash for his comments about Israel on 16 April 2024?"
Answer:
The outcome will be 'Yes' if:
    - Joe Rogan faces significant backlash on the specific day of 16 April 2024 for comments he made about Israel.
The outcome will be 'No' if:
    - Joe Rogan does not face significant backlash on 16 April 2024 for comments he made about Israel.
    - Joe Rogan faces significant backlash before or after 16 April 2024 for comments he made about Israel.

Question: "{market_question}"
Answer:
"""

# Prompt template for inferring the status of a prediction market question
INFER_STATUS_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to infer the status for a prediction market question and \
provide definitions for the key terms used in the question that apply to its context.
You are provided with some examples below.

EXAMPLES:
Question: "Will there be another case of H5N1 bird flu in Texas ?"
Answer:
Status: The question implies that there have been one or more confirmed cases of H5N1 bird flu in the state of Texas. It focusses on the timing of another new case of H5N1 bird flu occurring in Texas.
Definitions:
    - 'another case': A newly confirmed occurrence of H5N1 bird flu that is distinct from previous cases.

Question: "Will Samsung replace its voice assistant Bixby ?"
Answer:
Status: The question implies that Samsung is considering replacing its voice assistant, Bixby, with a new voice assistant. The question focuses on the timing of Samsung's potential replacement of Bixby.
Definitions:
    - 'replace': The discontinuation of Bixby as the primary voice assistant and the implementation of a new voice assistant.
    
Question: "Will Beyoncé release a full album for her 'country era' ?"
Answer:
Status: The question implies that Beyoncé is potentially exploring country music as a new musical direction referred to as her 'country era'. It focuses on the timing of the releas of a Beyoncé's full album within this thematic context.
Definitions:
    - 'full album': An album containing multiple tracks recognized as a complete album, not singles or EPs.
    - 'album for her country era': An album that predominantly features songs in the country music genre.

Question: "Will Microsoft announce a significant AI-related takeover by 16 April 2024?"
Answer:
Status: The question implies that Microsoft may be considering a significant acquisition or takeover related to artificial intelligence (AI). The question focuses on the timing of Microsoft's potential announcement of an AI-related takeover.
Definitions:
    - 'significant AI-related takeover': An acquisition or takeover that involves a substantial investment in AI technology or companies.
    - 'announcement': Public declaration or disclosure made by Microsoft regarding the takeover.

Question: "Will Gilberto Ramirez defend his WBA (Super) cruiserweight title successfully ?"
Answer:
Status: The question implies that Gilberto Ramirez is the current titleholder of the WBA (Super) cruiserweight title. The question focuses on the timing of Ramirez's successful defense of his title.
Definitions:
    - 'successfully defend': Ramirez must be the reigning titleholder and emerge victorious in a defense match against a contender to retain his title.
    - 'WBA (Super) cruiserweight title': A specific boxing title in the cruiserweight division.
    
Question: "Will Saudi Arabia successfully host the WTA Finals ?"
Answer:
Status: The question implies that Saudi Arabia is in consideration to host the WTA Finals, a prestigious women's tennis event. The focus is on the timing of Saudi Arabia's successfully hosting the WTA Finals.
Definitions:
    - 'successfully host': Saudi Arabia carries out the event to its conclusion without significant disruptions.
    - 'WTA Finals': Season-ending tennis championship for the top eight WTA singles and doubles teams.
    
Question: "Will Xiaomi's SU7 electric vehicle still have a waiting list in China ?"
Answer:
Status: The question implies that Xiaomi has introduced an electric vehicle model named SU7 in China, with a current waiting list of potential buyers. The question focuses on the timing for the continuation of this waiting list.
Definitions:
    - 'waiting list': A list of customers waiting to purchase the SU7 electric vehicle in China.
    - 'SU7 electric vehicle': A specific model of electric vehicle produced by Xiaomi.

Question: "Will a new climate bill be passed by both the Senate and the House ?"
Answer:
Status: The question implies that a climate bill already exists and that a new climate bill is currently under consideration by the United States Congress, which consists of two chambers: the Senate and the House of Representatives. The question focuses on the timing for this new bill being passed by both chambers.
Definitions:
    - 'pass': Official approval of the bill by both chambers of the United States Congress.
    - 'new climate bill': A bill that addresses climate-related issues and is distinct from an existing bill or legislation.
    - 'Senate and the House': The two legislative bodies that make up the United States Congress.

Question: "Will Tesla successfully launch its new electric car, Model S ?"
Answer:
Status: The question implies that Tesla is currently working on a new electric vehicle, which is called Model S. The question focuses on the timing of Tesla's successful launch of the Model S electric car.
Definitions:
    - 'launch': The public release and availability of the vehicle for purchase or use.
    - 'electric vehicle': A vehicle powered by an electric motor.
    - 'Model S': The specific model name of an electric vehicle.
   
Question: "Will FIFA fund the construction of new football stadiums for local clubs in England ?"
Answer:
Status: The question implies that local football clubs in England are in need of new stadiums. It further implies that FIFA, the international governing body of football, may consider funding the construction of these new stadiums. The question focusses on the timing of FIFA's potential funding for these specific projects.
Definitions:
    - 'fund': Actively disbursing financial resources.
    - 'new football stadiums for local clubs': Stadiums that are newly constructed for football clubs that are not part of the professional league system.
    
Question: "Will Prince William and Prince Harry appear separately at the event honoring Princess Diana ?"
Answer:
Status: Status: The question implies that there is an upcoming event honoring Princess Diana where both Prince William and Prince Harry are expected to attend. The question focuses on the event's timing and the manner in which the princes will appear at the event honoring Princess Diana.
Definitions:
    - 'appear separately': The princes attend the event at different times or locations without being present together. This includes virtual appearances.
    - 'event honoring Princess Diana': A gathering or ceremony dedicated to commemorating Princess Diana's life or legacy.

Question: "Will Disney Plus implement its password-sharing crackdown ?"
Answer:
Status: The question implies that Disney Plus is considering implementing a policy to prevent the sharing of login credentials among multiple users. The question focuses on the timing for the implementation of this policy.
Definitions:
    - 'implement': The policy is put into effect and actively enforced by Disney Plus.
    - 'password-sharing crackdown': The enforcement of measures to restrict or prevent the sharing of login credentials among multiple users.
    
Question: "Will Google destroy all browsing data collected in Incognito mode ?"
Answer:
Status: The question implies that Google has collected browsing data in Incognito mode and is expected to destroy this data. The question focuses on the timing for the Incognito data destruction.
Definitions:
    - 'destroy': The permanent deletion or removal of the Incognito browsing data.
    - 'all browsing data': The entirety of data collected during the browsing session in Incognito mode and not just a portion of it.

Question: "Will Joe Rogan face significant backlash for his comments about Israel ?"
Answer:
Status: The question implies that Joe Rogan has made comments about Israel that could potentially lead to a negative public reaction. The question focuses on the timing of any significant backlash he may face for his comments about Israel.
Definitions:
    - 'significant backlash': A substantial negative reaction or criticism that garners widespread attention and impacts Joe Rogan's reputation or public perception significantly.
    - 'comments about Israel': Remarks, opinions, or statements made by Joe Rogan regarding Israel, its policies, or related issues.

Question: "{market_question}"
Answer:
"""


class WebPage:
    _id_counter = 0
    id: int
    url: str
    html: Optional[str] = None
    scraped_text: Optional[str] = None
    publisher: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    publication_date: Optional[str] = None
    chunks_sorted: List[str] = Field(default_factory=list)
    relevant_chunks_summary: Optional[str] = None
    chunks_final: List[str] = Field(default_factory=list)
    final_output: Optional[str] = None

    def __init__(self, url, html=None, title=None, description=None, publication_date=None, publisher=None):
        type(self)._id_counter += 1
        self.id = type(self)._id_counter
        self.url = url
        self.html = html
        self.scraped_text = None
        self.publisher = publisher
        self.title = title
        self.description = description
        self.publication_date = publication_date
        self.chunks_sorted = []
        self.chunks_final = []
        self.extract_attribute_names = ["title", "description", "publication_date", "publisher"]

    def get_title(self, soup, scripts):
        try:
            title = soup.title
            if title:
                return title.string.strip()
        except AttributeError:
            pass

        # If the title was not found or an AttributeError was caught, attempt to find the title using meta tags.
        title = soup.find("meta", attrs={"name": "title"}) or soup.find("meta", attrs={"property": "title"})
        if title and title.get("content"):
            return title["content"].strip()

        # If no title was found return "n/a".
        return "n/a"

    def get_description(self, soup, scripts):
        description = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "description"})
        if description and description.get("content"):
            return description["content"].strip()
        return "n/a"

    def get_publisher(self, soup, scripts):
        for script in scripts:
            try:
                data = json.loads(script.string)
                publisher = self._find_publisher(data)
                return publisher
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
            except Exception as e:
                print(f"Error extracting publisher for webpage {self.url}")
                continue
        
        publisher = soup.find("meta", attrs={"name": "publisher"}) or soup.find("meta", attrs={"property": "publisher"})
        
        if publisher and publisher.get("content"):
            return publisher["content"].strip()
        else:
            return "n/a"

    def get_date(self, soup, scripts):
        for script in scripts:
            try:
                data = json.loads(script.string)
                data = get_first_dict_from_list(data)
                date = find_release_date_in_data(data)
                if date:
                    return format_date(date)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
            except Exception as e:
                print(f"Error extracting publisher for webpage {self.url}")
                continue

        # If not found, then look for release or publication date
        for name in RELEASE_DATE_NAMES:
            meta_tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
            if meta_tag and meta_tag.get("content"):
                return format_date(meta_tag["content"])
        return "n/a"

    def extract_page_attributes(
        self,
    ) -> object:
        """Retrieves the release date from the soup object containing the HTML tree"""
        if self.html:
            soup = BeautifulSoup(self.html, "html.parser")
            scripts = soup.find_all('script', type='application/ld+json')
            for attribute_name in self.extract_attribute_names:
                if attribute_name == "title":
                    self.title = self.get_title(soup, scripts)
                elif attribute_name == "description":
                    self.description = self.get_description(soup, scripts)
                elif attribute_name == "publication_date":
                    self.publication_date = self.get_date(soup, scripts)
                elif attribute_name == "publisher":
                    self.publisher = self.get_publisher(soup, scripts)
                else:
                    raise ValueError(f"Invalid attribute: {attribute_name}")
        else:
            print(f"No HTML content to extract page attributes from.\nURL: {self.url}\n")
        
        return self

    def to_prompt(self):
        """
        Function to convert article attributes into a structured format for LLM prompts.
        """
        page_info = f"ID: {self.id}\n"
        page_info += f"URL: {self.url}\n"
        page_info += f"Title: {self.title or 'Untitled'}\n"
        page_info += f"Description: {self.description or 'n/a'}\n"
        page_info += f"Published: {self.publication_date or 'Unknown'}\n"
        page_info += f"Publisher: {self.publisher or 'Unknown'}"
        
        return page_info
    
    def _find_publisher(self, data):
        def extract_names(item, key):
            """Helper function to extract names from a field that could be a list or a single object."""
            if isinstance(item, list):
                return [entry.get('name', 'Unknown name') for entry in item]
            elif isinstance(item, dict):
                return item.get('name', 'Unknown name')
            return 'n/a'

        # If data is a dictionary, look for 'publisher' or 'author' directly
        if isinstance(data, dict):
            if 'publisher' in data:
                return extract_names(data['publisher'], 'publisher')
            elif 'author' in data:
                return extract_names(data['author'], 'author')

        # If data is a list, iterate through it and look for 'publisher' or 'author' in each item
        elif isinstance(data, list):
            for item in data:
                if 'publisher' in item:
                    return extract_names(item['publisher'], 'publisher')
                elif 'author' in item:
                    return extract_names(item['author'], 'author')

        # If no 'publisher' or 'author' found, or data is neither a list nor a dict
        return 'n/a'


class TextChunk(BaseModel):
    text: str
    url: str
    num_tokens: Optional[int] = None
    embedding: Optional[List[float]] = None
    similarity: Optional[float] = None

def trim_json_formatting(text) -> str:
    """Trim the JSON formatting characters from string."""
    # Regex pattern that matches the start and end markers with optional newline characters
    pattern = r'^\s*```\s*json\s*({.*?})\s*```\s*$'

    # Use re.DOTALL to make '.' match newlines as well
    match = re.match(pattern, text, re.DOTALL)
    if match:
        print("JSON formatting characters found and removed")
        formatted_json = match.group(1)
        return formatted_json
    else:
        return text

def trim_chunks_string(
    chunks_string: str,
    enc: tiktoken.Encoding,
    max_tokens: int = MAX_CHUNKS_TOKENS_TO_SUMMARIZE
) -> str:
    """Trim a string to a maximum number of tokens for summarization"""
    encoding = enc.encode(chunks_string)
    if len(encoding) > max_tokens:
        encoding = encoding[:max_tokens]
    return enc.decode(encoding)

def find_release_date_in_data(data):
    for name in RELEASE_DATE_NAMES:
        if name in data:
            return data[name]
    return None

def format_date(date_string) -> str:
    # Desired format "February 16, 2024, 3:30 PM"
    format_str = "%B %d, %Y"

    # Handle the case where the date string is "unknown"
    if date_string.lower() == "unknown":
        return date_string
    try:
        # Parse the date string to datetime object
        parsed_date = parser.parse(date_string)

        # Adjust for AM/PM format, removing leading 0 in hour for consistency
        formatted_date = parsed_date.strftime(format_str).lstrip("0").replace(" 0", " ")
        return formatted_date
    except (ValueError, TypeError):
        # If there's an error during parsing, return the original string
        return date_string
    
def parse_date_str(date_str: str) -> datetime:
    # Desired format "February 16, 2024, 3:30 PM"
    datetime_format = "%B %d, %Y"
    try:
        return datetime.strptime(date_str, datetime_format)
    except (ValueError, TypeError):
        return datetime.min
    
def remove_date_from_query(query: str) -> str:
    """Remove time-related information from query"""
    date_pattern = r"\b(?:on or by |on or before |before |by |on )?(?:(\d{1,2})(st|nd|rd|th)? (January|February|March|April|May|June|July|August|September|October|November|December)|(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})(st|nd|rd|th)?,?) \d{4}\b"
    new_query = re.sub(date_pattern, "", query)
    return new_query

def recursive_character_text_splitter(text, max_tokens, overlap):
    if len(text) <= max_tokens:
        return [text]
    else:
        return [text[i:i+max_tokens] for i in range(0, len(text), max_tokens - overlap)]

def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))

def get_first_dict_from_list(data):
    """Returns the first item if data is a list of dictionaries"""
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return data[0]
    else:
        return data

def extract_answer(response_message: str) -> str:
    """Extract the answer from the response message."""
    if "Answer:" in response_message:
        answer = response_message.split("Answer:", 1)[1]
    else:
        answer = response_message
    return answer

def format_additional_information(web_pages: List[WebPage]) -> str:
    """Format the additional information from the web pages"""
    formatted_information = ""
    for i, web_page in enumerate(web_pages):
        # formatted_information += f"ARTICLE {i+1}: {web_page.title}, PUBLISHER: {web_page.publisher}, PUBLICATION_DATE: {web_page.publication_date}\n"
        formatted_information += f"ARTICLE {i+1}: PUBLISHER: {web_page.publisher}, PUBLICATION_DATE: {web_page.publication_date}\n"
        formatted_information += f"{web_page.final_output}\n\n"
    return formatted_information

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

def extract_question(text:str) -> str:
    # Look for a quoted question
    match = re.search(r'["“](.*?\?)["”]', text)
    if match:
        return match.group(1).strip()
    
    # Return prompt if ending with a question mark
    return text if text.strip().endswith('?') else ""

def get_prompt_template_by_timing(query: str) -> str:
    """Get the prompt template based on the timing of the event in the query."""
    date_pattern_on = r"\b(?:on )(?:(\d{1,2})(st|nd|rd|th)? (January|February|March|April|May|June|July|August|September|October|November|December)|(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})(st|nd|rd|th)?,?) \d{4}\b"
    date_pattern_by = r"\b(?:on or before |before |by |)(?:(\d{1,2})(st|nd|rd|th)? (January|February|March|April|May|June|July|August|September|October|November|December)|(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})(st|nd|rd|th)?,?) \d{4}\b"
    
    if re.search(date_pattern_on, query):
        match = re.search(date_pattern_on, query)
        return INFER_RULES_PROMPT_ON
    elif re.search(date_pattern_by, query):
        match = re.search(date_pattern_by, query)
        return INFER_RULES_PROMPT_BY
    else:
        return "No time-related information found in query."

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

def search_google(query: str, api_key: str, engine: str, num: int) -> List[str]:
    """Search Google using a custom search engine."""
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
    return [result["link"] for result in search.get("items", [])]

def process_in_batches(
    web_pages: List[WebPage],
    batch_size: int = 15,
    timeout: int = 10
) -> Generator[None, None, List[Tuple[Future, WebPage]]]:
    """ Process URLs in batches and yield the results as they complete."""
    session = Session()
    session.max_redirects = 5

    # User-Agent headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/117.0'
    }
    session.headers.update(headers)

    with ThreadPoolExecutor() as executor:
        # Loop through the URLs in batches
        for i in range(0, len(web_pages), batch_size):
            batch = web_pages[i:i + batch_size]
            
            # Submit HEAD requests for all URLs in the batch
            head_futures = {executor.submit(session.head, web_page.url, headers=headers, timeout=timeout, allow_redirects=True): web_page for web_page in batch}
            
            # Process HEAD requests as they complete
            get_futures = []
            for future in as_completed(head_futures):
                web_page = head_futures[future]
                try:
                    head_response = future.result()
                    if 'text/html' in head_response.headers.get('Content-Type', ''):
                        # Only submit GET requests for URLs with 'text/html' Content-Type
                        get_future = executor.submit(session.get, web_page.url, headers=headers, timeout=timeout, allow_redirects=True)
                        get_futures.append((get_future, web_page))
                except requests.exceptions.Timeout:
                    print(f"HEAD request for {web_page.url} timed out.")
                except Exception as e:
                    print(f"Error processing HEAD request for {web_page.url}: {e}")

            yield get_futures

def embed_batch(client: OpenAI, batch):
    """
    Helper function to process a single batch of texts and return the embeddings.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text_chunk.text for text_chunk in batch]
    )

    # Assert the order of documents in the response matches the request
    for i, data in enumerate(response.data):
        assert i == data.index, "Document order in the response does not match the request."

    # Return the embeddings
    return [data.embedding for data in response.data]

def sort_text_chunks(
    client: OpenAI, query: str, text_chunks_embedded: List[TextChunk]
) -> List[TextChunk]:
    """Similarity search to find similar chunks to a query"""
    query_embedding = (
        client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        .data[0]
        .embedding
    )

    index = faiss.IndexFlatIP(EMBEDDING_SIZE)
    index.add(np.array([text_chunk.embedding for text_chunk in text_chunks_embedded]))
    D, I = index.search(np.array([query_embedding]), len(text_chunks_embedded))
    for i, sim in enumerate(D[0]):
        text_chunks_embedded[I[0][i]].similarity = sim
        
    return [text_chunks_embedded[i] for i in I[0]]

def get_embeddings(client: OpenAI, text_chunks: List[TextChunk], enc: tiktoken.Encoding) -> List[TextChunk]:
    """Get embeddings for the text chunks."""  
    # Batch the text chunks that the sum of tokens is less than MAX_EMBEDDING_TOKEN_INPUT
    batches = []
    current_batch = []
    current_batch_token_count = 0
    for text_chunk in text_chunks:
        text_chunk.num_tokens  = len(enc.encode(text_chunk.text))
        if text_chunk.num_tokens + current_batch_token_count <= MAX_EMBEDDING_TOKEN_INPUT:
            # Add document to the batch if token limit is not exceeded
            current_batch.append(text_chunk)
            current_batch_token_count += text_chunk.num_tokens
        else:
            # Process the current batch and start a new one if the token limit would be exceeded
            batches.append(current_batch)
            current_batch = [text_chunk]
            current_batch_token_count = text_chunk.num_tokens

    # Add the last batch
    if current_batch:
        batches.append(current_batch)


    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_batch = {executor.submit(embed_batch, client, batch): batch for batch in batches}

        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                embeddings = future.result()
                # Assign embeddings to the corresponding documents
                for text_chunk, embedding in zip(batch, embeddings):
                    #print(f"Embedding for {text_chunk.text}: {embedding}")
                    text_chunk.embedding = embedding

            except Exception as e:
                print(f"Exception: {e}")

    return text_chunks

def get_chunks(web_pages: List[WebPage]) -> List[WebPage]:
    """Create chunks from the text of all web pages"""
    text_chunks = []
    for web_page in web_pages:
        if web_page.scraped_text:
            chunks = recursive_character_text_splitter(web_page.scraped_text, TEXT_CHUNK_LENGTH, TEXT_CHUNK_OVERLAP)
            text_chunks.extend(TextChunk(text=chunk, url=web_page.url) for chunk in chunks)

    return text_chunks

def scrape_web_pages(web_pages: List[WebPage], week_interval, max_num_char: int = 10000) -> List[WebPage]:
    """Scrape text from web pages"""
    filtered_web_pages = []
    for web_page in web_pages:
        if web_page.html:
            date_parsed = parse_date_str(web_page.publication_date)
            if date_parsed > datetime.now() - timedelta(weeks=week_interval) or date_parsed == datetime.min:
                doc_html2str = Document(web_page.html)
                doc_sum = doc_html2str.summary()
                text = md(doc_sum, strip=['a', 'b', 'strong', 'em', 'img', 'i', 'mark', 'small', 'u'], heading_style="ATX")
                text = "  ".join([x.strip() for x in text.split("\n")])
                text = re.sub(r'\s+', ' ', text)
                scraped_text = text

                if len(scraped_text) < 300:
                    if not (web_page.title == "n/a" and web_page.description == "n/a"):
                        prefix = f"{web_page.title}. {web_page.description}."
                        scraped_text = prefix + scraped_text
                        print(f"Scraped text has less than 300 characters. Added title and description to the text.")
                    else:
                        scraped_text = None
                        print("Scraped text has less than 300 characters and no title and description. Skipping...")
                        continue

                    web_page.scraped_text = scraped_text[:max_num_char]
                    web_page.scraped_text = scraped_text[:max_num_char]
                
                web_page.scraped_text = scraped_text[:max_num_char]
                
                filtered_web_pages.append(web_page)

            else:
                web_page.scraped_text = None
                print(f"Publication date {web_page.publication_date} is older than {week_interval} weeks.")
        else:
            web_page.html = None
            print("HTML content is not available for web page.")

    return filtered_web_pages

def extract_html_texts(
    web_pages: List[WebPage],
) -> List[WebPage]:    
    # Initialize empty list for storing extracted sentences along with their similarity scores, release dates and urls
    parsed_web_pages = []

    # Process URLs in batches
    for batch in process_in_batches(web_pages=web_pages):
        for future, web_page in batch:
            if future is None:
                print(f"Future for {web_page.url} is None.")
                continue
            try:
                result = future.result()
                if result.status_code == 200:
                    # Extract relevant information for the event question
                    web_page.html = result.text
                    if web_page.html is None:
                        print(f"HTML content for {web_page.url} is None.")
                    parsed_web_page = web_page.extract_page_attributes()
                    parsed_web_pages.append(parsed_web_page)
                elif result.status_code != 200:
                    print(f"Request for {web_page.url} returned status code {result.status_code}.")
                elif 'text/html' not in result.headers.get('Content-Type', ''):
                    print(f"Content-Type for {web_page.url} is not 'text/html'.")
    
            except requests.exceptions.Timeout:
                print(f"Request for {web_page.url} timed out.")
            
            except Exception as e:
                print(f"An error occurred in extract_html_texts: {e}")

    return parsed_web_pages

def get_urls_from_queries(
    queries: List[str],
    api_key: str,
    engine: str,
    num: int = 3
) -> List[str]:
    """Fetch unique URLs from search engine queries, limiting the number of URLs per query."""
    results = set()
    max_num_fetch = 10
    omit_keywords = ["pdf", "instagram", "youtube", "reddit"]

    if num > max_num_fetch:
        raise ValueError(f"The maximum number of URLs per query is {max_num_fetch}.")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(search_google, query, api_key, engine, max_num_fetch) for query in queries}
        for future in as_completed(futures):
            result = future.result()
            count = 0
            for url in result:
                if url not in results and not any(keyword in url for keyword in omit_keywords):
                    results.add(url)
                    count += 1
                    if count >= num:
                        break
                else:
                    continue

    for url in results:
        print(url)

    return list(results)

def fetch_queries(
    client: OpenAI,
    input_query: str,
    engine="gpt-3.5-turbo",
    market_rules = None,
    counter_callback=None,
    temperature=1.0,
    max_attempts=2,

) -> List[str]:
    """Fetch queries from the OpenAI engine"""
    attempts = 0
    while attempts < max_attempts:
        try:
            research_plan_prompt = RESEARCH_PLAN_PROMPT_TEMPLATE.format(query=input_query, search_limit="12")
            messages = [
                {"role": "system", "content": "You are a professional researcher."},
                {"role": "user", "content": f"Get the market rules for the prediction market question {input_query}"},
                {"role": "assistant", "content": market_rules},
                {"role": "user", "content": research_plan_prompt},
            ]
            
            # Fetch queries from the OpenAI engine
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
            )
            if counter_callback is not None:
                counter_callback(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=engine,
                    token_counter=count_tokens,
                )
            search_plan = response.choices[0].message.content
            
            messages = [
                {"role": "system", "content": "You are a professional researcher."},
                {"role": "user", "content": research_plan_prompt},
                {"role": "assistant", "content": search_plan},
                {"role": "user", "content": QUERY_RERANKING_PROMPT_TEMPLATE},
            ]
            
            # Fetch reranked and selected queries from the OpenAI engine
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=0.0,
            )
            if counter_callback is not None:
                counter_callback(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=engine,
                    token_counter=count_tokens,
                )
            output = response.choices[0].message.content

            # Parse the response content
            trimmed_output = trim_json_formatting(output)
            print(trimmed_output)
            json_data = json.loads(trimmed_output)
            queries = json_data["queries"]
            return queries, counter_callback  # Return the parsed data if successful
        except Exception as e:
            print(f"Attempt {attempts + 1} failed with error: {e}")
            attempts += 1
            if attempts == max_attempts:
                print("Maximum attempts reached, returning an empty string.")
                return []

def summarize_relevant_chunks(
        web_pages: List[WebPage],
        input_query: str,
        enc: tiktoken.Encoding,
        counter_callback,
        engine="gpt-3.5-turbo",
        temperature=0.0,
) -> List[WebPage]:
    def summarize_for_web_page(web_page: WebPage) -> None:
        chunks_string = ""
        if web_page.chunks_sorted:
            for chunk in web_page.chunks_sorted:
                chunks_string += f"\n…{chunk}…\n"
        else:
            web_page.relevant_chunks_summary = "Error"
            return
        
        trimmed_chunks = trim_chunks_string(chunks_string, enc)

        # Transform the market question to a "When" question
        market_question_no_date = remove_date_from_query(input_query)
        market_question_when = f"When {market_question_no_date}"

        summarize_prompt = SUMMARIZE_PROMPT.format(input_query=market_question_when, chunks=trimmed_chunks)
        messages = [
            {"role": "system", "content": "You are a professional journalist and researcher."},
            {"role": "user", "content": summarize_prompt},
        ]
        response = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=100,
        )
        if counter_callback is not None:
            counter_callback(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=engine,
                token_counter=count_tokens,
            )
        output = response.choices[0].message.content

        web_page.relevant_chunks_summary = output

    with ThreadPoolExecutor() as executor:
        # Submit all web pages to the executor for processing
        future_to_web_page = {executor.submit(summarize_for_web_page, web_page): web_page for web_page in web_pages}

        # Wait for all futures to complete
        for future in as_completed(future_to_web_page):
            web_page = future_to_web_page[future]
            try:
                # Result is None, as web_page objects are modified in-place
                future.result()
            except Exception as e:
                print(f'Web page {web_page.url} generated an exception: {e}')
    if web_pages:
        web_pages = [web_page for web_page in web_pages if "Error" not in web_page.relevant_chunks_summary]
    return web_pages, counter_callback

def summarize_over_summarized_chunks(
    web_pages: List[WebPage],
    input_query: str,
    counter_callback,
    engine="gpt-3.5-turbo",
    temperature=0.0,
) -> List[WebPage]:
    
    # Add WebPage ID after each line in relevant_chunks_summary
    # Initialize an empty list to hold all modified lines
    all_lines_with_id = []

    for web_page in web_pages:
        if web_page.chunks_final:
            # Split the summary into lines and align bulletpoints to "-"
            lines = web_page.relevant_chunks_summary.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("*"):
                    lines[i] = "-" + line[1:]                  
            
            # Append the web page ID to each line and add it to the list
            all_lines_with_id.extend([line + f" ({web_page.id})\n" for line in lines if line.strip() != ''])

    # Join all modified lines into a single string
    all_relevant_chunks_summary = '\n'.join(all_lines_with_id)

    prompt = FINAL_SUMMARY_PROMPT.format(input_query=input_query, chunks=all_relevant_chunks_summary)
    
    print(f"\nPROMPT SUMMARIZE OVER WEBSITE SUMMARIES:\n################################################")
    print(f"{prompt}\n################################################\n\n")

    messages = [
        {"role": "system", "content": "You are a professional journalist."},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model=engine,
        messages=messages,
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
    
    # Split the combined string into individual lines
    lines = output.strip().split('\n')

    # Mapping of web_page.id to web_page object for quick access
    web_pages_dict = {str(web_page.id): web_page for web_page in web_pages}

    # Initialize a set to track modified web_page IDs
    modified_ids = set()

    # Process each line to extract the web_page.id and content, then update the relevant web page
    for line in lines:      
        match = re.match(r"^(.*) \((\d+)\)\.?$", line)
        if match:
            content, web_page_id = match.groups()
            # Check if this web_page_id is in our dictionary of web pages
            if web_page_id in web_pages_dict:
                # Update the relevant_chunks_summary of the corresponding web page
                web_page = web_pages_dict[web_page_id]
                if web_page.final_output:
                    web_page.final_output += '\n' + content
                else:
                    web_page.final_output = content
                # Mark this ID as modified
                modified_ids.add(web_page_id)

    # Collect modified web_page objects based on modified_ids
    modified_web_pages = [web_pages_dict[web_page_id] for web_page_id in modified_ids]

    return modified_web_pages, counter_callback

def research(
    market_question: str,
    google_api_key: str,
    google_engine_id: str,
    engine: str,
    market_rules: str,
    counter_callback,
    num_urls: int = NUM_URLS_PER_QUERY,
):
    """Research additional information based on a prediction market question"""
    # Generate a list of sub-queries
    queries, counter_callback = fetch_queries(client, market_question, engine, market_rules, counter_callback)
    
    # Get URLs from sub-queries
    urls = get_urls_from_queries(
        queries,
        api_key=google_api_key,
        engine=google_engine_id,
        num=num_urls,
    )
    web_pages = [WebPage(url) for url in urls]
    web_pages = extract_html_texts(web_pages)
    
    # Scrape text from web pages not older than <week_interval> weeks
    week_interval = WEEKS_TO_SCRAPE_NEWS
    web_pages = scrape_web_pages(web_pages, week_interval)

    # Get text chunks from web pages
    text_chunks = get_chunks(web_pages)
    
    # Get embeddings for text chunks, sort and cap the number of text chunks 
    enc = tiktoken.get_encoding("cl100k_base") 
    text_chunks_embedded = get_embeddings(client, text_chunks, enc) if text_chunks else []
    text_chunks_sorted = sort_text_chunks(client, market_question, text_chunks_embedded) if text_chunks_embedded else []
    text_chunks_limited = text_chunks_sorted[:MAX_TEXT_CHUNKS_TOTAL]

    # Create a dictionary mapping URLs to WebPage objects for quicker lookups
    web_pages_dict = {web_page.url: web_page for web_page in web_pages}

    # Assign the sorted text chunks to the corresponding WebPage objects
    for text_chunk in text_chunks_limited:
        if text_chunk.url in web_pages_dict:
            web_pages_dict[text_chunk.url].chunks_sorted.append(text_chunk.text)

    # Summarize the relevant chunks from each web page 
    web_pages = list(web_pages_dict.values())
    web_pages, counter_callback = summarize_relevant_chunks(web_pages, market_question, enc, counter_callback, engine)
    web_pages = sorted(web_pages, key=lambda web_page: parse_date_str(web_page.publication_date))

    # Create a list of TextChunk objects with the summarized chunks
    relevant_summarized_chunks = []
    relevant_summarized_chunks.extend(TextChunk(text=page.relevant_chunks_summary, url=page.url) for page in web_pages)

    # Append the summarized chunks to the corresponding WebPage object in the dictionary
    for sum in relevant_summarized_chunks:
        if sum.url in web_pages_dict:
            web_pages_dict[sum.url].chunks_final.append(sum.text)

    # Summarize over all summarized chunks
    web_pages = list(web_pages_dict.values())
    web_pages, counter_callback = summarize_over_summarized_chunks(web_pages, market_question, counter_callback, engine)
    web_pages = sorted(web_pages, key=lambda web_page: parse_date_str(web_page.publication_date))

    additional_information = format_additional_information(web_pages)
    if additional_information:
        additional_information += (
            f"Disclaimer: This search output was retrieved on {datetime.now().strftime('%B %d, %Y')} and does not claim to be exhaustive or definitive."
        )

    return additional_information, counter_callback

def get_market_rules(
    market_question: str,
    counter_callback,
    temperature=0.0,
    engine="gpt-3.5-turbo",
):
    """Infer market rules for a prediction market question."""
    # Remove double quotes from the input query to avoid issues
    market_question = market_question.replace('"', "'")
    
    # Get the prompt template based on the timing of the event in the query
    infer_rules_template = get_prompt_template_by_timing(market_question)
    infer_rules_prompt = infer_rules_template.format(market_question=market_question)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_INFER_RULES},
        {"role": "user", "content": infer_rules_prompt},
    ]
    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
    )
    if counter_callback is not None:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=engine,
            token_counter=count_tokens,
        )
    response_message = response.choices[0].message.content

    # Extract the market rules from the response message
    market_rules = extract_answer(response_message)
    
    ## Infer the market status
    # Remove the date from the query to avoid bias
    market_question_no_date = remove_date_from_query(market_question)
    infer_status_prompt = INFER_STATUS_PROMPT.format(market_question=market_question_no_date)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_INFER_RULES},
        {"role": "user", "content": infer_status_prompt},
    ]
    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
    )
    if counter_callback is not None:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=engine,
            token_counter=count_tokens,
        )
    response_message = response.choices[0].message.content

    # Extract the market status from the response message
    market_status = extract_answer(response_message)
    
    return market_status, market_rules, counter_callback

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
        market_status, market_rules, counter_callback = get_market_rules(market_question, counter_callback, engine=engine)
        print(f"MARKET STATUS: {market_status}\n")
        print(f"MARKET RULES:\n{market_rules}\n")
        
        # Get additional information from the Research tool
        additional_inforamtion, counter_callback = research(market_question, google_api_key, google_engine_id, engine, market_rules, counter_callback)

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
        
        print("SUBJECT MATTER EXPERT:\n################################################\n")
        if sme:
            print(f"SME ROLE: {sme}")
        else:
            print("SME role not found.")
        print(f"SME INTRODUCTION: {sme_introduction}\n")
        print("################################################\n\n")

        print(f"REPORT PROMPT:\n################################################\n{report_prompt}\n################################################\n\n")

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
        print(f"RESEARCH OUTPUT:\n################################################\n\n{output}\n\n################################################\n\n")

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
        print(f"PREDICTION OUTPUT:\n################################################\n\n{output}\n\n################################################\n\n")

        # Remove conclusion field from the JSON string
        output = remove_unwanted_fields(output)
        
        return output, thread_history_string, None, counter_callback
              

                




