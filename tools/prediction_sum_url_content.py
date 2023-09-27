# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 Valory AG
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

import time
from typing import Any, Dict, Generator, List, Optional, Tuple
from datetime import datetime, time, timezone
import json
import re
from concurrent.futures import Future, ThreadPoolExecutor

from bs4 import BeautifulSoup, NavigableString
from googleapiclient.discovery import build
import openai
import requests
from requests import Session
import spacy
import tiktoken
import torch
import traceback

from dateutil import parser
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

NUM_URLS_EXTRACT = 5
MAX_TOTAL_TOKENS_CHAT_COMPLETION = 4096
WORDS_PER_TOKEN_FACTOR = 0.75
DEFAULT_OPENAI_SETTINGS = {
    "max_compl_tokens": 200,
    "temperature": 0.2,
}

ALLOWED_TOOLS = [
    "prediction-offline-sum-url-content",
    "prediction-online-sum-url-content",
]
TOOL_TO_ENGINE = {
    "prediction-offline-sum-url-content": "gpt-3.5-turbo",
    # "prediction-online-sum-url-content": "gpt-3.5-turbo",
    # "prediction-online-sum-url-content": "gpt-3.5-turbo-16k",
    "prediction-online-sum-url-content": "gpt-4",
}

# OLD:
# You are an LLM inside a multi-agent system. Your task is to estimate the probability of a user's 'event question', 
# which specifies an event in the physical world and any accompanying conditions to be met for the 'event question' to be true. The 'event question' allows only two outcomes: the event 
# will either occur or not, given the conditions. Find the 'event question' enclosed in double quotes as a part of 
# the user's prompt under 'USER_PROMPT'. The user's prompt also contains a more elaborate description of the task. 
# You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION", delimited by three backticks. This information results from a search engine query that has been done a few seconds ago with the aim to get up-to-date information that could be relevant estimating the 'event question'.
# You must adhere to the 'INSTRUCTIONS'.  

# * Carefully read the user's prompt under 'USER_PROMPT', enclosed by triple backticks.
# * If the 'event question' has more than two outcomes, respond with "Error" and ignore further instructions.
# * Based on your training data, provide a probability estimation of the event specified in the 'event question' occuring, considering all conditions provided.
# * You can use any item in "ADDITIONAL_INFORMATION" in addition to your training data to make the probability estimation.
# * Prioritize recent information in "ADDITIONAL_INFORMATION" based on the current time {timestamp}.
# * You must pay very close attention to the specific wording of the 'event question' in "USER_PROMPT".
# * If a date is provided in the 'event question' specifying when the event has to have occured, you must consider in your estimation, given the current time {timestamp}, how likely it is that the event will occur within the remaining timespan to that provided date.
# * If an item in "ADDITIONAL_INFORMATION" is not relevant for the estimation, you must ignore that item.
# * If there is insufficient information in "ADDITIONAL_INFORMATION", be aware of the limitations of your training data especially when relying on it for predicting events that require up-to-date information. In this case make a prediction that takes into account that you don't have up-to-date information.
# * Your pobability estimation must not only take into account if the specified event happens or not, but also if the event is likely to happen before, by or on the date specified in the 'event question'.
# * If there exist any information in "ADDITIONAL_INFORMATION" that is related to the 'event question' you can assume that you have been provided with up-to-date information that can be found on the internet.
# * If the 'event question' is formulated in a way that an event must have happend BY or ON a specific date, consider the timepoint of the event being 23:59:59 of that date. Decrease the probability of the event specified in the 'event question' happening the closer the current time {timestamp} is to the timepoint, if you could not find information that the event could happen within the remaining time. If the current time has exceeded the timepoint, decrease the probability to 0. Do this only if you have been provided with input under ADDITIONAL_INFORMATION that indicates that you have access to information that is up-to-date. If you have not been provided with such information, do not decrease the probability, but rather make a prediction that takes into account that you don't have up-to-date information.
# * You must provide your response in the format specified under "OUTPUT_FORMAT".
# * Do not include any other contents in your response.

# _______________________________________________________

# * If a timepoint date is specified in the 'event question', factor it into your probability estimate, assessing the likelihood of the event occurring within the timespan from the current time {timestamp} leading up to that date.
# * For events with a timepoint specified as BY or ON a particular date, treat 23:59:59 of that date as the cutoff. If no supporting information is found as the timepoint nears, lower the probability estimate accordingly. For reference, the current time is {timestamp}.
# * Prioritize the specific phrasing in the 'event question' as it holds crucial information, especially when a timepoint date is involved.
# * Your probability estimate should consider not only the event's occurrence but also its likelihood of happening before, on, or by the date in the 'event question'.
# no or only vague supporting information for the event occurring within the timepoint is found as the timepoint nears, lower the probability estimate exponentially. For reference, the current time is {timestamp}.

# Conditions for the event, often in the form of a specific timepoint date, are crucial for your probability assessment.


PREDICTION_PROMPT = """
You are a Large Language Model (LLM) operating within a multi-agent system. Your primary task is to precisely estimate the probability of the event occurring \
given the timepoint `{timepoint}`as specified in the following event question: "{event_question}". The event question has only two outcomes: the event either \
occurs before the timepoint or it does not occur before the timepoint.

You receive a list of information in the "ADDITIONAL_INFORMATION" section. Each entry in this list comes with timestamps in parenthesis, showing its initial \
release and last modification date. This information was obtained from a search engine query conducted a few seconds prior to your task, intended \
to be as current as possible for aiding in your probability estimation.

Note: Take extra care when interpreting dates. The date in the event question serves as the timepoint for the event to occur and is crucial for your probability assessment. \
Do not mix this up with the timestamps in "ADDITIONAL_INFORMATION," as those are meant to indicate the recency of that specific information. \
Mistaking these could lead to inaccurate probability assessments with significant financial consequences.

Strictly adhere to the 'INSTRUCTIONS' for a trustworthy and accurate probability estimation.

INSTRUCTIONS:
* Thoroughly scrutinize the event question: "{event_question}".
* If the event question permits more than two outcomes, return "Error" and discontinue further processing.
* Utilize your training data to formulate a probability estimate for the event occuring by or on the timepoint specified in the event question.
* Supplement your estimate with information from "ADDITIONAL_INFORMATION", paying special attention to the timestamps to gauge recency.
* If there exist any information in "ADDITIONAL_INFORMATION" that is related to the event question you can assume that you have been provided with most of the relevant information that can currently `{timestamp}` be found on the internet about that topic.
* Disregard any irrelevant items in "ADDITIONAL_INFORMATION".
* In case of information gaps in "ADDITIONAL_INFORMATION". be cognizant of your training data's limitations. Make an estimate acknowledging the absence of current data.
* It could be the case that there exist lots of supporting information for the event occurring some time but without specifying a date. Be aware that your task is to estimate the probability of the event occurring BY OR ON the timepoint `{timepoint}`. Always be aware of the current time `{timestamp}` and the remaining time until the timepoint.
* For the timepoint, treat 23:59:59 of the date within the event question as the cutoff. 
* Be neutral and unbiased in your probability estimation and ____________________________________________________-??????
* Adhere to the "OUTPUT_FORMAT" for your response, and refrain from including extraneous content.

ADDITIONAL_INFORMATION:
```
{additional_information}
```

USER_PROMPT:
{user_prompt}


OUTPUT_FORMAT:
* Your response should consist solely of a single JSON object, compatible with Python's "json.loads()" function.
* The JSON object must include four numerical fields: "p_yes," "p_no," "confidence," and "info_utility," each with values between 0 and 1.
   - "p_yes": The estimated probability of the event in the 'event question' taking place by or on the given date.
   - "p_no": The estimated likelihood of the event in the 'event question' not occurring by or on the given date.
   - "confidence": A measure of your assurance in the provided estimates, ranging from 0 for lowest to 1 for highest confidence.
   - "info_utility": A value indicating the usefulness of "ADDITIONAL_INFORMATION" in informing your estimate, ranging from 0 for no utility to 1 for maximum utility.
* Ensure that the sum of "p_yes" and "p_no" equals 1.
* Exclude any content other than this JSON object in your output.
"""
# , except for max three sentences explaining your reasoning.

URL_QUERY_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to formulate search engine queries based on \
a user's 'event question', which specifies an event and any accompanying conditions. The 'event question' allows \
only two outcomes: the event will either occur or not, given the conditions. Find the 'event question' under 'USER_PROMPT' \
and adhere to the 'INSTRUCTIONS'.

INSTRUCTIONS:
* Carefully read the 'event question' under 'USER_PROMPT', enclosed by triple backticks.
* If the 'event question' has more than two outcomes, respond with "Error" and ignore further instructions.
* Create a list of 1-4 unique search queries likely to yield relevant and contemporary information for assessing the event's likelihood under the given conditions.
* Each query must be unique, and they should not overlap or yield the same set of results.
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.

USER_PROMPT:
```
{event_question}
```

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain two fields: "queries", and "urls".
   - "queries": A 1-5 item array of the generated search engine queries.
* Include only the JSON object in your output.
"""

# Global constants for possible attribute names for release and update dates
RELEASE_DATE_NAMES = [
    'date', 'pubdate', 'publishdate', 'OriginalPublicationDate',
    'article:published_time', 'sailthru.date', 'article.published',
    'published-date', 'og:published_time', 'publication_date',
    'publishedDate', 'dc.date', 'DC.date', 'article:published',
    'article_date_original', 'cXenseParse:recs:publishtime', 'DATE_PUBLISHED',
    'pub-date', 'pub_date', 'datePublished', 'date_published',
    'time_published', 'article:published_date', 'parsely-pub-date',
    'publish-date', 'pubdatetime', 'published_time', 'publishedtime',
    'article_date', 'created_date', 'published_at', 'lastPublishedDate'
    'og:published_time', 'og:release_date', 'article:published_time',
    'og:publication_date', 'og:pubdate', 'article:publication_date',
    'product:availability_starts', 'product:release_date', 'event:start_date',
    'event:release_date', 'og:time_published', 'og:start_date', 'og:created',
    'og:creation_date', 'og:launch_date', 'og:first_published',
    'og:original_publication_date', 'article:published', 'article:pub_date',
    'news:published_time', 'news:publication_date', 'blog:published_time',
    'blog:publication_date', 'report:published_time', 'report:publication_date',
    'webpage:published_time', 'webpage:publication_date', 'post:published_time',
    'post:publication_date', 'item:published_time', 'item:publication_date'
]

UPDATE_DATE_NAMES = [
    'lastmod', 'lastmodified', 'last-modified', 'updated',
    'dateModified', 'article:modified_time', 'modified_date',
    'article:modified', 'og:updated_time', 'mod_date',
    'modifiedDate', 'lastModifiedDate', 'lastUpdate', 'last_updated',
    'LastUpdated', 'UpdateDate', 'updated_date', 'revision_date',
    'sentry:revision', 'article:modified_date', 'date_updated',
    'time_updated', 'lastUpdatedDate', 'last-update-date', 'lastupdate',
    'dateLastModified', 'article:update_time', 'modified_time',
    'last_modified_date', 'date_last_modified',
    'og:updated_time', 'og:modified_time', 'article:modified_time',
    'og:modification_date', 'og:mod_time', 'article:modification_date',
    'product:availability_ends', 'product:modified_date', 'event:end_date',
    'event:updated_date', 'og:time_modified', 'og:end_date', 'og:last_modified',
    'og:modification_date', 'og:revision_date', 'og:last_updated',
    'og:most_recent_update', 'article:updated', 'article:mod_date',
    'news:updated_time', 'news:modification_date', 'blog:updated_time',
    'blog:modification_date', 'report:updated_time', 'report:modification_date',
    'webpage:updated_time', 'webpage:modification_date', 'post:updated_time',
    'post:modification_date', 'item:updated_time', 'item:modification_date'
]

# Global constant for HTML tags to remove
HTML_TAGS_TO_REMOVE = [
    "script", "style", "header", "footer", "aside", "nav", "form", "button",
    "iframe", "input", "textarea", "select", "option", "label", "fieldset",
    "legend", "img", "audio", "video", "source", "track", "canvas", "svg",
    "object", "param", "embed", "link"
]


def search_google(query: str, api_key: str, engine: str, num: int = 3) -> List[str]:
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
    return [result["link"] for result in search["items"]]


def extract_event_date(doc_question) -> str:
    '''
    Extracts the event date from the event question if present.
    
    Args:
        doc_question (spaCy Doc): Document text as a spaCy Doc object.
        
    Returns:
        str: The event date in year-month-day format if present, otherwise None.
    '''
    
    event_date_ymd = None

    # Extract the date from the event question if present
    for ent in doc_question.ents:
        if ent.label_ == 'DATE':
            event_date_ymd = standardize_date(ent.text)

    # If event date not formatted as YMD or not found, return None
    if not datetime.strptime(event_date_ymd, '%Y-%m-%d') or event_date_ymd is None:
        return None
    else:
        return event_date_ymd
    


def get_max_tokens_for_additional_information(
    max_compl_tokens: int,
    prompt: str,
    enc: tiktoken.Encoding,
    safety_factor: float = 1.05,
) -> int:
    """
    Calculates the estimated maximum number of tokens that can be consumed by the additional information string.

    Args:
        max_compl_tokens (int): The maximum number of chat completion output tokens.
        prompt (str): The user prompt containing the event question.
        enc (tiktoken.Encoding): The tiktoken encoding to be used.
        safety_factor (float, optional): The safety factor to be used for prompt variations and message headers. Defaults to 1.05.

    Returns:
        int: The estimated number of tokens that can be consumed by the additional information string.
    """

    # Encode the strings into tokens
    user_prompt_enc = enc.encode(prompt)
    prediction_prompt_enc = enc.encode(PREDICTION_PROMPT)

    # Calculate token sum of thus far allocated tokens for the final prediction prompt
    token_sum = len(user_prompt_enc) + len(prediction_prompt_enc) + max_compl_tokens
    token_sum_safety = token_sum * safety_factor

    return int(MAX_TOTAL_TOKENS_CHAT_COMPLETION - token_sum_safety)


def truncate_additional_information(
    additional_informations: str,
    max_add_tokens: int,
    enc: tiktoken.Encoding,
) -> str:
    """
    Truncates additional information string to a specified number of tokens using tiktoken encoding.

    Args:
        additional_informations (str): The additional information string to be truncated.
        max_add_tokens (int): The maximum number of tokens allowed for the additional information string.
        enc (tiktoken.Encoding): The tiktoken encoding to be used.

    Returns:
    - str: The truncated additional information string.
    """

    # Encode the string into tokens
    add_enc = enc.encode(additional_informations)
    len_add_enc = len(add_enc)
    
    # Truncate additional information string if token sum exceeds maximum allowed
    if len_add_enc <= max_add_tokens:
        return additional_informations
    else:
        add_trunc_enc = add_enc[:-int(len_add_enc - max_add_tokens)]
        return enc.decode(add_trunc_enc)


def get_urls_from_queries(queries: List[str], api_key: str, engine: str, num: int = 3) -> List[str]:
    """
    Fetch unique URLs from search engine queries, limiting the number of URLs per query.
    
    Args:
        queries (List[str]): List of search engine queries.
        api_key (str): API key for the search engine.
        engine (str): Custom google search engine ID.
        num (int, optional): Number of returned URLs per query. Defaults to 3.

    Raises:
        ValueError: If the number of URLs per query exceeds the maximum allowed.
    
    Returns:
        List[str]: Unique list of URLs, omitting PDF and download-related URLs.
    """

    results = set()
    max_num_fetch = 10

    if num > max_num_fetch:
        raise ValueError(f"The maximum number of URLs per query is {max_num_fetch}.")

    for query in queries:
        fetched_urls = search_google(
            query=query,
            api_key=api_key,
            engine=engine,
            num=max_num_fetch  # Limit the number of returned URLs per query
        )

        # Add only unique URLs up to 'num' per query, omitting PDF and 'download' URLs
        count = 0
        for url in fetched_urls:
            if url not in results and not url.endswith(".pdf"):
                results.add(url)
                count += 1
                if count >= num:
                    break

    print("get_urls_from_queries result:")
    for url in results:
        print(url)

    return list(results)


def standardize_date(date_text):
    """
    Standardizes a given date string to the format 'YYYY-MM-DD' or 'MM-DD' if possible.
    
    Args:
        date_text (str): The date string to be standardized.

    Raises:
        ValueError: If the date string cannot be parsed.
    
    Returns:
        str: The standardized date string if possible, otherwise None.
    """

    try:
        # Compile regex patterns for month and day
        month_regex = re.compile(
            r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b',
            re.IGNORECASE
        )
        day_regex = re.compile(r'\b\d{1,2}\b')
        
        # Parse date_text using dateutil parser
        parsed_date = parser.parse(date_text)
        
        # Check if year, month, and day are in the original date_text
        month_exists = month_regex.search(date_text) is not None
        day_exists = day_regex.search(date_text) is not None
        year_exists = str(parsed_date.year) in date_text
        
        # Format the parsed date accordingly
        if year_exists and month_exists and day_exists:
            return parsed_date.strftime("%Y-%m-%d")
        elif month_exists and day_exists:
            return parsed_date.strftime("%m-%d")
        else:
            return None
    except Exception as e:
        return None


def get_context_around_isolated_event_date(
    doc_text, event_date_ymd, len_sentence_threshold, max_context=50
):
    """
    Extract sentences around isolated dates within the text.

    Args:
        doc_text (spaCy Doc): Document text as a spaCy Doc object.
        event_date_ymd (str): Event date in year-day-month format.
        len_sentence_threshold (int): Minimum number of words required for a sentence to be considered contextful.
        max_context (int, optional): Maximum number of words to include in the context. Defaults to 50.

    Raises:
        ValueError: If maximum context is less than threshold or greater than 100.

    Returns:
        list: List of sentences surrounding the target date.
    """

    # Check max_context value constraints
    if max_context < len_sentence_threshold:
        raise ValueError(
            f"The maximum number of words must be greater than or equal to the minimum number of words ({len_sentence_threshold}) required for a sentence to be considered contextful."
        )
    if max_context > 100:
        raise ValueError(
            f"The maximum number of words must be less than or equal to 300."
        )
    
    contexts_list = []
    len_doc_text = len(doc_text)
        
    # Extract the month and day from the event date
    event_date_md = event_date_ymd[5:]
    
    for ent in doc_text.ents:
        if ent.label_ == 'DATE':
            standardized_date = standardize_date(ent.text)
            if standardized_date is None:
                continue
            
            # Check if the entity matches the target date
            if standardized_date == event_date_ymd or standardized_date == event_date_md:
                sentence = next(
                    sent for sent in doc_text.sents 
                    if sent.start <= ent.start and sent.end >= ent.end
                )

                context_words = len(sentence.text.split())

                # Extend the context if the sentence is too short
                if context_words < len_sentence_threshold:
                    start_token, end_token = sentence.start, sentence.end
                    while context_words < max_context:
                        # Extend the context from the start of the sentence
                        new_start = start_token - 1
                        while new_start >= 0 and doc_text[new_start].is_sent_start is None:
                            new_start -= 1
                        if new_start >= 0:
                            context_words += len(doc_text[new_start:start_token].text.split())
                            start_token = new_start

                        # Break if max_context is reached
                        if context_words >= max_context:
                            break

                        # Extend the context from the end of the sentence
                        new_end = end_token + 1
                        while new_end < len_doc_text and doc_text[new_end].sent == sentence.sent:
                            new_end += 1
                        if new_end < len_doc_text:
                            context_words += len(doc_text[end_token:new_end].text.split())
                            end_token = new_end

                        # Break if max_context is reached
                        if context_words >= max_context:
                            break

                        # Break if max_context cannot be reached
                        if new_end == len_doc_text and start_token <= 0:
                            break
                    
                    context = doc_text[max(0, start_token):min(len_doc_text, end_token)].text
                    contexts_list.append(context)

    return contexts_list


def extract_relevant_information(
    text: str,
    query_emb,
    event_date: str,
    model,
    nlp,
    max_words: int
) -> str:
    """
    Extract relevant information from website text based on a given event question.

    Args:
        text (str): The website text to extract information from.
        event_question (str): The question to find relevant information to.
        event_date (str): Event date in year-day-month format.
        model: The BERT model for text embeddings.
        nlp: The spaCy NLP model.
        max_words (int): Maximum number of words allowed for output.

    Returns:
        str: The relevant sentences extracted from the website text.
    """        
    
    # Constants for sentence length and number thresholds
    len_sentence_threshold = 5
    num_sentences_threshold = 1000
    sentences = []     
    event_date_sentences = []
    seen = set()

    # Truncate text for performance optimization
    text = text[:50000]
    
    # Apply NLP pipeline to text
    doc_text = nlp(text)

    # Extract unique sentences
    for sent in doc_text.sents:
        sentence_text = sent.text
        if len(sentence_text.split()) >= len_sentence_threshold and sentence_text not in seen:
            sentences.append(sentence_text)
            seen.add(sentence_text)       
    sentences.extend(event_date_sentences)

    # Extract contextual sentences around event date occurences within too short sentences
    event_date_sentences.extend(
        get_context_around_isolated_event_date(
            doc_text, event_date, len_sentence_threshold, max_context=50
        )
    )

    if not sentences:
        return ""
    
    # Limit the number of sentences for performance optimization
    sentences = sentences[:num_sentences_threshold]
     
    # Encode event question calculate similarity scores
    sent_emb = model.encode(sentences)
    similarities = util.dot_score(query_emb, sent_emb)[0].cpu().tolist()

    # Extract top relevant sentences
    relevant_sentences = [
        sent for sent, sim in sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True) if sim > 0.4
    ]
    
    # # Print similarity scores along with the sentences
    # for sent, sim in sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True):
    #     if sim > 0.4:
    #         print(f"{sim:.4f}: {sent}")
    #         print()

    if not relevant_sentences:
        return ""
    
    # Truncate text to fit max_words limit
    output = ' '.join(relevant_sentences[:20]) 
    output_words = output.split(' ')
    if len(output_words) > max_words:
        output = ' '.join(output_words[:max_words])

    return output


def get_date(soup):    
    """
    Retrieves the release and modification dates from the soup object containing the HTML tree.
    
    Args:
        soup (BeautifulSoup): The BeautifulSoup object for the webpage.
        
    Returns:
        str: A string representing the release and modification dates.
    """
    
    release_date = "unknown"
    modified_date = "unknown"

    # Search for an update or modified date in the meta tags
    for name in UPDATE_DATE_NAMES:
        meta_tag = soup.find("meta", {"name": name}) or soup.find("meta", {"property": name})
        if meta_tag:
            modified_date = meta_tag.get("content", "")
            break
    
    # If not found, then look for release or publication date
    for name in RELEASE_DATE_NAMES:
        meta_tag = soup.find("meta", {"name": name}) or soup.find("meta", {"property": name})
        if meta_tag:
            release_date = meta_tag.get("content", "")
            break
    
    # Fallback to using the first time tag if neither release nor modified dates are found
    if release_date == "unknown" and modified_date == "unknown":
        time_tag = soup.find("time")
        if time_tag:
            release_date = time_tag.get("datetime", "")
           
    return f"({release_date}, {modified_date})"


def extract_text(
    html: str,
    query_emb,
    event_date: str,
    model,
    nlp,
    max_words: int,
) -> str:
    """
    Extract relevant information from HTML string.

    Args:
        html (str): The HTML content to extract text from.
        event_question (str): Event question for context.
        event_date (str): Event date in year-month-day format.
        model: Pre-trained model for sentence transformer.
        nlp: NLP object for additional text processing.
        max_words (int): Maximum number of words for the output summary.

    Raises:
        ValueError: If the HTML content is empty.
        ValueError: If the release or update date could not be extracted from the HTML.

    Returns:
        str: Relevant website information with release date.
    """

    if not html:
        raise ValueError("HTML is empty.")
    
    soup = BeautifulSoup(html, "html.parser")

    # Get the date of the website
    date = get_date(soup)
    if date is None:
        raise ValueError("Could not extract release or update date from HTML.")

    # Remove unnecessary tags to clean up text
    for element in soup(HTML_TAGS_TO_REMOVE):
        element.replace_with(NavigableString(' '))
    
    # Extract and clean text
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ". ".join(chunk for chunk in chunks if chunk)
    text = re.sub(r"\.{2,}", ".", text)

    # Get summarized text
    relevant_text = extract_relevant_information(
        text=text,
        query_emb=query_emb,
        event_date=event_date,
        model=model,
        nlp=nlp,
        max_words=max_words,
    )

    if not relevant_text:
        return ""

    return f"{date}: {relevant_text}"


def process_in_batches(
    urls: List[str], batch_size: int = 15, timeout: int = 10
) -> Generator[None, None, List[Tuple[Future, str]]]:
    """
    Process URLs in batches using a generator and thread pool executor.
    
    Args:
        urls (List[str]): List of URLs to process.
        batch_size (int, optional): Size of the processing batch_size. Default is 5.
        timeout (int, optional): Timeout for each request in seconds. Default is 10.

    Raises:
        ValueError: If the batch_size is less than or equal to zero.
        ValueError: If the timeout is less than or equal to zero.

    Yields:
        List[Tuple[Future, str]]: List containing Future objects and URLs for each batch.
    """

    if batch_size <= 0:
        raise ValueError("The 'batch_size' size must be greater than zero.")
    
    if timeout <= 0:
        raise ValueError("The 'timeout' must be greater than zero.")

    session = Session()
    session.max_redirects = 5

    # User-Agent headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/117.0'
    }
    session.headers.update(headers)

    # Using ThreadPoolExecutor to execute requests in parallel
    with ThreadPoolExecutor() as executor:
        # Loop through the URLs in batch_size of size 'batch_size'
        for i in range(0, len(urls), batch_size):
            batch = urls[i : i + batch_size]
            
            # Submit the batch of URLs for processing
            futures = []
            for url in batch:
                try:
                    # Submit a HEAD request to the url and check Content-Type
                    head_future = executor.submit(session.head, url, headers=headers, timeout=timeout, allow_redirects=True)
                    head_response = head_future.result()
                    if 'text/html' not in head_response.headers.get('Content-Type', ''):
                        continue
                    else:
                        # Submit a GET request to the url
                        futures.append((executor.submit(session.get, url, headers=headers, timeout=timeout), url))
                except Exception as e:
                    print(f"An error occurred: {e}")

            yield futures


def extract_texts(
    urls: List[str],
    event_question: str,
    max_words_per_url: int,
    nlp,
) -> List[str]:
    """
    Extract texts from a list of URLs using BERT and Spacy.
    
    Args:
        urls (List[str]): List of URLs to extract text from.
        event_question (str): Event-related question for text extraction.
        max_words_per_url (int): Maximum number of words allowed to extract for each URL.

    Raises:
        ValueError: If the event date could not be extracted from the event question.
        Timeout: If the request timed out.
    
    Returns:
        List[str]: List of extracted texts.
    """

    # Maximum number of allowed extractions
    max_allowed = 25
    
    # Initialize empty list for storing extracted texts
    extracted_texts = []
    
    # Initialize count and stop flag
    count = 0
    stop = False

    # Process the event question with spacy
    doc_question = nlp(event_question)
    event_date = extract_event_date(doc_question)

    # Initialize Sentence Transformer model
    model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')  
    
    # Create sentence embeddings for event question with Sentence Transformer
    query_emb = model.encode(event_question)

    if event_date is None:
        raise ValueError(f"Could not extract precise event date from event question: {event_question}")
    
    # Process URLs in batches
    for batch in process_in_batches(urls=urls):
        for future, url in tqdm(batch, desc="Processing URLs"):
            print(f"Processing {url}")
            try:
                result = future.result()
                if result.status_code != 200:
                    del result
                    continue
                # Extract relevant information for the event question
                extracted_text = extract_text(
                    html=result.text,
                    query_emb=query_emb,
                    event_date=event_date,
                    model=model,
                    nlp=nlp,
                    max_words=max_words_per_url,
                )
                
                # Delete the result object to free memory
                del result
                
                # Append the extracted text if available and increment the count
                if extracted_text:
                    # extracted_texts.append(f"{url}\n{extracted_text}")
                    extracted_texts.append(extracted_text)
                count += 1

                # Break if the maximum number of extractions is reached
                if count >= max_allowed:
                    stop = True
                    break
            
            except requests.exceptions.Timeout:
                print(f"Request for {url} timed out.")
            
            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()  # Print stack trace for debugging
        
        # Break if the maximum number of extractions is reached
        if stop:
            break
    
    return extracted_texts


def fetch_additional_information(
    event_question: str,
    max_add_words: int,
    google_api_key: str,
    google_engine: str,
    nlp,
    engine: str = "gpt-3.5-turbo",
    temperature: float = 1.0,
    max_compl_tokens: int = 500,
) -> str:

    """
    Get urls from a web search and extract relevant information based on an event question.
    
    Args:
        event_question (str): The question related to the event.
        max_add_words (int): The maximum number of words allowed for the additional information.
        google_api_key (str): The API key for the Google service.
        google_engine (str): The Google engine to be used.
        temperature (float): The temperature parameter for the engine.
        engine (str): The openai engine. Defaults to "gpt-3.5-turbo".
        temperature (float): The temperature parameter for the engine. Defaults to 1.0.
        max_compl_tokens (int): The maximum number of tokens for the engine's response.
        
    Returns:
        str: The relevant information fetched from all the URLs concatenated.
    """

    # Create URL query prompt
    url_query_prompt = URL_QUERY_PROMPT.format(event_question=event_question)
    
    # Perform moderation check
    moderation_result = openai.Moderation.create(url_query_prompt)
    if moderation_result["results"][0]["flagged"]:
        return "Moderation flagged the prompt as in violation of terms.", None
    
    # Create messages for the OpenAI engine
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": url_query_prompt},
    ]

    # Fetch queries from the OpenAI engine
    response = openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        temperature=temperature, # Override the default temperature parameter set for the engine
        max_tokens=max_compl_tokens, # Override the default max_compl_tokens parameter set for the engine
        n=1,
        timeout=90,
        request_timeout=90,
        stop=None,
    )
    
    # Parse the response content
    print(f"RESPONSE: {response}")
    json_data = json.loads(response.choices[0].message.content)
    # Print queries each on a new line
    print("QUERIES:\n")
    for query in json_data["queries"]:
        print(f"query: {query}\n")


    # Get URLs from queries
    urls = get_urls_from_queries(
        json_data["queries"],
        api_key=google_api_key,
        engine=google_engine,
    )
 
    # Get max number of words per URL
    max_words_per_url = max_add_words // len(urls) if len(urls) > 0 else 0

    # Extract texts from URLs
    texts = extract_texts(
        urls=urls,
        event_question=event_question,
        max_words_per_url=max_words_per_url,
        nlp=nlp,
    )

    # Join the texts and return
    additional_informations = "\n\n".join(["- " + text for text in texts])

    return additional_informations


def run(**kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Run the task with the given arguments.

    Args:
        kwargs (Dict): Keyword arguments that specify settings and API keys.

    Raises:
        ValueError: If the tool or prompt is not provided.
        ValueError: If the tool is not supported.
        ValueError: If the event question is not found in the prompt.

    Returns:
        Tuple[str, Optional[Dict[str, Any]]]: The generated content and any additional data.
    """

    tool = kwargs["tool"]
    prompt = kwargs["prompt"]
    max_compl_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_compl_tokens"])
    temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
    
    
    openai.api_key = kwargs["api_keys"]["openai"]
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"TOOL {tool} is not supported.")
    

    # Print the settings
    print(f"MECH TOOL: {tool}")
    print(f"PROMPT: {prompt}")
    print(f"MAX OPENAI RETURN TOKENS: {max_compl_tokens}")
    print(f"LLM TEMPERATURE: {temperature}")

    # Load the spacy model
    nlp = spacy.load("en_core_web_sm")
    
    # Get the LLM engine to be used
    engine = TOOL_TO_ENGINE[tool]
    print(f"ENGINE: {engine}")

    # Extract the event question from the prompt
    event_question = re.search(r"\"(.+?)\"", prompt).group(1)
    if not event_question:
        raise ValueError("No event question found in prompt.")
    print(f"EVENT_QUESTION: {event_question}")
    print()

    # Get the tiktoken base encoding
    enc = tiktoken.get_encoding("cl100k_base") 

    # Calculate the maximum number of tokens and words that can be consumed by the additional information string
    max_add_tokens = get_max_tokens_for_additional_information(
        max_compl_tokens=max_compl_tokens,
        prompt=prompt,
        enc=enc,
    )
    max_add_words = int(max_add_tokens * 0.75)

    # Fetch additional information
    additional_information = (
        fetch_additional_information(
            event_question=event_question,
            engine=engine,
            temperature=temperature,
            max_compl_tokens=max_compl_tokens,
            nlp=nlp,
            max_add_words=max_add_words,
            google_api_key=kwargs["api_keys"]["google_api_key"],
            google_engine=kwargs["api_keys"]["google_engine_id"],
        )
        if tool == "prediction-online-sum-url-content"
        else ""
    )

    # Truncate additional information to stay within the chat completion token limit of 4096
    additional_information = truncate_additional_information(
        additional_information, max_add_tokens, enc=enc,
    )

    # Get the current utc timestamp
    current_time_utc = datetime.now(timezone.utc)
    formatted_time_utc = current_time_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-6] + "Z"
 
    # Extract event date and format it to ISO 8601 with UTC timezone and 23:59:59 time
    doc_question = nlp(event_question)
    raw_event_date = extract_event_date(doc_question)
    parsed_event_date = datetime.strptime(raw_event_date, "%Y-%m-%d")
    final_event_date = parsed_event_date.replace(hour=23, minute=59, second=59, microsecond=0, tzinfo=timezone.utc)
    formatted_event_date = final_event_date.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-6] + "Z"
    
    # Generate the prediction prompt
    prediction_prompt = PREDICTION_PROMPT.format(
        event_question=event_question,
        timepoint=formatted_event_date,
        additional_information=additional_information,
        timestamp=formatted_time_utc,
    )
    print(f"\nPREDICTION PROMPT: {prediction_prompt}\n")

    # Perform moderation
    moderation_result = openai.Moderation.create(prediction_prompt)
    if moderation_result["results"][0]["flagged"]:
        return "Moderation flagged the prompt as in violation of terms.", None
    
    # Create messages for the OpenAI engine
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prediction_prompt},
    ]

    # Generate the response
    response = openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_compl_tokens,
        n=1,
        timeout=150,
        request_timeout=150,
        stop=None,
    )
    print(f"RESPONSE: {response}")

    return response.choices[0].message.content, None
