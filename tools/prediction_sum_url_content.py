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

from typing import Any, Dict, Generator, List, Optional, Tuple
from datetime import datetime
import time
import json
import re
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import openai
import requests
from requests import Session
from requests.adapters import HTTPAdapter
import spacy
import tiktoken
import torch

from dateutil import parser
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel, BertForPreTraining, BertForMaskedLM
from urllib3.util.retry import Retry

NUM_URLS_EXTRACT = 5
MAX_TOTAL_TOKENS_CHAT_COMPLETION = 4096
DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 200,
    "temperature": 0.2,
}

ALLOWED_TOOLS = [
    "prediction-offline-sum-url-content",
    "prediction-online-sum-url-content",
]
TOOL_TO_ENGINE = {
    "prediction-offline-sum-url-content": "gpt-3.5-turbo",
    "prediction-online-sum-url-content": "gpt-3.5-turbo",
    # "prediction-online-sum-url-content": "gpt-3.5-turbo-16k",
    # "prediction-online-sum-url-content": "gpt-4",
}

PREDICTION_PROMPT = """
You are an LLM inside a multi-agent system. Your task is to estimate the probability of a user's 'event question', 
which specifies an event in the physical world and any accompanying conditions to be met for the 'event question' to be true. The 'event question' allows only two outcomes: the event 
will either occur or not, given the conditions. Find the 'event question' enclosed in double quotes as a part of 
the user's prompt under 'USER_PROMPT'. The user's prompt also contains a more elaborate description of the task. 
You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION", delimited by three backticks. This information results from a search engine query that has been done a few seconds ago with the aim to get up-to-date information that could be relevant estimating the 'event question'.
You must adhere to the 'INSTRUCTIONS'.  


INSTRUCTIONS:
* Carefully read the user's prompt under 'USER_PROMPT', enclosed by triple backticks.
* If the 'event question' has more than two outcomes, respond with "Error" and ignore further instructions.
* Based on your training data, provide a probability estimation of the event specified in the 'event question' occuring, considering all conditions provided.
* You can use any item in "ADDITIONAL_INFORMATION" in addition to your training data to make the probability estimation.
* Prioritize recent information in "ADDITIONAL_INFORMATION" based on the current time {timestamp}.
* You must pay very close attention to the specific wording of the 'event question' in "USER_PROMPT".
* If a date is provided in the 'event question' specifying when the event has to have occured, you must consider in your estimation, given the current time {timestamp}, how likely it is that the event will occur within the remaining timespan to that provided date.
* If an item in "ADDITIONAL_INFORMATION" is not relevant for the estimation, you must ignore that item.
* If there is insufficient information in "ADDITIONAL_INFORMATION", be aware of the limitations of your training data especially when relying on it for predicting events that require up-to-date information. In this case make a prediction that takes into account that you don't have up-to-date information.
* Your pobability estimation must not only take into account if the specified event happens or not, but also if the event is likely to happen before, by or on the date specified in the 'event question'.
* If there exist any information in "ADDITIONAL_INFORMATION" that is related to the 'event question' you can assume that you have been provided with up-to-date information that can be found on the internet.
* If the 'event question' is formulated in a way that an event must have happend BY or ON a specific date, consider the deadline of the event being 23:59:59 of that date. Decrease the probability of the event specified in the 'event question' happening the closer the current time {timestamp} is to the deadline, if you could not find information that the event could happen within the remaining time. If the current time has exceeded the deadline, decrease the probability to 0. Do this only if you have been provided with input under ADDITIONAL_INFORMATION that indicates that you have access to information that is up-to-date. If you have not been provided with such information, do not decrease the probability, but rather make a prediction that takes into account that you don't have up-to-date information.
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.

USER_PROMPT:
```
{user_prompt}
```

ADDITIONAL_INFORMATION:
```
{additional_information}
```

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain four fields: "p_yes", "p_no", "confidence", and "info_utility", each ranging from 0 to 1.
   - "p_yes": Estimated probability that the event specified in the 'event question' occurs, considering all conditions provided.
   - "p_no": Estimated probability that the 'event question' does not occur, considering all conditions provided.
   - "confidence": Indicating the confidence in the estimated probabilities you provided ranging from 0 (lowest confidence) to 1 (maximum confidence).
   - "info_utility": Utility of the information provided in "ADDITIONAL_INFORMATION" to help you make the prediction ranging from 0 (lowest utility) to 1 (maximum utility).
* The sum of "p_yes" and "p_no" must equal 1.
* Output only the JSON object in your response.
"""

URL_QUERY_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to formulate search engine queries based on 
a user's 'event question', which specifies an event and any accompanying conditions. The 'event question' allows 
only two outcomes: the event will either occur or not, given the conditions. Find the 'event question' under 'USER_PROMPT' 
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
    "object", "param", "embed"
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


def truncate_additional_information(
    additional_informations: str,
    max_tokens: int,
    prompt: str,
    enc: tiktoken.Encoding,
    safety_factor: float = 1.05,
) -> str:
    """
    Truncates additional information string to a specified number of tokens using tiktoken encoding.

    Parameters:
        additional_informations (str): The additional information string to be truncated.
        max_tokens (int): The maximum number of chat completion output tokens.
        prompt (str): The user prompt containing the event question.
        enc (tiktoken.Encoding): The tiktoken encoding to be used.
        safety_factor (float, optional): The safety factor to be used for truncation. Defaults to 1.05.

    Returns:
    - str: The truncated additional information string.
    """

    # Encode the strings into tokens
    additional_information_token_enc = enc.encode(additional_informations)
    user_prompt_tokens_token_enc = enc.encode(prompt)
    prediction_prompt_tokens_token_enc = enc.encode(PREDICTION_PROMPT)

    print("Max Total Tokens:", MAX_TOTAL_TOKENS_CHAT_COMPLETION)
    print("Number of tokens in additional informations:", len(additional_information_token_enc))
    print("Number of tokens in user prompt:", len(user_prompt_tokens_token_enc))
    print("Number of tokens in prediction prompt:", len(prediction_prompt_tokens_token_enc))
    print("Number of tokens reserved for chat completion output:", max_tokens)

    # Calculate the rough token sum of final prediction prompt
    prompt_token_sum = len(additional_information_token_enc) + len(user_prompt_tokens_token_enc) + len(prediction_prompt_tokens_token_enc) + max_tokens
    print(f"Total number of tokens in prompt: {prompt_token_sum}")
    prompt_token_sum_safety_factor = prompt_token_sum * safety_factor
    print(f"Total number of tokens in prompt with safety factor: {prompt_token_sum_safety_factor}")

    if prompt_token_sum_safety_factor > MAX_TOTAL_TOKENS_CHAT_COMPLETION:
        num_tokens_to_truncate = prompt_token_sum_safety_factor - MAX_TOTAL_TOKENS_CHAT_COMPLETION
        print(f"Truncating additional information by {num_tokens_to_truncate} tokens.")

        # Truncate the additional informations tokens
        truncated_additional_informations_token = additional_information_token_enc[:-int(num_tokens_to_truncate)]
        print(f"Number of tokens in truncated additional informations: {len(truncated_additional_informations_token)}")

        # Decode the truncated tokens back into text
        truncated_additional_informations_string = enc.decode(truncated_additional_informations_token)
        return truncated_additional_informations_string
    else:
        return additional_informations


def get_urls_from_queries(queries: List[str], api_key: str, engine: str, num: int = 3) -> List[str]:
    """
    Fetch unique URLs from search engine queries, limiting the number of URLs per query.
    
    Args:
        queries (List[str]): List of search engine queries.
        api_key (str): API key for the search engine.
        engine (str): Search engine to be used.
        num (int, optional): Number of returned URLs per query. Defaults to 3.

    Raises:
        ValueError: If the number of URLs per query exceeds the maximum allowed.
    
    Returns:
        List[str]: Unique list of URLs, omitting PDF and download-related URLs.
    """

    results = set()
    max_num = 10

    if num > max_num:
        raise ValueError(f"The maximum number of URLs per query is {max_num}.")

    for query in queries:
        fetched_urls = search_google(
            query=query,
            api_key=api_key,
            engine=engine,
            num=max_num  # Limit the number of returned URLs per query
        )

        # Add only unique URLs up to 'num' per query, omitting PDF and 'download' URLs
        count = 0
        for url in fetched_urls:
            results.add(url)
            count += 1
            if count >= num:
                break
            
            # if "download" not in url.lower() and url not in results and not url.endswith(".pdf"):
            #     results.add(url)
            #     count += 1
            #     if count >= num:
            #         break
    return list(results)


def standardize_date(date_text):
    """
    Standardizes a given date string to the format 'YYYY-MM-DD' or 'MM-DD' if possible.
    
    Args:
        date_text (str): The date string to be standardized.
    
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


def get_context_around_isolated_dates(
    doc_text, target_date_ydm, len_sentence_threshold, max_context=50
):
    """
    Extract context around isolated dates within the text.

    Args:
        doc_text (spaCy Doc): Document text as a spaCy Doc object.
        target_date_ydm (str): Target date in year-day-month format.
        len_sentence_threshold (int): Minimum number of words required for a sentence to be considered contextful.
        max_context (int, optional): Maximum number of words to include in the context. Defaults to 50.

    Raises:
        ValueError: If max_context is less than len_sentence_threshold or greater than 300.

    Returns:
        list: List of sentences surrounding the target date.
    """

    # Check max_context value constraints
    if max_context < len_sentence_threshold:
        raise ValueError(
            f"The maximum number of words must be greater than or equal to the minimum number of words ({len_sentence_threshold}) required for a sentence to be considered contextful."
        )
    if max_context > 300:
        raise ValueError(
            f"The maximum number of words must be less than or equal to 300."
        )

    contexts_list = []
    target_date_dm = target_date_ydm[5:]
    len_doc_text = len(doc_text)

    for ent in doc_text.ents:
        if ent.label_ == 'DATE':
            standardized_date = standardize_date(ent.text)
            if standardized_date is None:
                continue

            # Check if the entity matches the target date
            if standardized_date == target_date_ydm or standardized_date == target_date_dm:
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
                    print(f"Successfully extracted context for isolated date {target_date_ydm}: {context}\n")
                    contexts_list.append(context)

    return contexts_list


def get_sentence_embeddings_and_similarities(
    sentences: List[str],
    question_embedding: torch.Tensor,
    model,
    tokenizer,
    batch_size: int = 32
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Calculate the sentence embeddings and similarities.

    Args:
        sentences (List[str]): List of sentences to compare.
        question_embedding (torch.Tensor): Tensor of the question embedding.
        model: The BERT model for text embeddings.
        tokenizer: The tokenizer for the BERT model.
        batch_size (int, optional): Number of sentences to process in each batch. Defaults to 32.

    Raises:
        ValueError: If batch_size is less than 1.

    Returns:
        Tuple[List[torch.Tensor], List[float]]: List of sentence embeddings and their similarities.    
    """

    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    similarities = []
 
    # Repeat the question embedding tensor to match the batch size
    question_embedding_repeated = question_embedding.repeat(batch_size, 1)

    # Print the number of sentences
    # print(f"Number of sentences: {len(sentences)}")

    # Print the number of batches
    # print(f"Number of batches to create: {len(sentences) // batch_size + 1}")

    # Batch the sentences for efficient processing
    sentence_batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
    
    # print(f"Number of batches created: {len(sentence_batches)}")
    # Number of sentences in each batch
    # print(f"Number of sentences in all batches except the last: {len(sentence_batches[:-1])}")
    
    # print(f"Number of sentences in the last batch: {len(sentence_batches[-1])}")
    
    for batch in tqdm(sentence_batches, desc="Calculating sentence similarities"):
        # Adjust the repeated question embedding if the batch size changes
        actual_batch_size = len(batch)
        if actual_batch_size != batch_size:
            question_embedding_repeated = question_embedding.repeat(actual_batch_size, 1)
        try:
            with torch.no_grad():
                # Tokenize and preprocess sentence batch
                sentence_tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                # Compute sentence embeddings
                sentence_embedding = model(**sentence_tokens).last_hidden_state.mean(dim=1)
                # Compute cosine similarities
                similarity = torch.cosine_similarity(question_embedding_repeated, sentence_embedding).tolist()
            similarities.extend(similarity)
        finally:
            # Free up GPU memory
            del sentence_tokens, sentence_embedding, similarity

    return similarities


def get_website_summary(
    text: str,
    event_question: str,
    model,
    tokenizer,
    nlp,
    max_words: int
) -> str:
    """
    Generate a summary of a website's text based on a given event question.

    Args:
        text (str): The website text to summarize.
        event_question (str): The question to focus the summary on.
        model: The BERT model for text embeddings.
        tokenizer: The tokenizer for the BERT model.
        nlp: The spaCy NLP model.
        max_words (int, optional): Maximum number of words for the output summary. Defaults to 200.

    Raises:
        ValueError: If max_words is less than 1 or greater than 300.

    Returns:
        str: The generated summary.
    """        
    
    # Constants for sentence length and number thresholds
    len_sentence_threshold = 5
    num_sentences_threshold = 100
    event_date_sentences = []
    
    # Validate inputs
    if not event_question or not text:
        return ""

    # Calculate the BERT embedding for the event question
    with torch.no_grad():
        question_tokens = tokenizer(event_question, return_tensors="pt", padding=True, truncation=True)
        question_embedding = model(**question_tokens).last_hidden_state.mean(dim=1)
    
    # Truncate text to stay within nlp character limit of 1,000,000
    text = text[:1000000]
    
    # Apply NLP pipeline to text and event question
    doc_text = nlp(text)
    doc_question = nlp(event_question)

    # Extract the date from the event question if present
    for ent in doc_question.ents:
        if ent.label_ == 'DATE':
            event_date_ydm = standardize_date(ent.text)

    # Extract contextual sentences around isolated dates
    if event_date_ydm is not None:
        event_date_sentences.extend(
            get_context_around_isolated_dates(doc_text, event_date_ydm, len_sentence_threshold, max_context=50)
        )

    seen = set()
    sentences = []

    # Extract unique and sufficiently long sentences
    for sent in doc_text.sents:
        sentence_text = sent.text
        if len(sentence_text.split()) >= len_sentence_threshold and sentence_text not in seen:
            sentences.append(sentence_text)
            seen.add(sentence_text)       
    sentences.extend(event_date_sentences)

    # Limit the number of sentences for performance
    sentences = sentences[:num_sentences_threshold]

    # Calculate sentence similarities
    similarities = get_sentence_embeddings_and_similarities(
        sentences, question_embedding, model, tokenizer, batch_size=16
    )

    # Extract top relevant sentences
    relevant_sentences = [
        sent for sent, sim in sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True) if sim > 0.9
    ]

    # # Print sentences and similarities if similarity is greater than 0.9
    # for sent, sim in sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True):
    #     if sim > 0.9:
    #         print(f"Similarity: {sim}\nSentence: {sent}\n")
    #         print()

    if not relevant_sentences:
        return ""
    
    # Truncate summary to fit max_words limit
    output = ' '.join(relevant_sentences[:10]) 
    output_words = output.split(' ')
    if len(output_words) > max_words:
        output = ' '.join(output_words[:max_words])

    return output


def get_date(soup):    
    """
    Retrieves the release and modification dates from the soup object containing the text of the website.
    
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
           
    return f"Release date {release_date}, Modified date {modified_date}"


def extract_text(
    html: str,
    event_question: str,
    model,
    tokenizer,
    nlp,
    max_words: int,
) -> str:
    """
    Extract relevant information from HTML string.

    Args:
        html (str): The HTML content to extract text from.
        event_question (str): Event question for context.
        model: Pre-trained model for text summarization.
        tokenizer: Tokenizer for the pre-trained model.
        nlp: NLP object for additional text processing.

    Raises:
        ValueError: If the HTML content is empty.

    Returns:
        str: Summarized text with the date.
    """

    if not html:
        raise ValueError("HTML is empty.")

    soup = BeautifulSoup(html, "html.parser")

    # Get the date of the website
    date = get_date(soup)
    if date is None:
        raise ValueError("Could not extract date from the HTML")

    # Remove unnecessary tags to clean up text
    for script in soup(HTML_TAGS_TO_REMOVE):
        script.extract()
    
    # Extract and clean text
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ". ".join(chunk for chunk in chunks if chunk)
    text = re.sub(r"\.{2,}", ".", text)

    # Get summarized text
    text_summary = get_website_summary(
        text=text,
        event_question=event_question,
        model=model,
        tokenizer=tokenizer,
        nlp=nlp,
        max_words=max_words,
    )

    if not text_summary:
        return ""

    return f"{date}:\n{text_summary}"


def process_in_batches(
    urls: List[str], batch_size: int = 5, timeout: int = 10
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

    # Set up retry logic
    retries = Retry(
        total=2,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

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
            futures = [
                (executor.submit(session.get, url, headers=headers, timeout=timeout), url) for url in batch
            ]
            yield futures


def extract_texts(
    urls: List[str],
    event_question: str,
) -> List[str]:
    """
    Extract texts from a list of URLs using BERT and Spacy.
    
    Parameters:
    urls (List[str]): List of URLs to extract text from.
    event_question (str): Event-related question for text extraction.
    
    Returns:
    List[str]: List of extracted texts.
    """

    # Maximum number of allowed extractions
    max_allowed = 25

    # Maximum number of words for each extraction
    # ~ 2642 tokens free for additional information ~ 1981 words
    # split by number of URLs
    max_words = 1981 // len(urls)
    
    # Initialize empty list for storing extracted texts
    extracted_texts = []
    
    # Initialize count and stop flag
    count = 0
    stop = False

    # Initialize BERT and Spacy models
    model = AutoModel.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    nlp = spacy.load("en_core_web_sm")
    
    # Process URLs in batches
    for batch in process_in_batches(urls=urls):
        for future, url in tqdm(batch, desc="Processing URLs"):
            try:
                result = future.result()
                print(f"\nURL: {url}")
                print(f"Status code: {result.status_code}")
                print(f"Content type: {result.headers.get('content-type')}\n")
                if result.status_code != 200:
                    del result
                    continue
                
                # Extract relevant information for the event question
                extracted_text = extract_text(
                    html=result.text,
                    event_question=event_question,
                    model=model,
                    tokenizer=tokenizer,
                    nlp=nlp,
                    max_words=max_words,
                )
                # print(f"extracted_text: {extracted_text}")
                
                # Delete the result object to free memory
                del result
                
                # Append the extracted text if available and increment the count
                if extracted_text:
                    extracted_texts.append(f"{url}\n{extracted_text}")
                count += 1
                # print(f"extracted_texts: {extracted_texts}\n")
                print(f"count: {count}\n")

                # Break if the maximum number of extractions is reached
                if count >= max_allowed:
                    stop = True
                    print(f"Maximum number of extractions reached: {max_allowed}.")
                    break

            except requests.exceptions.ReadTimeout:
                print(f"Request timed out: {url}.")
            
            except requests.exceptions.Timeout:
                print(f"Request for {url} timed out.")
            
            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()  # Print stack trace for debugging
        
        # Break if the maximum number of extractions is reached
        if stop:
            print(f"Maximum number of extractions reached: {max_allowed}.")
            break

    return extracted_texts


def fetch_additional_information(
    event_question: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    google_api_key: str,
    google_engine: str,
) -> str:

    """
    Fetch additional information based on an event question.
    
    Args:
        event_question (str): The question related to the event.
        engine (str): The engine to be used for fetching information.
        temperature (float): The temperature parameter for the engine.
        max_tokens (int): The maximum number of tokens for the engine's response.
        google_api_key (str): The API key for the Google service.
        google_engine (str): The Google engine to be used.
        
    Returns:
        str: The additional information fetched.
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
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0, # Override the default temperature parameter set for the engine
        max_tokens=500, # Override the default max_tokens parameter set for the engine
        n=1,
        timeout=90,
        request_timeout=90,
        stop=None,
    )
    
    # Parse the response content
    json_data = json.loads(response.choices[0].message.content)
    print(f"json_data: {json_data}")

    # Get URLs from queries
    urls = get_urls_from_queries(
        json_data["queries"],
        api_key=google_api_key,
        engine=google_engine,
    )
    for url in urls:
        print(f"url: {url}")

    # Extract texts from URLs
    texts = extract_texts(
        urls=urls,
        event_question=event_question,
    )

    # Join the texts and return
    additional_informations = "\n\n".join(["- " + text for text in texts])

    return additional_informations


def run(**kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Run the task with the given parameters.

    Args:
        kwargs (Dict): Keyword arguments that specify settings and API keys.

    Raises:
        ValueError: If the tool or prompt is not provided.
        ValueError: If the tool is not supported.
        ValueError: If the event question is not found in the prompt.

    Returns:
        Tuple[str, Optional[Dict[str, Any]]]: The generated content and any additional data.
    """

    print("Starting...")
    print()

    tool = kwargs["tool"]
    prompt = kwargs["prompt"]
    max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
    temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
    
    if not tool or not prompt:
        raise ValueError("Both 'mech tool' and 'prompt' must be provided.")

    # Print the settings
    print(f"MECH TOOL: {tool}")
    print(f"PROMPT: {prompt}")
    print(f"MAX OPENAI RETURN TOKENS: {max_tokens}")
    print(f"LLM TEMPERATURE: {temperature}")

    openai.api_key = kwargs["api_keys"]["openai"]
    
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"TOOL {tool} is not supported.")

    # Get the LLM engine to be used
    engine = TOOL_TO_ENGINE[tool]
    print(f"ENGINE: {engine}")

    # Extract the event question from the prompt
    event_question = re.search(r"\"(.+?)\"", prompt).group(1)
    if not event_question:
        raise ValueError("No event question found in prompt.")
    print(f"EVENT_QUESTION: {event_question}")
    print()

    # Fetch additional information
    additional_information = (
        fetch_additional_information(
            event_question=event_question,
            engine=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            google_api_key=kwargs["api_keys"]["google_api_key"],
            google_engine=kwargs["api_keys"]["google_engine_id"],
        )
        if tool == "prediction-online-sum-url-content"
        else ""
    )

    start_time = time.time()
    # # Truncate additional information to stay within the chat completion token limit of 4096
    enc = tiktoken.get_encoding("cl100k_base") # Get the tiktoken base encoding
    additional_information = truncate_additional_information(
        additional_information, 
        max_tokens,
        prompt=prompt,
        enc=enc,
    )
    end_time = time.time()
    print(f"Time taken to truncate additional information: {end_time - start_time} seconds.")

    # Generate the prediction prompt
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prediction_prompt = PREDICTION_PROMPT.format(
        user_prompt=prompt, additional_information=additional_information, timestamp=timestamp,
    )
    print(f"PREDICTION PROMPT: {prediction_prompt}\n")

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
        max_tokens=max_tokens,
        n=1,
        timeout=150,
        request_timeout=150,
        stop=None,
    )
    print(f"RESPONSE: {response}")

    return response.choices[0].message.content, None
