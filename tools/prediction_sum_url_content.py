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

import json
import re
from datetime import datetime
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Generator, List, Optional, Tuple
from tqdm import tqdm

import openai
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel, BertForPreTraining, BertForMaskedLM

import spacy
import torch

NUM_URLS_EXTRACT = 5
DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
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

PREDICTION_PROMPT = """
You are an LLM inside a multi-agent system. Your task is to estimate the probability of a user's 'event question', 
which specifies an event in the physical world and any accompanying conditions to be met for the 'event question' to be true. The 'event question' allows only two outcomes: the event 
will either occur or not, given the conditions. Find the 'event question' enclosed in double quotes as a part of 
the user's prompt under 'USER_PROMPT'. The user's prompt also contains a more elaborate description of the task. 
You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION", delimited by three backticks. 
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
* If there is insufficient information in "ADDITIONAL_INFORMATION", be aware of the limitations of your training data especially when relying on it for predicting events that require up-to-date information. So make a prediction that takes into account that you don't have up-to-date information.
* Your pobability estimation must not only take into account if the specified event happens or not, but also if the event is likely to happen before or on the date specified in the 'event question'.
* If the 'event question' is formulated in a way that an event must have happend by or before a specific date, consider the deadline of the event being 23:59:59 of that date. Decrease the probability of the event specified in the 'event question' happening the closer the current time {timestamp} is to the deadline, if you could not find information that the event could happen within the remaining time. If the remaining time is 0, decrease the probability to 0.
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
* Create a list of 1-5 unique search queries likely to yield relevant and contemporary information for assessing the event's likelihood under the given conditions.
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

def search_google(query: str, api_key: str, engine: str, num: int = 3) -> List[str]:
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


def get_urls_from_queries(queries: List[str], api_key: str, engine: str) -> List[str]:
    """Get URLs from search engine queries"""
    results = []
    for query in queries:
        for url in search_google(
            query=query,
            api_key=api_key,
            engine=engine,
            num=3,  # Number of returned urls per query
        ):
            results.append(url)
    unique_results = list(set(results))
    
    # Remove urls that are pdfs
    unique_results = [url for url in unique_results if not url.endswith(".pdf")]
    return unique_results


def get_website_summary(text: str, event_question: str, model, tokenizer, nlp, max_words: int = 150) -> str:
    """Get text summary from a website"""    
    # Check for empty inputs
    if not event_question or not text:
        return ""

    # Calculate the BERT embedding for the prompt
    with torch.no_grad():
        question_tokens = tokenizer(event_question, return_tensors="pt", padding=True, truncation=True)
        question_embedding = model(**question_tokens).last_hidden_state.mean(dim=1)
        
    # Sentence splitting and NER
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents if len(sent.text.split()) >= 5]
    entities = [ent.text for ent in doc.ents]
    
    # Crop the sentences list to the first 300 sentences to reduce the time taken for the similarity calculations.
    sentences = sentences[:300]

    # Similarity calculations and sentence ranking
    similarities = []

    # Batch the sentences to reduce the time taken for the similarity calculations
    batch_size = 32
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        with torch.no_grad():
            sentence_tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            sentence_embedding = model(**sentence_tokens).last_hidden_state.mean(dim=1)
            similarity = torch.cosine_similarity(question_embedding.repeat(len(batch), 1), sentence_embedding).tolist()
        
        for j, sent in enumerate(batch):
            if any(entity in sent for entity in entities):
                similarity[j] += 0.05
        similarities.extend(similarity)
    
    # Free up GPU memory
    del question_embedding, sentence_embedding
    torch.cuda.empty_cache()
        
    # Extract the top relevant sentences
    relevant_sentences = [sent for sent, sim in sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True) if sim > 0.9]

    # Print each sentence in relevant_sentences in a new line along with its similarity score > 0.7
    for sent, sim in sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True):
        if sim > 0.7:
            print(f"{sim} : {sent}\n")

    if len(relevant_sentences) == 0:
        return ""
    
    # Join the top 4 relevant sentences
    output = ' '.join(relevant_sentences[:4]) 
    output_words = output.split(' ')
    if len(output_words) > max_words:
        output = ' '.join(output_words[:max_words])

    return output


def get_date(soup):    
    # Get the updated or release date of the website.
    # The following are some of the possible values for the "name" attribute:
    release_date_names = [
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

    update_date_names = [
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

    release_date = "unknown"
    modified_date = "unknown"

    # First, try to find an update or modified date
    for name in update_date_names:
        meta_tag = soup.find("meta", {"name": name}) or soup.find("meta", {"property": name})
        if meta_tag:
            modified_date = meta_tag.get("content", "")
    
    # If not found, then look for release or publication date
    for name in release_date_names:
        meta_tag = soup.find("meta", {"name": name}) or soup.find("meta", {"property": name})
        if meta_tag:
            release_date = meta_tag.get("content", "")
    
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
) -> str:
    """Extract text from a single HTML document"""
    # Remove HTML tags and extract text
    soup = BeautifulSoup(html, "html.parser")

    # Get the date of the website
    date = get_date(soup)

    # Get the main element of the website
    # main_element = soup.find("main")
    # if main_element:
    #     soup = main_element

    for script in soup(["script", "style", "header", "footer", "aside", "nav", "form", "button", "iframe", "input", "textarea", "select", "option", "label", "fieldset", "legend", "img", "audio", "video", "source", "track", "canvas", "svg", "object", "param", "embed"]):
        script.extract()
    
    # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>< SOUP 1: \n{soup}")
    
    # for tag in soup.find_all():
    #     if tag.name not in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article', 'main', 'blockquote', 'ul', 'ol', 'li', 'strong', 'b', 'em', 'i', 'q', 'a', 'span', 'pre', 'code', 'time', 'abbr', 'section', 'div', 'figure', 'figcaption', 'mark']:
    #         tag.extract()

    # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>< SOUP 2: \n{soup}")
    



    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ". ".join(chunk for chunk in chunks if chunk)
    text = re.sub(r"\.{2,}", ".", text) # Use regex to replace multiple "."s with a single ".".
    # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>< TEXT: \n{text}")

    text_summary = get_website_summary(
        text=text,
        event_question=event_question,
        model=model,
        tokenizer=tokenizer,
        nlp=nlp,
    )
    return f"{date}:\n{text_summary}" if text_summary else ""


def process_in_batches(
    urls: List[str], window: int = 5, timeout: int = 10
) -> Generator[None, None, List[Tuple[Future, str]]]:
    """Iter URLs in batches."""
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(urls), window):
            batch = urls[i : i + window]
            futures = [(executor.submit(requests.get, url, timeout=timeout), url) for url in batch]
            yield futures


def extract_texts(
    urls: List[str],
    event_question: str,
) -> List[str]:
    """Extract texts from URLs"""
    max_allowed = 45
    extracted_texts = []
    count = 0
    stop = False
    
    # BERT Initialization
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Spacy Initialization for NER and sentence splitting
    nlp = spacy.load("en_core_web_sm")
    
    for batch in tqdm(process_in_batches(urls=urls), desc="Processing Batches"):
        for future, url in tqdm(batch, desc="Processing URLs"):
            try:
                result = future.result()
                if result.status_code != 200:
                    continue
                extracted_text = extract_text(
                    html=result.text,
                    event_question=event_question,
                    model=model,
                    tokenizer=tokenizer,
                    nlp=nlp,
                )
                if extracted_text:
                    extracted_texts.append(f"{url}\n{extracted_text}")
                count += 1
                if count >= max_allowed:
                    stop = True
                    break
            except requests.exceptions.ReadTimeout:
                print(f"Request timed out: {url}.")
            except Exception as e:
                print(f"An error occurred: {e}")
        if stop:
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
    """Fetch additional information."""
    url_query_prompt = URL_QUERY_PROMPT.format(event_question=event_question)
    moderation_result = openai.Moderation.create(url_query_prompt)
    if moderation_result["results"][0]["flagged"]:
        return ""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": url_query_prompt},
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=max_tokens,
        n=1,
        timeout=90,
        request_timeout=90,
        stop=None,
    )
    
    json_data = json.loads(response.choices[0].message.content)
    print(f"json_data: {json_data}")
    urls = get_urls_from_queries(
        json_data["queries"],
        api_key=google_api_key,
        engine=google_engine,
    )
    print(f"urls: {urls}")
    texts = extract_texts(
        urls=urls,
        event_question=event_question,
    )
    additional_informations = "\n\n".join(["- " + text for text in texts])
    # print(f"additional_informations: {additional_informations}")
    return additional_informations


def run(**kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Run the task"""
    print("Starting...")
    
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]
    max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
    temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
    
    print(f"Tool: {tool}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")

    openai.api_key = kwargs["api_keys"]["openai"]
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")

    engine = TOOL_TO_ENGINE[tool]
    print(f"Engine: {engine}")

    # Event question is the text between the first pair of double quotes in the prompt
    event_question = re.search(r"\"(.+?)\"", prompt).group(1)
    print(f"event_question: {event_question}")

    # Make an openai request to get similar formulations of the event question and store them in a list
    similar_formulations = []

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

    # Get today's date and generate the prediction prompt
    timestamp = datetime.now().strftime('%Y-%m-%d')
    prediction_prompt = PREDICTION_PROMPT.format(
        user_prompt=prompt, additional_information=additional_information, timestamp=timestamp,
    )
    print(f"prediction_prompt: {prediction_prompt}\n")

    moderation_result = openai.Moderation.create(prediction_prompt)
    if moderation_result["results"][0]["flagged"]:
        return "Moderation flagged the prompt as in violation of terms.", None
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prediction_prompt},
    ]

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
    print(f"response: {response}")
    return response.choices[0].message.content, None
