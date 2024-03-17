"""This module implements a research agent for extracting relevant information from URLs."""

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup
from docstring_parser import parse
import faiss
from googleapiclient.discovery import build
from itertools import islice
import json
import numpy as np
import tiktoken
from io import BytesIO
import PyPDF2
from openai import OpenAI
from pydantic import BaseModel, Field
from readability import Document as ReadabilityDocument
import requests
from requests import Session
from requests.exceptions import RequestException, TooManyRedirects
# from requests.packages.urllib3.util.retry import Retry
from markdownify import markdownify as md
from typing import Any, Dict, Generator, List, Optional, Tuple, Callable
from tiktoken import encoding_for_model
import html2text
from readability import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


from openai import OpenAI
from tqdm import tqdm


from dateutil import parser

# from typing import Any, Dict, Generator, List, Optional, Tuple
# from datetime import datetime, timezone, timedelta
# import time
# import json
# import re
# import os
# #import html2text
# from readability import Document
# from concurrent.futures import Future, ThreadPoolExecutor
# from itertools import groupby
# from operator import itemgetter

# from bs4 import BeautifulSoup, NavigableString
# from googleapiclient.discovery import build
# from langchain.llms import Ollama
# from langchain.pydantic_v1 import BaseModel, Field
# from langchain.tools import BaseTool, StructuredTool, tool
# # from langchain_core.callbacks import (
# #     AsyncCallbackManagerForToolRun,
# #     CallbackManagerForToolRun,
# # )


# from urllib.parse import urlparse
# from typing import Optional, Type
# import requests
# from requests import Session
# import spacy
# import spacy.util
# import tiktoken

# from dateutil import parser
# from tqdm import tqdm


#NUM_URLS_EXTRACT = 5
NUM_URLS_PER_QUERY = 3
SPLITTER_CHUNK_SIZE = 1800
SPLITTER_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_BATCH_SIZE = 1000
EMBEDDING_SIZE = 1536
MAX_TOKENS_ADDITIONAL_INFORMATION = 2000
WORDS_PER_TOKEN_FACTOR = 0.75
DEFAULT_OPENAI_SETTINGS = {
    "max_compl_tokens": 500,
    "temperature": 0,
}

ALLOWED_TOOLS = [
    "prediction-sentence-embedding-conservative",
    "prediction-sentence-embedding-bold",
]
TOOL_TO_ENGINE = {
    "prediction-sentence-embedding-conservative": "gpt-3.5-turbo",
    "prediction-sentence-embedding-bold": "gpt-4",
}


QUERIES_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to formulate search engine queries based on \
an input query, which specifies an event and any accompanying conditions. Find the input query \
under 'INPUT_QUERY' and adhere to the 'INSTRUCTIONS'.

INSTRUCTIONS:
* Carefully read the input query under 'USER_PROMPT', enclosed by triple backticks.
* Create a list of 5 unique search queries likely to yield relevant and contemporary information about the event.
* Each query must be unique, and they should not overlap or yield the same set of results.
* One query must be phrased exactly as the input query itself.
* The two queries should be phrased as negations of the input query and two other queries should be phrased as affirmations.
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.

INPUT_QUERY:
```
{input_query}
```

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()"
* The JSON must contain one field: "queries"
   - "queries": A 5 item array of the generated search engine queries
* Include only the JSON object in your output
* Do not include any formatting characters in your response!
* Do not include any other contents or explanations in your response!
"""

# Select those web pages that are likely to contain relevant and current information to answer the question. \
# ... return ... only for the selected relevant web pages.

URL_RERANKING_PROMPT = """
I will present you with a collection of web pages along with their titles, descriptions, release dates and publishers. \
These web pages were preselected for probably containing relevant information to answer the question: {market_question}. \

The web pages are divided by '---web_page---'

Evaluate the web pages details (publication dates, titles, descriptions and publishers) for relevance to answer the question. \
Consider content, recency of information and publisher in your relevance evaluation. \
Content can be considered relevant if it contains information that could help answer the question. \
Recent information can be considered more relevant than older information. For this consider that the current date is {current_date} \
A reputable publisher can be considered more relevant than a less reputable or unknown publisher. \

Rank the web pages in descending order of relevance and return the ranked list of its web page IDs. \

WEB_PAGES:
{web_pages_info}

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()"
* The JSON must contain two fields: "explanation" and "ranked_web_page_ids"
    - "relevance_evaluation": A dict of web page IDs along with their evaluated relevance a scale between 1 and 10 and a one sentence explanation
    - "ranked_web_page_ids": A list of web page IDs ranked by relevance of their corresponding web pages
* Include only the JSON object in your output
* Do not include any formatting characters in your response!
* Do not include any other contents or explanations in your response!
"""


REDUCER_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to drop sentences in 'SEARCH_OUTPUT' \
without modifying or rephrasing the remaining sentences in any way. The remaining and unchanged sentences inside the paragraphs \
must contain relevant information with respect to the SEARCH_QUERY. You must adhere to the following 'INSTRUCTIONS'. 

INSTRUCTIONS:
* Carefully read the search query under 'SEARCH_QUERY', enclosed by triple backticks.
* Select only the relevant sentences from 'SEARCH_OUTPUT' that are useful with respect to the search query 
* A sentence can be considered relevant if it contains information that might support or refute the event question.
* Drop the irrelevant sentences.
* You must not add or modify any content in 'SEARCH_OUTPUT'
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.

SEARCH_QUERY:
```
{input_query}
```

SEARCH_OUTPUT:
```
{additional_information_paragraph}
```

OUTPUT_FORMAT:
* If there is no relevant information at all that could be useful with respect to the search query, respond solely with the word "Error".
* Only output the remaining relevant sentences from 'SEARCH_OUTPUT' that could be useful with respect to the search query.
* Do not include any formatting characters in your response!
* Do not include any other contents or explanations in your response!
"""


SUMMARIZE_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to summarize sentences in 'SEARCH_OUTPUT' \
The summary must only contain relevant information with respect to the SEARCH_QUERY. You must adhere to the following 'INSTRUCTIONS'. 

INSTRUCTIONS:
* Carefully read the search query under 'SEARCH_QUERY', enclosed by triple backticks
* Select only the relevant information from 'SEARCH_OUTPUT' that is useful and relevant with respect to the search query
* A sentence can be considered relevant if it contains information that might support or refute the event question
* Summarize the relevant information in a way that is concise and informative
* You must not infer or add any new information, but only summarize the existing statements in an unbiased way
* You must provide your response in the format specified under "OUTPUT_FORMAT"
* Do not include any other contents in your response.

SEARCH_QUERY:
```
{input_query}
```

SEARCH_OUTPUT:
```
{additional_information_paragraph}
```

OUTPUT_FORMAT:
* Only output the summary containing the relevant information from 'SEARCH_OUTPUT' with respect to the search query.
* Do not include any other contents in your response!
* If there is no relevant information at all that could be useful with respect to the search query, respond solely with the word "Error".
"""


# Global constants for possible attribute names for release and update dates
RELEASE_DATE_NAMES = [
    'date', 'pubdate', 'publishdate', 'OriginalPublicationDate', 'dateCreated',
    'article:published_time', 'sailthru.date', 'article.published',
    'published-date', 'og:published_time', 'publication_date',
    'publishedDate', 'dc.date', 'DC.date', 'article:published',
    'article_date_original', 'cXenseParse:recs:publishtime', 'DATE_PUBLISHED',
    'pub-date', 'pub_date', 'datePublished', 'date_published', 'ArticleDate'
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
    "object", "param", "embed", "link", ".breadcrumb", ".pagination", ".nav",
    ".ad", ".sidebar", ".popup", ".modal", ".social-icons", ".hamburger-menu",
]

 
class WebPage:
    _id_counter = 0

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
        self.extract_attribute_names = ["title", "description", "publication_date", "publisher"]


    def get_title(self, soup, scripts):
        title = soup.title
        if title:
            return title.string.strip()
        else:
            title = soup.find("meta", attrs={"name": "title"}) or soup.find("meta", attrs={"property": "title"})
            if title and title.get("content"):
                return title["content"].strip()
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
                raise Exception(f"Error extracting publisher for webpage {self.url}") from e
        
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
            raise ValueError("No HTML content to extract page attributes from.")
        
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
        page_info += f"Publisher: {self.publisher or 'Unknown'}\n"
        
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


def get_first_dict_from_list(data):
    """Returns the first item if data is a list of dictionaries"""
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return data[0]
    else:
        return data  # or raise an appropriate exception



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


def download_spacy_model(self, model_name: str) -> None:
    """Downloads the specified spaCy language model if it is not already installed."""
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("spacy model_name must be a non-empty string")
    if not spacy.util.is_package(model_name):
        spacy.cli.download(model_name)
    else:
        print(f"{model_name} is already installed.")


def extract_event_date(self, doc_question) -> str:
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
            event_date_ymd = self.standardize_date(ent.text)

    # If event date not formatted as YMD or not found, return None
    try:
        datetime.strptime(event_date_ymd, '%Y-%m-%d')
    except (ValueError, TypeError):
        return None
    else:
        return event_date_ymd
    
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
        print(f"\nFORMATTED MODEL ANSWER:\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n{formatted_json}\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return formatted_json
    else:
        # Return the original string if no match is found
        return output_string


def truncate_additional_information(
    self,
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


def get_urls_from_queries(
    queries: List[str],
    api_key: str,
    engine: str,
    num: int = 3
) -> List[str]:
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

    
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(search_google, query, api_key, engine, max_num_fetch) for query in queries}
        for future in as_completed(futures):
            result = future.result()
            count = 0
            for url in result:
                if url not in results and not url.endswith(".pdf"):
                    results.add(url)
                    count += 1
                    if count >= num:
                        break
                else:
                    print(f"URL {url} already in results or is a PDF, skipping...")
            #results.append(future.result())

    # for query in queries:
    #     fetched_urls = search_google(
    #         query=query,
    #         api_key=api_key,
    #         engine=engine,
    #         num=max_num_fetch  # Limit the number of fetched URLs per query
    #     )

        # # Add only unique URLs up to 'num' per query, omitting PDF and 'download' URLs
        # count = 0
        # for url in fetched_urls:
        #     if url not in results and not url.endswith(".pdf"):
        #         results.add(url)
        #         count += 1
        #         if count >= num:
        #             break

    print("\nget_urls_from_queries result:")
    for url in results:
        print(url)

    return list(results)


def find_release_date_in_data(data):
    for name in RELEASE_DATE_NAMES:
        if name in data:
            return data[name]
    return None


def format_date(date_string):
    # Desired format "February 16, 2024, 3:30 PM"
    format_str = "%B %d, %Y, %I:%M %p"

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
    

def parse_date_str(item):
    # Desired format "February 16, 2024, 3:30 PM"
    date_str = item[1]
    format_str = "%B %d, %Y, %I:%M %p"
    try:
        # Parse the date string to datetime object
        return datetime.strptime(date_str, format_str)
    except (ValueError, TypeError):
        # If there's an error during parsing, return datetime.min to sort at the beginning
        return datetime.min
    

def concatenate_short_sentences(self, sentences, len_sentence_threshold):
    modified_sentences = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        word_count = len(sentence.split())
        
        # Check if the sentence is shorter than the threshold
        while word_count < len_sentence_threshold:
            i += 1
            # Break the loop if we reach the end of the list
            if i >= len(sentences):
                break
            next_sentence = sentences[i]
            sentence += " " + next_sentence
            word_count += len(next_sentence.split())
        
        modified_sentences.append(sentence)
        i += 1

    return modified_sentences


def extract_similarity_scores(
    self,
    text: str,
    query_emb,
    event_date: str,
    nlp,
    date: str,
    url: str,
    embedding_model,
) -> List[Tuple[str, float, str]]:
    """
    Extract relevant information from website text based on a given event question.

    Args:
        text (str): The website text to extract information from.
        input_query (str): The question to find relevant information to.
        event_date (str): Event date in year-day-month format.
        nlp: The spaCy NLP model.
        date (str): The release and modification dates of the website.
        url (str): The URL of the website.

    Returns:
        List[Tuple[str, float, str]]: List of tuples containing the extracted sentences, their similarity scores, and release dates.
    """        
    
    # Constants for sentence length and number thresholds
    len_sentence_threshold = 5
    max_chunk_size = 400
    overlap = 15
    num_sentences_threshold = 1000
    sentences = []     
    event_date_sentences = []
    seen = set()

    # Truncate text for performance optimization
    text = text[:50000]
    
    # Apply NLP pipeline to text
    doc_text = nlp(text)

    current_chunk = ""
    words = []
    #words = doc_text.split(" ")

    for sent in doc_text.sents:
        sentence_text = sent.text
        sent_words = sentence_text.split(" ")
        if len(sent_words) >= len_sentence_threshold:
            words.extend(sent_words)

    for word in words:
        if len(current_chunk) + len(word) <= max_chunk_size:
            current_chunk += " " + word
        else:
            # Before adding the chunk to the list, add the url at the end of the chunk
            #current_chunk += " Source: " + url
            sentences.append(current_chunk)
            current_chunk = (
                " ".join(current_chunk.split(" ")[-overlap:]) + " " + word
            )

    if not sentences:
        return ""
    
    # Concatenate short sentences
    
    sentences = self.concatenate_short_sentences(sentences, len_sentence_threshold)



    # Limit the number of sentences for performance optimization
    sentences = sentences[:num_sentences_threshold]
    
    similarities = []

    # Encode sentences using spaCy model
    for i, sentence in enumerate(sentences):
        # # spacy embedding specific
        # doc_sentence = nlp(sentence)
        # similarity_score = query_emb.similarity(doc_sentence)

        # sentence-transformers embedding specific
        sent_emb = embedding_model.encode(sentence)
        similarity_score = util.cos_sim(query_emb, sent_emb)[0].item()

        similarities.append(similarity_score)

    # Create tuples and store them in a list
    sentence_similarity_date_url_tuples = [(sentence, similarity, date, url) for sentence, similarity in zip(sentences, similarities) if similarity > 0.3]

    return sentence_similarity_date_url_tuples


def process_in_batches(
    web_pages: List[WebPage],
    batch_size: int = 15,
    timeout: int = 10
) -> Generator[None, None, List[Tuple[Future, str]]]:
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


def extract_html_texts(
    web_pages: List[WebPage],
) -> List[Tuple[str, float, str]]:    
    # Initialize empty list for storing extracted sentences along with their similarity scores, release dates and urls
    parsed_web_pages = []

    # Process URLs in batches
    for batch in process_in_batches(web_pages=web_pages):
        for future, web_page in tqdm(batch, desc="Processing URLs"):
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
            
            # except Exception as e:
            #     print(f"An error occurred in extract_html_texts: {e}")

    print("Web pages parsed successfully.\n")

    return parsed_web_pages



def drop_irrelevant_sentences(
    self,
    paragraph: str,
    engine: str,
    temperature: float,
    input_query: str,
    reducer_model
) -> str:
    """Drop irrelevant sentences from a paragraph"""
    reducer_prompt = REDUCER_PROMPT.format(input_query=input_query, additional_information_paragraph=paragraph)

    # Create messages for the model engine
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": reducer_prompt},
    ]

    response = reducer_model.invoke(messages, seed=1234)

    # try:
    #     json_data = json.loads(response)
    # except:
    #     print("Error parsing JSON")
    #     exit()

    # print(json_data["additional_information"])
    #print(response.content)
    return response.content


def summarize_relevant_sentences(
    self,
    paragraph: str,
    engine: str,
    temperature: float,
    input_query: str,
    reducer_model
) -> str:
    """Summarize relevant sentences from a paragraph"""
    summarize_prompt = SUMMARIZE_PROMPT.format(input_query=input_query, additional_information_paragraph=paragraph)

    # Create messages for the model engine
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": summarize_prompt},
    ]

    response = reducer_model.invoke(messages, seed=1234)

    # try:
    #     json_data = json.loads(response)
    # except:
    #     print("Error parsing JSON")
    #     exit()

    # print(json_data["additional_information"])
    return response.content


def join_and_group_sentences(
    self,
    sentences: List[Tuple[str, float, str, str]],
    max_words: int,
    engine: str,
    temperature: float,
    input_query: str,
    reducer_model
) -> str:
    """
    Join the sentences and group them by date.
    
    Args:
        sentences (List[Tuple[str, float, str]]): List of tuples containing the extracted sentences, their similarity scores, and release dates.
        max_words (int): Maximum number of words allowed for the output summary.
    
    Returns:
        str: The joined sentences grouped by date.
    """
    # Initialize final output string and word count
    final_output = ""
    current_word_count = 0

    # Initialize a list to hold the sentences that will be included in the final output
    filtered_sentences = []

    # Filter sentences based on word count
    for sentence, _, date, url in sentences:
        additional_word_count = len(sentence.split())
        if current_word_count + additional_word_count <= max_words:
            filtered_sentences.append((f"...{sentence}...\n\n", date, url))
            current_word_count += additional_word_count
        else:
            break
    
    # Sort filtered_sentences by date for grouping
    # filtered_sentences.sort(key=itemgetter(1, 2))
    filtered_sentences.sort(key=lambda x: (self.parse_date_str(x), x[2]))

    #release_date = self.format_date(release_date)

    # Group by date and iterate
    for date, date_group in groupby(filtered_sentences, key=lambda x: self.parse_date_str(x)):
        print(f"Sentences with release date {date}:")
        for url, url_group in groupby(date_group, key=itemgetter(2)):
            print(f"URL: {url}\n")
            sentences_group = [sentence for sentence, _, _ in url_group]
            concatenated_sentences = "\n".join(sentences_group)
            print(concatenated_sentences)
            print()

            final_sentences = self.summarize_relevant_sentences(concatenated_sentences, engine, temperature, input_query, reducer_model)
            if "Error" in final_sentences:
                continue
            
            # Parse the URL to extract the domain
            parsed_url = urlparse(url)
            # The 'netloc' attribute of the result contains the domain
            domain = parsed_url.netloc
            
            # Some URLs might include 'www.', so we strip it to get the clean domain
            if domain.startswith('www.'):
                domain = domain[4:]
            
            if date == datetime.min:
                date_str = "unknown"
            else:
                date_str = date.strftime("%B %d, %Y, %I:%M %p")

            # Formatting the string as per your requirement
            # formatted_string = f"Source: {domain}\n{date}\"Article:\n\"\"\"\n{final_sentences}\n\"\"\"\n\n"
            # formatted_string = f"{date}\"{final_sentences}\"\n\n"
            formatted_string = f"Source: {domain}\nRelease date: {date_str}\n\"{final_sentences}\"\n\n"


            # Add this formatted string to the final output
            final_output += formatted_string
    
    # Get the current date and time in the format "Month Day, Year, Hour:Minute AM/PM"
    current_date = datetime.now().strftime("%B %d, %Y, %I:%M %p")
    final_output += f"For reference, the current date and time is {current_date}.\n"

    return final_output


def fetch_queries(
    client: OpenAI,
    input_query: str,
    model="gpt-3.5-turbo",
    temperature=1.0,
    max_attempts=2,
):
    """Fetch queries from the OpenAI engine"""
    attempts = 0
    while attempts < max_attempts:
        try:
            queries_prompt = QUERIES_PROMPT.format(input_query=input_query)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": queries_prompt},
            ]
            
            # Fetch queries from the OpenAI engine
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            output = response.choices[0].message.content
            # Parse the response content
            print(output)
            trimmed_output = trim_json_formatting(output)
            json_data = json.loads(trimmed_output)
            queries = json_data["queries"]
            return queries  # Return the parsed data if successful
        except Exception as e:
            print(f"Attempt {attempts + 1} failed with error: {e}")
            attempts += 1
            if attempts == max_attempts:
                print("Maximum attempts reached, returning an empty string.")
                return ""  # Return an empty string after exhausting retries



def fetch_additional_information(
    input_query: str,
    max_add_words: int,
    google_api_key: str,
    google_engine: str,
    nlp,
    engine: str,
    reducer_model,
    fetch_queries_model,
    temperature: float = 0.5,
    max_compl_tokens: int = 500,
) -> str:

    """
    Get urls from a web search and extract relevant information based on an event question.
    
    Args:
        input_query (str): The question related to the event.
        max_add_words (int): The maximum number of words allowed for additional information.
        google_api_key (str): The API key for the Google service.
        google_engine (str): The Google engine to be used.
        temperature (float): The temperature parameter for the engine.
        engine (str): The openai engine. Defaults to "gpt-3.5-turbo".
        temperature (float): The temperature parameter for the engine. Defaults to 1.0.
        max_compl_tokens (int): The maximum number of tokens for the engine's response.
        
    Returns:
        str: The relevant information fetched from all the URLs concatenated.
    """


    

    # Extract relevant sentences from URLs
    relevant_sentences_sorted = self.extract_and_sort_sentences(
        urls=urls,
        input_query=input_query,
        nlp=nlp,
    )

    # Join the sorted sentences and group them by date
    additional_informations = self.join_and_group_sentences(
        relevant_sentences_sorted, max_add_words, engine, temperature, input_query, reducer_model
    )

    return additional_informations


def rerank_web_pages(
    client: OpenAI,
    parsed_web_pages: List[WebPage],
    market_question: str,
    model="gpt-3.5-turbo",
    temperature=0,
):
    """Rerank the web pages based on their relevance"""
    web_pages_info = "\n---web_page---\n".join([web_page.to_prompt() for web_page in parsed_web_pages])
    current_date = datetime.now().strftime("%B %d, %Y, %I:%M %p")
    url_reranking_prompt = URL_RERANKING_PROMPT.format(
        market_question=market_question, web_pages_info=web_pages_info, current_date=current_date
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": url_reranking_prompt},
    ]
    
    # Create openai chat completion to get the sorted web page indexes
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    output = response.choices[0].message.content
    print(output)
    
    # Iterate over each web_page and set its sort_index
    for index, web_page in enumerate(parsed_web_pages):
        web_page.sort_index = output[index]

    # To verify the changes
    for web_page in parsed_web_pages:
        print(f"URL: {web_page.url}, Sort Index: {web_page.sort_index}")
    return parsed_web_pages


def scrape_web_pages(web_pages: List[WebPage], week_interval) -> List[WebPage]:
    """Scrape text from web pages"""
    filtered_web_pages = []
    for web_page in web_pages:
        if web_page.html:
            date_parsed = parse_date_str([None, web_page.publication_date])
            if date_parsed > datetime.now() - timedelta(weeks=5) or date_parsed == datetime.min:
                soup = BeautifulSoup(web_page.html, "html.parser")
                soup_string = str(soup)

                # The pattern matches either a bracketed prefix followed by an "https" URL
                # or an "https" URL directly.
                pattern = r'(?:\[\d+\]\s*)?(https?://\S+)'

                # Remove the matched "https" URLs, with or without bracketed prefixes
                soup_string = re.sub(pattern, '', soup_string)
                
                # Clean the text
                doc = Document(soup_string)
                doc_sum = doc.summary()
                h = html2text.HTML2Text()
                h.ignore_links = True
                h.ignore_images = True
                h.ignore_emphasis = True
                web_page.scraped_text = h.handle(doc_sum)
                filtered_web_pages.append(web_page)

            else:
                web_page.scraped_text = ""
                print(f"Publication date {web_page.publication_date} is older than {week_interval} weeks.")
        else:
            web_page.html = ""
            print("HTML content is not available for web page.")

    return filtered_web_pages


def get_chunks(scraped_text: Document) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, length_function=len)
    chunks = text_splitter.create_documents([scraped_text])
    return chunks


def get_embeddings(client: OpenAI, split_docs: List[Document]) -> List[Document]:
    """Get embeddings for the split documents."""
    print("Starting to get embeddings.")
    for batch_start in range(0, len(split_docs), EMBEDDING_BATCH_SIZE):
        batch_end = batch_start + EMBEDDING_BATCH_SIZE
        batch = [doc.text for doc in split_docs[batch_start:batch_end]]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        for i, be in enumerate(response.data):
            assert i == be.index
        batch_embeddings = [e.embedding for e in response.data]
        for i, doc in enumerate(split_docs[batch_start:batch_end]):
            doc.embedding = batch_embeddings[i]
    print("Finished getting embeddings.")
    exit()
    return split_docs



def get_relevant_chunks(client: OpenAI, web_pages: List[WebPage]) -> List[WebPage]:
    """Get relevant chunk for each web page"""
    for web_page in web_pages:
        chunks = get_chunks(web_page.scraped_text)
        docs_with_embeddings = get_embeddings(client, chunks)
        
        exit()

    return web_pages


def research_additional_information(
    input_query: str,
    client: OpenAI,
    thread_id: str,
    google_api_key: str,
    google_engine_id: str,
    engine: str,
):
    """Research additional information based on a prediction market question"""
    # Generate a list of sub-queries
    queries = fetch_queries(client, input_query, engine)
    
    # Get URLs from sub-queries
    urls = get_urls_from_queries(
        queries,
        api_key=google_api_key,
        engine=google_engine_id,
        num=NUM_URLS_PER_QUERY,
    )
    web_pages = [WebPage(url) for url in urls]
    web_pages = extract_html_texts(web_pages)
    
    # print attributes of all web pages
    for page in web_pages:
        print(page.to_prompt())

    # reranked_web_pages = rerank_web_pages(client, parsed_web_pages, market_question=input_query)
    
    
    week_interval = 5
    # Scrape text from web pages not older <week_interval> weeks
    web_pages = scrape_web_pages(web_pages, week_interval)

    for page in web_pages:
        print(f"\n{page.to_prompt()}\n")
        print(page.scraped_text[:500])

    web_pages = get_relevant_chunks(client, web_pages)


        
    

    

    exit()


    reducer_model_name = "gpt-3.5-turbo"
    reducer_model_temperature = 0

    reducer_model = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=reducer_model_name,
        temperature=reducer_model_temperature,
    )

    tool = default_tool
    temperature = default_temperature
    # input_query = kwargs["prompt"]
    max_compl_tokens = DEFAULT_OPENAI_SETTINGS["max_compl_tokens"]
    input_query = input_query.replace("'",'').replace('"','') # Remove double quotes from the input query to avoid issues with react agent execution
    
 
    
    
    #openai.api_key = kwargs["api_keys"]["openai"]
    # if tool not in ALLOWED_TOOLS:
    #     raise ValueError(f"TOOL {tool} is not supported.")

    # Load the spacy model
    # self.download_spacy_model("en_core_web_lg")
    # nlp = spacy.load("en_core_web_md")
    nlp = spacy_universal_sentence_encoder.load_model("en_use_lg")


    # Get the LLM engine to be used
    # engine = TOOL_TO_ENGINE[tool]

    # Get the tiktoken base encoding
    enc = tiktoken.get_encoding("cl100k_base") 

    # Calculate the maximum number of tokens and words that can be consumed by the additional information string
    # max_add_tokens = get_max_tokens_for_additional_information(
    #     max_compl_tokens=max_compl_tokens,
    #     prompt=prompt,
    #     enc=enc,
    # )
    max_add_tokens = MAX_TOKENS_ADDITIONAL_INFORMATION
    max_add_words = int(max_add_tokens * 0.75)

    # Fetch additional information
    additional_information = (
        self.fetch_additional_information(
            input_query=input_query,
            engine=tool,
            temperature=temperature,
            max_compl_tokens=max_compl_tokens,
            nlp=nlp,
            max_add_words=max_add_words,
            google_api_key=google_api_key,
            google_engine=google_engine_id,
            reducer_model=reducer_model,
            fetch_queries_model=fetch_queries_model,
        )
    )
#         additional_information = """The search generated the following relevant information:
# \"\"\"
# None
# \"\"\""""
    
    # Truncate additional information to stay within the chat completion token limit of 4096
    additional_information = self.truncate_additional_information(
        additional_information, MAX_TOKENS_ADDITIONAL_INFORMATION, enc=enc,
    )

    return additional_information