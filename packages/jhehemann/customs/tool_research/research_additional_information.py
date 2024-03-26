"""This module implements a research agent for extracting relevant information from URLs."""

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup, NavigableString
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
import spacy
from spacy import Language
from spacy.cli import download

import logging

from openai import OpenAI
from tqdm import tqdm


from dateutil import parser

#logging.basicConfig(level=logging.DEBUG)

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

# import spacy.util
# import tiktoken

# from dateutil import parser
# from tqdm import tqdm


#NUM_URLS_EXTRACT = 5
NUM_URLS_PER_QUERY = 5
TEXT_CHUNK_LENGTH = 300
TEXT_CHUNK_OVERLAP = 50
MAX_CHUNKS_TOKENS_TO_SUMMARIZE = 1000
MAX_TEXT_CHUNKS_TOTAL = 50
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_EMBEDDING_TOKEN_INPUT = 8192
EMBEDDING_BATCH_SIZE = 1000
EMBEDDING_SIZE = 1536
MAX_TOKENS_ADDITIONAL_INFORMATION = 2000
WORDS_PER_TOKEN_FACTOR = 0.75
VOCAB = "en_core_web_sm"

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


FINAL_SUMMARY_PROMPT = """
You are provided with search outputs from multiple sources. These search outputs were received in response to the search \
query: "{search_query}". Your task is to select only the most relevant information from the articles. If there are no \
relevant results in one of the articles, you can skip it. Return the selected relevant articles only with the relevant information \
for answering the search query. The headers of each article must remain the same.

SEARCH_OUTPUT:
```
{additional_information}
```
"""

FINAL_SUMMARY_CONTINUOUS_TEXT_PROMPT = """
You are provided with search outputs from multiple sources. These search outputs were received in response to the search \
query: "{search_query}". Your task is to summarize only the concrete relevant results from the search outputs. If there are dates \
mentioned in the search outputs, ensure to include them in the summary. It must be concise and informative.

SEARCH_OUTPUT:
```
{additional_information}
```
"""


RESEARCH_PLAN_PROMPT_TEMPLATE = """
Your goal is to prepare a research plan for {query}.

The plan must consist of {search_limit} search engine queries separated by commas.
Return ONLY the queries, separated by commas and without quotes.
The queries must be phrased as questions.
"""


QUERY_RERANKING_PROMPT_TEMPLATE = """
Evaluate the queries and decide which ones will provide the best and wholesome data to answer the question. Do not modify the queries.
Select only the best five queries, in order of relevance.

OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()"
* The JSON must contain one field: "queries"
    - "queries": A 5 item array of the generated search engine queries
* Include only the JSON object in your output
"""


SUB_QUERIES_PROMPT = """
Your goal is to prepare a research plan for {input_query}.

The plan will consist of multiple web searches that cover a variety of perspectives and sources.
Think about additional information that needs to be gathered to answer the question comprehensively.
Formulate 5 unique search queries that will help you find relevant information about the event and its date.


OUTPUT_FORMAT:
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()"
* Limit your searches to 5.
* The JSON must contain one field: "queries"
    - "queries": A 5 item array of the generated search engine queries
* Include only the JSON object in your output
"""


QUERIES_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to formulate search engine queries. Find the input query \
under 'INPUT_QUERY' and adhere to the 'INSTRUCTIONS'.

INSTRUCTIONS:
* Carefully read the input query under 'USER_PROMPT', enclosed by triple backticks.
* Create a list of 4 unique search queries likely to yield relevant and contemporary information about the event.
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
   - "queries": A 4 item array of the generated search engine queries
* Include only the JSON object in your output
* Do not include any formatting characters in your response!
* Do not include any other contents or explanations in your response!
"""

# Select those web pages that are likely to contain relevant and current information to answer the question. \
# ... return ... only for the selected relevant web pages.


SUMMARIZE_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to summarize relevant information in 'SEARCH_OUTPUT' \
The summary must only contain relevant information with respect to the SEARCH_QUERY. You must adhere to the following 'INSTRUCTIONS'. 

INSTRUCTIONS:
* Carefully read the search query under 'SEARCH_QUERY'
* Select only the relevant information from 'SEARCH_OUTPUT' that is useful and relevant with respect to the search query
* A chunk can be considered relevant if it contains information that might support or refute the event question
* Summarize the relevant information in a way that is concise and informative
* You must not infer or add any new information, but only summarize the existing statements in an unbiased way
* You must provide your response in the format specified under "OUTPUT_FORMAT"
* Do not include any other contents in your response.

SEARCH_QUERY:
{input_query}

SEARCH_OUTPUT:
```
{chunks}
```

OUTPUT_FORMAT:
* Only output the summary containing the relevant information from 'SEARCH_OUTPUT' with respect to the search query.
* The summary must be structured in bullet points.
* Respond solely with "Error", if there is no relevant information in 'SEARCH_OUTPUT'.
* Do not include any other contents in your response!

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
                print(f"Error decoding JSON: {e}\nScript: {script}")
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


def load_model(vocab: str) -> Language:
    """Utilize spaCy to load the model and download it if it is not already available."""
    try:
        return spacy.load(vocab)
    except OSError:
        print("Downloading language model...")
        download(vocab)
        return spacy.load(vocab)


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


def find_release_date_in_data(data):
    for name in RELEASE_DATE_NAMES:
        if name in data:
            return data[name]
    return None


def format_date(date_string) -> str:
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
    

def parse_date_str(date_str: str) -> datetime:
    # Desired format "February 16, 2024, 3:30 PM"
    datetime_format = "%B %d, %Y, %I:%M %p"
    try:
        return datetime.strptime(date_str, datetime_format)
    except (ValueError, TypeError):
        return datetime.min


def process_in_batches(
    web_pages: List[WebPage],
    batch_size: int = 15,
    timeout: int = 10
) -> Generator[None, None, List[Tuple[Future, WebPage]]]:
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


def recursive_character_text_splitter(text, max_tokens, overlap):
    if len(text) <= max_tokens:
        return [text]
    else:
        return [text[i:i+max_tokens] for i in range(0, len(text), max_tokens - overlap)]


def embed_batch(client: OpenAI, batch):
    """
    Helper function to process a single batch of texts and return the embeddings.
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
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

    print("\nINPUT QUERY:")  
    print(query)
    print()

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
    print("SIMILAR CHUNK INDICES (SORTED IN DESCENDING ORDER):")
    print(I)
    print()
    print("SIMILARITY SCORES (SORTED IN DESCENDING ORDER):")
    print(D)
    print()
    for i, sim in enumerate(D[0]):
        text_chunks_embedded[I[0][i]].similarity = sim
        # print(f"SIMILARITY: {sim}, INDEX: {I[0][i]}")
        # print(text_chunks_embedded[I[0][i]].text)
        # print()
    return [text_chunks_embedded[i] for i in I[0]]


def get_embeddings(client: OpenAI, text_chunks: List[TextChunk], enc: tiktoken.Encoding) -> List[TextChunk]:
    """Get embeddings for the text chunks."""
    print("Start getting embeddings ...")
    
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
            # print the first three chunks
            text_chunks.extend(TextChunk(text=chunk, url=web_page.url) for chunk in chunks)

    return text_chunks


def scrape_web_pages(web_pages: List[WebPage], week_interval, max_num_char: int = 10000) -> List[WebPage]:
    """Scrape text from web pages"""
    filtered_web_pages = []
    investigate_urls = []
    for web_page in web_pages:
        if web_page.html:
            date_parsed = parse_date_str(web_page.publication_date)
            if date_parsed > datetime.now() - timedelta(weeks=5) or date_parsed == datetime.min:
                if web_page.url in investigate_urls:
                    print(web_page.html)
                # Clean the text
                doc_html2str = Document(web_page.html)
                doc_sum = doc_html2str.summary()
                h = html2text.HTML2Text()
                h.ignore_links = True
                h.ignore_images = True
                h.ignore_emphasis = True
                scraped_text = h.handle(doc_sum)
                scraped_text = "  ".join([x.strip() for x in scraped_text.split("\n")])
                scraped_text = re.sub(r'\s+', ' ', scraped_text)
                # print()
                # print(scraped_text)
                # print()

                if len(scraped_text) < 300:
                    print(f"\nScraped text for {web_page.url} has less than 300 characters: {len(scraped_text)}.")
                    print("Trying a different approach...")
                    # print()
                    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    # print()
                    soup = BeautifulSoup(web_page.html, "lxml")
                    #Remove unnecessary tags to clean up html
                    for element in soup(HTML_TAGS_TO_REMOVE):
                        element.replace_with(NavigableString(' '))

                    text = h.handle(soup.prettify())
                    text = "  ".join([x.strip() for x in text.split("\n")])
                    text = re.sub(r'\s+', ' ', text)
                    scraped_text = text
                    # print()
                    # print(scraped_text)
                    # print()

                    if len(scraped_text) < 300 and not (web_page.title == "n/a" and web_page.description == "n/a"):
                        prefix = f"{web_page.title}. {web_page.description}."
                        scraped_text = prefix + scraped_text
                        print(f"Scraped text still less than 300 characters. Added title and description to the text.")
                    else:
                        scraped_text = None
                        print("Scraped text has still less than 300 characters and no title and description. Skipping...")
                
                if scraped_text:
                    web_page.scraped_text = scraped_text[:max_num_char]
                
                # # The pattern matches either a bracketed prefix followed by an "https" URL
                # # or an "https" URL directly.
                # pattern = r'(?:\[\d+\]\s*)?(https?://\S+)'
                # # Remove the matched "https" URLs, with or without bracketed prefixes
                # web_page.scraped_text = re.sub(pattern, '', web_page.scraped_text)

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
                    # print(f"URL {url} already in results or omitting keyword found, skipping...")
                    continue

    print("\nget_urls_from_queries result:")
    for url in results:
        print(url)

    return list(results)


def fetch_queries(
    client: OpenAI,
    input_query: str,
    model="gpt-3.5-turbo",
    temperature=1.0,
    max_attempts=2,
) -> List[str]:
    """Fetch queries from the OpenAI engine"""
    attempts = 0
    while attempts < max_attempts:
        try:
            research_plan_prompt = RESEARCH_PLAN_PROMPT_TEMPLATE.format(query=input_query, search_limit="12")
            messages = [
                {"role": "system", "content": "You are a professional researcher and expert for prediction markets."},
                {"role": "user", "content": research_plan_prompt},
            ]
            
            # Fetch queries from the OpenAI engine
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            search_plan = response.choices[0].message.content
            messages = [
                {"role": "system", "content": "You are a professional researcher and expert for prediction markets."},
                {"role": "user", "content": research_plan_prompt},
                {"role": "assistant", "content": search_plan},
                {"role": "user", "content": QUERY_RERANKING_PROMPT_TEMPLATE},
            ]
            
            # Fetch reranked and selected queries from the OpenAI engine
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            output = response.choices[0].message.content






            # Parse the response content
            trimmed_output = trim_json_formatting(output)
            print(trimmed_output)
            json_data = json.loads(trimmed_output)
            queries = json_data["queries"]
            return queries  # Return the parsed data if successful
        except Exception as e:
            print(f"Attempt {attempts + 1} failed with error: {e}")
            attempts += 1
            if attempts == max_attempts:
                print("Maximum attempts reached, returning an empty string.")
                return []


def trim_chunks_string(chunks_string: str, enc: tiktoken.Encoding) -> str:
    encoding = enc.encode(chunks_string)
    if len(encoding) > MAX_CHUNKS_TOKENS_TO_SUMMARIZE:
        encoding = encoding[:MAX_CHUNKS_TOKENS_TO_SUMMARIZE]
    return enc.decode(encoding)


def summarize_relevant_chunks(
        web_pages: List[WebPage],
        input_query: str,
        client: OpenAI,
        enc: tiktoken.Encoding,
        model = "gpt-3.5-turbo",
        temperature = 0.0
) -> List[WebPage]:
    def summarize_for_web_page(web_page: WebPage) -> None:
        chunks_string = ""
        for i, chunk in enumerate(web_page.chunks_sorted):
            chunks_string += f"\n…{chunk}…\n"
        trimmed_chunks = trim_chunks_string(chunks_string, enc)
        summarize_prompt = SUMMARIZE_PROMPT.format(input_query=input_query, chunks=trimmed_chunks)
        # print()
        # print(summarize_prompt)
        # print()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": summarize_prompt},
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
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
                # Result is None, as we're modifying web_page objects in-place
                future.result()
            except Exception as e:
                print(f'Web page {web_page.url} generated an exception: {e}')
    return [web_page for web_page in web_pages if "Error" not in web_page.relevant_chunks_summary]


def format_additional_information(web_pages: List[WebPage]) -> str:
    """Format the additional information from the web pages"""
    formatted_information = ""
    for i, web_page in enumerate(web_pages):
        formatted_information += f"ARTICLE {i+1}: {web_page.title}, PUBLISHER: {web_page.publisher}, PUBLICATION_DATE: {web_page.publication_date}\n"
        formatted_information += f"{web_page.relevant_chunks_summary}\n\n"
    return formatted_information


def final_summary(
    client: OpenAI,
    additional_information: str,
    search_query: str,
    engine: str = "gpt-3.5-turbo",
    temperature: float = 0.0
) -> str:
    prompt = FINAL_SUMMARY_PROMPT.format(additional_information=additional_information, search_query=search_query)
    messages = [
            {"role": "system", "content": "You are an expert in summarizing information most relevant information from articles with respect to a web search query."},
            {"role": "user", "content": prompt},
        ]
    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
    )
    output = response.choices[0].message.content
    return output



def research(
    market_question: str,
    client: OpenAI,
    google_api_key: str,
    google_engine_id: str,
    engine: str,
):
    """Research additional information based on a prediction market question"""
    
    # Generate a list of sub-queries
    queries = fetch_queries(client, market_question, engine)
    
    # Get URLs from sub-queries
    urls = get_urls_from_queries(
        queries,
        api_key=google_api_key,
        engine=google_engine_id,
        num=NUM_URLS_PER_QUERY,
    )
    web_pages = [WebPage(url) for url in urls]
    web_pages = extract_html_texts(web_pages)

    week_interval = 5
    # Scrape text from web pages not older <week_interval> weeks
    web_pages = scrape_web_pages(web_pages, week_interval)

    for page in web_pages:
        if page.scraped_text:
            print(f"\n{page.to_prompt()}")
            print(f"Length of scraped text: {len(page.scraped_text)}\n")

    text_chunks = get_chunks(web_pages)
    
    # Initialize Tokenizer
    enc = tiktoken.get_encoding("cl100k_base") 
    
    text_chunks_embedded = get_embeddings(client, text_chunks, enc) if text_chunks else []
    text_chunks_sorted = sort_text_chunks(client, market_question, text_chunks_embedded) if text_chunks_embedded else []
    text_chunks_limited = text_chunks_sorted[:MAX_TEXT_CHUNKS_TOTAL]
    # print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    # for chunk in text_chunks_sorted:
    #     print(f"Similarity: {chunk.similarity}")
    #     print(chunk.text)
    #     print()

    # Create a dictionary mapping URLs to WebPage objects for quicker lookups
    web_pages_dict = {web_page.url: web_page for web_page in web_pages}

    for text_chunk in text_chunks_limited:
        if text_chunk.url in web_pages_dict:
            web_pages_dict[text_chunk.url].chunks_sorted.append(text_chunk.text)

    web_pages = list(web_pages_dict.values())
    web_pages = summarize_relevant_chunks(web_pages, market_question, client, enc)
    web_pages = sorted(web_pages, key=lambda web_page: parse_date_str(web_page.publication_date))

    additional_information = format_additional_information(web_pages)
    if additional_information:
        additional_information += (
            f"Disclaimer: This search output was retrieved on {datetime.now().strftime('%B %d, %Y, %I:%M %p')} and does not claim to be exhaustive or definitive."
        )
    print(f"\nADDITIONAL INFORMATION for SEARCH QUERY {market_question}")
    print(additional_information)
    print()
    additional_information = final_summary(client, additional_information, market_question, engine)
    return additional_information