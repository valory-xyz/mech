# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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

"""This module implements a tool that infers the market rules and status for a prediction market question"""

import re
from typing import Any, Dict, Optional, Tuple
from openai import OpenAI
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


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0.0,
}

LLM_SETTINGS = {
    "gpt-3.5-turbo": DEFAULT_OPENAI_SETTINGS,
    "gpt-4-turbo-preview": DEFAULT_OPENAI_SETTINGS,
}

ALLOWED_TOOLS = [
    "infer-market-rules-gpt-3.5-turbo",
    "infer-market-rules-gpt-4-turbo-preview",
]
TOOL_TO_ENGINE = {
    "infer-market-rules-gpt-3.5-turbo": "gpt-3.5-turbo",
    "infer-market-rules-gpt-4-turbo-preview": "gpt-4-turbo-preview",
}

ALLOWED_MODELS = list(LLM_SETTINGS.keys())

MAX_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "gpt-4-turbo-preview": 8192,
}

SYSTEM_PROMPT = """You are a world class algorithm for generating structured output from a given input."""

# Prompt template for infering rules for a question that asks for an event to happen "by" a specific date
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


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))

def extract_question(text) -> str:
    """Extract the question from prompt enclosed in escaped quotation marks."""
    pattern = r'\"(.*?)\"'
    match = re.search(pattern, text)
    return match.group(1) if match else ""

def remove_date_from_query(query: str) -> str:
    """Remove time-related information from query"""
    date_pattern = r"\b(?:on or by |on or before |before |by |on )?(?:(\d{1,2})(st|nd|rd|th)? (January|February|March|April|May|June|July|August|September|October|November|December)|(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})(st|nd|rd|th)?,?) \d{4}\b"
    new_query = re.sub(date_pattern, "", query)
    return new_query

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

def extract_answer(response_message: str) -> str:
    """Extract the answer from the response message."""
    if "Answer:" in response_message:
        answer = response_message.split("Answer:", 1)[1]
    else:
        answer = response_message
    return answer


def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the task"""
    with OpenAIClientManager(kwargs["api_keys"]["openai"]):
        tool = kwargs["tool"]
        prompt = kwargs["prompt"]
        max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
        counter_callback = kwargs.get("counter_callback", None)

        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"Tool {tool} is not supported.")

        engine = TOOL_TO_ENGINE[tool]

        market_question = extract_question(prompt)
        if not market_question:
            return "Market question not found in prompt", None, None, None
        print(f"MARKET QUESTION:\n{market_question}\n")
        
        # Remove double quotes from the input query to avoid issues
        market_question = market_question.replace('"', "'")
        
        # Get the prompt template based on the timing of the event in the query
        infer_rules_template = get_prompt_template_by_timing(market_question)
        infer_rules_prompt = infer_rules_template.format(market_question=market_question)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
            {"role": "system", "content": SYSTEM_PROMPT},
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
        
        return market_status + "\n\nRules:\n" + market_rules, infer_status_prompt + "\n////\n" + infer_rules_prompt, None, counter_callback