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

"""This module implements a research agent for extracting relevant information from URLs."""

from openai import OpenAI
from tiktoken import encoding_for_model
import re

SYSTEM_PROMPT = """You are a world class algorithm for generating structured output from a given input."""

INFER_RULES_PROMPT_BEST = """
You are a Large Language Model in a multi-agent system. Your task is to infer the rules for a prediction market question. \
Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. The rules that \
you define should be based on information that can be found on the internet. You are provided with some examples below. You must adhere to the instructions.

INSTRUCTIONS:
* Carefully read the market question
* Pay detailled attention on the phrasing of the market question
* Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'

EXAMPLES:
Question: "Will a new climate bill be passed by both the Senate and the House by 30 September 2024?"
Answer:
    Rules:
    'Yes': The question resolves as 'Yes' if there exists a another climate bill and it is passed by both the Senate, which is the upper chamber, and the House, which is the lower chamber of the United States Congress BY 30 September 2024. This includes the bill being passed by both chambers BEFORE 30 September 2024. This must be evidenced by an official announcement, press release, or documented agreement confirming that the bill has been passed BY 30 September 2024.
    
    'No': The question resolves as 'No' if there does not exist another bill or it isn't passed by both chambers BY 30 September 2024, or only one chamber passes it BY 30 September 2024. The market also resolves as 'No' if both chambers pass a different bill and not another climate bill BY 30 September 2024.

Question: "Will Tesla successfully launch its electric vehicle, Model Z, on 14 June 2024?"
Answer:
    Rules:
    'Yes': The question resolves as 'Yes' if, exactly ON 14 June 2024, Tesla, an influential electric vehicle manufacturer, officially releases a Model Z. This must be evidenced by a public event, press release, or substantial media coverage that confirms the release will occur specifically ON 14 June 2024.

    'No': The question resolves as 'No' if, ON 14 June 2024, Tesla, an influential electric vehicle manufacturer, does not release a Model Z. This includes any release occurring BEFORE OR AFTER 14 June 2024, or the absence of any official announcement, public event, press release, or significant media coverage confirming a release exactly ON 14 June 2024. The market also resolves as 'No' if Tesla releases a different vehicle Model W exactly ON 14 June 2024.

Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by the end of 2024?"
Answer:
    Rules:
    'Yes': The question resolves as 'Yes' if FIFA, the international governing body of football, will directly allocate funds for the construction of new football stadiums specifically for local clubs in England BY 31 December 2024. This includes the allocation of funds BEFORE 31 December 2024. This allocation must be evidenced by an official announcement, press release, or documented agreement confirming that the funding will happen BY 31 December 2024.
    
    'No': The question resolves as 'No' if FIFA, the international governing body of football, will not allocate funds for the construction of new football stadiums for local clubs in England BY 31 December 2024. This includes any funding allocated AFTER 31 December 2024, the absence of any official announcement, press release, or documented agreement confirming that the allocation will happen BY 31 December 2024. The market also resolves as 'No' if FIFA allocates funds for stadium construction in a different country or for a different purpose BY 31 December 2024.

Question: "{market_question}"
Answer:
"""

INFER_RULES_PROMPT_TEST = """
You are a Large Language Model in a multi-agent system. Your task is to infer the rules for a prediction market question. \
Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. The rules that \
you define should be based on information that can be found on the internet. You are provided with some examples below. You must adhere to the instructions.

INSTRUCTIONS:
* Carefully read the market question
* Pay detailled attention on the phrasing of the market question
* Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'

EXAMPLES:
Question: "Will another climate bill be passed by both the Senate and the House by 30 September 2024?"
Answer:
    Rules:
    'Yes': The question resolves as 'Yes' if there exists a another climate bill and it is passed by both the Senate, which is the upper chamber, and the House, which is the lower chamber of the United States Congress BY 30 September 2024. This includes the bill being passed by both chambers BEFORE 30 September 2024.
    
    'No': The question resolves as 'No' if there does not exist another bill or it isn't passed by both chambers BY 30 September 2024, or only one chamber passes it BY 30 September 2024. The market also resolves as 'No' if both chambers pass a different bill and not another climate bill BY 30 September 2024.

Question: "Will Tesla successfully launch its electric vehicle, Model Z, on 14 June 2024?"
Answer:
    Rules:
    'Yes': The question resolves as 'Yes' if, exactly ON 14 June 2024, Tesla, an influential electric vehicle manufacturer, officially releases a Model Z.

    'No': The question resolves as 'No' if, ON 14 June 2024, Tesla, an influential electric vehicle manufacturer, does not release a Model Z. This includes any release occurring BEFORE OR AFTER 14 June 2024. The market also resolves as 'No' if Tesla releases a different vehicle Model W exactly ON 14 June 2024.

Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by the end of 2024?"
Answer:
    Rules:
    'Yes': The question resolves as 'Yes' if FIFA, the international governing body of football, will directly allocate funds for the construction of new football stadiums specifically for local clubs in England BY 31 December 2024. This includes the allocation of funds BEFORE 31 December 2024.
    
    'No': The question resolves as 'No' if FIFA, the international governing body of football, will not allocate funds for the construction of new football stadiums for local clubs in England BY 31 December 2024. This includes any funding allocated AFTER 31 December 2024. The market also resolves as 'No' if FIFA allocates funds for stadium construction in a different country or for a different purpose BY 31 December 2024.

Question: "{market_question}"
Answer:
"""


INFER_RULES_PROMPT_PAST_TENSE = """
You are a Large Language Model in a multi-agent system. Your task is to infer the rules for a prediction market question. \
Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. The rules that \
you define should be based on information that can be found on the internet. You are provided with some examples below. You must adhere to the instructions.

INSTRUCTIONS:
* Carefully read the market question
* Pay detailled attention on the phrasing of the market question
* Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'

EXAMPLES:
Question: "Will another climate bill be passed by both the Senate and the House by 30 September 2024?"
Answer:
    Rules:
    'Yes': The question resolves as 'Yes' if there exists a another climate bill and it has been passed by both the Senate, which is the upper chamber, and the House, which is the lower chamber of the United States Congress BY 30 September 2024.
    
    'No': The question resolves as 'No' if there does not exist another bill or it has not been passed by both chambers BY 30 September 2024, or only one chamber has passed it BY 30 September 2024. The market also resolves as 'No' if both chambers has passed a different bill and not another climate bill BY 30 September 2024.

Question: "Will Tesla successfully launch its electric vehicle, Model Z, on 14 June 2024?"
Answer:
    Rules:
    'Yes': The question resolves as 'Yes' if, exactly ON 14 June 2024, Tesla, an influential electric vehicle manufacturer, will officially have released a Model Z. This must be evidenced by a public event, press release, or substantial media coverage that confirms the release has occurred specifically ON 14 June 2024.

    'No': The question resolves as 'No' if, ON 14 June 2024, Tesla, an influential electric vehicle manufacturer, will not have released a Model Z. This includes any release that will have occurred BEFORE OR AFTER 14 June 2024, or the absence of any official announcement, public event, press release, or significant media coverage confirming a release exactly ON 14 June 2024. The market also resolves as 'No' if Tesla will have released a different vehicle Model W exactly ON 14 June 2024.

Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by the end of 2024?"
Answer:
    Rules:
    'Yes': The question resolves as 'Yes' if FIFA, the international governing body of football, will have allocated funds directly for the construction of new football stadiums specifically for local clubs in England BY 31 December 2024. This allocation must be evidenced by an official announcement, press release, or documented agreement confirming that the funding has happened BY 31 December 2024.
    
    'No': The question resolves as 'No' if FIFA, the international governing body of football, will not have allocated funds for the construction of new football stadiums for local clubs in England BY 31 December 2024. This includes any funding that will be allocated after 31 December 2024, the absence of any official announcement, press release, or documented agreement confirming that the allocation has happen BY 31 December 2024. The market also resolves as 'No' if FIFA will have allocated funds for stadium construction in a different country or for a different purpose BY 31 December 2024.

Question: "{market_question}"
Answer:
"""


INFER_RULES_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to infer the rules for a prediction market question. \
Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. The rules that \
you define should be based on information that can be found on the internet. You are provided with some examples below. You must adhere to the instructions.

INSTRUCTIONS:
* Carefully read the market question
* Pay detailled attention on the phrasing of the market question
* Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'

EXAMPLES:
Question: "Will there be another case of H5N1 bird flu in Texas by 10 April 2024?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' if there will have been confirmed evidence of at least one new case of H5N1 bird flu in the state of Texas by 10 April 2024. For the outcome to be 'Yes', the case must be newly confirmed and not previously reported.
    
    'No': The question will be resolved as 'No' if there has been no confirmed evidence of a new case of H5N1 bird flu in Texas by 10 April 2024, including any cases confirmed after this date. Additionally, the question will be resolved as 'No' if there is a confirmed case of a different bird flu strain, or if the new case is confirmed in a location outside of Texas by the specified date.

Question: "Will the Powerball jackpot reach $1 billion by the drawing on 6 April 2024?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' if the official Powerball jackpot amount, as announced by the Multi-State Lottery Association or its designated representatives, is equal to or has exceeded $1 billion for a drawing on 6 April 2024. For the outcome to be 'Yes' both the jackpot must be at least $1 billion by 6 April 2024 and a drawing must occur on 6 April 2024.
    
    'No': The question will be resolved as 'No' if the official Powerball jackpot amount for a drawing on 6 April 2024 is less than $1 billion, as announced by the Multi-State Lottery Association or its designated representatives. The question also will be resolved as 'No' if the jackpot reaches $1 billion by 6 April 2024, but there is no Powerball drawing scheduled on 6 April 2024 for any reason, including but not limited to cancellations, postponements, or changes in the Powerball lottery operations. Additionally, the question will be resolved as 'No' if the jackpot amount reaches or exceeds $1 billion after the drawing on 6 April 2024.

Question: "Will Tesla successfully launch its electric vehicle, Model Z, on 14 June 2024?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' only if Tesla, an influential electric vehicle manufacturer, officially releases the Model Z on the specific day of 14 June 2024.

    'No': The question will be resolved as 'No' if Tesla, an influential electric vehicle manufacturer, will not release a Model Z on the specific day of 14 June 2024. This includes any release occurring before or after this day. The question also will be resolved as 'No' if Tesla releases a different vehicle Model W specificly on 14 June 2024.    

Question: "Will President Biden post into the fediverse on 23 September 2024?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' only if Joe Biden, the 46th president of the United States, or his official account, makes a new post on the fediverse, a decentralized networking protocol, on the specific day of 23 September 2024. The post must be original and created on that exact date to meet the criteria for a 'Yes' resolution.
    
    'No': The question will be resolved as 'No' if Joe Biden, the 46th president of the United States, does not make a new post on the fediverse on 23 September 2024. The question also will be resolved as 'No' if there is evidence that Joe Biden has used the fediverse before but will not make a post on that exact date.

Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by the end of 2024?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' if FIFA, the international governing body of football, has directly allocated funds for the construction of new football stadiums specifically for local clubs in England by 31 December 2024. 

    'No': The question will be resolved as 'No' if FIFA, the international governing body of football, has not allocated funds for the construction of new football stadiums for local clubs in England by 31 December 2024. This includes any funding allocated after 31 December 2024. The question also will be resolved as 'No' if FIFA allocates funds for stadium construction in a different country or for a different purpose by 31 December 2024.

Question: "Will Beyoncé release a full album for her 'country era' by 9th January 2025?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' if Beyoncé, the globally recognized music artist, has officially released a full music album that is publicly described or marketed as belonging to her 'country era' by 9th January 2025. This includes any albums released before the current date. The release must consist of multiple tracks that together are recognized as a complete album.
    
    'No': The question will be resolved as 'No' if Beyoncé has not released a full album that is publicly described or marketed as belonging to her 'country era' by 9th January 2025. This includes any albums released after 9th January 2025. The question also will be resolved as 'No' if Beyoncé releases a new album that is not categorized or described as part of her 'country era' by this date, or if she releases only singles, EPs, or compilations that do not constitute a full album as recognized by standard music industry definitions.

Question: "Will Prince William and Prince Harry appear separately at the event honoring Princess Diana on 13 April 2024?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' only if both Prince William and Prince Harry, the sons of Princess Diana, will appear separately at an event honoring Princess Diana on the specific day of 13 April 2024. For the outcome to be 'Yes', there must be an event honoring Princess Diana on 13 April 2024, and both princes must make individual appearances at the event without appearing together.

    'No': The question will be resolved as 'No' if either Prince William or Prince Harry, or both, will not appear at the event honoring Princess Diana on 13 April 2024. The question also will be resolved as 'No' if both princes will appear at the event together, or if the event itself will not take place on 13 April 2024, but on a different date.

Question: "Will Google destroy all browsing data collected in Incognito mode by 2 October 2022?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' if Google, the multinational technology company, has destroyed all browsing data collected in Incognito mode by 2 October. The destruction of data must be comprehensive and include all data collected during the use of Incognito mode. 

    'No': The question will be resolved as 'No' if Google has not destroyed all browsing data collected in Incognito mode by 2 October. This includes any partial destruction of data or retention of browsing data beyond this date. The question also will be resolved as 'No' if Google has confirmed that it would destroy the data, but the destruction has not occurred by 2 October 2022.



Question: "{market_question}"
Answer:
"""

INFER_STATUS_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to infer the current status for a prediction market question. \
You are provided with some examples below. You must adhere to the instructions.

INSTRUCTIONS:
* Carefully read the market question.
* Pay detailled attention on the phrasing of the market question.
* Analyze what the phrasing implies about the current status, who the involved parties are and what the conditions are.

EXAMPLES:
Question: "Will another new climate bill be passed by both the Senate and the House by 30 September 2024?"
Answer:
    Status: The question implies that there already exists a climate bill. It pertains to a new climate bill under consideration by the United States Congress, which consists of two chambers: the Senate and the House of Representatives. If a new bill is under consideration, the question suggests that it has not yet been passed by both chambers yet.

Question: "Will Tesla successfully launch its new electric vehicle, Model Z, on 30 June 2024?"
Answer:
    Status: The question implies that Tesla, an influential electric vehicle manufacturer, is planning to release a new electric vehicle, which is called Model Z. Furthermore, the question suggests that up to today, the release has not yet occurred.
   
Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by the end of 2024?"
Answer:
    Status: The question implies that FIFA, the international governing body of football, is considering funding the construction of new football stadiums specifically for local clubs in England. The question suggests that FIFA has not yet allocated funds for this purpose.
   
Question: "{market_question}"
Answer:
"""

# The question implies that Beyoncé, a globally recognized music artist, is potentially exploring or has hinted at a 'country era' in her musical career. The use of 'country era' suggests a thematic or stylistic shift in her music to focus on country genre elements.

# Question: "Will Julia Roberts announce her retirement from acting after her next film on 15 July 2024?"
# Answer:
#     Status: The question consists of different components. The event that is asked for is an announcement of retirement from acting by Julia Roberts, a famous Hollywood actress. It asks whether this announcement will be made by her exactly on 15 July 2024 and also whether the time of retirement will start after her next film. The question implies that Julia Roberts has not yet announced her retirement from acting.
#     Rules:
#     'Yes': The question will be resolved as 'Yes' if, exactly on 15 July 2024, Julia Roberts makes an official announcement declaring her retirement from acting after her next film. This must be evidenced by a public statement, press release, or significant media coverage confirming that, exactly on 15 July 2024, Julia Roberts herself has declared her retirement from acting after her next film.
#     'No': The question will be resolved as 'No' if, on 15 July 2024, Julia Roberts does not make an official announcement regarding her retirement from acting. This includes any announcements made before or after 15 July 2024, or the absence of any public statement, press release, or significant media coverage confirming such an announcement exactly on 15 July 2024.


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


def remove_date_from_query(query: str) -> str:
    """Remove time-related information from query"""
    date_pattern = r"\b(?:on or by |on or before |by |on )?(?:(\d{1,2})(st|nd|rd|th)? (January|February|March|April|May|June|July|August|September|October|November|December)|(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})(st|nd|rd|th)?,?) \d{4}\b"
    new_query = re.sub(date_pattern, "", query)
    return new_query


def get_market_rules(
    market_question: str,
    client: OpenAI,
    counter_callback,
    temperature=0.0,
    engine="gpt-3.5-turbo",
):
    """Infer market rules for a prediction market question."""
    
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

    if "Answer:" in response_message:
        market_rules = response_message.split("Answer:", 1)[1]
    else:
        market_rules = response_message


    ## Infer the market status
    market_question_no_date = remove_date_from_query(market_question)

    infer_status_prompt = INFER_STATUS_PROMPT.format(market_question=market_question_no_date)
    
    # Create messages for the model engine
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


    if "Answer:" in response_message:
        market_status = response_message.split("Answer:", 1)[1]
    else:
        market_status = response_message
    
    return market_status, market_rules, counter_callback