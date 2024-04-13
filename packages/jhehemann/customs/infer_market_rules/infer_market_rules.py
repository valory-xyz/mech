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

# INFER_RULES_PROMPT_BEST = """
# You are a Large Language Model in a multi-agent system. Your task is to infer the rules for a prediction market question. \
# Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. The rules that \
# you define should be based on information that can be found on the internet. You are provided with some examples below. You must adhere to the instructions.

# INSTRUCTIONS:
# * Carefully read the market question
# * Pay detailled attention on the phrasing of the market question
# * Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'

# EXAMPLES:
# Question: "Will a new climate bill be passed by both the Senate and the House by 30 September 2024?"
# Answer:
#     Rules:
#     'Yes': The question resolves as 'Yes' if there exists a another climate bill and it is passed by both the Senate, which is the upper chamber, and the House, which is the lower chamber of the United States Congress BY 30 September 2024. This includes the bill being passed by both chambers BEFORE 30 September 2024. This must be evidenced by an official announcement, press release, or documented agreement confirming that the bill has been passed BY 30 September 2024.
    
#     'No': The question resolves as 'No' if there does not exist another bill or it isn't passed by both chambers BY 30 September 2024, or only one chamber passes it BY 30 September 2024. The market also resolves as 'No' if both chambers pass a different bill and not another climate bill BY 30 September 2024.

# Question: "Will Tesla successfully launch its electric vehicle, Model Z, on 14 June 2024?"
# Answer:
#     Rules:
#     'Yes': The question resolves as 'Yes' if, exactly ON 14 June 2024, Tesla, an influential electric vehicle manufacturer, officially releases a Model Z. This must be evidenced by a public event, press release, or substantial media coverage that confirms the release will occur specifically ON 14 June 2024.

#     'No': The question resolves as 'No' if, ON 14 June 2024, Tesla, an influential electric vehicle manufacturer, does not release a Model Z. This includes any release occurring BEFORE OR AFTER 14 June 2024, or the absence of any official announcement, public event, press release, or significant media coverage confirming a release exactly ON 14 June 2024. The market also resolves as 'No' if Tesla releases a different vehicle Model W exactly ON 14 June 2024.

# Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by the end of 2024?"
# Answer:
#     Rules:
#     'Yes': The question resolves as 'Yes' if FIFA, the international governing body of football, will directly allocate funds for the construction of new football stadiums specifically for local clubs in England BY 31 December 2024. This includes the allocation of funds BEFORE 31 December 2024. This allocation must be evidenced by an official announcement, press release, or documented agreement confirming that the funding will happen BY 31 December 2024.
    
#     'No': The question resolves as 'No' if FIFA, the international governing body of football, will not allocate funds for the construction of new football stadiums for local clubs in England BY 31 December 2024. This includes any funding allocated AFTER 31 December 2024, the absence of any official announcement, press release, or documented agreement confirming that the allocation will happen BY 31 December 2024. The market also resolves as 'No' if FIFA allocates funds for stadium construction in a different country or for a different purpose BY 31 December 2024.

# Question: "{market_question}"
# Answer:
# """

# INFER_RULES_PROMPT_TEST = """
# You are a Large Language Model in a multi-agent system. Your task is to infer the rules for a prediction market question. \
# Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. The rules that \
# you define should be based on information that can be found on the internet. You are provided with some examples below. You must adhere to the instructions.

# INSTRUCTIONS:
# * Carefully read the market question
# * Pay detailled attention on the phrasing of the market question
# * Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'

# EXAMPLES:
# Question: "Will another climate bill be passed by both the Senate and the House by 30 September 2024?"
# Answer:
#     Rules:
#     'Yes': The question resolves as 'Yes' if there exists a another climate bill and it is passed by both the Senate, which is the upper chamber, and the House, which is the lower chamber of the United States Congress BY 30 September 2024. This includes the bill being passed by both chambers BEFORE 30 September 2024.
    
#     'No': The question resolves as 'No' if there does not exist another bill or it isn't passed by both chambers BY 30 September 2024, or only one chamber passes it BY 30 September 2024. The market also resolves as 'No' if both chambers pass a different bill and not another climate bill BY 30 September 2024.

# Question: "Will Tesla successfully launch its electric vehicle, Model Z, on 14 June 2024?"
# Answer:
#     Rules:
#     'Yes': The question resolves as 'Yes' if, exactly ON 14 June 2024, Tesla, an influential electric vehicle manufacturer, officially releases a Model Z.

#     'No': The question resolves as 'No' if, ON 14 June 2024, Tesla, an influential electric vehicle manufacturer, does not release a Model Z. This includes any release occurring BEFORE OR AFTER 14 June 2024. The market also resolves as 'No' if Tesla releases a different vehicle Model W exactly ON 14 June 2024.

# Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by the end of 2024?"
# Answer:
#     Rules:
#     'Yes': The question resolves as 'Yes' if FIFA, the international governing body of football, will directly allocate funds for the construction of new football stadiums specifically for local clubs in England BY 31 December 2024. This includes the allocation of funds BEFORE 31 December 2024.
    
#     'No': The question resolves as 'No' if FIFA, the international governing body of football, will not allocate funds for the construction of new football stadiums for local clubs in England BY 31 December 2024. This includes any funding allocated AFTER 31 December 2024. The market also resolves as 'No' if FIFA allocates funds for stadium construction in a different country or for a different purpose BY 31 December 2024.

# Question: "{market_question}"
# Answer:
# """


# INFER_RULES_PROMPT_PAST_TENSE = """
# You are a Large Language Model in a multi-agent system. Your task is to infer the rules for a prediction market question. \
# Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. The rules that \
# you define should be based on information that can be found on the internet. You are provided with some examples below. You must adhere to the instructions.

# INSTRUCTIONS:
# * Carefully read the market question
# * Pay detailled attention on the phrasing of the market question
# * Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'

# EXAMPLES:
# Question: "Will another climate bill be passed by both the Senate and the House by 30 September 2024?"
# Answer:
#     Rules:
#     'Yes': The question resolves as 'Yes' if there exists a another climate bill and it has been passed by both the Senate, which is the upper chamber, and the House, which is the lower chamber of the United States Congress BY 30 September 2024.
    
#     'No': The question resolves as 'No' if there does not exist another bill or it has not been passed by both chambers BY 30 September 2024, or only one chamber has passed it BY 30 September 2024. The market also resolves as 'No' if both chambers has passed a different bill and not another climate bill BY 30 September 2024.

# Question: "Will Tesla successfully launch its electric vehicle, Model Z, on 14 June 2024?"
# Answer:
#     Rules:
#     'Yes': The question resolves as 'Yes' if, exactly ON 14 June 2024, Tesla, an influential electric vehicle manufacturer, will officially have released a Model Z. This must be evidenced by a public event, press release, or substantial media coverage that confirms the release has occurred specifically ON 14 June 2024.

#     'No': The question resolves as 'No' if, ON 14 June 2024, Tesla, an influential electric vehicle manufacturer, will not have released a Model Z. This includes any release that will have occurred BEFORE OR AFTER 14 June 2024, or the absence of any official announcement, public event, press release, or significant media coverage confirming a release exactly ON 14 June 2024. The market also resolves as 'No' if Tesla will have released a different vehicle Model W exactly ON 14 June 2024.

# Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by the end of 2024?"
# Answer:
#     Rules:
#     'Yes': The question resolves as 'Yes' if FIFA, the international governing body of football, will have allocated funds directly for the construction of new football stadiums specifically for local clubs in England BY 31 December 2024. This allocation must be evidenced by an official announcement, press release, or documented agreement confirming that the funding has happened BY 31 December 2024.
    
#     'No': The question resolves as 'No' if FIFA, the international governing body of football, will not have allocated funds for the construction of new football stadiums for local clubs in England BY 31 December 2024. This includes any funding that will be allocated after 31 December 2024, the absence of any official announcement, press release, or documented agreement confirming that the allocation has happen BY 31 December 2024. The market also resolves as 'No' if FIFA will have allocated funds for stadium construction in a different country or for a different purpose BY 31 December 2024.

# Question: "{market_question}"
# Answer:
# """


INFER_RULES_PROMPT_BY = """
You are a Large Language Model in a multi-agent system. Your task is to infer the rules for a prediction market question. \
Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. \
You are provided with some examples below. You must adhere to the instructions.

INSTRUCTIONS:
* Carefully read the market question
* Examine the EXAMPLES provided to understand the structure of the rules you need to define
* Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'

EXAMPLES:
Question: "Will Beyoncé release a full album for her 'country era' by 9th January 2025?"
Answer:
    Resolution as 'Yes':
        Beyoncé officially releases a full music album containing songs from her 'country era' on or before 9 January 2025.
        Beyoncé has already released such an album recently.

    Resolution as 'No':
        Beyoncé does not release a full album with 'country era' songs on or before 9 January 2025.
        Beyoncé releases an album without 'country era' songs by the specified date.
        Beyoncé only releases singles, EPs, or compilations instead of a full album by the specified date.
        The album meeting the above criteria is released after 9 January 2025.

Question: "Will there be another case of H5N1 bird flu in Texas by 11 November 2024?"
Answer:
    Resolution as 'Yes':
        At least one new case of H5N1 bird flu occurs in the state of Texas on or before 11 November 2024.
        A new case of H5N1 bird flu has recently been confirmed, ensuring it meets the criteria of being at least the second confirmed case in Texas.

    Resolution as 'No':
        No new case of H5N1 bird flu is confirmed in Texas on or before 11 November 2024.
        The next new case of H5N1 occurs after 11 November 2024.

Question: "Will an arrest be made in the alleged arson at Sen. Bernie Sanders' Vermont office by 15 April 2024?"
Answer:
    Resolution as 'Yes':
        An arrest is made in connection with the alleged arson at Senator Bernie Sanders' Vermont office on or before 15 April 2024.
        An arrest has already been made recently in relation to the arson case.

    Resolution as 'No':
        No arrest is made in connection with the alleged arson at Senator Bernie Sanders' Vermont office on or before 15 April 2024.
        The arrest related to the arson case occurs after 15 April 2024.

Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by 31 December 2024?"
Answer:
    Resolution as 'Yes':
        FIFA officially funds or begins disbursing funds for the construction of new football stadiums for local clubs in England on or before 31 December 2024.
        FIFA has already begun funding these projects recently.

    Resolution as 'No':
        FIFA does not fund nor begin disbursing funds for the construction of new football stadiums for local clubs in England on or before 31 December 2024.
        FIFA allocates funds after 31 December 2024.

Question: "Will a new climate bill be passed by both the Senate and the House by 30 September 2024?"
Answer:
    Resolution as 'Yes':
        The new climate bill is officially passed by both the Senate and the House of Representatives on or before 30 September 2024.
        The bill has already been passed by both chambers recently.

    Resolution as 'No':
        A new climate bill does not exist, or it is not passed by both chambers on or before 30 September 2024.
        Only one chamber passes the bill by the specified date.
        The bill is passed after 30 September 2024.
        No session of Congress is convened to vote on the bill by 30 September 2024.

Question: "Will Google destroy all browsing data collected in Incognito mode by 2 October 2022?"
Answer:
    Resolution as 'Yes':
        Google officially destroys all browsing data collected in Incognito mode on or before 2 October 2022.
        Google has already completed the destruction of all such data recently.

    Resolution as 'No':
        Google does not destroy all browsing data collected in Incognito mode on or before 2 October 2022.
        The destruction of data is only partial and not fully completed by 2 October 2022.
        The destruction of the data will be completed after 2 October 2022.

Question: "Will Gilberto Ramirez defend his WBA (Super) cruiserweight title successfully by 13 April 2024?"
Answer:
    Resolution as 'Yes':
        Gilberto Ramirez successfully defends his WBA (Super) cruiserweight title in a bout on or before 13 April 2024.
        Gilberto Ramirez has already successfully defended his title recently.
        He must be the reigning titleholder and emerge victorious against a contender to retain his title.

    Resolution as 'No':
        Gilberto Ramirez does not successfully defend his title, loses his title, or is not the reigning titleholder on or before 13 April 2024.
        The title is lost for reasons other than a defense match by 13 April 2024.
        The defense match or any bout meant to defend the title occurs after 13 April 2024.

Question: "{market_question}"
Answer:
"""

INFER_RULES_PROMPT_BY_WITH_DETAILED_YES_AND_WITHOUT_INCLUDE_IN_NO_PART = """
You are a Large Language Model in a multi-agent system. Your task is to infer the rules for a prediction market question. \
Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. \
You are provided with some examples below. You must adhere to the instructions.

INSTRUCTIONS:
* Carefully read the market question
* Examine the EXAMPLES provided to understand the structure of the rules you need to define
* Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'

EXAMPLES:
Question: "Will Beyoncé release a full album for her 'country era' by 9th January 2025?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' if Beyoncé officially releases a full music album that contains songs belonging to her 'country era' on or before 9 January 2025. This includes any album that has been released recently. The release must consist of multiple tracks recognized as a complete album of which some songs belong to her 'country era'.
    
    'No': The question will be resolved as 'No' if Beyoncé has does not release a full album with songs from her 'country era' on or before 9 January 2025. Additionally, it will be resolved as 'No' if Beyoncé releases an album without 'country era' songs by 9 January 2025, or if she only releases singles, EPs, or compilations. The question will also be resolved as 'No' if the album is released after 9 January 2025. 

Question: "Will there be another case of H5N1 bird flu in Texas by 11 November 2024?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' if there occurs at least one new case of H5N1 bird flu in the state of Texas on or before 11 November 2024. For the outcome to be 'Yes', the case must be newly confirmed and at least the second case of H5N1 bird flu in Texas. This includes any new and distinct occurrences that have been confirmed recently.
    
    'No': The question will be resolved as 'No' if there is no evidence of a new case of H5N1 bird flu in Texas on or before 11 November 2024. Additionally, it will be resolved as 'No' if there occurs a new case of a different bird flu strain, or if the new case is confirmed in a location outside of Texas on or before 11 November 2024. The question will also be resolved as 'No' if a new case occurs after 11 November 2024.

Question: "Will an arrest be made in the alleged arson at Sen. Bernie Sanders' Vermont office by 15 April 2024?"
Answer:
    'Yes': The question will be resolved as 'Yes' if an arrest is made in connection with the alleged arson at Sen. Bernie Sanders' Vermont office on or before 15 April 2024. This includes any arrest that has already been made in relation to the arson case recently.

    'No': The question will be resolved as 'No' if no arrest is made in connection with the alleged arson at Sen. Bernie Sanders' Vermont office on or before 15 April 2024. Additionally, it will be resolved as 'No' if an arrest is made for a different offense or at a different location. The question will also be resolved as 'No' if the arrest occurs after 15 April 2024.

Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by 31 December 2024?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' if FIFA funds or begins disbursing funds for the construction of new football stadiums for local clubs in England on or before 31 December 2024. It will also be resolved as 'Yes' if FIFA has already funded these projects recently.

    'No': The question will be resolved as 'No' if FIFA does not fund the construction of new football stadiums for local clubs in England on or before 31 December 2024. Additionally, it will be resolved as 'No' if FIFA allocates funds after 31 December 2024. The question will also be resolved as 'No' if the funding is allocated for a different purpose or a different country.

Question: "Will a new climate bill be passed by both the Senate and the House by 30 September 2024?"
Answer:
    Rules:
    'Yes': The question resolves as 'Yes' if there exists a new climate bill and it is passed by both the Senate and the House of Representatives of the United States Congress on or before 30 September 2024. The question will also be resolved as 'Yes' if the new climate bill has already been passed by both chambers recently.
    
    'No': The question will be resolved as 'No' if a new climate bill does not exist, is not passed by both chambers on or before 30 September 2024, or if only one chamber passes it by that date. The question will also resolve as 'No' if the bill is passed after 30 September 2024, or if there is no session of Congress convened to vote on the bill by 30 September 2024.

Question: "Will Google destroy all browsing data collected in Incognito mode by 2 October 2022?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' if Google destroys all browsing data collected in Incognito mode on or before 2 October 2022. This includes comprehensive destruction of all such data. The question will also be resolved as 'Yes' if Google has already completed this data destruction."

    'No': The question will be resolved as 'No' if Google does not destroy all browsing data collected in Incognito mode on or before 2 October 2022, or if the destruction is only partial and not fully completed by 2 October 2022. Additionally, the question will be resolved as 'No' if the destruction of the data occurs after 2 October 2022.

Question: "Will Gilberto Ramirez defend his WBA (Super) cruiserweight title successfully by 13 April 2024?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' if Gilberto Ramirez successfully defends his WBA (Super) cruiserweight title in a bout on or before 13 April 2024. He must emerge victorious against a contender to retain his title. The question will also be resolved as 'Yes' if he has already successfully defended his title recently.

    'No': The question will be resolved as 'No' if Gilberto Ramirez loses his WBA (Super) cruiserweight title, is not the current titleholder, or if the title is lost for reasons other than a defense match by 13 April 2024. The outcome will also be 'No' if the defense match occurs after 13 April 2024.
    

Question: "{market_question}"
Answer:
"""

# Question: "Will Tesla successfully launch its electric vehicle, Model Z, by 14 June 2024?"
# Answer:
#     Rules:
#     'Yes': The question will be resolved as 'Yes' if Tesla releases a Model Z on or before 14 June 2024. The question will also be resolved as 'Yes' if Tesla has already released the Model Z.

#     'No': The question will be resolved as 'No' if Tesla does not release a Model Z on or before 14 June 2024. This includes any releases occurring after that date. The question will also be resolved as 'No' if Tesla announces the release of a model Z but does not release it by 14 June 2024.


# Question: "Will President Biden post into the fediverse by 23 September 2024?"
# Answer:
#     Rules:
#     'Yes': The question will be resolved as 'Yes' if Joe Biden or his official account, makes a new post on the fediverse, a decentralized networking protocol, on or before 23 September 2024. The question will also be resolved as 'Yes' if Joe Biden has already posted on the fediverse.
    
#     'No': The question will be resolved as 'No' if Joe Biden does not make a new post on the fediverse on or before 23 September 2024. This includes any posts he makes after that date.

# Question: "Will Gilberto Ramirez defend his WBA (Super) cruiserweight title successfully by 13 April 2024?"
# Answer:
#     Rules:
#     'Yes': Yes': The question will be resolved as 'Yes' if Gilberto Ramirez is the reigning WBA (Super) cruiserweight titleholder and defends his title successfully in a defense bout on or before 13 April 2024. He must defend his title against a contender and emerge victorious in this fight, thereby retaining his WBA (Super) cruiserweight title on or before 13 April 2024. The question will also be resolved as 'Yes' if he has already successfully defended his title recently.

#     'No': The question will be resolved as 'No' if Gilberto Ramirez loses his WBA (Super) cruiserweight title in a defense match on or before 13 April 2024. Additionally, the question will be resolved as 'No' if the defense match takes place after 13 April 2024. The question will also be resolved as 'No' if Ramirez is not the current titleholder or he loses the title due to reasons other than a defense match by 13 April 2024.
    

# The question will also be resolved as 'Yes' if there already has been another new confirmed new case previously.

# Question: "Will OpenAI release another voice synthesis tool by 13 April 2024?"
# Answer:
#     Rules:
#     'Yes': The question will be resolved as 'Yes' if OpenAI has released another voice synthesis tool on or before 13 April 2024. The release must be an official announcement from OpenAI or a public launch of a new voice synthesis tool. The question will also be resolved as 'Yes' if OpenAI has already released another voice synthesis tool.

#     'No': The question will be resolved as 'No' if OpenAI has not released another voice synthesis tool on or before 13 April 2024. This includes any releases after this date. The question will also be resolved as 'No' if OpenAI announces the release of a different type of tool or technology by 13 April 2024.


INFER_RULES_PROMPT_ON = """
You are a Large Language Model in a multi-agent system. Your task is to infer the rules for a prediction market question. \
Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. \
You are provided with some examples below. You must adhere to the instructions.

INSTRUCTIONS:
* Carefully read the market question
* Examine the EXAMPLES provided to understand the structure of the rules
* Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'

EXAMPLES:
Question: "Will the Powerball jackpot reach $1 billion by the drawing on 6 April 2024?"
Answer:
    Rules:
    Yes': The question will be resolved as 'Yes' if the official Powerball jackpot amount reaches or exceeds $1 billion on or before 6 April 2024, and a drawing takes place on that date. The question will also be resolved as 'Yes' if the jackpot has already exceeded $1 billion and maintains this amount until the drawing on 6 April 2024.
    
    'No': The question will be resolved as 'No' if either no drawing takes place on 6 April 2024, or if the official Powerball jackpot amount is less than $1 billion on that specific day. Additionally, the question will be resolved as 'No' if either the jackpot reaches $1 billion or the drawing takes place on that date, but not both. Finally, the question will also be resolved as 'No' if the jackpot amount reaches or exceeds $1 billion after the drawing on 6 April 2024.

Question: "Will Tesla successfully launch its electric vehicle, Model Z, on 14 June 2024?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' only if Tesla officially releases a Model Z on the specific day of 14 June 2024.

    'No': The question will be resolved as 'No' if Tesla does not release a Model Z on the specific day of 14 June 2024. This includes any release occurring before or after this day. The question also will be resolved as 'No' if Tesla announces the release of a model Z but does not release it on 14 June 2024.

Question: "Will President Biden post into the fediverse on 23 September 2024?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' only if Joe Biden or his official account, makes a new post on the fediverse, a decentralized networking protocol, on the specific day of 23 September 2024. The post must be original and created on that exact date.
    
    'No': The question will be resolved as 'No' if Joe Biden does not make a new post on the fediverse on 23 September 2024.

Question: "Will Prince William and Prince Harry appear separately at the event honoring Princess Diana on 13 April 2024?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' only if both Prince William and Prince Harry appear separately at an event honoring Princess Diana on the specific day of 13 April 2024. For the outcome to be 'Yes', there must be an event honoring Princess Diana on 13 April 2024, and both princes must make individual appearances at the event without appearing together.

    'No': The question will be resolved as 'No' if there is no event scheduled honoring Princess Diana on the specific day of 13 April 2024. It will also be resolved as 'No' if either Prince William or Prince Harry do not appear at the event on that date, or if both princes will attend the event together.

Question: "Will Xiaomi's SU7 electric vehicle still have a waiting list in China on 9 April 2024?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' if Xiaomi's SU7 electric vehicle continues to have an active waiting list in China on the specific day of 9 April 2024. This waiting list must be officially acknowledged by Xiaomi or its authorized dealers, and there should be verifiable evidence of customers waiting to purchase the SU7 electric vehicle in China on the specified date.

    'No': The question will be resolved as 'No' if Xiaomi's SU7 electric vehicle does not have a waiting list in China on 9 April 2024. This includes scenarios where the waiting list has been cleared or officially discontinued before 9 April 2024, or if there is no evidence of customers waiting to purchase the SU7 electric vehicle in China on or after the specified date.

Question: "Will Gilberto Ramirez defend his WBA (Super) cruiserweight title successfully on 13 April 2024?"
Answer:
    Rules:
    'Yes': Yes': The question will be resolved as 'Yes' only if Gilberto Ramirez is the reigning WBA (Super) cruiserweight titleholder and participates in a title defense bout on the specific day of 13 April 2024. He must defend his title against a contender and emerge victorious in this fight, thereby retaining his WBA (Super) cruiserweight title.

    'No': The question will be resolved as 'No' if there is no defense match scheduled for Gilberto Ramirez on 13 April 2024, or if he participates in a title defense bout but does not emerge victorious, thereby losing his WBA (Super) cruiserweight title. Additionally, the question will be resolved as 'No' if he is not the current titleholder on 13 April 2024, or if the scheduled bout is canceled or postponed to a date after or before 13 April 2024.

Question: "Will there be another case of H5N1 bird flu in Texas on 11 November 2024?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' if there occurs a new case of H5N1 bird flu in the state of Texas on the specific day of 11 November 2024. For the outcome to be 'Yes', the case must be newly confirmed and not previously reported.
    
    'No': The question will be resolved as 'No' if there does not occur a new case of H5N1 bird flu in Texas on 11 November 2024, including any cases confirmed before or after this date. Additionally, the question will be resolved as 'No' if there is a confirmed case of a different bird flu strain, or if the new case is confirmed in a location outside of Texas on the specified date.

Question: "Will a new climate bill be passed by both the Senate and the House on 30 September 2024?"
Answer:
    Rules:
    'Yes': The question resolves as 'Yes' if there exists a new climate bill and it is passed by both the Senate and the House of Representatives of the United States Congress on the specific day of 30 September 2024. For the outcome to be 'Yes', there must be a convened session of Congress where a vote can take place.
    
    'No': The question resolves as 'No' if there does not exist a new climate bill or it is not passed by both chambers on 30 September 2024, or only one chamber passes it on that date. This includes the bill being passed before or after 30 September 2024. The question will also be resolved as 'No' if it is announced that the new bill will be passed but it is not passed by both chambers on 30 September 2024.

Question: "Will Google destroy all browsing data collected in Incognito mode on 2nd October 2022?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' if Google destroys all browsing data collected in Incognito mode on the specific day of 2 October 2022. The destruction of data must be comprehensive and include all Incognito data.

    'No': The question will be resolved as 'No' if Google does not destroy all browsing data collected in Incognito mode on 2 October 2022. This includes the destruction before or after 2 October 2022. Additionally, the question will be resolved as 'No' if the destruction is partial or incomplete on this date. Finally, the question will also be resolved as 'No' if it is announced that the data will be destroyed but it is not done on 2 October 2022.

Question: "Will Beyoncé release a full album for her 'country era' on 9th January 2025?"
Answer:
    Rules:
    'Yes': The question will be resolved as 'Yes' only if Beyoncé officially releases a full music album that contains songs as belonging to her 'country era' on the specific day of 9 January 2025. The release must consist of multiple tracks that together are recognized as a complete album.
    
    'No': The question will be resolved as 'No' if Beyoncé does not release a full album that contains songs belonging to her 'country era' on 9 January 2025. This includes the album released before or after 9 January 2025. The question will also be resolved as 'No' if Beyoncé releases an album that does not contain songs described as part of her 'country era' by this date, or if she only releases singles, EPs, or compilations. Finally, the question will also be resolved as 'No' if the album is announced to be released but it is not done on 9 January 2025.
    

Question: "{market_question}"
Answer:
"""


# Question: "Will an arrest be made in the alleged arson at Sen. Bernie Sanders' Vermont office by 17 November 2024?"
#     Rules:
#     'Yes' if an arrest has been made by law enforcement authorities in connection with the alleged arson incident at Senator Bernie Sanders' Vermont office on or before 17 November 2024. The arrest must be directly related to the arson incident and involve a suspect being taken into custody for investigation or prosecution. The question will also be resolved as 'Yes' if the arrest has already occurred.

#     'No': The question will be resolved as 'No' if no arrest has been made in connection with the alleged arson incident at Senator Bernie Sanders' Vermont office on or before 17 November 2024. This includes scenarios where the investigation remains ongoing without any arrests being reported in relation to the incident. The question will also be resolved as no if the arrest will be made after 17 November 2024.

# * Pay detailled attention on the phrasing of the market question, especially the time-related information
# * Pay close attention to how the rules are phrased and what they convey in relation to the phrasing of the time-related information in the question
# The question also will be resolved as 'Yes' if there is evidence that Joe Biden has already used the fediverse before and will make a post on that exact date.
# The question also will be resolved as 'No' if there is evidence that Joe Biden has used the fediverse before but will not make a post on that exact date.
INFER_STATUS_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to infer the status for a prediction market question. \
You are provided with some examples below. You must adhere to the instructions.

INSTRUCTIONS:
* Carefully read the market question.
* Pay detailled attention on the phrasing of the market question.
* Analyze what the question implies and asks for, who the involved parties are and what the conditions are.
* Do not use the date specified in the question in your response.

EXAMPLES:
Question: "Will there be another case of H5N1 bird flu in Texas by 10 April 2024?"
Answer:
    Status: The question implies that there have been one or more confirmed cases of H5N1 bird flu in the state of Texas. The question focuses on another new case occurring.

Question: "Will Beyoncé release a full album for her 'country era' by 9th January 2025?"
Answer:
    Status: The question implies that Beyoncé is potentially exploring country music as a new musical direction referred to as her 'country era'. It focuses on the release date of a full album within this thematic context.

Question: "Will President Biden post into the fediverse on 23 September 2024?"
Answer:
    Status: The question implies that Joe Biden, the 46th president of the United States, has an account on the fediverse, a decentralized networking protocol and potentially has made posts in the past. The question focuses on his future posting activities on the fediverse.

Question: "Will Prince William and Prince Harry appear separately at the event honoring Princess Diana on 13 April 2024?"
Answer:
    Status: Status: The question implies that there is an upcoming event honoring Princess Diana where both Prince William and Prince Harry are expected to attend. The question focuses on the manner of their attendance and the specific date of the event.

Question: "Will Xiaomi's SU7 electric vehicle still have a waiting list in China on 9 April 2024?"
Answer:
    Status: The question implies that Xiaomi has introduced an electric vehicle model named SU7 in China, with a current waiting list of potential buyers. The question focuses on the duration of this waiting list and how long it will persist.

Question: "Will a new climate bill be passed by both the Senate and the House by 30 September 2024?"
Answer:
    Status: The question implies that a climate bill already exists and that a new climate bill is currently under consideration by the United States Congress, which consists of two chambers: the Senate and the House of Representatives. The question focuses on whether this new bill will be passed by both chambers and on the timeline for this legislative action.

Question: "Will Tesla successfully launch its new electric vehicle, Model Z, on 29 June 2024?"
Answer:
    Status: The question implies that Tesla is currently working on a new electric vehicle, which is called Model Z. The question focuses on its release date.
   
Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by 31 December 2024?"
Answer:
    Status: The question implies that local football clubs in England are in need of new stadiums. The question focuses on whether FIFA, the international governing body of football, will fund the construction of these new stadiums and the timeline for this funding.

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

def get_prompt_template_by_timing(query: str) -> str:
    """Extract time-related information from query"""
    date_pattern_on = r"\b(?:on )(?:(\d{1,2})(st|nd|rd|th)? (January|February|March|April|May|June|July|August|September|October|November|December)|(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})(st|nd|rd|th)?,?) \d{4}\b"
    date_pattern_by = r"\b(?:on or before |by )(?:(\d{1,2})(st|nd|rd|th)? (January|February|March|April|May|June|July|August|September|October|November|December)|(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})(st|nd|rd|th)?,?) \d{4}\b"
    date_pattern_before = r"\b(?:before )(?:(\d{1,2})(st|nd|rd|th)? (January|February|March|April|May|June|July|August|September|October|November|December)|(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})(st|nd|rd|th)?,?) \d{4}\b"
    
    if re.search(date_pattern_on, query):
        match = re.search(date_pattern_on, query)
        return INFER_RULES_PROMPT_ON
    elif re.search(date_pattern_by, query):
        match = re.search(date_pattern_by, query)
        return INFER_RULES_PROMPT_BY
    # elif re.search(date_pattern_before, query):
    #     match = re.search(date_pattern_before, query)
    #     return INFER_RULES_PROMPT_BEFORE
    else:
        return "No time-related information found in query."


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
    infer_rules_template = get_prompt_template_by_timing(market_question)
    print()
    print("infer_rules_template: ", infer_rules_template)
    print()

    # Create prompt
    infer_rules_prompt = infer_rules_template.format(market_question=market_question)
    
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