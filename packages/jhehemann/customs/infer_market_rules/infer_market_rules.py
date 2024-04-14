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
    The question will resolve as 'Yes' if:
        - Beyoncé officially releases a full music album for her 'country era' on or before 9 January 2025.
        - Beyoncé has already released such an album recently.

    The question will resolve as 'No' if:
        - Beyoncé does not release a full album for her 'country era' by 9 January 2025.
        - Beyoncé releases a 'country era' album after 9 January 2025.
    
    Additional Notes:
        Definition of 'full album': An album containing multiple tracks recognized as a complete album, not singles or EPs.
        Definition of 'album for her country era': An album that predominantly features songs in the country music genre.

Question: "Will there be another case of H5N1 bird flu in Texas by 11 November 2024?"
Answer:
    The question will resolve as 'Yes' if:
        - A new case of H5N1 bird flu occurs in the state of Texas on or before 11 November 2024.
        - A new case of H5N1 bird flu has recently occurred in Texas.

    The question will resolve as 'No' if:
        - No new case of H5N1 bird flu is confirmed in Texas on or before 11 November 2024.
        - The next new case of H5N1 occurs after 11 November 2024.

    Additional Notes:
        Definition of 'new case': A new occurrence of H5N1 bird flu that is confirmed and is at least the second case in Texas.

Question: "Will an arrest be made in the alleged arson at Sen. Bernie Sanders' Vermont office on or before May 7, 2024?"
Answer:
    The question will resolve as 'Yes' if:
        - An arrest is made in connection with the alleged arson at Senator Bernie Sanders' Vermont office on or before 7 May 2026.
        - An arrest has already been made recently in relation to the arson case.

    The question will resolve as 'No' if:
        - No arrest is made in connection with the alleged arson at Senator Bernie Sanders' Vermont office by 7 May 2026.
        - The arrest related to the arson case occurs after 7 May 2026.
    
    Additional Notes:
        Definition of 'arrest': The detention of an individual by law enforcement.

Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by 31 December 2024?"
Answer:
    The question will resolve as 'Yes' if:
        - FIFA officially funds or begins disbursing funds for the construction of new football stadiums for local clubs in England on or before 31 December 2024.
        - FIFA has already begun funding these projects recently.

    The question will resolve as 'No' if:
        - FIFA does not fund nor begin disbursing funds for the construction of new football stadiums for local clubs in England on or before 31 December 2024.
        - FIFA allocates funds after 31 December 2024.
    
    Additional Notes:
        Definition of 'new football stadiums for local clubs': Stadiums that are newly constructed for football clubs that are not part of the professional league system.
        Definition of 'fund': Actively disbursing financial resources.

Question: "Will a new climate bill be passed by both the Senate and the House on or by 22 October 2024?"
Answer:
    The question will resolve as 'Yes' if:
        - Both the Senate and the House of Representatives officially pass a new climate bill on or by before 22 October 2024.
        - Both chambers have already passed a new climate bill recently.

    The question will resolve as 'No' if:
        - A new climate bill does not exist, or one of the two chambers does not pass it on or before 22 October 2024.
        - The Senate and the House pass the bill after 22 October 2024.
        - Congress does not convene a session to vote on the bill on or before 22 October 2024.
    
    Additional Notes:
        Definition of 'pass': Official approval of the bill by both chambers of the United States Congress.
        Definition of 'new climate bill': A bill that addresses climate-related issues and is distinct from an existing bill or legislation.

Question: "Will Microsoft announce a significant AI-related takeover by 16 April 2024?"
Answer:
    The question will resolve as 'Yes' if:
        - Microsoft officially announces a significant acquisition or takeover related to artificial intelligence (AI) on or before 16 April 2024.
        - Microsoft has recently announced a significant AI-related takeover.

    The question will resolve as 'No' if:
        - Microsoft does not announce any significant AI-related takeover by 16 April 2024.
        - Microsoft announces an AI-related takeover after 16 April 2024.

    Additional Notes:
        Definition of 'significant AI-related takeover': An acquisition or takeover that involves a substantial investment in AI technology or companies.
        Definition of 'announcement': Public declaration or disclosure made by Microsoft regarding the takeover.

Question: "Will Samsung replace its voice assistant Bixby by 7 April 2024?"
Answer:
    The question will resolve as 'Yes' if:
        - Samsung replaces its voice assistant Bixby with a new voice assistant on or before 7 April 2024.
        - Samsung has recently replaced Bixby with a new voice assistant.

    The question will resolve as 'No' if:
        - Samsung does not replace its voice assistant Bixby by 7 April 2024.
        - Samsumg replaces Bixby after 7 April 2024.

    Additional Notes:
        Definition of 'replace': The discontinuation of Bixby as the primary voice assistant and the implementation of a new voice assistant.

Question: "Will Google destroy all browsing data collected in Incognito mode by 2 October 2022?"
Answer:
    The question will resolve as 'Yes' if:
        - Google officially destroys all browsing data collected in Incognito mode on or before 2 October 2022.
        - Google has already completed the destruction of all such data recently.
        
    The question will resolve as 'No' if:
        - Google does not destroy all browsing data collected in Incognito mode on or before 2 October 2022.
        - Google completes the destruction of the data after 2 October 2022.
    
    Additional Notes:
        Definition of 'all browsing data': The entirety of data collected during the browsing session in Incognito mode and not just a portion of it.

Question: "Will President Joe Biden make another visit to Baltimore over the Francis Scott Key Bridge collapse by 16 June 2024?
Answer:
    The question will resolve as 'Yes' if:
        - Joe Biden makes another official visit to Baltimore in relation to the Francis Scott Key Bridge collapse on or before 16 June 2024.
        - Joe Biden has recently visited Baltimore over the bridge collapse incident another time.

    The question will resolve as 'No' if:
        - Joe Biden does not make another visit to Baltimore regarding the Francis Scott Key Bridge collapse by 16 June 2024.
        - Joe Bidens next visit related to the bridge collapse occurs after 16 June 2024.

    Additional Notes:
        Definition of 'visit': President Biden physically travels to Baltimore for the purpose of addressing or inspecting the Francis Scott Key Bridge collapse.
        Definition of 'Francis Scott Key Bridge collapse': A specific incident involving the structural failure or damage to the Francis Scott Key Bridge in Baltimore.

Question: "Will Gilberto Ramirez defend his WBA (Super) cruiserweight title successfully by 19 February 2021?"
Answer:
    The question will resolve as 'Yes' if:
        - Gilberto Ramirez successfully defends his WBA (Super) cruiserweight title in a bout on or before 19 February 2021.
        - Gilberto Ramirez has already successfully defended his title recently.

    The question will resolve as 'No' if:
        - Gilberto Ramirez does not successfully defend his title on or before 19 February 2021.
        - The defense match occurs after 19 February 2021.
    
    Additional Notes:
        Definition of 'successfully defend': Ramirez must be the reigning titleholder and emerge victorious against a contender to retain his title.

Question: "Will Disney Plus implement its password-sharing crackdown by August 25, 2024?"
Answer:
    The market will resolve as 'Yes' if:
        - Disney Plus implements its password-sharing crackdown policy on or before 25 August 2024.
        - Disney Plus has already implemented the password-sharing crackdown recently.

    The market will resolve as 'No' if:
        - Disney Plus does not implement its password-sharing crackdown policy by 25 August 2024.
        - Disney Plus implements a password-sharing crackdown policy after 25 August 2024.
    
    Additional Notes:
        Definition of 'password-sharing crackdown': The enforcement of measures to restrict or prevent the sharing of login credentials among multiple users.
        Definition of 'implement': The policy is put into effect and actively enforced by Disney Plus.

Question: "{market_question}"
Answer:
"""

INFER_RULES_PROMPT_BY_REDUCED_HEADINGS = """
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
    'Yes':
        - Beyoncé officially releases a full music album for her 'country era' on or before 9 January 2025.
        - Beyoncé has already released such an album recently.

    'No':
        - Beyoncé does not release a full album for her 'country era' by 9 January 2025.
        - Beyoncé releases a 'country era' album after 9 January 2025.
    
    Additional Notes:
        Definition of 'full album': An album containing multiple tracks recognized as a complete album, not singles or EPs.
        Definition of 'album for her country era': An album that predominantly features songs in the country music genre.

Question: "Will there be another case of H5N1 bird flu in Texas by 11 November 2024?"
Answer:
    'Yes':
        - A new case of H5N1 bird flu occurs in the state of Texas on or before 11 November 2024.
        - A new case of H5N1 bird flu has recently occurred in Texas.

    'No':
        - No new case of H5N1 bird flu is confirmed in Texas on or before 11 November 2024.
        - The next new case of H5N1 occurs after 11 November 2024.

    Additional Notes:
        Definition of 'new case': A new occurrence of H5N1 bird flu that is confirmed and is at least the second case in Texas.

Question: "Will an arrest be made in the alleged arson at Sen. Bernie Sanders' Vermont office on or before May 7, 2024?"
Answer:
    'Yes':
        - An arrest is made in connection with the alleged arson at Senator Bernie Sanders' Vermont office on or before 7 May 2026.
        - An arrest has already been made recently in relation to the arson case.

    'No':
        - No arrest is made in connection with the alleged arson at Senator Bernie Sanders' Vermont office by 7 May 2026.
        - The arrest related to the arson case occurs after 7 May 2026.
    
    Additional Notes:
        Definition of 'arrest': The detention of an individual by law enforcement.

Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by 31 December 2024?"
Answer:
    'Yes':
        - FIFA officially funds or begins disbursing funds for the construction of new football stadiums for local clubs in England on or before 31 December 2024.
        - FIFA has already begun funding these projects recently.

    'No':
        - FIFA does not fund nor begin disbursing funds for the construction of new football stadiums for local clubs in England on or before 31 December 2024.
        - FIFA allocates funds after 31 December 2024.
    
    Additional Notes:
        Definition of 'new football stadiums for local clubs': Stadiums that are newly constructed for football clubs that are not part of the professional league system.
        Definition of 'fund': Actively disbursing financial resources.

Question: "Will a new climate bill be passed by both the Senate and the House on or by 22 October 2024?"
Answer:
    'Yes':
        - Both the Senate and the House of Representatives officially pass a new climate bill on or by before 22 October 2024.
        - Both chambers have already passed a new climate bill recently.

    'No':
        - A new climate bill does not exist, or one of the two chambers does not pass it on or before 22 October 2024.
        - The Senate and the House pass the bill after 22 October 2024.
        - Congress does not convene a session to vote on the bill on or before 22 October 2024.
    
    Additional Notes:
        Definition of 'pass': Official approval of the bill by both chambers of the United States Congress.
        Definition of 'new climate bill': A bill that addresses climate-related issues and is distinct from an existing bill or legislation.

Question: "Will Microsoft announce a significant AI-related takeover by 16 April 2024?"
Answer:
    'Yes':
        - Microsoft officially announces a significant acquisition or takeover related to artificial intelligence (AI) on or before 16 April 2024.
        - Microsoft has recently announced a significant AI-related takeover.

    'No':
        - Microsoft does not announce any significant AI-related takeover by 16 April 2024.
        - Microsoft announces an AI-related takeover after 16 April 2024.

    Additional Notes:
        Definition of 'significant AI-related takeover': An acquisition or takeover that involves a substantial investment in AI technology or companies.
        Definition of 'announcement': Public declaration or disclosure made by Microsoft regarding the takeover.

Question: "Will Samsung replace its voice assistant Bixby by 7 April 2024?"
Answer:
    'Yes':
        - Samsung replaces its voice assistant Bixby with a new voice assistant on or before 7 April 2024.
        - Samsung has recently replaced Bixby with a new voice assistant.

    'No':
        - Samsung does not replace its voice assistant Bixby by 7 April 2024.
        - Samsumg replaces Bixby after 7 April 2024.

    Additional Notes:
        Definition of 'replace': The discontinuation of Bixby as the primary voice assistant and the implementation of a new voice assistant.

Question: "Will Google destroy all browsing data collected in Incognito mode by 2 October 2022?"
Answer:
    'Yes':
        - Google officially destroys all browsing data collected in Incognito mode on or before 2 October 2022.
        - Google has already completed the destruction of all such data recently.
        
    'No':
        - Google does not destroy all browsing data collected in Incognito mode on or before 2 October 2022.
        - Google completes the destruction of the data after 2 October 2022.
    
    Additional Notes:
        Definition of 'all browsing data': The entirety of data collected during the browsing session in Incognito mode and not just a portion of it.

Question: "Will President Joe Biden make another visit to Baltimore over the Francis Scott Key Bridge collapse by 16 June 2024?
Answer:
    'Yes':
        - Joe Biden makes another official visit to Baltimore in relation to the Francis Scott Key Bridge collapse on or before 16 June 2024.
        - Joe Biden has recently visited Baltimore over the bridge collapse incident another time.

    'No':
        - Joe Biden does not make another visit to Baltimore regarding the Francis Scott Key Bridge collapse by 16 June 2024.
        - Joe Bidens next visit related to the bridge collapse occurs after 16 June 2024.

    Additional Notes:
        Definition of 'visit': President Biden physically travels to Baltimore for the purpose of addressing or inspecting the Francis Scott Key Bridge collapse.
        Definition of 'Francis Scott Key Bridge collapse': A specific incident involving the structural failure or damage to the Francis Scott Key Bridge in Baltimore.

Question: "Will Gilberto Ramirez defend his WBA (Super) cruiserweight title successfully by 19 February 2021?"
Answer:
    'Yes':
        - Gilberto Ramirez successfully defends his WBA (Super) cruiserweight title in a bout on or before 19 February 2021.
        - Gilberto Ramirez has already successfully defended his title recently.

    'No':
        - Gilberto Ramirez does not successfully defend his title on or before 19 February 2021.
        - The defense match occurs after 19 February 2021.
    
    Additional Notes:
        Definition of 'successfully defend': Ramirez must be the reigning titleholder and emerge victorious against a contender to retain his title.

Question: "Will Disney Plus implement its password-sharing crackdown by August 25, 2024?"
Answer:
    'Yes':
        - Disney Plus implements its password-sharing crackdown policy on or before 25 August 2024.
        - Disney Plus has already implemented the password-sharing crackdown recently.

    'No':
        - Disney Plus does not implement its password-sharing crackdown policy by 25 August 2024.
        - Disney Plus implements a password-sharing crackdown policy after 25 August 2024.
    
    Additional Notes:
        Definition of 'password-sharing crackdown': The enforcement of measures to restrict or prevent the sharing of login credentials among multiple users.
        Definition of 'implement': The policy is put into effect and actively enforced by Disney Plus.

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
    The question will resolve as 'Yes' if:
        - The Powerball jackpot amount reaches or exceeds $1 billion on or before 9 January 2024, and a drawing takes place on 9 January 2024.
        - The Powerball jackpot has already exceeded $1 billion and maintains this amount until a drawing on 9 January 2024.

    The question will resolve as 'No' if:
        - No Powerball drawing takes place on 9 January 2024.
        - The Powerball drawing takes place before or after 9 January 2024.
        - The Powerball jackpot amount is less than $1 billion on 9 January 2024.
        - The Powerball jackpot reaches $1 billion or more after the drawing on 9 January 2024.

    Additional Notes:
        Definition of 'Powerball jackpot': The total prize amount offered in the Powerball lottery game.
        Definition of 'drawing': The official selection of winning numbers.

Question: "Will Tesla successfully launch its electric vehicle, Model Z, on 14 June 2024?"
Answer:
    The question will resolve as 'Yes' if:
        - Tesla officially releases an electric vehicle named Model Z on 14 June 2024.
    
    The question will resolve as 'No' if:
        - Tesla does not release an electric vehicle named Model Z on 14 June 2024.
        - Tesla releases a Model Z before or after 14 June 2024.

    Additional Notes:
        Definition of 'launch': The public release and availability of the vehicle for purchase or use.
        Definition of 'electric vehicle': A vehicle powered by an electric motor.

Question: "Will Prince William and Prince Harry appear separately at the event honoring Princess Diana on 16 September 2025?"
Answer:
    The question will resolve as 'Yes' if:
        - Both Prince William and Prince Harry make individual appearances at an event honoring Princess Diana on 16 September 2025 without appearing together.

    The question will resolve as 'No' if:
        - There is no event scheduled honoring Princess Diana on 16 September 2025.
        - The event honoring Princess Diana takes place before or after 16 September 2025.
        - Prince William and Prince Harry do not appear separately at an event honoring Princess Diana on 16 September 2025.
        - Either Prince William or Prince Harry does not attend such an event on 16 September 2025.
    
    Additional Notes:
        Definition of 'appear separately': The princes attend the event at different times or locations without being present together. This includes virtual appearances.

Question: "Will Xiaomi's SU7 electric vehicle still have a waiting list in China on April 18, 2024?"
Answer:
    The question will resolve as 'Yes' if:
        - Xiaomi's SU7 electric vehicle continues to have an active waiting list in China on 18 April 2024.
    
    The question will resolve as 'No' if:
        - Xiaomi's SU7 electric vehicle does not have a waiting list in China on 18 April 2024.
        - Xiaomi cleares or discontinues the waiting list for the SU7 electric vehicle before 18 April 2024.
    
    Additional Notes:
        Definition of 'waiting list': A list of customers waiting to purchase the SU7 electric vehicle in China.

Question: "Will Gilberto Ramirez defend his WBA (Super) cruiserweight title successfully on 26 November 2020?"
Answer:
    The question will resolve as 'Yes' if:
        - Gilberto Ramirez successfully defends his WBA (Super) cruiserweight title in a bout on 26 November 2020.
    
    The question will resolve as 'No' if:
        - There is no defense match scheduled for Gilberto Ramirez on 26 November 2020.
        - The defense match occurs before or after 26 November 2020.
        - Gilberto Ramirez participates in a title defense bout on 26 November 2020 but does not emerge victorious, thereby losing his WBA (Super) cruiserweight title.
        - Gilberto Ramirez is not the reigning WBA (Super) cruiserweight titleholder on 26 November 2020.

    Additional Notes:
        Definition of 'successfully defend': Ramirez must be the reigning titleholder and emerge victorious in a defense match against a contender to retain his title.
    
Question: "Will there be another case of H5N1 bird flu in Texas on 11 November 2024?"
Answer:
    The question will resolve as 'Yes' if:
        - A new case of H5N1 bird flu occurs in the state of Texas on 11 November 2024.
    
    The question will resolve as 'No' if:
        - No new case of H5N1 bird flu occurs in Texas on 11 November 2024.
        - The next case of H5N1 bird flu in Texas occurs before or after 11 November 2024.

    Additional Notes:
        Definition of 'new case': A newly confirmed occurrence of H5N1 bird flu that is distinct from previous cases.
    
Question: "Will a new climate bill be passed by both the Senate and the House on 30 September 2024?"
Answer:
    The question will resolve as 'Yes' if:
        - Both the Senate and the House of Representatives officially pass a new climate bill on 30 September 2024.

    The question will resolve as 'No' if:
        - Congress does not convene a session to vote on the bill on 30 September 2024.
        - A new climate bill does not exist, or one of the two chambers does not pass it on 30 September 2024.
        - Only one chamber passes the bill on 30 September 2024.
        - The Senate and the House pass the bill before or after 30 September 2024.

    Additional Notes:
        Definition of 'pass': Official approval of the bill by both chambers of the United States Congress.
        Definition of 'new climate bill': A bill that addresses climate-related issues and is distinct from an existing bill or legislation.

Question: "Will Google destroy all browsing data collected in Incognito mode on 2nd October 2022?"
Answer:
    The question will resolve as 'Yes' if:
        - Google destroys all browsing data collected in Incognito mode on 2 October 2022.

    The question will resolve as 'No' if:
        - Google does not destroy all browsing data collected in Incognito mode on 2 October 2022.
        - Google destroys the Incognito browsing data before or after 2 October 2022.

    Additional Notes:
        Definition of 'all browsing data': The entirety of data collected during the browsing session in Incognito mode and not just a portion of it.

Question: "Will Samsung replace its voice assistant Bixby on 7 April 2024?"
Answer:
    The question will resolve as 'Yes' if:
        - Samsung replaces its voice assistant Bixby with a new voice assistant on 7 April 2024.

    The question will resolve as 'No' if:
        - Samsung does not replace its voice assistant Bixby on 7 April 2024.
        - Samsung replaces Bixby before or after 7 April 2024.

    Additional Notes:
        Definition of 'replace': The discontinuation of Bixby as the primary voice assistant and the implementation of a new voice assistant.

Question: "Will Beyoncé release a full album for her 'country era' on 12 July 2025?"
Answer:
    The question will resolve as 'Yes' if:
        - Beyoncé officially releases a full music album for her 'country era' on 12 July 2025.
    
    The question will resolve as 'No' if:
        - Beyoncé does not release a full album for her 'country era' on 12 July 2025.
        - Beyoncé releases a 'country era' album before or after 12 July 2025.

    Additional Notes:
        Definition of 'full album': An album containing multiple tracks recognized as a complete album, not singles or EPs.
        Definition of 'album for her country era': An album that predominantly features songs in the country music genre.

Question: "Will Disney Plus implement its password-sharing crackdown on 7 February 2025?"
Answer:
    The question will resolve as 'Yes' if:
        - Disney Plus officially implements a password-sharing crackdown policy on 7 February 2025.

    The question will resolve as 'No' if:
        - Disney Plus does not implement a password-sharing crackdown policy by 7 February 2025.
        - Disney Plus implements a password-sharing crackdown policy before or after 7 February 2025.
    
    Additional Notes:
        Definition of 'password-sharing crackdown': The enforcement of measures to restrict or prevent the sharing of login credentials among multiple users.
        Definition of 'implement': The policy is put into effect and actively enforced by Disney Plus.

Question: "Will the Royals and Chiefs relocate from Kansas City following the rejection of the stadium tax on 24 June 2024?"
Answer:
    The question will resolve as 'Yes' if:
        - Both the Royals and Chiefs relocate from Kansas City following the rejection of the stadium tax on 24 June 2024.

    The question will resolve as 'No' if:
        - Either the Royals or the Chiefs do not complete the relocation from Kansas City following the rejection of the stadium tax on 24 June 2024.
        - The rejection of the stadium tax does not lead to the relocation of both teams from Kansas City on 24 June 2024.
        - The Royals and the Chiefs complete relocation before or after 24 June 2024.

    Additional Notes:
        Definition of 'relocation': Moving the team's home base from Kansas City to a different city or location.
        Definition of 'stadium tax': A tax proposal related to funding for sports stadium facilities.

Question: "{market_question}"
Answer:
"""

# Prompt template for inferring the status of a prediction market question
# 
# INSTRUCTIONS:
# * Carefully read the market question.
# * Pay detailled attention on the phrasing of the market question.
# * Analyze what the question implies and asks for, who the involved parties are and what the conditions are.
# * Do not mention the date specified in the question in your response.

INFER_STATUS_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to infer the status for a prediction market question. \
You are provided with some examples below.

EXAMPLES:
Question: "Will there be another case of H5N1 bird flu in Texas ?"
Answer:
    Status: The question implies that there have been one or more confirmed cases of H5N1 bird flu in the state of Texas. The question focuses on another new case occurring in the future.
    Definitions:
        'new case': A newly confirmed occurrence of H5N1 bird flu that is distinct from previous cases.

Question: "Will Samsung replace its voice assistant Bixby ?"
Answer:
    Status: The question implies that Samsung is considering replacing its voice assistant, Bixby, with a new voice assistant. The question focuses on the timing of this potential replacement.
    Definitions:
        'replace': The discontinuation of Bixby as the primary voice assistant and the implementation of a new voice assistant.
    
Question: "Will Beyoncé release a full album for her 'country era' ?"
Answer:
    Status: The question implies that Beyoncé is potentially exploring country music as a new musical direction referred to as her 'country era'. It focuses on the release date of a full album within this thematic context.
    Definitions:
        'full album': An album containing multiple tracks recognized as a complete album, not singles or EPs.
        'album for her country era': An album that predominantly features songs in the country music genre.

Question: "Will Gilberto Ramirez defend his WBA (Super) cruiserweight title successfully on 26 November 2020?"
Answer:
    Status: The question implies that Gilberto Ramirez is the current titleholder of the WBA (Super) cruiserweight title. The question focuses on the outcome and timing of a potential title defense match.    
    Definitions:
        'successfully defend': Ramirez must be the reigning titleholder and emerge victorious in a defense match against a contender to retain his title.
    
Question: "Will Saudi Arabia successfully host the WTA Finals ?"
Answer:
    Status: The question implies that Saudi Arabia is in consideration to host the WTA Finals, a prestigious women's tennis event. The focus is on the successful execution of the event by Saudi Arabia.
    Definitions:
        'successfully host': Saudi Arabia carries out the event to its conclusion without significant disruptions.
    
Question: "Will Xiaomi's SU7 electric vehicle still have a waiting list in China ?"
Answer:
    Status: The question implies that Xiaomi has introduced an electric vehicle model named SU7 in China, with a current waiting list of potential buyers. The question focuses on the duration of this waiting list and how long it will persist.
    Definitions:
        'waiting list': A list of customers waiting to purchase the SU7 electric vehicle in China.

Question: "Will a new climate bill be passed by both the Senate and the House ?"
Answer:
    Status: The question implies that a climate bill already exists and that a new climate bill is currently under consideration by the United States Congress, which consists of two chambers: the Senate and the House of Representatives. The question focuses on the timeline for this new bill being passed by both chambers.
    Definitions:
       'pass': Official approval of the bill by both chambers of the United States Congress.
       'new climate bill': A bill that addresses climate-related issues and is distinct from an existing bill or legislation.

Question: "Will Tesla successfully launch its new electric vehicle, Model Z ?"
Answer:
    Status: The question implies that Tesla is currently working on a new electric vehicle, which is called Model Z. The question focuses on its release date.
    Definitions:
        'launch': The public release and availability of the vehicle for purchase or use.
        'electric vehicle': A vehicle powered by an electric motor.
   
Question: "Will FIFA fund the construction of new football stadiums for local clubs in England ?"
Answer:
    Status: The question implies that local football clubs in England are in need of new stadiums. The question focuses on whether FIFA, the international governing body of football, will fund the construction of these new stadiums and the timeline for this funding.
    Definitions:
        'new football stadiums for local clubs': Stadiums that are newly constructed for football clubs that are not part of the professional league system.
        'fund': Actively disbursing financial resources.

Question: "Will Prince William and Prince Harry appear separately at the event honoring Princess Diana ?"
Answer:
    Status: Status: The question implies that there is an upcoming event honoring Princess Diana where both Prince William and Prince Harry are expected to attend. The question focuses on the manner of their attendance.
    Definitions:
        'appear separately': The princes attend the event at different times or locations without being present together. This includes virtual appearances.

Question: "Will Disney Plus implement its password-sharing crackdown ?"
Answer:
    Status: The question implies that Disney Plus is considering implementing a policy to prevent the sharing of login credentials among multiple users. The question focuses on the potential implementation of this policy in the future.
    Definitions:
        'password-sharing crackdown': The enforcement of measures to restrict or prevent the sharing of login credentials among multiple users.
        'implement': The policy is put into effect and actively enforced by Disney Plus.

Question: "Will Google destroy all browsing data collected in Incognito mode by 2 October 2022?"
Answer:
    Status: The question implies that Google has collected browsing data in Incognito mode and is expected to destroy this data. The question focuses on the timeline for the data destruction.
    Definitions:
        'all browsing data': The entirety of data collected during the browsing session in Incognito mode and not just a portion of it.

Question: "{market_question}"
Answer:
"""


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


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
        return INFER_RULES_PROMPT_BY_REDUCED_HEADINGS
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