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

SYSTEM_PROMPT = """You are a world class algorithm for generating structured output from a given input."""

INFER_RULES_PROMPT = """
You are a Large Language Model in a multi-agent system. Your task is to infer the current status and rules for a prediction market question. \
Provide reliable and well-structured rules for when the prediction market question will be resolved as 'Yes' and 'No'. The rules that \
you define should be based on information that can be found on the internet. You are provided with some examples below. You must adhere to the instructions.

INSTRUCTIONS:
* Carefully read the market question
* Pay detailled attention on the phrasing of the market question
* Analyze what the phrasing implies about the current status, who the involved parties are and what the conditions are
* Define measurable and verifiable rules for when the market question will be resolved as 'Yes' and when it will be resolved as 'No'

EXAMPLES:
Question: "Will the new climate bill be passed by both the Senate and the House by 30 September 2024?"
Answer:
    Status: The question implies that a new climate bill is under consideration by the United States Congress, which consists of two chambers: the Senate and the House of Representatives. If there exists such a bill, the question suggests that it has not yet been passed by both chambers.
    Rules:
    'Yes': The question resolves as 'Yes' if there exists a climate bill and it is passed by both the Senate, which is the upper chamber, and the House, which is the lower chamber of the United States Congress on or before 30 September 2024.
    'No': The question resolves as 'No' if there exists no bill or it isn't passed by both chambers on or before 30 September 2024, or only one chamber passes it on or before this date. The market also resolves as 'No' if both chambers pass a different bill and not a new climate bill on or before this date.

Question: "Will Tesla successfully launch its new electric vehicle, Model Z, on 30 June 2024?"
Answer:
    Status: The question implies that, exactly on 30 June 2024, Tesla, an influential electric vehicle manufacturer, is planning to release a new electric vehicle, which is called Model Z. Furthermore, the question suggests that up to today, the release has not yet occurred.
    Rules:
    'Yes': The question resolves as 'Yes' if, exactly on 30 June 2024, Tesla, an influential electric vehicle manufacturer, officially releases a Model Z. This must be evidenced by a public event, press release, or substantial media coverage that confirms the release will occur specifically on 30 June 2024.
    'No': The question resolves as 'No' if, on 30 June 2024, Tesla, an influential electric vehicle manufacturer, does not release a Model Z. This includes any release occurring before or after this date, or the absence of any official announcement, public event, press release, or significant media coverage confirming a release exactly on 30 June 2024. The market also resolves as 'No' if Company Y releases a different vehicle Model W exactly on 30 June 2024.

Question: "Will FIFA fund the construction of new football stadiums for local clubs in England by the end of 2024?"
Answer:
    Status: The question implies that FIFA, the international governing body of football, is considering funding the construction of new football stadiums specifically for local clubs in England. The question suggests that FIFA has not yet allocated funds for this purpose.
    Rules:
    'Yes': The question resolves as 'Yes' if FIFA, the international governing body of football, will directly allocate funds for the construction of new football stadiums specifically for local clubs in England on or before 31 December 2024. This allocation must be evidenced by an official announcement, press release, or documented agreement confirming that the funding will happen on or before this specific date.
    'No': The question resolves as 'No' if FIFA, the international governing body of football, will not allocate funds for the construction of new football stadiums for local clubs in England on or before 31 December 2024. This includes any funding allocated after this date, the absence of any official announcement, press release, or documented agreement confirming that the allocation will happen on or before this date. The market also resolves as 'No' if FIFA allocates funds for stadium construction in a different country or for a different purpose on or before 31 December 2024.

Question: "{market_question}"
Answer:
"""

# Question: "Will Julia Roberts announce her retirement from acting after her next film on 15 July 2024?"
# Answer:
#     Status: The question consists of different components. The event that is asked for is an announcement of retirement from acting by Julia Roberts, a famous Hollywood actress. It asks whether this announcement will be made by her exactly on 15 July 2024 and also whether the time of retirement will start after her next film. The question implies that Julia Roberts has not yet announced her retirement from acting.
#     Rules:
#     'Yes': The question resolves as 'Yes' if, exactly on 15 July 2024, Julia Roberts makes an official announcement declaring her retirement from acting after her next film. This must be evidenced by a public statement, press release, or significant media coverage confirming that, exactly on 15 July 2024, Julia Roberts herself has declared her retirement from acting after her next film.
#     'No': The question resolves as 'No' if, on 15 July 2024, Julia Roberts does not make an official announcement regarding her retirement from acting. This includes any announcements made before or after 15 July 2024, or the absence of any public statement, press release, or significant media coverage confirming such an announcement exactly on 15 July 2024.


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


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
    
    return market_rules, counter_callback