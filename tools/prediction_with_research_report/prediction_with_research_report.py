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

from typing import Any, Dict, Optional, Tuple
from evo_researcher.functions.research import research
import openai

DEFAULT_OPENAI_SETTINGS = {
    "temperature": 0,
    "max_compl_tokens": 500,
}

ALLOWED_TOOLS = [
    "prediction-sentence-embedding-conservative",
    "prediction-sentence-embedding-bold",
]
TOOL_TO_ENGINE = {
    "prediction-sentence-embedding-conservative": "gpt-3.5-turbo-1106",
    "prediction-sentence-embedding-bold": "gpt-4-1106-preview",
}

DEFAULT_RESEARCH_SETTINGS = {
    "initial_subqueries_limit": 20,
    "subqueries_limit": 4,
    "scrape_content_split_chunk_size": 800,
    "scrape_content_split_chunk_overlap": 225,
    "top_k_per_query": 8,
}

PREDICTION_PROMPT = """
INTRODUCTION:
You are a Large Language Model (LLM) within a multi-agent system. Your primary task is to accurately estimate the probabilities for the outcome of a 'market question', \
found in 'USER_PROMPT'. The market question is part of a prediction market, where users can place bets on the outcomes of market questions and earn rewards if the selected outcome occurrs. The 'market question' \
in this scenario has only two possible outcomes: `Yes` or `No`. Each market has a closing date at which the outcome is evaluated. This date is typically stated within the market question.  \
The closing date is considered to be 23:59:59 of the date provided in the market question. If the event specified in the market question has not occurred before the closing date, the market question's outcome is `No`. \
If the event has happened before the closing date, the market question's outcome is `Yes`. You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION", which is \
sourced from a Google search engine query performed a few seconds ago and is meant to assist you in your probability estimation. You must adhere to the following 'INSTRUCTIONS'.  


INSTRUCTIONS:
* Examine the user's input labeled 'USER_PROMPT'. Focus on the part enclosed in double quotes, which contains the 'market question'.
* If the 'market question' implies more than two outcomes, output the response "Error" and halt further processing.
* When the current time {timestamp} has passed the closing date of the market and the event specified in the market question has not happened, the market question's outcome is `No` and the user who placed a bet on `No` will receive a reward.
* When the current time {timestamp} has passed the closing date of the market and the event has happened before, the market question's final outcome is `Yes` and the user who placed a bet on `yes` will receive a reward.
* Consider the prediction market with the market question, the closing date and the outcomes in an isolated context that has no influence on the protagonists that are involved in the event in the real world, specified in the market question. The closing date is always arbitrarily set by the market creator and has no influence on the real world. So it is likely that the protagonists of the event in the real world are not even aware of the prediction market and do not care about the market's closing date.
* The probability estimations of the market question outcomes must be as accurate as possible, as an inaccurate estimation will lead to financial loss for the user.
* Utilize your training data and the information provided under "ADDITIONAL_INFORMATION" to generate probability estimations for the outcomes of the 'market question'.
* Examine the itemized list under "ADDITIONAL_INFORMATION" thoroughly and use all the relevant information for your probability estimation. This data is sourced from a Google search engine query done a few seconds ago. 
* Use any relevant item in "ADDITIONAL_INFORMATION" in addition to your training data to make the probability estimation. You can assume that you have been provided with the most current and relevant information available on the internet. Still pay close attention on the release and modification timestamps provided in parentheses right before each information item. Some information might be outdated and not relevant anymore.
* More recent information indicated by the timestamps provided in parentheses right before each information item overrides older information within ADDITIONAL_INFORMATION and holds more weight for your probability estimation.
* If there exist contradicting information, evaluate the release and modification dates of those information and prioritize the information that is more recent and adjust your confidence in the probability estimation accordingly.
* Even if not all information might not be released today, you can assume that there haven't been publicly available updates in the meantime except for those inside ADDITIONAL_INFORMATION.
* If the information in "ADDITIONAL_INFORMATION" indicate without a doubt that the event has already happened, it is very likely that the outcome of the market question will be `Yes`.
* If the information in "ADDITIONAL_INFORMATION" indicate that the event will happen after the closing date, it is very likely that the outcome of the market question will be `No`.
* The closer the current time `{timestamp}` is to the closing time the higher the likelyhood that the outcome of the market question will be `No`, if recent information do not clearly indicate that the event will occur before the closing date.
* If there exist recent information indicating that the event will happen after the closing date, it is very likely that the outcome of the market question will be `No`.
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
   - "p_yes": Probability that the market question's outcome will be `Yes`.
   - "p_no": Probability that the market questions outcome will be `No`.
   - "confidence": Indicating the confidence in the estimated probabilities you provided ranging from 0 (lowest confidence) to 1 (maximum confidence). Confidence can be calculated based on the quality and quantity of data used for the estimation.
   - "info_utility": Utility of the information provided in "ADDITIONAL_INFORMATION" to help you make the probability estimation ranging from 0 (lowest utility) to 1 (maximum utility).
* The sum of "p_yes" and "p_no" must equal 1.
* Output only the JSON object in your response. Do not include any other contents in your response.
"""


def run(**kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Run the task"""
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]
    temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
    max_compl_tokens = kwargs.get(
        "max_tokens", DEFAULT_OPENAI_SETTINGS["max_compl_tokens"]
    )
    
    openai_api_key = kwargs["api_keys"]["openai"]
    tavily_api_key = kwargs["api_keys"]["tavily"]
    
    initial_subqueries_limit = kwargs.get('initial_subqueries_limit', DEFAULT_RESEARCH_SETTINGS["initial_subqueries_limit"])
    subqueries_limit = kwargs.get('subqueries_limit', DEFAULT_RESEARCH_SETTINGS["subqueries_limit"])
    scrape_content_split_chunk_size = kwargs.get('scrape_content_split_chunk_size', DEFAULT_RESEARCH_SETTINGS["scrape_content_split_chunk_size"])
    scrape_content_split_chunk_overlap = kwargs.get('scrape_content_split_chunk_overlap', DEFAULT_RESEARCH_SETTINGS["scrape_content_split_chunk_overlap"])
    top_k_per_query = kwargs.get('top_k_per_query', DEFAULT_RESEARCH_SETTINGS["top_k_per_query"])
    
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"TOOL {tool} is not supported.")
    
    engine = TOOL_TO_ENGINE[tool]
    
    (research_report, _) = research(
        prompt,
        openai_key=openai_api_key,
        tavily_key=tavily_api_key,
        model=engine,
        kwargs={
            "initial_subqueries_limit": initial_subqueries_limit,
            "subqueries_limit": subqueries_limit,
            "scrape_content_split_chunk_size": scrape_content_split_chunk_size,
            "scrape_content_split_chunk_overlap": scrape_content_split_chunk_overlap,
            "top_k_per_query": top_k_per_query,
        }
    )

    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")

    engine = TOOL_TO_ENGINE[tool]

    prediction_prompt = PREDICTION_PROMPT.format(
        user_prompt=prompt, additional_information=research_report
    )
    moderation_result = openai.Moderation.create(prediction_prompt)
    if moderation_result["results"][0]["flagged"]:
        return "Moderation flagged the prompt as in violation of terms.", None, None
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prediction_prompt},
    ]
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
    return response.choices[0].message.content, prediction_prompt, None