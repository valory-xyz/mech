# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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
"""Contains the job definitions"""

import functools
import json
from typing import Any, Callable, Dict, Optional, Tuple

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError


MechResponseWithKeys = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]


def with_key_rotation(func: Callable) -> Callable:
    """
    Decorator that retries a function with API key rotation on failure.

    Expects `api_keys` in kwargs, supporting `rotate(service)` and `max_retries()`.
    Retries the function on key-related exceptions until retries are exhausted.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MechResponseWithKeys:
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponseWithKeys:
            """Retry the function with a new key."""
            try:
                result: MechResponse = func(*args, **kwargs)
                return result + (api_keys,)
            except GoogleAPIError:
                # try with a new key again
                service = "gemini"
                if retries_left[service] <= 0:
                    raise Exception("Error: API retries exhausted")
                retries_left[service] -= 1
                api_keys.rotate(service)
                return execute()
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


PREDICTION_OFFLINE_PROMPT = """
You are an LLM inside a multi-agent system that takes in a prompt of a user requesting a probability estimation
for a given event. You are provided with an input under the label "USER_PROMPT". You must follow the instructions
under the label "INSTRUCTIONS". You must provide your response in the format specified under "OUTPUT_FORMAT".

INSTRUCTIONS
* Read the input under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" specifies an event.
* The event will only have two possible outcomes: either the event will happen or the event will not happen.
* If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
* You must provide a probability estimation of the event happening, based on your training data.
* You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION" delimited by three backticks.
* You can use any item in "ADDITIONAL_INFORMATION" in addition to your training data.
* If an item in "ADDITIONAL_INFORMATION" is not relevant, you must ignore that item for the estimation.
* You must provide your response in the format specified under "OUTPUT_FORMAT".
* Do not include any other contents in your response.

USER_PROMPT:
```
{user_prompt}
```

ADDITIONAL_INFORMATION:
```
```

OUTPUT_FORMAT
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain four fields: "p_yes", "p_no", "confidence", and "info_utility".
* Each item in the JSON must have a value between 0 and 1.
   - "p_yes": Estimated probability that the event in the "USER_PROMPT" occurs.
   - "p_no": Estimated probability that the event in the "USER_PROMPT" does not occur.
   - "confidence": A value between 0 and 1 indicating the confidence in the prediction. 0 indicates lowest
     confidence value; 1 maximum confidence value.
   - "info_utility": Utility of the information provided in "ADDITIONAL_INFORMATION" to help you make the prediction.
     0 indicates lowest utility; 1 maximum utility.
* The sum of "p_yes" and "p_no" must equal 1.
* Output only the JSON object. Do not include any other contents in your response.
* This is incorrect:"```json{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}```"
* This is incorrect:```json"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"```
* This is correct:"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"
"""

AVAILABLE_TOOLS = ["gemini-prediction", "gemini-completion"]
AVAILABLE_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro"]


def error_response(msg: str) -> Tuple[str, None, None, None]:
    """Return an error mech response."""
    return msg, None, None, None


def response_post_process(response: str) -> str:
    """Loads the prediction into a json object"""

    try:
        json_response = json.loads(response)
        return json.dumps(json_response)
    except json.JSONDecodeError:
        return f"Error: response could not be properly postprocessed: {response}"


@with_key_rotation
def run(**kwargs: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""

    api_key = kwargs["api_keys"]["gemini"]
    tool_name = kwargs.get("tool", None)
    prompt = kwargs.get("prompt", None)
    model = kwargs.get("model", "gemini-1.5-flash")

    if api_key is None:
        return error_response("Gemini API key is not available.")

    if tool_name is None:
        return error_response("No tool name has been specified.")

    if tool_name not in AVAILABLE_TOOLS:
        return error_response(
            f"Tool {tool_name} is not an available tool [{AVAILABLE_TOOLS}]."
        )

    if prompt is None:
        return error_response("No prompt has been given.")

    if model not in AVAILABLE_MODELS:
        return error_response(
            f"Model {model} is not an avaliable model: {AVAILABLE_MODELS}"
        )

    if tool_name == "gemini-prediction":
        prompt = PREDICTION_OFFLINE_PROMPT.format(user_prompt=prompt)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model)
    response = model.generate_content(prompt)
    response = response.text

    if tool_name == "gemini-prediction":
        response = response_post_process(response)

    return response, prompt, None, None
