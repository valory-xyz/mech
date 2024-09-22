import requests
import re
import json
from typing import Optional, Dict, Any, Tuple

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

CORCEL_URL = "https://api.corcel.io/v1/chat/completions"

AVAILABLE_TOOLS = ["corcel-prediction", "corcel-completion"]

DEFAULT_VALUES = {
    "model": "llama-3",
    "temperature": 0.1,
    "max_tokens": 500
}


def send_corcel_request(api_key: str, prompt: str, **kwargs) -> str:
    """Makes a request to Corcel API"""

    payload = {key: kwargs.get(key, default_value) for key, default_value in DEFAULT_VALUES.items()}

    payload["messages"] = [
        {
            "role": "system",
            "content": prompt
        }
    ]

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": api_key
    }

    response = requests.post(CORCEL_URL, json=payload, headers=headers, timeout=60)
    return response.text


def error_response(msg: str) -> Tuple[str, None, None, None]:
    """Return an error mech response."""
    return msg, None, None, None


def response_post_process(response: str, tool_name: str) -> str:
    """Loads the prediction into a json object"""

    # Join all the response pieces of content
    content_regex = r'\{"content":\s"(.*?)(?:",\s|"})'
    matches = re.findall(content_regex, response, re.DOTALL)
    joined = ("").join(matches)

    # Clean and parse the response
    clean_response = joined.replace('\\n', '').replace('"', '').replace('\\', '')

    if tool_name != "corcel-prediction":
        return clean_response

    # Add missing quotes to the json keys
    clean_response = re.sub(r'(\w+):', r'"\1":', clean_response)

    # Load the response
    try:
        json_response = json.loads(clean_response)
        return json.dumps(json_response)
    except json.JSONDecodeError:
        return f"Error: response could not be properly postprocessed: {clean_response}"


def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""

    api_key = kwargs.get("api_keys", {}).get("corcel", None)
    tool_name = kwargs.get("tool", None)
    prompt = kwargs.get("prompt", None)

    if api_key is None:
        return error_response("Corcel API key is not available.")

    if tool_name is None:
        return error_response("No tool name has been specified.")

    if tool_name not in AVAILABLE_TOOLS:
        return error_response(f"Tool {tool_name} is not an available tool [{AVAILABLE_TOOLS}].")

    if prompt is None:
        return error_response("No prompt has been given.")

    if tool_name == "corcel-prediction":
        kwargs["prompt"] = PREDICTION_OFFLINE_PROMPT.format(user_prompt=prompt)

    response = send_corcel_request(api_key=api_key, **kwargs)

    response = response_post_process(response, tool_name)

    return response, prompt, None, None
