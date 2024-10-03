from llama_cpp import Llama
import json
from typing import Optional, Dict, Any, Tuple, Callable
import functools

MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

def with_key_rotation(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]

        def execute() -> MechResponse:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


PREDICTION_OFFLINE_PROMPT = """
You are an LLM that takes in a prompt of a user requesting a probability estimation
for a given event. You are provided with an input under the label "USER_PROMPT". You must follow the instructions
under the label "INSTRUCTIONS". You must provide your response in the format specified under "OUTPUT_FORMAT".

INSTRUCTIONS
* Read the input under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" specifies an event.
* The event will only have two possible outcomes: either the event will happen or the event will not happen.
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
* The sum of "p_yes" and "p_no" must equal 1.
* Output only the JSON object. Do not include any other contents in your response.
"""

AVAILABLE_TOOLS = ["llama-prediction", "llama-completion"]

AVAILABLE_MODELS = {
    "mistral": {
        "repo_id": "TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF",
        "filename": "*Q2_K.gguf",
    },
    "llama-3": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "*Q4_0.gguf",
    },
    "qwen-2.5": {
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "*q3_k_m.gguf",
    }
}

DEFAULT_VALUES = {
    "model": "llama-3",
    "temperature": 0.8
}


def inference(model: str, prompt: str, **kwargs) -> str:
    """Runs LLm inference locally"""

    messages = [
        {"role": "system", "content": prompt}
    ]

    llm = Llama.from_pretrained(
        repo_id=AVAILABLE_MODELS[model]["repo_id"],
        filename=AVAILABLE_MODELS[model]["filename"],
        verbose=False,
    )

    output = llm.create_chat_completion(
        messages=messages,
        temperature=kwargs.get("temperature", DEFAULT_VALUES["temperature"]),
    )

    return output["choices"][0]["message"]["content"]


def error_response(msg: str) -> Tuple[str, None, None, None]:
    """Return an error mech response."""
    return msg, None, None, None


def response_post_process(response: str) -> str:
    """Response posptprocessing"""
    try:
        json_response = json.loads(response)
        return json.dumps(json_response)
    except json.JSONDecodeError:
        return f"Response could not be properly postprocessed: {response}"


@with_key_rotation
def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""

    tool_name = kwargs.get("tool", None)
    kwargs["model"] = kwargs.get("model", "llama-3")
    prompt = kwargs.get("prompt", None)

    if tool_name is None:
        return error_response("No tool name has been specified.")

    if tool_name not in AVAILABLE_TOOLS:
        return error_response(f"Tool {tool_name} is not an available tool [{AVAILABLE_TOOLS}].")

    if prompt is None:
        return error_response("No prompt has been given.")

    if tool_name == "llama-prediction":
        kwargs["prompt"] = PREDICTION_OFFLINE_PROMPT.format(user_prompt=prompt)

    response = inference(**kwargs)

    if tool_name == "llama-prediction":
        response = response_post_process(response)

    return response, prompt, None, None
