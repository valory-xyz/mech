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

"""This script tests multiple tools with multiple markets."""

import json
import os
from datetime import datetime
from timeit import default_timer as timer

from dotenv import load_dotenv  # type: ignore

from packages.jhehemann.customs.prediction_sum_url_content.prediction_sum_url_content import (
    run as prediction_sum_url_content_run,
)
from packages.napthaai.customs.prediction_request_rag.prediction_request_rag import (
    run as prediction_request_rag_run,
)
from packages.napthaai.customs.prediction_request_reasoning.prediction_request_reasoning import (
    run as prediction_request_reasoning_run,
)
from packages.napthaai.customs.prediction_url_cot.prediction_url_cot import (
    run as prediction_url_cot_run,
)
from packages.napthaai.customs.resolve_market_reasoning.resolve_market_reasoning import (
    run as resolve_market_reasoning_run,
)
from packages.nickcom007.customs.prediction_request_sme.prediction_request_sme import (
    run as prediction_request_sme_run,
)
from packages.valory.customs.prediction_request.prediction_request import (
    run as prediction_request_run,
)
from packages.valory.customs.superforcaster.superforcaster import (
    run as superforcaster_run,
)
from packages.valory.skills.task_execution.utils.apis import KeyChain


load_dotenv(override=True)

TOOLS_TO_RUN_FUNCTION = {
    "prediction-offline": prediction_request_run,
    "prediction-online": prediction_request_run,
    "claude-prediction-online": prediction_request_run,
    "claude-prediction-offline": prediction_request_run,
    "prediction-online-sme": prediction_request_sme_run,
    "prediction-offline-sme": prediction_request_sme_run,
    "prediction-online-sum-url-content": prediction_sum_url_content_run,
    "prediction-request-rag": prediction_request_rag_run,
    "prediction-request-rag-claude": prediction_request_rag_run,
    "prediction-request-reasoning": prediction_request_reasoning_run,
    "prediction-request-reasoning-claude": prediction_request_reasoning_run,
    "prediction-url-cot": prediction_url_cot_run,
    "prediction-url-cot-claude": prediction_url_cot_run,
    "superforcaster": superforcaster_run,
    "resolve-market-reasoning-gpt-4.1": resolve_market_reasoning_run,
}

TOOLS_TO_TEST = [
    "prediction-offline",
    "prediction-online",
    "claude-prediction-online",
    "claude-prediction-offline",
    "prediction-request-rag",
    "prediction-request-rag-claude",
    "prediction-request-reasoning",
    "prediction-request-reasoning-claude",
    "resolve-market-reasoning-gpt-4.1",
    "prediction-url-cot",
    "prediction-url-cot-claude",
]

# TOOL = "prediction-online"
MODEL_GPT = "gpt-4.1-2025-04-14"
MODEL_CLAUDE = "claude-3-5-sonnet-20240620"

MARKETS = [
    """Will any women's football club publicly announce, before or on August 27, 2025, the signing of a player for a transfer fee exceeding £1.1 million?""",
    """Will the FBI publicly announce, before or on August 31, 2025, the implementation of new technology or protocols aimed at more effectively tracking or identifying perpetrators of swatting incidents?""",
    """Will Evergrande's liquidators publicly announce, before or on August 30, 2025, the sale of at least $500 million in assets as part of the company's ongoing liquidation process?""",
    """Will at least one additional professional tennis player ranked outside the WTA or ATP top 100 publicly announce joining OnlyFans before or on August 30, 2025?""",
    """Will Google publicly release, before or on August 31, 2025, an updated environmental impact report for Gemini AI that includes both direct and indirect water usage figures?""",
]
PROMPTS = [
    f"""With the given question \"{market}\" and the `yes` option represented by `Yes` and the `no` option represented by `No`, what are the respective probabilities of `p_yes` and `p_no` occurring?"""
    for market in MARKETS
]


CURRENT_TIME_PREFIX = datetime.now().strftime("%Y%m%d_%H%M%S")


API_KEYS = json.loads(os.getenv("API_KEYS", "{}"))

JSON_KEYS = {"p_yes", "p_no", "confidence", "info_utility"}
TEST_RESULTS_DIR = "./tools_test_results"


def check_keys(json_obj) -> bool:
    """Check if all json keys are present."""
    return all(key in json_obj for key in JSON_KEYS)


def test_prediction_tool(prompt: str, tool: str, model: str) -> None:
    """Test a specific tool with a specific model."""

    kwargs = {
        "tool": tool,
        "model": model,
        "prompt": prompt,
        "api_keys": KeyChain(API_KEYS),
    }
    result = TOOLS_TO_RUN_FUNCTION[tool](**kwargs)

    # result is a tuple with actual result as first element. Actual result is a dict in string format.
    try:
        actual_result = json.loads(result[0])
    except json.JSONDecodeError as e:
        raise AssertionError(f"Result is not valid JSON: {result[0]}") from e
    assert check_keys(
        actual_result
    ), f"Missing keys in result for tool {tool} with model {model}: {actual_result.keys()}"

    assert (
        0.0 <= actual_result["p_yes"] <= 1.0
    ), f"p_yes out of bounds: {actual_result['p_yes']}"
    assert (
        0.0 <= actual_result["p_no"] <= 1.0
    ), f"p_no out of bounds: {actual_result['p_no']}"
    assert (
        0.0 <= actual_result["confidence"] <= 1.0
    ), f"confidence out of bounds: {actual_result['confidence']}"
    assert (
        0.0 <= actual_result["info_utility"] <= 1.0
    ), f"info_utility out of bounds: {actual_result['info_utility']}"
    assert (actual_result["p_yes"] + actual_result["p_no"]) == 1.0, (
        f"p_yes and p_no do not sum to 1: {actual_result['p_yes']} + {actual_result['p_no']} = "
        f"{actual_result['p_yes'] + actual_result['p_no']}"
    )
    return result


def test_market_resolution_tool(prompt: str, tool: str, model: str) -> None:
    """Test a specific tool with a specific model."""

    kwargs = {
        "tool": tool,
        "model": model,
        "prompt": prompt,
        "api_keys": KeyChain(API_KEYS),
    }
    result = TOOLS_TO_RUN_FUNCTION[tool](**kwargs)

    # result is a tuple with actual result as first element. Actual result is a dict in string format.
    actual_result = json.loads(result[0])

    assert (
        "has_occurred" in actual_result
    ), f"Missing 'resolution' key in result: {actual_result.keys()}"
    assert actual_result["has_occurred"] in {
        True,
        False,
    }, f"Invalid resolution value: {actual_result['resolution']}. Expected 'Yes' or 'No'."
    return result


def get_current_agent_hash() -> str:
    """Get the current agent hash from packages.json."""
    with open("./packages/packages.json", "r") as f:
        packages = json.load(f)
    return packages["dev"]["agent/valory/mech/0.1.0"]


def save_result(
    tool: str, model: str, result: str, time_taken: str, prompt: str
) -> None:
    """Save the result to a file."""
    if not os.path.exists(TEST_RESULTS_DIR):
        os.makedirs(TEST_RESULTS_DIR)
    current_agent_hash = get_current_agent_hash()
    filename = os.path.join(
        TEST_RESULTS_DIR, f"{CURRENT_TIME_PREFIX}_{current_agent_hash}.jsonl"
    )
    with open(filename, "a+") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "tool": tool,
                    "model": model,
                    "result": result,
                    "time_taken": time_taken,
                    "prompt": prompt,
                }
            )
            + "\n"
        )


def main() -> None:
    """Test the prediction request tool."""

    for tool in TOOLS_TO_TEST:
        model = MODEL_GPT if "cot" not in tool else MODEL_CLAUDE
        for market, prompt in zip(MARKETS, PROMPTS):
            print(f"Testing tool: {tool} with model: {model}")
            start_time = timer()
            result = (
                test_prediction_tool(prompt, tool, model)
                if "resolve-market-reasoning" not in tool
                else test_market_resolution_tool(market, tool, model)
            )
            end_time = timer()
            elapsed_time = end_time - start_time
            print(
                f"Time taken for tool {tool} with model {model}: {elapsed_time:.2f} seconds"
            )
            save_result(tool, model, result[0], f"{elapsed_time:.2f} seconds", prompt)


if __name__ == "__main__":
    main()
