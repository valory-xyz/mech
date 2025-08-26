# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2025 Valory AG
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

"""This script tests the prediction request tool."""

import json
import os

from dotenv import load_dotenv  # type: ignore

from packages.valory.customs.prediction_request.prediction_request import run
from packages.valory.skills.task_execution.utils.apis import KeyChain


load_dotenv(override=True)


TOOL = "prediction-online"
MODEL = "gpt-4.1-2025-04-14"
PROMPT = """
With the given question

"Will Oak Ridge National Laboratory publicly announce,
before or on September 16, 2025, the initiation of long-term durability testing for the 3D-printed
concrete reactor shielding columns used in the Hermes Low-Power Demonstration Reactor project?"

and the `yes` option represented by `Yes` and the `no` option represented by `No`,
what are the respective probabilities of `p_yes` and `p_no` occurring?
"""

API_KEYS = json.loads(os.getenv("API_KEYS", "{}"))

# Mocking a Google API exception                # noqa: E800
# from googleapiclient.errors import HttpError  # noqa: E800
# from httplib2 import Response                 # noqa: E800
# import json                                   # noqa: E800
# raise HttpError(                              # noqa: E800
#     Response({"status": str(429)}),           # noqa: E800
#     json.dumps("dummy").encode("utf-8"),      # noqa: E800
#     uri="https://fake.googleapis.com/test"    # noqa: E800
# )                                             # noqa: E800


def main() -> None:
    """Test the prediction request tool."""
    kwargs = {
        "tool": TOOL,
        "model": MODEL,
        "prompt": PROMPT,
        "api_keys": KeyChain(API_KEYS),
    }

    result = run(**kwargs)
    print(result)


if __name__ == "__main__":
    main()
