import json
import os

from dotenv import load_dotenv

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

# Mocking a Google API exception
# from googleapiclient.errors import HttpError
# from httplib2 import Response
# import json
# raise HttpError(
#     Response({"status": str(429)}),
#     json.dumps("dummy").encode("utf-8"),
#     uri="https://fake.googleapis.com/test"
# )


def main():
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
