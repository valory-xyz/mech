"""
python native_transfer_request.py “transfer 0.0001 ETH to 0x4253cB6Fbf9Cb7CD6cF58FF9Ed7FC3BDbd8312fe"

WIP
"""

import ast
import os
import sys
from openai_request import run

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_PRIVATE_KEY = os.getenv("AGENT_PRIVATE_KEY")
ETHEREUM_LEDGER_RPC_0 = os.getenv("ETHEREUM_LEDGER_RPC_0")
TEST_RPC = os.getenv("TEST_RPC")


"""NOTE: In the case of native token transfers on evm chains we do not need any contract address or ABI. The only unknowns are the "recipient address" and the "value" to send for evm native transfers."""
native_token_transfer_prompt = """
You are an LLM inside a multi-agent system that takes in a prompt from a user requesting you to execute a native gas token (ETH) transfer to another public address on Ethereum. 
The agent process you are sending your response to requires the unknown transaction parameters in the exact format below, written by you in your response as an input to sign/execute the transaction in the agent process. 
The agent does not know the receiving address, “recipient_address", the value to send, “value”, or the denomination of the "value" given in wei "wei_value" which is converted by you without use of any functions, the user prompt indicates to send. The unknown 
transaction parameters not known beforehand must be constructed by you from the user's prompt information. 

User Prompt: {user_prompt}

only respond with the format below using curly brackets to encapsulate the variables within a python dictionary object and no other text:

"to": recipient_address, 
"value": value, 
"wei_value": wei_value

Do not respond with anything else other than the transaction object you constructed with the correct known variables the agent had before the request and the correct unknown values found in the user request prompt as input to the web3.py signing method.
"""


def native_transfer(**kwargs) -> str:
    """
    Execute a native token transfer on the EVM. This involves:
    1. using the user prompt to create the formatted tool prompt for the LLM (Done)
    2. using the tool prompt to create the transaction object in string form (Done)
    3. parsing the transaction object string to get the transaction object (WIP)
    4. encoding the transaction object (WIP)
    4. return the encoded transaction object (WIP)
    """

    # format the tool prompt
    tool_prompt = native_token_transfer_prompt.format(user_prompt=str(kwargs["prompt"]))

    # use openai_request tool
    response = run(api_keys={"openai": OPENAI_API_KEY}, prompt=tool_prompt, tool='openai-gpt-3.5-turbo')

    print("\033[95m\033[1m" + "\n RESPONSE FROM GPT:" + "\033[0m\033[0m")
    print(response)

    # parse the response to get the transaction object string itself
    parsed_txs = ast.literal_eval(response)
    
    print("\033[95m\033[1m" + "\n PARSED TXS OBJECT:" + "\033[0m\033[0m")
    print(parsed_txs)

    # Txs List
    txs_list = []

    # build the transaction object, unknowns are referenced from parsed_txs
    transaction = {
        "to": str(parsed_txs["to"]),
        "value": str(parsed_txs["wei_value"]),
    }

    # return the encoded transaction object
    txs_list = [transaction]

    return txs_list



def main(task: str):
    """Run the task"""

    # build dictionary of kwargs
    kwargs = {
        "prompt": task,
        "tool": "openai-gpt3.5-turbo",
        "engine": "gpt-3.5-turbo",
        "max_tokens": 500,
        "temperature": .7,
        "top_p": 1,
        "api_keys": {"openai": OPENAI_API_KEY},
    }

    txs_list = native_transfer(prompt=kwargs["prompt"]), 
    print("RETURNED DICT LIST : " + str(txs_list))


if __name__ == "__main__":
    task = sys.argv
    try:
        main(task)
    except KeyboardInterrupt:
        pass