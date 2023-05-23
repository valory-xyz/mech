"""
python native_transfer_request.py “transfer 0.001 ETH to 0x812ecd8740Bfbd4b808860a442a0d3aF9C146c32”

WIP
"""

import ast
import os
import sys
from web3 import Web3
from openai_request import run

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_PRIVATE_KEY = os.getenv("AGENT_PRIVATE_KEY")
ETHEREUM_LEDGER_RPC_0 = os.getenv("ETHEREUM_LEDGER_RPC_0")


"""NOTE: In the case of native token transfers on evm chains we do not need any contract address or ABI. The only unknowns are the "recipient address" and the "value" to send for evm native transfers."""
native_token_transfer_prompt = """
You are an LLM inside a multi-agent system that takes in a prompt from a user requesting you to execute a native gas token (ETH) transfer to another public address on Ethereum. 
The agent process you are sending your response to requires the transaction object written by you in your response as an input to sign and execute the transaction in the agent process. 
The agent already knows the owned address for the agent service, "my_address", the gas limit in the variable "gas_limit", the gas price in the variable "gas_price" and the 
nonce of the transaction in the variable "current_nonce". The agent does not know the receiving address, “recipient_address” or the value, “value”, the user prompt indicates to send. 
The rest of the transaction object that is not known beforehand must be constructed by you from the user's prompt information. 

User Prompt: {user_prompt}

only respond with the format below using curly brackets to encapsulate the variables:

"to": recipient_address, 
"value": value, 

Do not respond with anything else other than the transaction object you constructed with the correct known variables the agent had before the request and the correct unknown values found in the user request prompt as input to the web3.py signing method.
"""


def native_transfer(**kwargs) -> str:
    """
    Execute a native token transfer on Ethereum. This involves:
    1. using the user prompt to create the formatted tool prompt for the LLM (Done)
    2. using the tool prompt to create the transaction object in string form (Done)
    3. parsing the transaction object string to get the transaction object (WIP)
    4. signing the transaction object with the private key of the agent (WIP)
    5. sending the signed transaction object + other data if desired to the agent process (WIP)
    """

    print("\033[95m\033[1m" + "\n USER PROMPT:" + "\033[0m\033[0m")
    print(str(kwargs["prompt"]))

    tool_prompt = native_token_transfer_prompt.format(user_prompt=str(kwargs["prompt"]))

    print("\033[95m\033[1m" + "\n FORMATTED PROMPT FOR TOOL:" + "\033[0m\033[0m")
    print(tool_prompt)

    # create a web3 instance
    w3 = Web3(Web3.HTTPProvider((ETHEREUM_LEDGER_RPC_0)))

    # Define the known variables
    # Agent address
    my_address = "0x812ecd8740Bfbd4b808860a442a0d3aF9C146c32"
    gas_limit = 21000
    gas_price = w3.eth.gas_price
    current_nonce = w3.eth.get_transaction_count(my_address)

    # use openai_request tool
    response = run(api_keys={"openai": OPENAI_API_KEY}, prompt=tool_prompt, tool="gpt-3.5-turbo")

    print("\033[95m\033[1m" + "\n RESPONSE FROM GPT:" + "\033[0m\033[0m")
    print(response)

    # parse the response to get the transaction object string itself
    parsed_txs = ast.literal_eval(response)
    
    print("\033[95m\033[1m" + "\n PARSED TXS OBJECT:" + "\033[0m\033[0m")
    print(parsed_txs)
    

    # build the transaction object, unknowns are referenced from parsed_txs
    transaction = {
        "from": my_address,
        "to": str(parsed_txs["to"]),
        "value": str(parsed_txs["value"]),
        "gas": gas_limit,
        "gasPrice": gas_price,
        "nonce": current_nonce,
    }

    # sign the transaction
    signed_txn = w3.eth.account.sign_transaction(transaction, private_key=AGENT_PRIVATE_KEY)

    # send the transaction
    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)

    return response, tx_hash



def main(task: str):
    """Run the task"""

    # build dictionary of kwargs
    kwargs = {
        "prompt": task,
        "engine": "openai-gpt3.5-turbo",
        "max_tokens": 500,
        "temperature": .7,
        "top_p": 1,
        "api_keys": {"openai": OPENAI_API_KEY},
    }

    response, tx_hash = native_transfer(prompt=kwargs["prompt"]), 
    print("RESPONSE FROM GPT: " + response)
    print("TX HASH: " + tx_hash)


if __name__ == "__main__":
    task = sys.argv
    try:
        main(task)
    except KeyboardInterrupt:
        pass