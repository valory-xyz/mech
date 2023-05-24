"""
python univ2_trade_request.py swap .001 WETH to WBTC on sushiswap"

WIP
"""

import os
from web3 import Web3
from openai_request import run
import ast

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_PRIVATE_KEY = os.getenv("AGENT_PRIVATE_KEY")
ETHEREUM_LEDGER_RPC_0 = os.getenv("ETHEREUM_LEDGER_RPC_0")


# WIP
"""NOTE: In the case of the uniV2 trades we need a "contract abi", "from token address", "to token address", "value" and "recipient address" to send the txs."""
uniV2_transfer_prompt = """
You are an LLM inside a multi-agent system that takes in a prompt from a user requesting you to execute a trade on sushiswap on an EVM blockhain like ethereum. 
The agent process you are sending your response to requires the transaction object written by you in your response as an input to sign and execute the transaction in the agent process. 
The agent already has/knows the owned address for the agent service, "my_address". The agent does not know the receiving address, “recipient_address”, the value, “value”, the token to 
trade from, "from_token_addr", the token to trade to, "to_token_addr", and the sushiswap router address, “router_address”. The rest of the transaction object that is not known 
beforehand must be constructed by you from the user's prompt information. 

here is the prompt from the user: 
{user_prompt}

only respond with the format below using curly brackets to encapsulate the variables:

"value": value, 
"from_token_address": from_token_address,
"to_token_address": contract_address,
"router_address": router_address,

Do not respond with anything else other than the json transaction object containing the information you constructed with the correct known variables the agent had before the request and the correct unknown values found in the user request prompt as input to the web3.py signing method.
"""


def univ2_trade(**kwargs):
    """
    Execute a uniV2 swap on an evm chain.
    """

    print("\033[95m\033[1m" + "\n USER PROMPT:" + "\033[0m\033[0m")
    print(str(kwargs["prompt"]))

    tool_prompt = uniV2_transfer_prompt.format(user_prompt=str(kwargs["prompt"]))

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
    from_token = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    to_token = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
    deadline =  w3.eth.getBlock("latest")["timestamp"] + 1000
    router_address = "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F"
    # contract ABIs
    contract_abi_weth = []
    contract_abi_wbtc = []
    contract_abi_router = []

    # use openai_request tool
    response = run(api_keys={"openai": OPENAI_API_KEY}, prompt=tool_prompt, tool="gpt-3.5-turbo")

    print("\033[95m\033[1m" + "\n RESPONSE FROM GPT:" + "\033[0m\033[0m")
    print(response)

    # parse the response to get the transaction object string itself
    parsed_txs = ast.literal_eval(response)
    
    print("\033[95m\033[1m" + "\n PARSED TXS OBJECT:" + "\033[0m\033[0m")
    print(parsed_txs)

    # extract unknown variables from the response
    value = parsed_txs["value"]
    
    # contract instances
    weth = w3.eth.contract(address="from_token", abi=contract_abi_weth)
    wbtc = w3.eth.contract(address="to_token", abi=contract_abi_wbtc)
    router = w3.eth.contract(address="router_address", abi=contract_abi_router)

    # construct the "path" using token addresses
    path = [from_token, to_token]

    # check if the agent has enough balance to make the trade
    # check if the agent has enough allowance to make the trade
    # check if the agent has enough gas to make the trade

    # construct the transaction object
    # Construct the transaction data
    transaction = router.functions.swapExactTokensForTokens(
        value, 0, path, my_address, deadline
    ).buildTransaction({
        'from': my_address,
        'gas': 200000,
        'gasPrice': gas_price,
        'nonce': current_nonce,
    })

    # sign the transaction
    signed_txn = w3.eth.account.sign_transaction(transaction, private_key=AGENT_PRIVATE_KEY)

    # send the transaction
    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)

    return response, tx_hash