"""
python erc20_transfer_request.py “transfer 1 USDC to 0x812ecd8740Bfbd4b808860a442a0d3aF9C146c32”

WIP
"""

import os
from web3 import Web3
from openai_request import run
import ast

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_PRIVATE_KEY = os.getenv("AGENT_PRIVATE_KEY")
ETHEREUM_LEDGER_RPC_0 = os.getenv("ETHEREUM_LEDGER_RPC_0")

"""
# connect to your node
web3 = Web3(HTTPProvider(ETHEREUM_LEDGER_RPC_1))
# your private key
private_key = 'YOUR_PRIVATE_KEY'
# get account details
account = Account.privateKeyToAccount(private_key)
# ERC20 token contract address (this is just a sample)
contract_address = web3.toChecksumAddress('0xTokenContractAddress')
# recipient address
to_address = web3.toChecksumAddress('0xRecipientAddress')
# the amount of token to transfer
token_amount = web3.toWei(10, 'ether')
# contract ABI USDC
contract_abi = []
# get the contract instance
contract = web3.eth.contract(address=contract_address, abi=contract_abi)
# build a transaction dict
txn_dict = contract.functions.transfer(
    to_address,
    token_amount
).buildTransaction({
    'chainId': 1,  # replace with your chain ID
    'gas': 100000,
    'gasPrice': w3.toWei('1', 'gwei'),
    'nonce': web3.eth.getTransactionCount(account.address),
})
# sign the transaction
signed_txn = web3.eth.account.signTransaction(txn_dict, private_key)
# send the transaction
txn_hash = web3.eth.sendRawTransaction(signed_txn.rawTransaction)
# get transaction receipt
txn_receipt = None
txn_receipt = web3.eth.getTransactionReceipt(txn_hash)
"""

# WIP
"""NOTE: In the case of the erc20 transfers we need a "contract abi", "token address", "value" and "recipient address" to send the txs."""
erc20_transfer_prompt = """
You are an LLM inside a multi-agent system that takes in a prompt from a user requesting you to execute an ERC20 token transfer on an EVM blockhain like ethereum. 
The agent process you are sending your response to requires the transaction object written by you in your response as an input to sign and execute the transaction in the agent process. 
The agent already has/knows the owned address for the agent service, "my_address". The agent does not know the receiving address, “recipient_address”, the value, “value” and the erc20 
contract address, “contract_address”. The rest of the transaction object that is not known beforehand must be constructed by you from the user's prompt information. 

here is the prompt from the user: 
{user_prompt}

only respond with the format below using curly brackets to encapsulate the variables:

"to": recipient_address, 
"value": value, 
"token_address": contract_address,

Do not respond with anything else other than the json transaction object containing the information you constructed with the correct known variables the agent had before the request and the correct unknown values found in the user request prompt as input to the web3.py signing method.
"""


def erc20_transfer(**kwargs):
    """
    Execute an erc20 token transfer on an evm chain. This involves:
    1. using the user prompt to create the formatted tool prompt for the LLM 
    2. using the tool prompt to create the transaction data object in string format
    3. parse the transaction object string to get the transaction data
    4. sign transaction using parsed data with the private key of the agent
    5. sending the signed transaction + other data if desired to the agent process
    """

    print("\033[95m\033[1m" + "\n USER PROMPT:" + "\033[0m\033[0m")
    print(str(kwargs["prompt"]))

    tool_prompt = erc20_transfer_prompt.format(user_prompt=str(kwargs["prompt"]))

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
    contract_abi = [{"constant":false,"inputs":[{"name":"newImplementation","type":"address"}],"name":"upgradeTo","outputs":[],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":false,"inputs":[{"name":"newImplementation","type":"address"},{"name":"data","type":"bytes"}],"name":"upgradeToAndCall","outputs":[],"payable":true,"stateMutability":"payable","type":"function"},{"constant":true,"inputs":[],"name":"implementation","outputs":[{"name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"name":"newAdmin","type":"address"}],"name":"changeAdmin","outputs":[],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[],"name":"admin","outputs":[{"name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"},{"inputs":[{"name":"_implementation","type":"address"}],"payable":false,"stateMutability":"nonpayable","type":"constructor"},{"payable":true,"stateMutability":"payable","type":"fallback"},{"anonymous":false,"inputs":[{"indexed":false,"name":"previousAdmin","type":"address"},{"indexed":false,"name":"newAdmin","type":"address"}],"name":"AdminChanged","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"name":"implementation","type":"address"}],"name":"Upgraded","type":"event"}]

    # use openai_request tool
    response = run(api_keys={"openai": OPENAI_API_KEY}, prompt=tool_prompt, tool="gpt-3.5-turbo")

    print("\033[95m\033[1m" + "\n RESPONSE FROM GPT:" + "\033[0m\033[0m")
    print(response)

    # parse the response to get the transaction object string itself
    parsed_txs = ast.literal_eval(response)
    
    print("\033[95m\033[1m" + "\n PARSED TXS OBJECT:" + "\033[0m\033[0m")
    print(parsed_txs)
    
    # contract instance
    erc20_contract = w3.eth.contract(address=parsed_txs["token_address"], abi=contract_abi)

    # build the transaction object, unknowns are referenced from parsed_txs
    txn_dict = erc20_contract.functions.transfer(
        str(parsed_txs["to"]),
        str(parsed_txs["value"])
    ).buildTransaction({
    'chainId': 1,  # replace with your chain ID
    'gas': 100000,
    'gasPrice': w3.toWei('1', 'gwei'),
    'nonce': w3.eth.getTransactionCount(my_address),
    })

    # sign the transaction
    signed_txn = w3.eth.account.sign_transaction(txn_dict, private_key=AGENT_PRIVATE_KEY)

    # send the transaction
    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)

    return response, tx_hash