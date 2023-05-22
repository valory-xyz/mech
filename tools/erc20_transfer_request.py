"""
python erc20_transfer_request.py “transfer 1 USDC to 0x812ecd8740Bfbd4b808860a442a0d3aF9C146c32”

WIP
"""

import os
import openai
from web3 import Web3, HTTPProvider
from web3.auto import w3
from eth_account import Account

ETHEREUM_LEDGER_RPC_1 = os.getenv("ETHEREUM_LEDGER_RPC_1")

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
token_amount = web3.toWei(10, 'ether')  # 10 tokens, for example
# contract ABI
contract_abi = 'ABI_JSON'  # replace with your contract's ABI
# get the contract instance
contract = web3.eth.contract(address=contract_address, abi=contract_abi)
# build a transaction dict
txn_dict = contract.functions.transfer(
    to_address,
    token_amount
).buildTransaction({
    'chainId': 1,  # replace with your chain ID
    'gas': 70000,
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
erc20_transfer_prompt = """
You are an LLM inside a multi-agent system that takes in a prompt from a user requesting you to execute an ERC20 token transfer on an EVM blockhain like ethereum. The agent process you are sending your response to requires the transaction object written by you in your response as an input to sign and execute the transaction in the agent process. The agent already has and knows the owned address for the agent service, "my_address" and the private key of "my_address". The agent does not know the receiving address, “recipient_address” or the value, “value” the user prompt indicates to send. The rest of the transaction object that is not known beforehand must be constructed by you from the user’s prompt information. 

here is the prompt from the user: 
"transfer 1 USDC to 0x812ecd8740Bfbd4b808860a442a0d3aF9C146c32" 

only respond with the format below using curly brackets to encapsulate the variables:

"from": my_address, 
"to": recipient_address, 
"value": value, 
"gas": gas_limit, 
"gasPrice": gas_price, 
"nonce": current_nonce, 


Do not respond with anything else other than the json transaction object you constructed with the correct known variables the agent had before the request and the correct unknown values found in the user request prompt as input to the web3.py signing method.
"""


def erc20_transfer():
    pass