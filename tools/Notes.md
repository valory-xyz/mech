# Generalized LLM -> EVM Tools

**Abstract**: 
Any given transaction to any given evm chain involves two sub categories and can be broken down into a series of steps assuming we have predefined the chain we want to search and execute on. Each chain has a "chain id" to identify by so we could end up using this to differentiate in the application side based on metadata about the user's wallet and what chain they are connected to. Once the web3 wallet connection has been set the logic of the mech determines based on the user's request which of the two sub categories the transaction request falls under. The two sub categories are as follows:

account_call: EVM calls to an account with no calldata (for example native token (ETH) transfers between accounts)
contract_call: EVM calls to a contract with calldata (for example a call to a smart contract function like erc20.transfer())

NOTE: for the sake of simplicity we will leave the case of calls to a contract with no calldata out of scope for now with exception of WETH perhaps.

The following steps are taken to execute a transaction request:

1. define the environment for the transaction (chain id)
2. define the transaction type in reference to the above sub categories (account_call or contract_call)
3. define the known variables of the transaction (from, gas, gasPrice, nonce, etc.)

if **account_call**:
- define the unknown transaction parameters using the request data (to, value)
    - LLM tool responds with the inferred "to" address and the "value" amount based on the request data
    - Agent process parses the response string into a dictionary containing the data given by the LLM tool and sets the previously unknown transaction parameters
- sign the transaction with the agent private key
- send the transaction to the chain
- return the transaction hash and other metadata if needed

if **contract_call**:
- define the undefined contract(s) needed to complete the transaction from the request using their respective **addresses**
    - we will need to grab these "addresses" for our python to evm interaction library (web3.py for example) from some sort of database/mapping made to associate specific names of contracts on the chain in question (USDC for example) to the contract addresses we want the agent process to use or simply use an API (rpc, block explorer api, nodes, etc.) and pray that the "address locator" pulls the correct address
    - Make a new LLM tool **address_locator** that takes in a prompt and retrieves the addresses of the contract(s) needed to complete the transaction (could be extended to multi-txs and/or multi-chain logic given their is multiple databases/APIs to pull from)
- define the contract(s) ABI
    - we will need to grab these ABIs from some sort of database/API (block explorer APIs, etc.) new LLM tool, **abi_locator** for example, given we have the address of the contract(s) needed to complete the transaction
    - Preceding this step we need the addresses of the contract(s) to retrieve the ABIs successfully (can be done also with block explorer APIs, RPCs, and potentially other data sources)
- define the contract function(s) in scope needed to complete the requested transaction/action from the user request data and the ABIs
    - LLM tool **function_locator**, responds with the inferred contract function(s) in scope of ABI needed to complete the requested transaction/action from the user request data + ABIs
    -  Agent process parses the response string containing needed functions for user request into a dictionary containing the data given by the LLM tool and sets the previously unknown contract function(s)
- define the sequence/order of the contract calls needed to complete the requested transaction(s)/action from the user request data + ABIs
    - LLM tool **function_ordering**, responds with the inferred sequence/order of the contract calls needed to complete the requested transaction(s)/action from the user request data + ABIs + functions in scope given by the function locator.
    - Agent process parses the response string into a dictionary containing the data given by the LLM tool and sets the previously unknown sequence/order of the contract function calls the order output by the LLM tool.
- define input parameters to the first contract.function() call in the ordered list given by the function ordering tool above using the user's prompt and the function/ABI. This is needed to make the call(s) to each contract function in scope needed for the user's request using the request data & sequence.
    - LLM tool **function_caller**, responds with the inferred input parameters needed to make the call(s) to each contract function in scope needed for the user's request based on the user prompt, the ABIs and the function sequence.
    - Agent process parses the response string into a dictionary containing the data given by the LLM tool (inputs to current function call) and sets the previously unknown input parameters to execute the transaction.
- return tx hash and/or hashes along with other metadata if needed