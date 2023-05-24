# mech
Mech for EthLisbon hack

## Installation

Install project dependencies (you can find install instructions for Poetry [here](https://python-poetry.org/docs/)):
```bash
poetry shell
poetry install
```

Fetch all the packages
```bash
autonomy packages sync --update-packages
```

## Run locally as is

Note - the service is by default configured to match a specific on-chain representation (id 3 here: https://aimechs.autonolas.network/registry) - since you won't hold the private key for that agent your local instance won't be able to transact. However, you will receive events from transactions (Request type) sent to the on-chain mech here: https://aimechs.autonolas.network/mech. These Requests will be worked by the deployed instance. Your local deployment will fail when trying to send the transaction (Deliver type).

### Prepare env file

First, copy the env file:
```bash
cp .example.env .1env
```

Provide your OpenAI API key in place of `dummy_api_key` in `.1env`.

Source the env file:
```bash
source .1env
```

### Option 1: Run the agent standalone

Ensure you have a file with the private key at `ethereum_private_key.txt`.

From one terminal run
```bash
bash run_agent.sh
```

From another terminal run
```bash
bash run_tm.sh
```

### Option 2: Run the service

Ensure you have a file with the private key at `keys.json`.

Run, the service:
```bash
bash run_service.sh
```

## Create your own instance

### 1. Create a new tool

You can add your tool in `tools` as a single python file. The file must contain a `run` function that accepts `kwargs` and returns a string.

```python
def run(**kwargs) -> str:
    """Run the task"""

    # YOUR CODE

    return result  # a string
```

The `kwargs` are guaranteed to contain the keys `api_keys`, (`kwargs["api_keys"]`) a dictionary itself and discussed below, `prompt`, a string containing the prompt, and `tool`, a string specifying the tool to be used.

Once finished, upload the tool file to IPFS and generate a hash, e.g.:
```bash
python push_to_ipfs.py "tools/openai_request.py"
```

### 2. Configure your service

The default service has this configuration:
```bash
FILE_HASH_TO_TOOLS=[[bafybeihhxncljjtzniecvm7yr7u44g6ooquzqek473ma5fcnn2f6244v3e, [openai-text-davinci-002, openai-text-davinci-003, openai-gpt-3.5-turbo, openai-gpt-4]]]
API_KEYS=[[openai, dummy_api_key]]
```

To add a tool with hash `xyz` and tool list `[a, b, c]` and api keys `secret` simply update this to:
```bash
FILE_HASH_TO_TOOLS=[[bafybeihhxncljjtzniecvm7yr7u44g6ooquzqek473ma5fcnn2f6244v3e, [openai-text-davinci-002, openai-text-davinci-003, openai-gpt-3.5-turbo, openai-gpt-4]],[xyz, [a,b,c]]]
API_KEYS=[[openai, dummy_api_key],[xyz, secret]]
```

Then also register your service on [Autonolas](https://protocol.autonolas.network/services/mint) and create a mech for it [here](https://aimechs.autonolas.network/factory). This will allow you to set `SAFE_CONTRACT_ADDRESS` and `AGENT_MECH_CONTRACT_ADDRESS`.

Here an example of the agent NFT metadata:
```json
{"name":"Autonolas Mech III","description":"The mech executes AI tasks requested on-chain and delivers the results to the requester.","inputFormat":"ipfs-v0.1","outputFormat":"ipfs-v0.1","image":"tbd","tools": ["openai-text-davinci-002", "openai-text-davinci-003", "openai-gpt-3.5-turbo", "openai-gpt-4"]}
```

### 3. Run your service locally as per above

Once your service works locally, you have the option to run it on hosted service [Propel](https://propel.valory.xyz/).
