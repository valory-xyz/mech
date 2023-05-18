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

After development:
```bash
poetry run autonomy packages lock
poetry run autonomy push-all
poetry run autonomy build-image --service-dir packages/valory/services/mech
```

To generate a hash of the tools:
```bash
python push_to_ipfs.py "tools/openai_request.py"
```

Then run on Propel.

### Prepare env file

First, copy the env file:
```bash
cp .example.env .1env
```

Provide your OpenAI API key in place of `dummy_api_key` in `.1env`.

Source the env file:
```bash
source .env
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
