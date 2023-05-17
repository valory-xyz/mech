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

### Option 1: Run the agent standalone

From one terminal run
```bash
bash run_agent.sh
```

From another terminal run
```bash
bash run_tm.sh
```

### Option 2: Run the service

First, copy the env file:
```bash
cp .example.env .1env
```

Provide your OpenAI API key in `.1env`.

Run, the service:
```bash
bash run_service.sh
```
