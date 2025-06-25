<p align="center">
   <img src="./docs/images/mechs-logo.png" width=300>
</p>

<h1 align="center" style="margin-bottom: 0;">
    Autonolas AI Mechs
    <br><a href="https://github.com/valory-xyz/mech/blob/main/LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/github/license/valory-xyz/mech"></a>
    <a href="https://pypi.org/project/open-autonomy/0.10.7/"><img alt="Framework: Open Autonomy 0.10.7" src="https://img.shields.io/badge/framework-Open%20Autonomy%200.10.7-blueviolet"></a>
    <!-- <a href="https://github.com/valory-xyz/mech/releases/latest">
    <img alt="Latest release" src="https://img.shields.io/github/v/release/valory-xyz/mech"> -->
    </a>
</h1>

This repository contains an AI Mech for the [Predict Agent Economy](https://olas.network/agent-economies/predict). 

## Requirements

You need the following requirements installed in your system:

- [Python](https://www.python.org/) (recommended `3.10`)
- [Poetry](https://python-poetry.org/docs/)
- [Docker Engine](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Tendermint](https://docs.tendermint.com/v0.34/introduction/install.html) `==0.34.19`

## Set up your environment

Follow these instructions to have your local environment prepared to run the demo below, as well as to build your own AI Mech.

1. Create a Poetry virtual environment and install the dependencies:

    ```bash
    poetry install && poetry shell
    ```

2. Fetch the software packages using the [Open Autonomy](https://docs.autonolas.network/open-autonomy/) CLI:

    ```bash
    autonomy packages sync --update-packages
    ```

    This will populate the Open Autonomy [local registry](https://docs.autonolas.network/open-autonomy/guides/set_up/#the-registries-and-runtime-folders) (folder `./packages`) with the required components to run the worker services.

## Run the Mech Predict

### Using Mech Quickstart (Preferred Method)

To help you integrate your own tools more easily, we’ve created a new base repository that serves as a minimal example of how to run the project. It’s designed to minimize setup time and provide a more intuitive starting point. This new repo is streamlined to give you a clean slate, making it easier than ever to get started.

**Why Use the New Base Repo?**

- Less Configuration: A clean setup that removes unnecessary complexities.
- Easier to Extend: Perfect for adding your own features and customizations.
- Clear Example: Start with a working example and build from there.

**Feature Comparison**

   | Feature                        | New Base Repo (Recommended)                        | Old Mech Repo (Not Preferred)                |
|---------------------------------|---------------------------------------------------|----------------------------------------------|
| **Setup Ease**                  | Simplified minimal setup and quick to start       | Requires extra configuration and more error prone |
| **Flexibility & Customization** | Easy to extend with your own features             | Less streamlined for extensions              |
| **Future Support**              | Actively maintained & improved                    | No longer the focus for updates              |
| **Complexity**                  | Low complexity, easy to use                       | More complex setup                          |


We highly encourage you to start with this base repo for future projects. You can find it [here](https://github.com/valory-xyz/mech-quickstart).

### Running the old base mech

> **Warning**<br />
The old repo is no longer the recommended approach for running and extending the project. Although it’s still remains available for legacy projects, we advise you to use the new base repo to ensure you are working with the most current and efficient setup. Access the new mech repo [here](https://github.com/valory-xyz/mech-quickstart). Start with the preferred method mentioned [above](#using-mech-quickstart-preffered-method).

Follow the instructions below to run the AI Mech demo executing the tool in `./packages/valory/customs/openai_request.py`. Note that AI Mechs can be configured to work in two modes: *polling mode*, which periodically reads the chain, and *websocket mode*, which receives event updates from the chain. The default mode used by the demo is *polling*.

First, you need to configure the worker service. You need to create a `.1env` file which contains the service configuration parameters. We provide a prefilled template (`.example.env`). You will need to provide or create an [OpenAI API key](https://platform.openai.com/account/api-keys).

```bash
# Copy the prefilled template
cp .example.env .1env

# Edit ".1env" and replace "dummy_api_key" with your OpenAI API key.

# Source the env file
source .1env
```

##### Environment Variables

You may customize the agent's behaviour by setting these environment variables.
 
| Name                       | Type   | Sample Value                                                                                                                                                                                                                                                        | Description                                                            |
| -------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `TOOLS_TO_PACKAGE_HASH`    | `dict` | `{"openai-gpt-3.5-turbo-instruct":"bafybeigz5brshryms5awq5zscxsxibjymdofm55dw5o6ud7gtwmodm3vmq","openai-gpt-3.5-turbo":"bafybeigz5brshryms5awq5zscxsxibjymdofm55dw5o6ud7gtwmodm3vmq","openai-gpt-4":"bafybeigz5brshryms5awq5zscxsxibjymdofm55dw5o6ud7gtwmodm3vmq"}` | Tracks services for each tool packages.                                |
| `API_KEYS`                 | `dict` | `{"openai":["dummy_api_key"], "google_api_key":["dummy_api_key"]}`                                                                                                                                                                                                      | Tracks API keys for each service.                                      |
| `SERVICE_REGISTRY_ADDRESS` | `str`  | `"0x9338b5153AE39BB89f50468E608eD9d764B755fD"`                                                                                                                                                                                                                      | Smart contract which registers the services.                           |
| `COMPLEMENTARY_SERVICE_METADATA_ADDRESS`   | `str`  | `"0x0598081D48FB80B0A7E52FAD2905AE9beCd6fC69"`                                                                                                                                                                                                                      | Smart contract which tracks metadata hash of the mech.                             |
| `MECH_MARKETPLACE_ADDRESS` | `str`  | `"0x4554fE75c1f5576c1d7F765B2A036c199Adae329"`                                                                                                                                                                                                                      | Marketplace for posting and delivering requests served by agent mechs. |
| `MECH_TO_SUBSCRIPTION`     | `dict` | `{"0x77af31De935740567Cf4fF1986D04B2c964A786a":{"tokenAddress":"0x0000000000000000000000000000000000000000","tokenId":"1"}}`                                                                                                                                        | Tracks mech's subscription details.                                    |
| `MECH_TO_CONFIG`           | `dict` | `{"0xFf82123dFB52ab75C417195c5fDB87630145ae81":{"use_dynamic_pricing":false,"is_marketplace_mech":false}}`                                                                                                                                                          | Tracks mech's config.                                                  |
 
The rest of the common environment variables are present in the [service.yaml](https://github.com/valory-xyz/mech/blob/main/packages/valory/services/mech/service.yaml), which are customizable too.


> **Warning**<br />
> **The demo service is configured to match a specific on-chain agent (ID 3 on [Mech Hub](https://aimechs.autonolas.network/registry)). Since you will not have access to its private key, your local instance will not be able to transact.
> However, it will be able to receive Requests for AI tasks [sent from Mech Hub](https://aimechs.autonolas.network/mech). These Requests will be executed by your local instance, but you will notice that a failure will occur when it tries to submit the transaction on-chain (Deliver type).**

Now, you have two options to run the worker: as a standalone agent or as a service.

### Option 1: Run the Mech as a standalone agent

1. Ensure you have a file with a private key (`ethereum_private_key.txt`). You can generate a new private key file using the Open Autonomy CLI:

   ```bash
   autonomy generate-key ethereum 
   ```

2. From one terminal, run the agent:

    ```bash
    bash run_agent.sh
    ```

3. From another terminal, run the Tendermint node:

    ```bash
    bash run_tm.sh
    ```

### Option 2: Run the Mech as an agent service

1. Ensure you have a file with the agent address and private key (`keys.json`). You can generate a new private key file using the Open Autonomy CLI:

    ```bash
    autonomy generate-key ethereum -n 1
    ```

2. Ensure that the variable `ALL_PARTICIPANTS` in the file `.1env` contains the agent address from `keys.json`:

   ```bash
   ALL_PARTICIPANTS='["your_agent_address"]'
   ```

3. Run, the service:

    ```bash
    bash run_service.sh
    ```

## Included tools

| Tools |
|---|
| packages/jhehemann/customs/prediction_sum_url_content |
| packages/napthaai/customs/prediction_request_rag |
| packages/napthaai/customs/resolve_market_reasoning |
| packages/nickcom007/customs/prediction_request_sme |
| packages/nickcom007/customs/sme_generation_request |
| packages/polywrap/customs/prediction_with_research_report |
| packages/psouranis/customs/optimization_by_prompting |
| packages/valory/customs/native_transfer_request |
| packages/valory/customs/openai_request |
| packages/valory/customs/prediction_request |
| packages/valory/customs/prediction_request_claude |
| packages/valory/customs/prediction_request_embedding |
| packages/valory/customs/resolve_market |

## More on tools

- **OpenAI request** (`openai_request.py`). Executes requests to the OpenAI API through the engine associated with the specific tool. It receives as input an arbitrary prompt and outputs the returned output by the OpenAI API.
  - `openai-gpt-3.5-turbo`
  - `openai-gpt-4`
  - `openai-gpt-3.5-turbo-instruct`

- **Stability AI request** (`stabilityai_request.py`): Executes requests to the Stability AI through the engine associated with the specific tool. It receives as input an arbitrary prompt and outputs the image data corresponding to the output of Stability AI.
  - `stabilityai-stable-diffusion-v1-5`
  - `stabilityai-stable-diffusion-xl-beta-v2-2-2`
  - `stabilityai-stable-diffusion-512-v2-1`
  - `stabilityai-stable-diffusion-768-v2-1`

- **Native transfer request** (`native_transfer_request.py`): Parses user prompt in natural language as input into an Ethereum transaction.
  - `transfer_native`

- **Prediction request** (`prediction_request.py`): Outputs the estimated probability of occurrence (`p_yes`) or no occurrence (`p_no`) of a certain event specified as the input prompt in natural language.
  - `prediction-offline`: Uses only training data of the model to make the prediction.
  - `prediction-online`: In addition to training data, it also uses online information to improve the prediction.
