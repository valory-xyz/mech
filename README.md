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


The execution of AI tasks, such as image generation using DALL-E, prompt processing with ChatGPT, or more intricate operations involving on-chain transactions, poses a number of challenges, including:
- Access to proprietary APIs, which may come with associated fees/subscriptions.
- Proficiency in the usage of the the related open-source technologies, which may entail facing their inherent complexities.

AI Mechs run on the [Gnosis chain](https://www.gnosis.io/), and enables you to post *AI tasks requests* on-chain and get their result delivered back to you efficiently. An AI Mech will execute these tasks for you. All you need is some xDAI in your wallet to reward the worker service executing your task. AI Mechs are **hassle-free**, **crypto-native**, and **infinitely composable**.

> :bulb: These are just a few ideas on what capabilities can be brought on-chain with AI Mechs:
> - fetch real-time **web search** results
> - integrate **multi-sig wallets**,
> - **simulate** chain transactions
> - execute a variety of **AI models**:
>   - **generative** (e.g, Stability AI, Midjourney),
>   - **action-based** AI agents (e.g., AutoGPT, LangChain)

**AI Mechs is a project born at [ETHGlobal Lisbon](https://ethglobal.com/showcase/ai-mechs-dt36e).**

## AI Mechs components

The project consists of three components:

- Off-chain AI workers, each of which controls a Mech. Each AI worker is implemented as an autonomous service on the Autonolas stack.
- An on-chain protocol, which is used to generate a registry of AI Mechs, represented as NFTs on-chain.
- [Mech Hub](https://aimechs.autonolas.network/), a frontend which allows to interact with the protocol:
  - Gives an overview of the AI workers in the registry.
  - Allows Mech owners to create new workers.
  - Allows users to request work from an existing worker.

## Requirements

This repository contains a demo AI Mech. You can clone and extend the codebase to create your own AI Mech. You need the following requirements installed in your system:

- [Python](https://www.python.org/) (recommended `3.10`)
- [Poetry](https://python-poetry.org/docs/)
- [Docker Engine](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Tendermint](https://docs.tendermint.com/v0.34/introduction/install.html) `==0.34.19`

## Set up your environment

Follow these instructions to have your local environment prepared to run the demo below, as well as to build your own AI Mech.

1. Create a Poetry virtual environment and install the dependencies:

    ```bash
    poetry run pip install openapi-core==0.13.2
    poetry run pip install openapi-spec-validator==0.2.8
    poetry install && poetry shell
    ```

2. Fetch the software packages using the [Open Autonomy](https://docs.autonolas.network/open-autonomy/) CLI 

    ```bash
    autonomy packages sync --update-packages
    ```

    This will populate the Open Autonomy [local registry](https://docs.autonolas.network/open-autonomy/guides/set_up/#the-registries-and-runtime-folders) (folder `./packages`) with the required components to run the worker services.

## Run the demo

Follow the instructions below to run the AI Mech demo executing the tool in `./tools/openai_request.py`. Note that AI Mechs can be configured to work in two modes: *polling mode*, which periodically reads the chain, and *websocket mode*, which receives event updates from the chain. The default mode used by the demo is *polling*.

First, you need to configure the worker service. You need to create a `.1env` file which contains the service configuration parameters. We provide a prefilled template (`.example.env`). You will need to provide or create an [OpenAI API key](https://platform.openai.com/account/api-keys).

```bash
# Copy the prefilled template
cp .example.env .1env

# Edit ".1env" and replace "dummy_api_key" with your OpenAI API key.

# Source the env file
source .1env
```

> **Warning**<br />
> **The demo service is configured to match a specific on-chain agent (ID 3 on [Mech Hub](https://aimechs.autonolas.network/registry). Since you will not have access to its private key, your local instance will not be able to transact.
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

## Build your own

You can create and mint your own AI Mech that handles requests for tasks that you can define.

1. **Create a new tool.** Tools are the components that execute the Requests for AI tasks submitted on [Mech Hub](https://aimechs.autonolas.network/mech). Tools must be located in the folder `./tools` as a single Python file. Such file must contain a `run` function that accepts `kwargs` and must **always** return a string (`str`). That is, the `run` function must not raise any exception. If exceptions occur inside the function, they must be processed, and the return value must be set accordingly, for example, returning an error code.

    ```python
    def run(**kwargs) -> str:
        """Run the task"""

        # Your code here

        return result  # a string
    ```

    The `kwargs` are guaranteed to contain:
    * `api_keys` (`kwargs["api_keys"]`): the required API keys. This is a dictionary containing the API keys required by your Mech:
        ```python
        <api_key>=kwargs["api_keys"][<api_key_id>]).
        ```
    * `prompt` (`kwargs["prompt"]`): a string containing the user prompt.
    * `tool` (`kwargs["tool"]`): a string specifying the (sub-)tool to be used. The `run` command must parse this input and execute the task corresponding to the particular sub-tool referenced. These sub-tools will allow the user to fine-tune the use of your tool.

2. **Upload the tool file to IPFS.** You can use the following script:
    ```bash
    mechx push-to-ipfs "tools/<your_tool>.py"
    ```

    You should see an output similar to this:
    ```
    IPFS file hash v1: bafybei0123456789abcdef...0
    IPFS file hash v1 hex: f017012200123456789abcdef...0
    ```
    Note down the generated hashes for your tool.

3. **Configure your service.** Edit the `.1env` file. The demo service has this configuration:
    ```bash
    FILE_HASH_TO_TOOLS=[["bafybeihhxncljjtzniecvm7yr7u44g6ooquzqek473ma5fcnn2f6244v3e",["openai-text-davinci-002","openai-text-davinci-003","openai-gpt-3.5-turbo","openai-gpt-4"]],["bafybeiepc5v4ixwuu5m6p5stck5kf2ecgkydf6crj52i5umnl2qm5swb4i",["stabilityai-stable-diffusion-v1-5","stabilityai-stable-diffusion-xl-beta-v2-2-2","stabilityai-stable-diffusion-512-v2-1","stabilityai-stable-diffusion-768-v2-1"]]]
    API_KEYS=[["openai","dummy_api_key"],["stabilityai","dummy_api_key"]]
    ```

    To add your new tool with hash `<your_tool_hash>` and sub-tool list `[a, b, c]` and API key `<your_api_key>` simply update the variables above to:
    ```bash
    FILE_HASH_TO_TOOLS=[["bafybeihhxncljjtzniecvm7yr7u44g6ooquzqek473ma5fcnn2f6244v3e",["openai-text-davinci-002","openai-text-davinci-003","openai-gpt-3.5-turbo","openai-gpt-4"]],["bafybeiepc5v4ixwuu5m6p5stck5kf2ecgkydf6crj52i5umnl2qm5swb4i",["stabilityai-stable-diffusion-v1-5","stabilityai-stable-diffusion-xl-beta-v2-2-2","stabilityai-stable-diffusion-512-v2-1","stabilityai-stable-diffusion-768-v2-1"]]]
    API_KEYS=[[openai, dummy_api_key],[<your_api_key_id>, <your_api_key>]]
    ```

4. **Mint your agent service** in the [Autonolas Protocol](https://registry.olas.network/services/mint), and create a Mech for it in [Mech Hub](https://aimechs.autonolas.network/factory). This will allow you to set the `SAFE_CONTRACT_ADDRESS` and `AGENT_MECH_CONTRACT_ADDRESS` in the `.1env` file.

    > **Warning**
    > AI Mechs run on the [Gnosis chain](https://www.gnosis.io/). You must ensure that your wallet is connected to the [Gnosis chain](https://www.gnosis.io/) before using the [Autonolas Protocol](https://protocol.autonolas.network/services/mint) and [Mech Hub](https://aimechs.autonolas.network/factory).

    Here is an example of the agent NFT metadata once you create the Mech:
    ```json
    {
      "name": "Autonolas Mech III",
      "description": "The mech executes AI tasks requested on-chain and delivers the results to the requester.",
      "inputFormat": "ipfs-v0.1",
      "outputFormat": "ipfs-v0.1",
      "image": "tbd",
      "tools": ["openai-text-davinci-002", "openai-text-davinci-003", "openai-gpt-3.5-turbo", "openai-gpt-4"]
    }
    ```

5. **Run your service.** You can take a look at the `run_service.sh` script and execute your service locally as [above](#option-2-run-the-mech-as-an-agent-service).

    Once your service works locally, you have the option to run it on a hosted service like [Propel](https://propel.valory.xyz/).

## Included tools

- **OpenAI request** (`openai_request.py`). Executes requests to the OpenAI API through the engine associated to the specific tool. It receives as input an arbitrary prompt and outputs the returned output by the OpenAI API.
  - `openai-gpt-3.5-turbo`
  - `openai-gpt-4`
  - `openai-text-davinci-002`
  - `openai-text-davinci-003`

- **Stability AI request** (`stabilityai_request.py`): Executes requests to the Stability AI through the engine associated to the specific tool. It receives as input an arbitrary prompt and outputs the image data corresponding to the output of Stability AI.
  - `stabilityai-stable-diffusion-v1-5`
  - `stabilityai-stable-diffusion-xl-beta-v2-2-2`
  - `stabilityai-stable-diffusion-512-v2-1`
  - `stabilityai-stable-diffusion-768-v2-1`

- **Native transfer request** (`native_transfer_request.py`): Parses user prompt in natural language as input into an Ethereum transaction.
  - `transfer_native`

- **Prediction request** (`prediction_request.py`): Outputs the estimated probability of occurrence (`p_yes`) or no occurrence (`p_no`) of a certain event specified as the input prompt in natural language.
  - `prediction-offline`: Uses only training data of the model to make the prediction.
  - `prediction-online`: In addition to training data, it also uses online information to improve the prediction.

## How key files look
A keyfile is just a file with your ethereum private key as a hex-string, example:
```
0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd
```

Make sure you don't have any extra characters in the file, like newlines or spaces.
