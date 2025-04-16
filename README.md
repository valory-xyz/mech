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
- Proficiency in the usage of the related open-source technologies, which may entail facing their inherent complexities.

AI Mechs run on the [Gnosis chain](https://www.gnosis.io/), and enables you to post *AI tasks requests* on-chain and get their result delivered back to you efficiently. An AI Mech will execute these tasks for you. All you need is some xDAI in your wallet to reward the worker service executing your task. AI Mechs are **hassle-free**, **crypto-native**, and **infinitely composable**.

> :bulb: These are just a few ideas on what capabilities can be brought on-chain with AI Mechs:
>
> - fetch real-time **web search** results
> - integrate **multi-sig wallets**,
> - **simulate** chain transactions
> - execute a variety of **AI models**:
>   - **generative** (e.g, Stability AI, Midjourney),
>   - **action-based** AI agents (e.g., AutoGPT, LangChain)

**AI Mechs is a project born at [ETHGlobal Lisbon](https://ethglobal.com/showcase/ai-mechs-dt36e).**

## AI Mechs components

The project consists of three components:

- Off-chain AI workers, each of which controls a Mech contract. Each AI worker is implemented as an autonomous service on the Autonolas stack.
- An on-chain protocol, which is used to generate a registry of AI Mechs, represented as NFTs on-chain.
- An on-chain [MarketPlace](https://github.com/valory-xyz/ai-registry-mech/) which enable AI Mechs to easily deploy Mech contracts, relays service requests and deliveries to such Mech contracts, and guarantees service deliveries by implementing a reputation score and a take-over mechanism.
- [Mech Hub](https://mech.olas.network/mechs), a frontend which allows to interact with the protocol:
  - Gives an overview of the AI workers in the registry.
  - Allows Mech owners to create new workers.
  - Allows users to request work from an existing worker.

_Note that Mechs which were deployed before the Mech Marketplace contracts (called legacy Mechs) receive request and deliver services directly via their Mech contract._

## Mech request-response flow

This section presents the workflow for Mech agents independently from the Mech Marketplace. For Mechs which receive requests
via the Marketplace, check this [document](https://github.com/valory-xyz/ai-registry-mech/blob/main/docs/MechMarketplaceDescriptionAndContractsOverviewRepo.pdf).

![image](docs/images/mech_request_response_flow.png)

1. Write request metadata: the application writes the request metadata to the IPFS. The request metadata must contain the attributes `nonce`, `tool`, and `prompt`. Additional attributes can be passed depending on the specific tool:

    ```json
    {
      "nonce": 15,
      "tool": "prediction_request",
      "prompt": "Will my favourite football team win this week's match?"
    }
    ```

2. The application gets the metadata's IPFS hash.

3. The application writes the request's IPFS hash to the Mech contract which includes a small payment (currently $0.01 on the Gnosis chain deployment). Alternatively, the payment could be done separately through a Nevermined subscription.

4. The Mech service is constantly monitoring Mech contract events, and therefore gets the request hash.

5. The Mech reads the request metadata from IPFS using its hash.

6. The Mech selects the appropriate tool to handle the request from the `tool` entry in the metadata, and runs the tool with the given arguments, usually a prompt. In this example, the mech has been requested to interact with OpenAI's API, so it forwards the prompt to it, but the tool can implement any other desired behavior.

7. The Mech gets a response from the tool.

8. The Mech writes the response to the IPFS.

9. The Mech receives the response the IPFS hash.

10. The Mech writes the response hash to the Mech contract.

11. The application monitors for contract Deliver events and reads the response hash from the associated transaction.

12. The application gets the response metadata from the IPFS:

    ```json
    {
      "requestId": 68039248068127180134548324138158983719531519331279563637951550269130775,
      "result": "{\"p_yes\": 0.35, \"p_no\": 0.65, \"confidence\": 0.85, \"info_utility\": 0.75}"
    }
    ```

See some examples of requests and responses on the [Mech Hub](https://mech.olas.network/gnosis/mech/0x77af31de935740567cf4ff1986d04b2c964a786a?legacy=true).

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
    poetry install && poetry shell
    ```

2. Fetch the software packages using the [Open Autonomy](https://docs.autonolas.network/open-autonomy/) CLI:

    ```bash
    autonomy packages sync --update-packages
    ```

    This will populate the Open Autonomy [local registry](https://docs.autonolas.network/open-autonomy/guides/set_up/#the-registries-and-runtime-folders) (folder `./packages`) with the required components to run the worker services.

## Run the demo

### Using Mech Quickstart (Preffered Method)

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
cp .example.env .agentenv

# Edit ".agentenv" and replace "dummy_api_key" with your OpenAI API key.

# Source the env file
source .agentenv
```

##### Environment Variables

You may customize the agent's behaviour by setting these environment variables.

| Name                       | Type   | Sample Value                                                                                                                                                                                                                                                        | Description                                                            |
| -------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `TOOLS_TO_PACKAGE_HASH`    | `dict` | `{"openai-gpt-3.5-turbo-instruct":"bafybeigz5brshryms5awq5zscxsxibjymdofm55dw5o6ud7gtwmodm3vmq","openai-gpt-3.5-turbo":"bafybeigz5brshryms5awq5zscxsxibjymdofm55dw5o6ud7gtwmodm3vmq","openai-gpt-4":"bafybeigz5brshryms5awq5zscxsxibjymdofm55dw5o6ud7gtwmodm3vmq"}` | Tracks services for each tool packages.                                |
| `API_KEYS`                 | `dict` | `{"openai":["dummy_api_key"], "google_api_key":["dummy_api_key"]}`                                                                                                                                                                                                      | Tracks API keys for each service.                                      |
| `SERVICE_REGISTRY_ADDRESS` | `str`  | `"0x9338b5153AE39BB89f50468E608eD9d764B755fD"`                                                                                                                                                                                                                      | Smart contract which registers the services.                           |
| `AGENT_REGISTRY_ADDRESS`   | `str`  | `"0xE49CB081e8d96920C38aA7AB90cb0294ab4Bc8EA"`                                                                                                                                                                                                                      | Smart contract which registers the agents.                             |
| `MECH_MARKETPLACE_ADDRESS` | `str`  | `"0x4554fE75c1f5576c1d7F765B2A036c199Adae329"`                                                                                                                                                                                                                      | Marketplace for posting and delivering requests served by agent mechs. |
| `MECH_TO_SUBSCRIPTION`     | `dict` | `{"0x77af31De935740567Cf4fF1986D04B2c964A786a":{"tokenAddress":"0x0000000000000000000000000000000000000000","tokenId":"1"}}`                                                                                                                                        | Tracks mech's subscription details.                                    |
| `MECH_TO_CONFIG`           | `dict` | `{"0xFf82123dFB52ab75C417195c5fDB87630145ae81":{"use_dynamic_pricing":false,"is_marketplace_mech":false}}`                                                                                                                                                          | Tracks mech's config.                                                  |

The rest of the common environment variables are present in the [service.yaml](https://github.com/valory-xyz/mech/blob/main/packages/valory/services/mech/service.yaml), which are customizable too.

> **Warning**<br />
> **The demo service is configured to match a specific on-chain agent (ID 3 on [Mech Hub](https://mech.olas.network/mechs?legacy=true)). Since you will not have access to its private key, your local instance will not be able to transact.
> However, it will be able to receive Requests for AI tasks [sent from Mech Hub](https://mech.olas.network/mechs). These Requests will be executed by your local instance, but you will notice that a failure will occur when it tries to submit the transaction on-chain (Deliver type).**

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

## Integrating mechs into your application

### For generic apps and scripts

Use the [mech-client](https://github.com/valory-xyz/mech-client), which can be used either as a CLI or directly from a Python script.

### For other autonomous services

To perform mech requests from your service, use the [mech_interact_abci skill](https://github.com/valory-xyz/IEKit/tree/main/packages/valory/skills/mech_interact_abci). This skill abstracts away all the IPFS and contract interactions so you only need to care about the following:

- Add the mech_interact_abci skill to your dependency list, both in `packages.json`, `aea-config.yaml` and any composed `skill.yaml`.

- Import [MechInteractParams and MechResponseSpecs in your `models.py` file](https://github.com/valory-xyz/IEKit/blob/main/packages/valory/skills/impact_evaluator_abci/models.py#L88). You will also need to copy [some dataclasses to your rounds.py](https://github.com/valory-xyz/IEKit/blob/main/packages/valory/skills/twitter_scoring_abci/rounds.py#L66-L97).

- Add mech_requests and mech_responses to your skills' `SynchonizedData` class ([see here](https://github.com/valory-xyz/IEKit/blob/main/packages/valory/skills/twitter_scoring_abci/rounds.py#L181-193))

- To send a request, [prepare the request metadata](https://github.com/valory-xyz/IEKit/blob/main/packages/valory/skills/twitter_scoring_abci/behaviours.py#L857), write it to [`synchronized_data.mech_requests`](https://github.com/valory-xyz/IEKit/blob/main/packages/valory/skills/twitter_scoring_abci/rounds.py#L535) and [transition into mech_interact](https://github.com/valory-xyz/IEKit/blob/main/packages/valory/skills/twitter_scoring_abci/rounds.py#L736).

- You will need to appropriately chain the `mech_interact_abci` skill with your other skills ([see here](https://github.com/valory-xyz/IEKit/blob/main/packages/valory/skills/impact_evaluator_abci/composition.py#L66)) and `transaction_settlement_abci`.

- After the interaction finishes, the responses will be inside [`synchronized_data.mech_responses`](https://github.com/valory-xyz/IEKit/blob/main/packages/valory/skills/twitter_scoring_abci/behaviours.py#L903)

For a complete list of required changes, [use this PR as reference](https://github.com/valory-xyz/market-creator/pull/91).

## Build your own

You can create and mint your own AI Mech that handles requests for tasks that you can define.

You can take a look at the preferred method mentioned [above](#using-mech-quickstart-preffered-method) to get started quickly and easily.

Once your service works locally, you have the option to run it on a hosted service like [Propel](https://app.propel.valory.xyz/).

## Included tools

🚧 **Under Construction** 🚧  
We are working on adding a simple tool for the quickstart

## More on tools

🚧 **Under Construction** 🚧

## How key files look

A keyfile is just a file with your ethereum private key as a hex-string, example:

```text
0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd
```

Make sure you don't have any extra characters in the file, like newlines or spaces.

## Examples of deployed Mechs

### Legacy Mechs

| Network  | Service                                               | Mech Instance (Nevermined Pricing) - Agent Id    | Mech Instance (Fixed Pricing) - Agent Id         |  Service id |
| :------: | ----------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------ | -------- |
| Ethereum | https://registry.olas.network/ethereum/services/21    | n/a                                              | n/a  | n/a                                            |
|  Gnosis  | https://registry.olas.network/gnosis/services/3       | `0x327E26bDF1CfEa50BFAe35643B23D5268E41F7F9` - 3 | `0x77af31De935740567Cf4fF1986D04B2c964A786a` - 6 | 3 |
| Arbitrum | https://registry.olas.network/arbitrum-one/services/1 | `0x0eA6B3137f294657f0E854390bb2F607e315B82c` - 1 | `0x1FDAD3a5af5E96e5a64Fc0662B1814458F114597` - 2 | 1 |
| Polygon  | https://registry.olas.network/polygon/services/3      | `0xCF1b5Db1Fa26F71028dA9d0DF01F74D4bbF5c188` - 1 | `0xbF92568718982bf65ee4af4F7020205dE2331a8a` - 2 | 3 | 
|   Base   | https://registry.olas.network/base/services/1         | `0x37C484cc34408d0F827DB4d7B6e54b8837Bf8BDA` - 1 | `0x111D7DB1B752AB4D2cC0286983D9bd73a49bac6c` - 2 | 1 |
|   Celo   | https://registry.olas.network/celo/services/1         | `0xeC20694b7BD7870d2dc415Af3b349360A6183245` - 1 | `0x230eD015735c0D01EA0AaD2786Ed6Bd3C6e75912` - 2 | 1 |
| Optimism | https://registry.olas.network/optimism/services/1     | `0xbA4491C86705e8f335Ceaa8aaDb41361b2F82498` - 1 | `0xDd40E7D93c37eFD860Bd53Ab90b2b0a8D05cf71a` - 2 | 1 |

### Mechs receiving requests via the Mech Marketplace

There is no Mech deployed on the Mech Marketplace at the moment.