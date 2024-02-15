![MechKit](images/mechkit.svg){ align=left }
The MechKit is a toolkit designed for constructing _Mechs_. Mechs function as permissionless marketplaces for AI skills. They provide agents with seamless access to a wide array of AI tools, payable via crypto on-chain, essentially serving as a pay-to-think service for agents.

Agents and their operators no longer need to juggle multiple APIs and their associated subscriptions; instead, they can tap into a Mech service. Allowing agents to pay directly in crypto, Mechs can significantly enhance agents' capacity for autonomous decision-making and action, while bypassing the need for manual integrations.

Mechs address a critical need in the agent ecosystem: the ability to outsource complex decision-making processes. Whether it's evaluating DAO proposals, analyzing the impact of social media content, or making predictive analyses, Mechs provide agents with the tools they need to operate autonomously and effectively.

## How it works

A Mech service consists of a library of tools. These can range from something as simple as an API call to highly complex business logic. Some examples of tools can be found on [the `tools` folder](https://github.com/valory-xyz/mech/tree/main/tools) of the first operational Mech service.

For instance, if an agent service needs to perform a task requiring access to an API, instead of each operator having to manage their own keys, the service can make use of a Mech tool. Agents then only need to make a request to a Mech, which gets returned in the form of a deliver event. The agent can then use the deliver event to accomplish what it set out to do. Both the request and deliver events are executed on-chain and the associated data is stored on IPFS making them easily retrievable by agents or anyone else.

## Live use case

!!! tip "See it in action!"

    Watch a Mech service live in action [here](https://aimechs.autonolas.network/mech/0x77af31De935740567Cf4fF1986D04B2c964A786a)!

Consider the [Trader service](https://github.com/valory-xyz/trader), an autonomous service that trades in prediction markets. It utilizes a specific tool called `prediction_request` provided by the Mech service.

Here is how the agent and the Mech interact:

1. The trader agent submits a question to the ‘prediction_request’  tool.
2. The Mech then runs the tool which gathers the most recent news using the Google Search API.
3. Leveraging OpenAI’s API, the tool calculates the probabilities of specific outcomes related to the query.
4. In response, the trader agent receives from the Mech the probabilities in a delivery event.

Armed with these probabilities, the Trader Agent can decide whether to engage in trading within that market.

The clever part? All the intricacies of dealing with APIs and data scraping are handled by the Mech tool. The trader agent operator does not need to worry about subscribing to OpenAI or managing Google-related tasks—the trader agent simply pays per request.

## Demo

!!! warning "Important"

    This section is under active development - please report issues in the [Autonolas Discord](https://discord.com/invite/z2PT65jKqQ).

    The demo service is configured to match a specific on-chain agent (ID 3 on [Mech Hub](https://aimechs.autonolas.network/registry). Since you will not have access to its private key, your local instance will not be able to transact.

    However, it will be able to receive Requests for AI tasks [sent from Mech Hub](https://aimechs.autonolas.network/mech). These Requests will be executed by your local instance, but you will notice that a failure will occur when it tries to submit the transaction on-chain (Deliver type).

    Please, refer to the complete instructions on the repository [README.md](https://github.com/valory-xyz/mech).

In order to run a local demo service based on the MechKit:

1. [Set up your system](https://docs.autonolas.network/open-autonomy/guides/set_up/) to work with the Open Autonomy framework and prepare the repository:

    ```bash
    git clone https://github.com/valory-xyz/mech && cd mech
    poetry run pip install openapi-core==0.13.2
    poetry run pip install openapi-spec-validator==0.2.8
    poetry install && poetry shell

    autonomy init --remote --ipfs --reset --author=your_name
    autonomy packages sync --update-packages 
    ```

2. Configure the service. You need to create a `.1env` file which contains the service configuration parameters. We provide a prefilled template (`.example.env`). You will need to provide or create an [OpenAI API key](https://platform.openai.com/account/api-keys).

    ```bash
    # Copy the prefilled template
    cp .example.env .1env

    # Edit ".1env" and replace "dummy_api_key" with your OpenAI API key.

    # Source the env file
    source .1env
    ```

3. Run the service.

    1. Ensure you have a file with the agent address and private key (`keys.json`). You can generate a new private key file using the Open Autonomy CLI:

        ```bash
        autonomy generate-key ethereum -n 1
        ```

    2. Ensure that the variable `ALL_PARTICIPANTS` in the file `.1env` matches the agent address within the file `keys.json`:

        ```bash
        ALL_PARTICIPANTS='["your_agent_address"]'
        ```

    3. Launch the service using the provided script:

        ```bash
        bash run_service.sh
        ```

## Build

1. Fork the [MechKit repository](https://github.com/valory-xyz/mech).
2. Make the necessary adjustments to tailor the service to your needs. This could include:
    * Adjust configuration parameters (e.g., in the `service.yaml` file).
    * Expand the service finite-state machine with your custom states.
3. Run your service as detailed above.

!!! tip "Looking for help building your own?"

    Refer to the [Autonolas Discord community](https://discord.com/invite/z2PT65jKqQ), or consider ecosystem services like [Valory Propel](https://propel.valory.xyz) for the fastest way to get your first autonomous service in production.
