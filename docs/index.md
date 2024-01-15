![MechKit](images/mechkit.svg){ align=left }
The MechKit is a toolkit designed for constructing _Mechs_. Mechs function as permissionless marketplaces for AI skills. They provide agents with seamless access to a wide array of AI tools, payable via crypto on-chain, essentially serving as a pay-to-think service for agents.

Agents and their operators no longer need to juggle multiple APIs and their associated subscriptions; instead, they can tap into a Mech service. Allowing agents to pay directly in crypto, Mechs can significantly enhance agents' capacity for autonomous decision-making and action, while bypassing the need for manual integrations.

Mechs address a critical need in the agent ecosystem: the ability to outsource complex decision-making processes. Whether it's evaluating DAO proposals, analyzing the impact of social media content, or making predictive analyses, Mechs provide agents with the tools they need to operate autonomously and effectively.

## How it works

A Mech service consists of a library of tools. These can range from something as simple as an API call to highly complex business logic. Some examples of tools can be found on [the `tools` folder](https://github.com/valory-xyz/mech/tree/main/tools) of the first operational Mech service.

For instance, if an agent service needs to perform a task requiring access to an API, instead of each operator having to manage their own keys, the service can make use of a Mech tool. Agents then only need to make a request to a Mech, which gets returned in the form of a deliver event. The agent can then use the deliver event to accomplish what it set out to do. Both the request and deliver events are executed on-chain and the associated data is stored on IPFS making them easily retrievable by agents or anyone else.

Watch a Mech service in action [here](https://aimechs.autonolas.network/mech/0x77af31De935740567Cf4fF1986D04B2c964A786a).

## Live use case

Consider the [Trader service](https://github.com/valory-xyz/trader), an autonomous service that trades in prediction markets. It utilizes a specific tool called `prediction_request` provided by the Mech service.

Here is how the agent and the Mech interact:

1. The trader agent submits a question to the ‘prediction_request’  tool.
2. The Mech then runs the tool which gathers the most recent news using the Google Search API.
3. Leveraging OpenAI’s API, the tool calculates the probabilities of specific outcomes related to the query.
4. In response, the trader agent receives from the Mech the probabilities in a delivery event.

Armed with these probabilities, the Trader Agent can decide whether to engage in trading within that market.

The clever part? All the intricacies of dealing with APIs and data scraping are handled by the Mech tool. The trader agent operator does not need to worry about subscribing to OpenAI or managing Google-related tasks—the trader agent simply pays per request.

## Demo

!!! info

	This section will be completed soon.

## Build

1. Fork the [MechKit repository](https://github.com/valory-xyz/mech).
2. Make the necessary adjustments to tailor the service to your needs. This could include:
    * Adjust configuration parameters (e.g., in the `service.yaml` file).
    * Expand the service finite-state machine with your custom states.
3. Run your service as detailed above.

!!! tip "Looking for help building your own?"

    Refer to the [Autonolas Discord community](https://discord.com/invite/z2PT65jKqQ), or consider ecosystem services like [Valory Propel](https://propel.valory.xyz) for the fastest way to get your first autonomous service in production.
