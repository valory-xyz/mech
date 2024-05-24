# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""A mech tool that integrates langchain and langgraph."""

import getpass
import os
from typing import Annotated, Literal, Sequence, TypedDict, Optional, Tuple
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode
import operator
import functools

tavily_tool = TavilySearchResults(max_results=5)
llm = ChatOpenAI(model="gpt-4-1106-preview")


def _set_if_undefined(var: str):
    "Set env vars"
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }


def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"



def run_langgraph(topic: str, timeframe: str, question: str) -> Tuple[str, str]:
    """Run langgraph"""

    # Create research agent and node
    research_agent = create_agent(
        llm,
        [tavily_tool],
        system_message="You should provide accurate data for the data_analyzer to use.",
    )
    research_node = functools.partial(agent_node, agent=research_agent, name="researcher")

    # Create data_analyzer agent and node
    analyzer_agent = create_agent(
        llm,
        [tavily_tool],
        system_message="You should analyze the data you've been provided with and decide whether an event is more likely to happen or not."
        "* Your output response must be only a single JSON object to be parsed by Python's json.loads()."
        '* The JSON must contain just one field: "response" which value can only be "yes" or "no".'
        "* Output only the JSON object. Do not include any other contents in your response."
    )
    analyzer_node = functools.partial(agent_node, agent=analyzer_agent, name="data_analyzer")

    # Tools
    tools = [tavily_tool]
    tool_node = ToolNode(tools)

    # Either agent can decide to end
    workflow = StateGraph(AgentState)

    workflow.add_node("researcher", research_node)
    workflow.add_node("data_analyzer", analyzer_node)
    workflow.add_node("call_tool", tool_node)

    workflow.add_conditional_edges(
        "researcher",
        router,
        {"continue": "data_analyzer", "call_tool": "call_tool", "__end__": END},
    )
    workflow.add_conditional_edges(
        "data_analyzer",
        router,
        {"continue": "researcher", "call_tool": "call_tool", "__end__": END},
    )

    workflow.add_conditional_edges(
        "call_tool",
        # Each agent node updates the 'sender' field
        # the tool calling node does not, meaning
        # this edge will route back to the original agent
        # who invoked the tool
        lambda x: x["sender"],
        {
            "researcher": "researcher",
            "data_analyzer": "data_analyzer",
        },
    )
    workflow.set_entry_point("researcher")
    graph = workflow.compile()

    prompt = (
        f"Fetch data and news about {topic} over the {timeframe},"
        f" then ask the question: {question}"
        " Once you have an answer, finish."
    )

    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content=prompt
                )
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 150},
    )

    # Generator to list
    event_list = [e for e in events]

    # Response is the last message from the last event
    response = event_list[-1]["researcher"]["messages"][-1].content

    return response, prompt


def error_response(msg: str) -> Tuple[str, None, None, None]:
    """Return an error mech response."""
    return msg, None, None, None


def run(**kwargs) -> Tuple[Optional[str], Optional[str], None, None]:
    """Run the langchain example."""

    # Process the kwargs
    topic = kwargs.get("topic", None)
    timeframe = kwargs.get("topic", None)
    question = kwargs.get("question", None)
    openai_api_key = kwargs.get("api_keys", {}).get("openai", None)
    tavily_api_key = kwargs.get("api_keys", {}).get("tavily", None)

    if topic is None:
        return error_response("No topic has been specified.")

    if timeframe is None:
        return error_response("No timeframe has been specified.")

    if question is None:
        return error_response("No question has been specified.")

    if openai_api_key is None:
        return error_response("No openai_api_key has been specified.")

    if tavily_api_key is None:
        return error_response("No tavily_api_key has been specified.")

    # Set the environment
    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("TAVILY_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # Run the tool
    response, prompt = run_langgraph(topic, timeframe, question)

    # The expected output is: response, prompt, irrelevant, irrelevant
    return response, prompt, None, None


if __name__ == "__main__":
    """Main"""

    api_keys = {
        "openai": os.environ["OPENAI_API_KEY"],
        "tavily": os.environ["TAVILY_API_KEY"]
    }

    response = run(
        topic="consumer technology",
        timeframe="past month",
        question="will Apple will unveil a new Iphone before the end of 2024?",
        api_keys=api_keys
    )
    print(response)