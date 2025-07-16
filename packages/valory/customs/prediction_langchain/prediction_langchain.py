# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
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

import functools
import operator
import os
from typing import Annotated, Any, Callable, Dict, Literal, Optional, Sequence, Tuple

import anthropic
import openai
from googleapiclient.errors import HttpError as GoogleAPIHttpError
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


MechResponseWithKeys = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]


def with_key_rotation(func: Callable) -> Callable:
    """
    Decorator that retries a function with API key rotation on failure.

    :param func: The function to be decorated.
    :type func: Callable
    :returns: Callable -- the wrapped function that handles retries with key rotation.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MechResponseWithKeys:
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponseWithKeys:
            """Retry the function with a new key."""
            try:
                result: MechResponse = func(*args, **kwargs)
                return result + (api_keys,)
            except anthropic.RateLimitError as e:
                # try with a new key again
                service = "anthropic"
                if retries_left[service] <= 0:
                    raise e
                retries_left[service] -= 1
                api_keys.rotate(service)
                return execute()
            except openai.RateLimitError as e:
                # try with a new key again
                if retries_left["openai"] <= 0 and retries_left["openrouter"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                retries_left["openrouter"] -= 1
                api_keys.rotate("openai")
                api_keys.rotate("openrouter")
                return execute()
            except GoogleAPIHttpError as e:
                # try with a new key again
                rate_limit_exceeded_code = 429
                if e.status_code != rate_limit_exceeded_code:
                    raise e
                service = "google_api_key"
                if retries_left[service] <= 0:
                    raise e
                api_keys.rotate(service)
                return execute()
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


def _set_if_undefined(var: str, key: str) -> None:
    """Set env vars"""
    if not os.environ.get(var):
        os.environ[var] = key


def create_agent(tools: Any, system_message: str) -> Any:
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful AI assistant, collaborating with other assistants.
                Use the provided tools to progress towards answering the question.
                If you are unable to fully answer, that's OK, another assistant with different tools
                will help where you left off. Execute what you can to make progress.
                If you or any of the other assistants have the final answer or deliverable,
                prefix your response with FINAL ANSWER so the team knows to stop.
                Your final answer must start with FINAL ANSWER and be followed by a JSON object to be parsed by Python's `json.loads()`.
                Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
                The JSON must contain four fields: "p_yes", "p_no", "confidence", and "info_utility".
                Output only the FINAL ANSWER and the JSON object. Do not include any other contents in your response.
                Each item in the JSON must have a value between 0 and 1.
                   - "p_yes": Estimated probability that the event in the "USER_PROMPT" occurs.
                   - "p_no": Estimated probability that the event in the "USER_PROMPT" does not occur.
                   - "confidence": A value between 0 and 1 indicating the confidence in the prediction.
                     0 indicates lowest confidence value; 1 maximum confidence value.
                   - "info_utility": Utility of the information provided in "ADDITIONAL_INFORMATION" to help you
                     make the prediction. 0 indicates lowest utility; 1 maximum utility.
                The sum of "p_yes" and "p_no" must equal 1.
                You must provide your response in the format specified below:
                   - This is incorrect:"```json{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}```"
                   - This is incorrect:```json"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"```
                   - This is incorrect: Based on the search results: "{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"
                   - This is correct:"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"
                You have access to the following tools: {tool_names}.\n{system_message}
                """,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.1)
    return prompt | llm.bind_tools(tools)


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    """Agent state"""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# Helper function to create a node for a given agent
def agent_node(state: Any, agent: Any, name: Any) -> Any:
    """Create a node for given agent"""
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender, so we know who to pass to next.
        "sender": name,
    }


def router(state: Any) -> Literal["call_tool", "__end__", "continue"]:
    """Router"""
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

    tavily_tool = TavilySearchResults(max_results=5)

    # Create research agent and node
    research_agent = create_agent(
        [tavily_tool],
        system_message="You should provide accurate data for the data_analyzer to use.",
    )
    research_node = functools.partial(
        agent_node, agent=research_agent, name="researcher"
    )

    # Create data_analyzer agent and node
    analyzer_agent = create_agent(
        [tavily_tool],
        system_message="You should analyze the data you've been provided with and decide whether an event is more likely to happen or not.",
    )
    analyzer_node = functools.partial(
        agent_node, agent=analyzer_agent, name="data_analyzer"
    )

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
        " Once you have an answer, finish. Remember to only output a single JSON object, parseable by `json.loads` in Python, and no extra text."
    )

    events = graph.stream(
        {
            "messages": [HumanMessage(content=prompt)],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 300},
    )

    # Generator to list
    event_list = [e for e in events]

    # Response is the last message from the last event
    last_event = event_list[-1]
    last_sender = list(last_event.keys())[0]
    last_message = last_event[last_sender]["messages"][-1]
    response = (
        last_message.content.replace("FINAL ANSWER:", "")
        .replace("FINAL ANSWER", "")
        .strip()
    )

    return response, prompt


def error_response(msg: str) -> Tuple[str, None, None, None]:
    """Return an error mech response."""
    return msg, None, None, None


@with_key_rotation
def run(**kwargs: Any) -> Tuple[Optional[str], Optional[str], None, None]:
    """Run the langchain example."""

    # hardcode topic and timeframe
    topic = "consumer technology"
    timeframe = "past month"
    # flake8: noqa: E800
    # topic: Optional[str] = kwargs.get("topic", None)
    # timeframe: Optional[str] = kwargs.get("timeframe", None)
    # flake8: enable: E800

    # Process the kwargs
    question: Optional[str] = kwargs.get("prompt", None)
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
    _set_if_undefined("OPENAI_API_KEY", openai_api_key)
    _set_if_undefined("TAVILY_API_KEY", tavily_api_key)
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # Run the tool
    response, prompt = run_langgraph(topic, timeframe, question)

    # The expected output is: response, prompt, irrelevant, irrelevant
    return response, prompt, None, None
