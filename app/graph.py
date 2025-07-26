from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
os.environ["GOOGLE_API_KEY"] =os.environ.get("GEMINI_API_KEY")

@tool()
def human_assistance_tool(query: str):
    """Request assistance from a human."""
    human_response = interrupt({ "query": query }) # Graph will exit out after saving data in DB
    return human_response["data"] # resume with the data

tools = [human_assistance_tool]

llm = init_chat_model(model_provider="openai", model="gpt-4.1")
llm_with_tools = llm.bind_tools(tools=tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot") # We missed this

graph_builder.add_edge("chatbot", END)

# Without any memory
graph = graph_builder.compile()

# Creates a new graph with given checkpointer
def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
