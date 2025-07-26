import json
import requests
from dotenv import load_dotenv
from langfuse.openai import openai
from langfuse import observe
import os
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END,MessagesState

load_dotenv()


def run_command(command):
    result = os.system(command=command)
    return result


def get_weather(city: str):
    # TODO!: Do an actual API Call
    print("🔨 Tool Called: get_weather", city)
    
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"


tool_node = ToolNode([get_weather,run_command])

model = init_chat_model(model="gemini-2.0-flash",model_provider='gemini')
model_with_tools = model.bind_tools(tool_node)


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


graph_builder = StateGraph(MessagesState)

graph_builder.add_node("call_model",call_model)
graph_builder.add_node("tools",tool_node)

graph_builder.add_edge(START, "call_model")
graph_builder.add_conditional_edges("call_model", should_continue, ["tools", END])
graph_builder.add_edge("tools", "call_model")

graph = graph_builder.compile()

def call_graph():
    # user_query=input("Ask Question: ")

    result=graph.invoke({"messages": [{"role": "user", "content": "what's the weather in sf?"}]})
    
    print("Final Result", result)

call_graph()