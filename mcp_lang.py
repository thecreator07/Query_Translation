from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
import os
import asyncio
import copy
import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.schema import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.mongodb import AsyncMongoDBSaver
from langfuse.openai import openai
from langfuse import observe
from langfuse import Langfuse
from langgraph.prebuilt import create_react_agent
from langfuse.langchain import CallbackHandler
from langsmith import Client
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers.langchain import LangChainTracer


load_dotenv()
os.environ["GOOGLE_API_KEY"] =os.environ.get("GEMINI_API_KEY")
# import vertexai

# vertexai.init(
#     project="text-to-speech-467917",
#     location="your-model-location"  # e.g., "us-central1"
# )

# langfuse = Langfuse(
#     public_key="pk-lf-a16e98cb-a84c-4196-bed8-19a306a39230",
#     secret_key="sk-lf-76a7135b-076a-4a54-918a-0d2bb60186be",
#     host="http://localhost:3000"
# )

# langfuse_handler = CallbackHandler()



langsmith_client = Client()
langsmith_tracer = LangChainTracer()
# callback_manager = CallbackManager([langfuse_handler, langsmith_tracer])


async def get_clean_tools():
    client = MultiServerMCPClient(
    {
        "all": {
            "transport": "streamable_http",
            "url": "http://localhost:4000/mcp/"
        },
    }
    )
    raw_tools = await client.get_tools()
    clean_tools = []
    for tool in raw_tools:
        tool_copy = copy.deepcopy(tool)
        args_schema = getattr(tool_copy, "args", None)
        if isinstance(args_schema, dict):
            # Remove unsupported keys
            for bad in ["additionalProperties", "$schema"]:
                args_schema.pop(bad, None)
        clean_tools.append(tool_copy)
    return clean_tools

mcptool = asyncio.run(get_clean_tools())

@tool()
def human_assistance_tool(query: str):
    """Request assistance from a human."""
    human_response = interrupt({ "query": query }) # Graph will exit out after saving data in DB
    return human_response["data"] # resume with the data

@tool
def get_weather(city: str):
    """takes the city name and return the output"""
    print("🔨 Tool Called: get_weather", city)
    
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"


@tool
def command_run(cmd:str):
    """Takes a command line prompt and executes it on the user's machine and return output of the command.
    Example: command_run(cmd="ls") where ls is the command to list the file
    """
    result=os.system(command=cmd)
    return result

mcptool = asyncio.run(get_clean_tools())
# print(mcptool)
tools = mcptool+[human_assistance_tool,command_run,get_weather]


model = init_chat_model("google_genai:gemini-2.5-flash",callbacks=[langsmith_tracer])


DB_URI=os.environ.get('DB_URL')
config = {"configurable": {"thread_id": "2",}}
async def main():
    async with AsyncMongoDBSaver.from_conn_string(DB_URI) as checkpointer:
        agent= create_react_agent(model, tools, checkpointer=checkpointer)
        while True:
            user_input = input("You: ")
            if user_input.lower() in {"exit", "quit"}:
                break

            async for event in agent.astream(
                {"messages": [{"role": "user", "content": user_input}]},
                stream_mode="values",
                config=config
            ):
                if "messages" in event:
                    event["messages"][-1].pretty_print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
