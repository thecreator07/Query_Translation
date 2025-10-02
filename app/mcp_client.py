from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_core.tools import StructuredTool
from typing import List

async def mcp_tool_to_langchain_tool(mcp_tool, mcp_url: str):
    """Wraps an MCP tool into a LangChain-compatible StructuredTool."""

    async def _run(**kwargs):
        async with streamablehttp_client(mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(mcp_tool.name, kwargs)
                return result.content[0].text

    return StructuredTool.from_function(
        coroutine=_run,
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.inputSchema
    )
    
    
async def fetch_wrapped_mcp_tools(mcp_url: str) -> List:
    wrapped_tools = []
    # streamablehttp_client yields (read_stream, write_stream, misc)
    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools_response = await session.list_tools()
            # mcp_tools_response.tools is the list of tool specs
            for t in mcp_tools_response.tools:
                wrapped = await mcp_tool_to_langchain_tool(t, mcp_url)
                wrapped_tools.append(wrapped)
    return wrapped_tools