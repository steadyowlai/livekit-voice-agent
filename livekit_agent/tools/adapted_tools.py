import asyncio
import functools
from langchain_core.tools import BaseTool
from livekit.agents import llm
from langchain.tools import search_market_rates, search_fed_policy


def adapt_langchain_tool(lc_tool: BaseTool):
    """
    Standard LangChain -> LiveKit Tool Adapter.

    1. Uses `functools.wraps` to preserve function signature, type hints,
       and docstrings so LiveKit's schema generator can extract parameter descriptions.
    2. Uses `asyncio.to_thread` to execute synchronous LangChain tools safely
       without blocking LiveKit's real-time audio pipeline.
    """
    @functools.wraps(lc_tool.func)
    async def _async_tool_wrapper(*args, **kwargs):
        return await asyncio.to_thread(lc_tool.func, *args, **kwargs)

    return llm.function_tool(
        name=lc_tool.name,
        description=lc_tool.description,
    )(_async_tool_wrapper)

# Adapt existing LangChain search tools for the LiveKit voice agent
adapted_search_market_rates = adapt_langchain_tool(search_market_rates)
adapted_search_fed_policy = adapt_langchain_tool(search_fed_policy)



