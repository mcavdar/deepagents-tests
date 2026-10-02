from langchain.mcp import MCPAdapter
from deepagents import create_deep_agent
import asyncio
import os
from langchain_openai import ChatOpenAI

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from dotenv import load_dotenv

load_dotenv(".env")

model = ChatOpenAI(
    model="mock-model",
    base_url="https://mock.envai.tr/v1",
    api_key="mock-key",
    streaming=False,
)


async def main():
    async with MCPAdapter("https://docs.langchain.com/mcp") as adapter:
        tools = await adapter.list_tools()
        agent = create_deep_agent(
            model=model,
            tools=tools,
        )
        result = await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": "find all docs about mcp in langchain."}
                ]
            },
            config={"configurable": {"thread_id": "1"}},
        )
        return result


if __name__ == "__main__":
    print(repr(asyncio.run(main())))
