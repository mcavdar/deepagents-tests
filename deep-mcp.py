from langchain.mcp import MCPAdapter
from deepagents import create_deep_agent
import asyncio
import os


from dotenv import load_dotenv

load_dotenv(".env")

from langchain.chat_models import init_chat_model
ollama = init_chat_model("ollama:gemma4:31b:cloud", base_url=os.getenv("OLLAMA_BASE_URL"))

from langfuse import get_client
from langfuse.langchain import CallbackHandler


langfuse = get_client()
langfuse_handler = CallbackHandler()

async def main():
    async with MCPAdapter("https://docs.langchain.com/mcp") as adapter:
        tools = await adapter.list_tools()

        tool = tools[0]

        print(tool.name)
        print(tool.description)
        print(tool.args_schema)
        print(tool.response_format)

        agent = create_deep_agent(
            model=ollama,
            tools=tools,
        ).with_config(
    {
        "callbacks": [langfuse_handler],
        "run_name": "deepagents-mock",
    }
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
