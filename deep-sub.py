# Print the agent's response
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

from dotenv import load_dotenv
load_dotenv(".env")


tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
from langchain.chat_models import init_chat_model
ollama = init_chat_model("ollama:gemma4:31b:cloud", base_url=os.getenv("OLLAMA_BASE_URL"))

from langfuse import get_client
from langfuse.langchain import CallbackHandler


langfuse = get_client()
langfuse_handler = CallbackHandler()


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
# System prompt to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.
You have access to an internet search tool as your primary means of gathering information.
## `internet_search`
Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
If possible try to create a subagent to run it.
"""
agent = create_deep_agent(
    model=ollama,
    tools=[internet_search],
    system_prompt=research_instructions,
).with_config(
    {
        "callbacks": [langfuse_handler],
        "run_name": "deepagents",
    }
)
result = agent.invoke({"messages": [{"role": "user", "content": "Create a subagent and search about this: What is aegra?"}]})
# Print the agent's response
print(result["messages"][-1].content)
