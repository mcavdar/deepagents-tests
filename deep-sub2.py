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

research_subagent = {
    "name": "research-agent",
    "description": "Use this subagent exclusively to perform in-depth web research and gather external facts before drafting responses.",
    "system_prompt": "You are a great researcher",
    "tools": [internet_search],
    "model": ollama,  # Optional override, defaults to main agent model
}
subagents = [research_subagent]

agent = create_deep_agent(
    model=ollama,
    subagents=subagents,
).with_config(
    {
        "callbacks": [langfuse_handler],
        "run_name": "deepagents",
    }
)
result = agent.invoke({"messages": [{"role": "user", "content": "What is aegra?"}]})
# Print the agent's response
print(result["messages"][-1].content)
