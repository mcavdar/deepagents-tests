# Print the agent's response
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent
from pathlib import Path
from deepagents.backends import FilesystemBackend

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
from langchain.chat_models import init_chat_model
ollama = init_chat_model("ollama:gemma4:latest", base_url="https://ollamauser:mhmt@ollama.envai.tr")


from langfuse import get_client
from langfuse.langchain import CallbackHandler

from dotenv import load_dotenv
load_dotenv(".env")

ollama = init_chat_model("ollama:gemma4:latest", base_url=os.getenv("OLLAMA_BASE_URL"))


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
research_instructions = "You're a helpful assistant."
agent = create_deep_agent(
    model=ollama,
    tools=[internet_search],
    system_prompt=research_instructions,
    backend=FilesystemBackend(
        root_dir=Path.cwd(),
    ),
).with_config(
    {
        "callbacks": [langfuse_handler],
        "run_name": "deepagents",
    }
)
result = agent.invoke({"messages": [{"role": "user", "content": "create an empty file with name cabbar.txt in the current directory and then show the file list and show the content of the first file"}]})
# Print the agent's response
print(result["messages"][-1].content)
