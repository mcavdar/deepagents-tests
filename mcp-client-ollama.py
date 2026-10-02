import base64
import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from ollama import AsyncClient as OllamaClient

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp_types import TextContent


load_dotenv()


# ============================================================
# Configuration
# ============================================================

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollamauser:mhmt@ollama.envai.tr")

MAX_TOOL_TURNS = 10

DEFAULT_REMOTE_MCP_URL = "https://docs.langchain.com/mcp"

headers = {}

credentials = f"ollamauser:mhmt"
encoded = base64.b64encode(
    credentials.encode("utf-8")
).decode("ascii")

headers["Authorization"] = f"Basic {encoded}"


class MCPClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

        # Ollama
        self.ollama = OllamaClient(
            host=OLLAMA_HOST,
            headers=headers,
        )

    # ========================================================
    # MCP CONNECTION
    # ========================================================

    async def connect_to_server(self, server: str):
        """
        Connect to either:

        1. Local Python MCP server:
             python client.py server.py

        2. Local JavaScript MCP server:
             python client.py server.js

        3. Remote Streamable HTTP MCP server:
             python client.py https://example.com/mcp
        """

        if server.startswith(("http://", "https://")):
            await self.connect_to_remote_server(server)
        else:
            await self.connect_to_local_server(server)

    async def connect_to_local_server(self, server_script_path: str):
        """Connect to a local MCP server over stdio."""

        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")

        if not (is_python or is_js):
            raise ValueError(
                "Local MCP server must be a .py or .js file"
            )

        if is_python:
            path = Path(server_script_path).resolve()

            server_params = StdioServerParameters(
                command="uv",
                args=[
                    "--directory",
                    str(path.parent),
                    "run",
                    path.name,
                ],
                env=None,
            )

        else:
            server_params = StdioServerParameters(
                command="node",
                args=[server_script_path],
                env=None,
            )

        print(f"\nConnecting to local MCP server: {server_script_path}")

        # stdio_client gives us the MCP read/write streams.
        read_stream, write_stream = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
            )
        )

        await self.session.initialize()

        await self.show_tools()

    async def connect_to_remote_server(self, url: str):
        """Connect to a remote MCP server over Streamable HTTP."""

        print(f"\nConnecting to remote MCP server:")
        print(f"  {url}")

        # Streamable HTTP MCP transport.
        #
        # Current MCP Python SDK returns:
        #
        #     read_stream, write_stream
        #
        # from streamable_http_client().
        read_stream, write_stream = await self.exit_stack.enter_async_context(
            streamable_http_client(url)
        )

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
            )
        )

        await self.session.initialize()

        await self.show_tools()

    async def show_tools(self):
        """Print MCP server information and available tools."""

        if self.session is None:
            raise RuntimeError("MCP session is not connected")

        response = await self.session.list_tools()

        print(
            f"\nConnected to MCP server "
            f"using protocol {self.session.protocol_version}"
        )

        print("\nAvailable tools:")

        for tool in response.tools:
            print(f"  - {tool.name}")

            if tool.description:
                print(f"    {tool.description}")

    # ========================================================
    # MCP -> OLLAMA TOOL FORMAT
    # ========================================================

    async def get_ollama_tools(self):
        """
        Convert MCP tool definitions into Ollama's
        tool-calling format.
        """

        if self.session is None:
            raise RuntimeError("MCP session is not connected")

        response = await self.session.list_tools()

        ollama_tools = []

        for tool in response.tools:
            ollama_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
            )

        return ollama_tools

    # ========================================================
    # MCP TOOL RESULT
    # ========================================================

    def extract_tool_result(self, result) -> str:
        """
        Convert an MCP CallToolResult into text that can be
        passed back to Ollama.
        """

        parts = []

        # Normal MCP content blocks
        for block in result.content:
            if isinstance(block, TextContent):
                parts.append(block.text)

        # Some MCP servers return structured content.
        if result.structured_content:
            parts.append(
                f"\nStructured content:\n"
                f"{result.structured_content}"
            )

        if result.is_error:
            parts.insert(0, "MCP tool returned an error.")

        if not parts:
            return "Tool returned no textual output."

        return "\n".join(parts)

    # ========================================================
    # OLLAMA + MCP AGENT LOOP
    # ========================================================

    async def process_query(self, query: str) -> str:
        """
        Send a query to Ollama.

        If Ollama decides that an MCP tool is required:

            Ollama
              ↓
            MCP tool
              ↓
            result
              ↓
            Ollama
              ↓
            final answer
        """

        if self.session is None:
            raise RuntimeError(
                "MCP server is not connected"
            )

        tools = await self.get_ollama_tools()

        messages = [
            {
                "role": "user",
                "content": query,
            }
        ]

        # ----------------------------------------------------
        # First Ollama request
        # ----------------------------------------------------

        response = await self.ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=tools,
        )

        for turn in range(MAX_TOOL_TURNS):

            message = response["message"]

            # ------------------------------------------------
            # Normal response
            # ------------------------------------------------

            content = message.get("content", "")

            tool_calls = message.get(
                "tool_calls",
                []
            )

            if content:
                print(
                    f"\nOllama: {content}"
                )

            # ------------------------------------------------
            # No tools requested -> we're done
            # ------------------------------------------------

            if not tool_calls:
                return content

            # Add Ollama's assistant message to conversation.
            messages.append(message)

            # ------------------------------------------------
            # Execute every requested tool
            # ------------------------------------------------

            for tool_call in tool_calls:

                function = tool_call["function"]

                tool_name = function["name"]
                tool_args = function.get(
                    "arguments",
                    {},
                )

                print(
                    f"\n[Tool call {turn + 1}]"
                )

                print(
                    f"  Tool: {tool_name}"
                )

                print(
                    f"  Args: {tool_args}"
                )

                try:

                    result = await self.session.call_tool(
                        tool_name,
                        arguments=tool_args,
                    )

                    tool_output = self.extract_tool_result(
                        result
                    )

                except Exception as e:

                    tool_output = (
                        f"Error executing MCP tool "
                        f"{tool_name}: {e}"
                    )

                print(
                    f"  Result:\n{tool_output[:1000]}"
                )

                # ------------------------------------------------
                # Give the MCP result back to Ollama.
                # ------------------------------------------------

                messages.append(
                    {
                        "role": "tool",
                        "content": tool_output,
                    }
                )

            # ------------------------------------------------
            # Ask Ollama to continue reasoning.
            # ------------------------------------------------

            response = await self.ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=tools,
            )

        return (
            f"Ollama reached the maximum tool-call limit "
            f"({MAX_TOOL_TURNS})."
        )

    # ========================================================
    # CHAT LOOP
    # ========================================================

    async def chat_loop(self):
        """Interactive terminal chat."""

        print("\n========================================")
        print(" Ollama + MCP Client")
        print("========================================")
        print(f"Model: {OLLAMA_MODEL}")
        print(f"Ollama: {OLLAMA_HOST}")

        print(
            "\nType 'quit' or 'exit' to stop."
        )

        while True:

            try:
                query = await asyncio.to_thread(
                    input,
                    "\nYou: ",
                )

            except (
                EOFError,
                KeyboardInterrupt,
            ):
                break

            query = query.strip()

            if not query:
                continue

            if query.lower() in {
                "quit",
                "exit",
            }:
                break

            try:

                response = await self.process_query(
                    query
                )

                print(
                    f"\nAssistant: {response}"
                )

            except Exception as e:

                print(
                    f"\nError: {e}"
                )

    # ========================================================
    # CLEANUP
    # ========================================================

    async def cleanup(self):

        await self.exit_stack.aclose()


# ============================================================
# MAIN
# ============================================================

async def main():

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "\n  Local MCP:"
        )

        print(
            "    python client.py server.py"
        )

        print(
            "\n  Remote MCP:"
        )

        print(
            "    python client.py "
            "https://docs.langchain.com/mcp"
        )

        print(
            "\nOr simply edit DEFAULT_REMOTE_MCP_URL."
        )

        sys.exit(1)

    server = sys.argv[1]

    client = MCPClient()

    try:

        # ----------------------------------------------------
        # Check Ollama
        # ----------------------------------------------------

        print(
            f"Checking Ollama at {OLLAMA_HOST}..."
        )

        try:

            await client.ollama.list()

        except Exception as e:

            print(
                "\nCould not connect to Ollama."
            )

            print(
                "Make sure Ollama is running:"
            )

            print(
                "\n  ollama serve"
            )

            print(
                f"\nError: {e}"
            )

            return

        # ----------------------------------------------------
        # Connect MCP
        # ----------------------------------------------------

        await client.connect_to_server(
            server
        )

        # ----------------------------------------------------
        # Start chat
        # ----------------------------------------------------

        await client.chat_loop()

    except Exception as e:

        print(
            f"\nFatal error: {e}"
        )

    finally:

        await client.cleanup()


if __name__ == "__main__":

    asyncio.run(main())
