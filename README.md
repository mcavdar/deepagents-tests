# DeepAgents Experiments

Small Python experiments for exploring DeepAgents with chat models, web search, filesystem access, and Model Context Protocol (MCP) tools. The scripts also demonstrate optional Langfuse tracing.

## Scripts

- `deep1.py` and `backendfile.py` — DeepAgents examples using a model and web search; `backendfile.py` also demonstrates a filesystem backend.
- `deep-mcp.py`, `deep-sub.py`, and `deep-sub2.py` — experiments involving MCP tools or subagents.
- `deepmock.py`, `mcp-mock.py`, and `mcp-client-ollama.py` — mock and client experiments.

Run a script individually from the repository root, for example `python deep1.py`. These scripts may call configured model, search, or tracing services. `backendfile.py` creates a sample file in its working directory when run.

## Configuration

Provide the service configuration expected by the script you run, including `OLLAMA_BASE_URL` and, for web-search examples, `TAVILY_API_KEY`. Langfuse tracing uses the Langfuse client's environment configuration. Keep credentials in a local `.env` file or environment variables; do not commit them.

There is no dependency manifest in this repository, so install compatible Python packages required by the selected script before running it. Different experiments may need different dependencies.