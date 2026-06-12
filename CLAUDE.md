# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

CustomAgent is a local AI agent built with LangChain + LangGraph, using Ollama for local model inference. It supports web search (Baidu with Sogou fallback), webpage fetching, and extensible tool calling. Two interfaces: CLI (`main.py`) and Web GUI (`gui.py`).

## Commands

```bash
# Install deps (in venv)
pip install -r requirements.txt

# Pull recommended model
ollama pull qwen3.5:4b-mlx

# Run CLI mode
python main.py

# Run Web GUI (browser at http://127.0.0.1:8000)
python gui.py

# Syntax check all files
python -m py_compile src/config.py src/state.py src/agent.py src/tools/*.py main.py gui.py
```

## Architecture

The agent follows the standard LangGraph ReAct pattern with a custom `bind_tools()` + `ToolNode` setup:

```
START → agent (LLM + bound tools) → tools (ToolNode) → agent → ... → END
```

**Key files and their roles:**

- `src/agent.py` — Core: `create_agent()` builds and compiles the StateGraph. Contains the system prompt template and `_build_system_prompt()` which dynamically injects the current real time on every LLM call. The model is configured with `num_predict=config.max_tokens` to override Ollama's default 128-token limit.

- `src/config.py` — Singleton `Config` dataclass. `model`, `temperature`, `max_tokens`, `max_search_results`, `log_level`. Imported everywhere as the `config` instance.

- `src/state.py` — `AgentState(TypedDict)` with a single field `messages: Annotated[list[BaseMessage], add_messages]`.

- `src/tools/__init__.py` — `ALL_TOOLS` list. This is the single registry; adding a new tool requires importing it here and appending to this list.
  - `time_tool.py` — `get_current_time()`: returns formatted current datetime.
  - `web_search.py` — `web_search(query, max_results)`: Baidu via `mcpcn-web-search-mcp` (auto fallback to Sogou). `fetch_webpage(url, max_length)`: fetches and extracts text from a URL using httpx + lxml, stripping `<script>/<style>/<nav>/<footer>/<header>`.

- `gui.py` — FastAPI app with a single `GET /` route returning an embedded HTML template, and a `WS /ws` WebSocket endpoint that runs `agent.astream_events(version="v2")` for token-level streaming. The HTML template includes all CSS (dark theme via CSS custom properties) and JS (marked.js for Markdown rendering, WebSocket client with auto-reconnect). The WebSocket protocol: client sends `{"type": "chat", "content": "..."}`, server streams `token`, `tool_call`, `tool_result`, `references`, `error`, `done` events. Tool calls are displayed as lightweight search indicators; reference sources are rendered as expandable `<details>` links.

- `main.py` — CLI loop using `agent.stream(stream_mode="values")` synchronously.

## Critical design details

### Dynamic time injection
The system prompt is NOT a static string. `_build_system_prompt()` in `agent.py` calls `datetime.now()` each time and uses `timedelta` to pre-calculate yesterday, day-before-yesterday, and Monday's dates. These are filled into the prompt template via `str.format()`, including example search queries with the concrete dates. This ensures the model always knows the real date and converts relative time descriptions ("最近两天") into absolute date ranges in search queries.

### `num_predict` is required
Ollama defaults to 128 tokens of output. Without setting `num_predict` on `ChatOllama`, responses get silently truncated. The `config.max_tokens` field (default 4096) maps to this parameter.

### Tool calling requires a capable model
The agent uses `llm.bind_tools(ALL_TOOLS)`. Models below ~3B parameters often don't support native tool calling. The code expects models like qwen3.5:4b-mlx, qwen2.5:7b, llama3.1:8b, etc.

### GUI HTML is self-contained
The entire frontend lives as `HTML_TEMPLATE` string constant in `gui.py`. There are no separate template files, no build step, and no npm dependencies. `marked` is loaded from CDN. CSS uses CSS custom properties for the dark theme; all variables are defined in `:root`.

### Synchronous tools with async streaming
The tools (web_search_mcp, httpx) are synchronous but the GUI uses `agent.astream_events()` — LangGraph handles the async/sync bridge internally via `ToolNode`. For a single-user local app this is acceptable; if blocking becomes an issue, individual tools can be wrapped with `asyncio.to_thread()`.

### Streaming implementation
The `call_model()` function in `agent.py` uses `llm_with_tools.stream()` (chunk-by-chunk accumulation) instead of `invoke()`, and the GUI uses `agent.astream_events(version="v2")` to capture `on_chat_model_stream` events for token-level streaming. The frontend appends tokens to a single streaming bubble with a blinking cursor effect, then finalizes the bubble when streaming completes.

## Adding a new tool

1. Create `src/tools/new_tool.py` with a function decorated `@tool`
2. Import and append to `ALL_TOOLS` in `src/tools/__init__.py`
3. Optionally add an icon mapping in `gui.py`'s `TOOL_ICONS` and `TOOL_LABELS` JS objects, a summary handler in `_tool_summary()`, and a label in `main.py`'s `TOOL_LABELS`
