# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

CustomAgent is a local AI agent built with LangChain + LangGraph. It has two subsystems:

1. **General Q&A Agent** — ReAct pattern with web search + webpage fetch, driven by Ollama (or any OpenAI-compatible API). Supports real-time information retrieval and structured answers.
2. **Travel Planning Agent** — 2-phase Plan-and-Execute workflow. Phase 1 runs structured parallel searches to collect POI/city data. Phase 2 uses a single LLM call with `with_structured_output()` to produce a Pydantic TravelPlan JSON. Output targets mobile clients via SSE.

The model layer is **provider-abstracted**: `src/models.py` has a `create_llm()` factory. Switching from local Ollama to DeepSeek (or any OpenAI-compatible API) is a one-line config change.

## Commands

```bash
# Install deps (in venv)
pip install -r requirements.txt

# Pull recommended local model
ollama pull qwen3.5:4b-mlx

# Run API server (SSE endpoint for travel planning + WebSocket fallback)
python api.py

# Run CLI mode (general Q&A)
python main.py

# Syntax check all files
python -m py_compile src/config.py src/models.py src/state.py src/schemas.py \
  src/agent.py src/travel_agent.py src/tools/*.py src/api.py main.py
```

## Architecture

### General Q&A: ReAct Agent (`src/agent.py`)

```
START → agent (LLM + bound tools) → tools (ToolNode) → agent → ... → END
```

- `create_agent()` builds a StateGraph with agent + tools nodes
- `_build_system_prompt()` dynamically injects the current real time on every LLM call
- `num_predict` maps from `config.max_tokens` to override Ollama's 128-token default
- The graph uses `should_continue()` routing: if the last message has `tool_calls`, loop back to tools; otherwise END

### Travel Planning: 2-Phase Plan-and-Execute (`src/travel_agent.py`)

```
POST /api/v1/travel/plan {city, days, style, budget, start_date}
         │
         ▼
┌──────────────────────────────────────────────┐
│  Phase 1: Research                          │
│                                              │
│  1.1 query_gen (1 small LLM call)            │
│      → generates N targetted search queries  │
│      → e.g. ["上海 亲子 必去景点",            │
│               "上海 亲子餐厅 推荐", ...]       │
│                                              │
│  1.2 parallel_search (N web_search calls)     │
│      → all queries executed simultaneously   │
│      → results accumulated in state          │
│                                              │
│  1.3 parallel_enrich (M fetch_webpage calls)  │
│      → key POIs get detailed pages fetched   │
│      → extracts hours, tickets, tips         │
│                                              │
│  1.4 research_summary (1 small LLM call)      │
│      → raw results → structured              │
│        TravelResearchData                    │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  Phase 2: Plan & Generate                   │
│                                              │
│  Single LLM call:                            │
│    llm.with_structured_output(TravelPlan)    │
│                                              │
│  Input: all research_data + user constraints │
│  Output: complete TravelPlan JSON            │
│                                              │
│  One call handles: route ordering,           │
│  time allocation, guide polishing            │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
            SSE stream → mobile client
```

**Why 2-phase instead of pure ReAct:**
- Pure ReAct interleaves search and planning, which leads to inconsistent coverage (model may stop searching too early or wander off)
- Structured parallel searches guarantee broad POI coverage with predictable latency
- A single structured-output call leverages the full reasoning power of strong cloud models (DeepSeek V3) to produce a coherent plan in one shot

**Why structured search instead of free ReAct for Phase 1:**
- Fixed number of LLM calls (2 small ones) — token cost is predictable
- N parallel searches — total latency = slowest single request, not sum of all
- All categories (attractions, food, transport, tips) are guaranteed to be covered
- Research results are cacheable (same city + same style within 24h)

### Routing: General vs Travel

`src/api.py` exposes separate endpoints or routes by intent:

- `POST /api/v1/travel/plan` → `travel_agent.run()` (travel planning)
- `POST /api/v1/chat` → `agent.invoke()` (general Q&A)

No shared state between the two graphs. Each has its own tools and prompts.

### Model provider abstraction (`src/models.py`)

```python
def create_llm(config: Config):
    if config.provider == "ollama":
        return ChatOllama(model=config.model, ...)
    elif config.provider == "deepseek":
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
        )
```

DeepSeek's API is OpenAI-compatible, so `langchain-openai`'s `ChatOpenAI` works directly. Switching providers is a one-line config change.

### Key files and their roles

- `src/config.py` — Singleton `Config` dataclass. `provider`, `model`, `temperature`, `max_tokens`, `max_search_results`, `log_level`, plus DeepSeek-specific fields (`deepseek_api_key`, `deepseek_base_url`). Imported everywhere as the `config` instance.

- `src/models.py` — `create_llm(config)` factory. Returns the right LangChain chat model based on `config.provider`. Also houses `create_llm_with_structured_output(config, schema)` for typed outputs.

- `src/schemas.py` — Pydantic models for structured output: `TravelPlan`, `DayPlan`, `ScheduleItem`, `POIInfo`, `MealItem`. This is the **API contract** with mobile clients — all fields must be backward-compatible.

- `src/state.py` — `AgentState(TypedDict)` for general Q&A (single field: `messages`). `TravelState(TypedDict)` for travel planning (fields: `city`, `days`, `style`, `budget`, `start_date`, `search_queries`, `research_data`, `travel_plan`).

- `src/agent.py` — General Q&A graph: `create_agent()` with ReAct pattern. Contains `SYSTEM_PROMPT` template and `_build_system_prompt()` for dynamic time injection.

- `src/travel_agent.py` — Travel planning graph: `create_travel_agent()` with 2-phase nodes (`query_gen → parallel_search → parallel_enrich → research_summary → plan_generate`). Uses `Send()` API for parallel tool execution.

- `src/api.py` — FastAPI application. `POST /api/v1/travel/plan` returns SSE stream with `progress` / `result` / `done` events. `POST /api/v1/chat` for general Q&A. Replaces `gui.py` as the primary interface.

- `src/tools/time_tool.py` — `get_current_time()`: returns formatted current datetime.

- `src/tools/web_search.py` — `web_search(query, max_results)`: Baidu via `mcpcn-web-search-mcp` (auto fallback to Sogou). `fetch_webpage(url, max_length)`: fetches and extracts text from a URL using httpx + lxml, stripping `<script>/<style>/<nav>/<footer>/<header>`.

- `src/tools/travel_search.py` — `search_city_poi(city, category, count)`: wraps `web_search` with travel-specific query construction. `get_poi_info(name, city)`: deep-dive detail fetch for a single POI. `search_transport(from_poi, to_poi)`: transportation between two locations.

- `src/tools/__init__.py` — Two registries:
  - `ALL_TOOLS` — for general Q&A agent
  - `TRAVEL_TOOLS` — for travel planning agent

- `main.py` — CLI loop using `agent.stream(stream_mode="values")` synchronously. General Q&A only.

### Critical design details

#### Dynamic time injection
The system prompt is NOT a static string. `_build_system_prompt()` in `agent.py` calls `datetime.now()` each time and uses `timedelta` to pre-calculate yesterday, day-before-yesterday, and Monday's dates. These are filled into the prompt template via `str.format()`, including example search queries with the concrete dates. This ensures the model always knows the real date and converts relative time descriptions ("最近两天") into absolute date ranges in search queries. The travel agent does the same, using `start_date` from the user request to anchor all date references.

#### `num_predict` is required (Ollama only)
Ollama defaults to 128 tokens of output. Without setting `num_predict` on `ChatOllama`, responses get silently truncated. The `config.max_tokens` field (default 4096) maps to this parameter.

#### Structured output with `with_structured_output()`
Phase 2 of the travel agent uses LangChain's `with_structured_output(TravelPlan)`. This constrains the LLM response to valid Pydantic JSON. On DeepSeek, this uses JSON mode. On Ollama, it falls back to prompt-guided JSON parsing. Include `method="json_mode"` for providers that support it, otherwise `method="function_calling"`.

#### Parallel tool execution via `Send()`
Phase 1.2 (parallel search) uses LangGraph's `Send()` API to fan out N search calls simultaneously. Each call returns independently and results are merged. This is critical for latency — sequential search would take N × 3s, parallel takes ~3s.

```python
# travel_agent.py pattern
def continue_to_searches(state: TravelState):
    return [Send("search_node", {"query": q}) for q in state["search_queries"]]
```

#### Research caching
Phase 1 research data for (city, style) pairs is cacheable. Implement a simple TTL cache (e.g., 24h) on `research_summary` output to skip Phase 1 entirely on repeat requests for the same city + style.

#### Tool calling requires a capable model
The agent uses `llm.bind_tools(ALL_TOOLS)`. Models below ~3B parameters often don't support native tool calling. The code expects models like qwen3.5:4b-mlx, qwen2.5:7b, llama3.1:8b, or cloud models like DeepSeek V3.

#### Synchronous tools with async streaming
The tools (web_search, httpx) are synchronous but the API uses SSE streaming — LangGraph handles the async/sync bridge internally via `ToolNode`. For a single-user local app this is acceptable; for multi-user production, wrap individual tools with `asyncio.to_thread()`.

### API response format (SSE)

```
event: progress
data: {"phase": "query_gen|searching|enriching|planning", "message": "..."}

event: result
data: {<TravelPlan JSON>}

event: error
data: {"message": "..."}

event: done
data: {}
```

Mobile clients should:
- Listen for `progress` events and update a 4-step progress indicator
- Buffer `result` data and deserialize into the TravelPlan model
- Handle `error` events with a user-facing error state
- Close the connection on `done`

### Adding a new tool

**General Q&A tool:**
1. Create `src/tools/new_tool.py` with a function decorated `@tool`
2. Import and append to `ALL_TOOLS` in `src/tools/__init__.py`

**Travel-specific tool:**
1. Create `src/tools/new_travel_tool.py` with a function decorated `@tool`
2. Import and append to `TRAVEL_TOOLS` in `src/tools/__init__.py`

### Provider migration checklist

When switching from Ollama to DeepSeek:
1. Set `config.provider = "deepseek"` and configure `deepseek_api_key`
2. Set `max_tokens` higher (DeepSeek supports up to 8K output)
3. Enable `method="json_mode"` in `with_structured_output()` calls
4. Test Phase 2 output — stronger models may produce richer plans, verify schema compliance
5. Monitor token usage — Phase 1+2 combined is ~5K-15K tokens per request
