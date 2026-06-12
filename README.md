# TravelAgent

基于 **LangChain + LangGraph** 的本地 AI Agent，支持通用联网搜索问答 + 结构化旅行路线规划。提供 **CLI 命令行**、**FastAPI SSE 端点**（面向移动端）两种交互方式。

## 功能

- **通用问答** — 联网搜索 + 网页抓取，获取最新信息并给出详细回答
- **旅行路线规划** — 根据城市/天数/风格，自动搜索 POI 并生成结构化旅行攻略
- **结构化输出** — 旅行攻略以 Pydantic JSON 返回，移动端直接反序列化消费
- **流式进度推送** — SSE（Server-Sent Events）推送实时规划进度
- **模型层抽象** — 同一套代码支持 Ollama 本地模型 / DeepSeek 云端模型，改配置切换
- **并行搜索** — 旅行规划中多个搜索词并行执行，缩短延迟
- **动态时间注入** — 每次推理自动注入当前真实时间，搜索词自动换算相对日期

## 快速开始

### 前置要求

- Python 3.11+
- [Ollama](https://ollama.com) 已安装并运行（本地模型模式）
- 支持 function calling 的模型（推荐 4B+）

### 安装

```bash
# 1. 拉取模型（本地模式）
ollama pull qwen3.5:4b-mlx

# 2. 虚拟环境
python -m venv .venv
source .venv/bin/activate       # macOS/Linux

# 3. 安装依赖
pip install -r requirements.txt
```

### 启动

```bash
# API 服务（推荐，面向移动端/前端）
python api.py
# SSE 端点: POST /api/v1/travel/plan
# 浏览器查看文档: http://127.0.0.1:8000/docs

# 命令行模式
python main.py
```

### 模型选择与切换

```python
# src/config.py
class Config:
    # 本地 Ollama
    provider: str = "ollama"
    model: str = "qwen3.5:4b-mlx"

    # 切换 DeepSeek 云端模型
    # provider = "deepseek"
    # model = "deepseek-chat"
    # deepseek_api_key = "sk-xxx"
```

| provider | 模型 | 适用场景 |
|----------|------|----------|
| ollama | qwen3.5:4b-mlx | 本地开发、离线使用 |
| ollama | qwen2.5:7b | 更好效果、更高延迟 |
| ollama | llama3.1:8b | 备选 |
| deepseek | deepseek-chat (V3) | 生产环境、移动端后端 |

## API 设计

### 旅行规划 SSE 端点

```
POST /api/v1/travel/plan
Content-Type: application/json

{
  "city": "上海",
  "days": 3,
  "style": "亲子游",
  "budget": "中等",
  "start_date": "2026-07-01"
}

-- SSE 流式响应 --

event: progress
data: {"phase":"query_gen","message":"分析需求，生成搜索计划..."}

event: progress
data: {"phase":"searching","message":"并行搜索 POI、美食、交通信息..."}

event: progress
data: {"phase":"enriching","message":"获取重点 POI 详细信息..."}

event: progress
data: {"phase":"planning","message":"编排路线，生成攻略..."}

event: result
data: { <TravelPlan JSON 结构> }

event: done
data: {}
```

移动端消费：监听 SSE 事件，`progress` 更新进度条，`result` 反序列化后直接渲染。

### 结构化输出 Schema

核心输出类型（`src/schemas.py`）：

```
TravelPlan
├── city, days, style           # 基本信息
├── overview                    # 城市概述 + 行程总览
├── daily_plans: [DayPlan]      # 每日行程
│   ├── day, theme              # 第几天 + 主题
│   ├── schedule: [ScheduleItem]  # 时间轴
│   │   ├── time, activity
│   │   ├── poi: POIInfo        # 景点信息
│   │   └── transport_from_prev # 交通方式
│   └── meals: [MealItem]        # 餐食推荐
├── food_recommendations        # 其他推荐美食
├── transport_tips              # 交通贴士
├── budget_estimate             # 预算估算
└── weather_note                # 天气提示
```

`POIInfo.coordinates` 字段支持移动端地图标注。

## 项目结构

```
TravelAgent/
├── src/
│   ├── config.py              # 模型/provider/日志配置
│   ├── models.py              # LLM 工厂（provider 抽象）
│   ├── state.py               # AgentState + TravelState
│   ├── schemas.py             # 结构化输出类型定义
│   ├── agent.py               # 通用问答 Agent（ReAct）
│   ├── travel_agent.py        # 旅行规划 Agent（2 阶段图）
│   ├── tools/
│   │   ├── __init__.py        # 工具注册表
│   │   ├── time_tool.py       # 当前时间查询
│   │   ├── web_search.py      # 联网搜索 + 网页获取
│   │   └── travel_search.py   # POI 搜索 / 详情 / 交通
│   └── api.py                 # FastAPI + SSE 端点
├── api.py                     # API 服务启动入口
├── main.py                    # CLI 入口
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## 架构

### 通用问答：ReAct Agent

```
START → agent (LLM + 工具) → tools (ToolNode) → agent → ... → END
```

适用于：联网搜索问答、网页抓取、实时信息查询。

### 旅行规划：2 阶段 Plan-and-Execute

```
POST /api/v1/travel/plan
         │
         ▼
┌────────────────────────────────────────────┐
│  Phase 1: Research（结构化搜索）            │
│                                            │
│  1.1 查询生成（1 次小 LLM 调用）             │
│      → 输出 N 个搜索关键词                   │
│                                            │
│  1.2 并行搜索（N 路 web_search 同时执行）    │
│      → 累积原始搜索结果                     │
│                                            │
│  1.3 并行详情补全（M 路 fetch_webpage）      │
│      → 关键 POI 开放时间/门票/攻略          │
│                                            │
│  1.4 研究总结（1 次小 LLM 调用）             │
│      → 结构化 TravelResearchData           │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│  Phase 2: Plan & Generate（单次 LLM 调用）   │
│                                            │
│  输入: 全部 research_data + 用户需求         │
│  输出: TravelPlan（with_structured_output）  │
│                                            │
│  一次调用完成: 路线编排 + 时间分配 + 攻略润色 │
└────────────────────┬───────────────────────┘
                     │
                     ▼
            结构化 JSON → 移动端消费
```

**为什么是 2 阶段：**
- 云端强模型（DeepSeek V3）可以一次性完成路线编排和攻略润色，不需要拆成多个 LLM 节点
- Phase 1 的搜索用「计划+并行执行」替代自由 ReAct，覆盖更全面、延迟更低
- LLM 调用次数固定（3-4 次），token 成本可控

### 时间处理

核心痛点：模型不知道"现在"是什么时候，相对时间描述会搜错日期。

解决方案：每次调用 LLM 前，`_build_system_prompt()` 用 `datetime.now()` 动态生成系统提示词，预计算今天/昨天/本周一的具体日期，并填入搜索词示例。旅行规划中同理，`start_date` 字段支持用户指定出行日期，日期类搜索词自动换算。

### 模型层抽象

```python
# src/models.py
def create_llm(config: Config):
    if config.provider == "ollama":
        return ChatOllama(model=config.model, ...)
    elif config.provider == "deepseek":
        return ChatOpenAI(model="deepseek-chat", api_key=..., base_url=...)
```

DeepSeek API 兼容 OpenAI 格式，使用 `langchain-openai` 的 `ChatOpenAI` 即可。切换 provider 只需改 `config.py` 一行。

## 配置

编辑 `src/config.py`：

```python
config = Config(
    # 模型提供商
    provider="ollama",             # "ollama" | "deepseek" | "openai"
    model="qwen3.5:4b-mlx",        # Ollama 模型名 或 API 模型名

    # DeepSeek 配置（provider="deepseek" 时生效）
    deepseek_api_key="",           # 或从环境变量 DEEPSEEK_API_KEY 读取
    deepseek_base_url="https://api.deepseek.com/v1",

    # 通用
    temperature=0.0,
    max_tokens=4096,
    max_search_results=10,
    log_level=logging.INFO,
)
```

## 工具说明

### 通用工具

| 工具 | 说明 |
|------|------|
| `get_current_time` | 获取当前系统日期时间（备用，时间已自动注入 prompt） |
| `web_search(query)` | 百度搜索（自动故障转移到搜狗），返回标题+链接+摘要 |
| `fetch_webpage(url)` | 抓取网页文本，自动去噪（script/style/nav） |

### 旅行工具

| 工具 | 说明 |
|------|------|
| `search_city_poi(city, category)` | 搜索城市热门 POI（景点/美食/购物），返回结构化列表 |
| `get_poi_info(name, city)` | 获取单个 POI 详细信息（开放时间/门票/攻略） |
| `search_transport(from_poi, to_poi)` | 搜索两点间交通方式（地铁/公交/打车） |

旅行工具底层复用 `web_search`，是对搜索能力的结构化包装。

## 添加自定义工具

1. 在 `src/tools/` 下创建新文件，使用 `@tool` 装饰器
2. 在 `src/tools/__init__.py` 中注册到 `ALL_TOOLS` 列表
3. 如果是旅行专用工具，注册到 `TRAVEL_TOOLS` 列表

## 依赖

| 包 | 用途 |
|----|------|
| langgraph | Agent 工作流图 |
| langchain-ollama | Ollama 本地模型 |
| langchain-openai | DeepSeek 等 OpenAI 兼容 API |
| langchain-core | 消息、工具、结构化输出 |
| fastapi | API 服务 |
| uvicorn | ASGI 服务器 |
| sse-starlette | SSE 流式事件推送 |
| mcpcn-web-search-mcp | 百度/搜狗搜索 |
| httpx | HTTP 客户端 |
| lxml | HTML 解析 |
| pydantic | 结构化输出 Schema |
