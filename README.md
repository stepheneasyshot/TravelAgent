# CustomAgent

基于 **LangChain + LangGraph** 的本地 AI Agent，使用 Ollama 运行开源大模型，支持联网搜索、网页抓取和工具调用。提供 **CLI 命令行**和 **Web 图形界面**两种交互方式。

## 功能

- **本地模型推理** — 通过 Ollama 运行开源模型，数据不出本地
- **联网搜索** — 基于百度/搜狗搜索互联网，自动故障转移，获取最新信息
- **网页获取** — 抓取任意网页的文本内容（自动去噪）
- **流式输出** — GUI 支持 token 级流式输出，打字机效果
- **动态时间注入** — 每次推理自动注入当前真实时间，搜索词自动换算相对日期
- **工具调用** — `bind_tools()` + `ToolNode` 标准 ReAct 模式
- **Web GUI** — 聊天界面 + 搜索指示器 + 参考来源 + Markdown 渲染
- **可扩展** — 模块化工具注册，添加新工具只需三步

## 快速开始

### 前置要求

- Python 3.11+
- [Ollama](https://ollama.com) 已安装并运行
- 支持 function calling 的模型（推荐 4B+）

### 安装

```bash
# 1. 拉取模型
ollama pull qwen3.5:4b-mlx

# 2. 虚拟环境
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

### 启动

```bash
# Web 图形界面（推荐）
python gui.py
# 浏览器打开 http://127.0.0.1:8000

# 命令行模式
python main.py
```

### 模型选择

| 模型 | 大小 | function calling | 推荐 |
|------|------|:---:|------|
| qwen3.5:4b-mlx | 2.4GB | ✓ | **推荐** (支持 MLX 加速) |
| qwen2.5:7b | 4.7GB | ✓ | 更好效果 |
| qwen2.5:14b | 8.9GB | ✓ | 更佳效果 |
| llama3.2:3b | 2.0GB | ✓ | 低配置 |
| llama3.1:8b | 4.9GB | ✓ | 备选 |

> 3B 以下模型可能不支持原生 function calling，且多轮工具调用后容易丢失上下文。Apple Silicon 设备推荐使用带 `-mlx` 后缀的模型以获得 GPU 加速。

## 项目结构

```
CustomAgent/
├── src/
│   ├── __init__.py
│   ├── agent.py             # Agent 图定义、动态时间注入、路由
│   ├── config.py            # 模型/日志/搜索参数配置
│   ├── state.py             # AgentState 状态定义
│   └── tools/
│       ├── __init__.py      # 工具注册表 (ALL_TOOLS)
│       ├── time_tool.py     # 当前时间查询
│       └── web_search.py    # 联网搜索 + 网页获取
├── main.py                  # CLI 入口
├── gui.py                   # Web GUI (FastAPI + WebSocket + 内嵌 HTML)
├── requirements.txt
└── README.md
```

## 架构

### Agent 工作流

使用 LangGraph 构建 ReAct 模式：

```
┌─────────┐     ┌───────────┐     ┌──────────┐
│  START  │────▶│   agent   │◀────│  tools   │
└─────────┘     └─────┬─────┘     └──────────┘
                      │ 有 tool_calls?   ▲
                      ├──────────────────┘
                      │ 无
                      ▼
                 ┌─────────┐
                 │   END   │
                 └─────────┘
```

- **agent 节点** — LLM 推理（已绑定工具），决定回复还是调用工具
- **tools 节点** (`ToolNode`) — 自动解析 `tool_calls` 并执行对应函数
- **条件路由** — 检查消息是否包含 `tool_calls`，有则进入 tools 循环
- **每次推理注入当前时间** — `_build_system_prompt()` 动态生成包含真实日期的提示词

### 时间处理

Agent 的核心痛点：模型不知道"现在"是什么时候，"最近两天"会搜成错误的日期。

**解决方案**：每次调用 LLM 前，`_build_system_prompt()` 用 `datetime.now()` 动态生成提示词，预计算好今天/昨天/本周一的具体日期并填入搜索词示例：

```
当前真实时间: 2026年06月07日 星期日
今天是 2026年06月07日，昨天是 2026年06月06日，本周一是 2026年06月01日。

"最近两天有什么AI新闻" → 搜索词应为 "AI新闻 2026年06月05日 2026年06月07日"
```

### Web GUI 架构

```
┌─────────────────────────────┬──────────────────┐
│  聊天面板                    │  运行状态         │
│  消息气泡 + 搜索指示器       │  实时日志         │
│  Markdown 渲染              │  工具调用 / 错误   │
├─────────────────────────────┤                  │
│  [输入框______________] [发送]│                  │
└─────────────────────────────┴──────────────────┘
```

- **后端**: FastAPI + WebSocket，`agent.astream_events()` 流式推送（token 级）
- **前端**: 原生 HTML/CSS/JS，零框架依赖
- **实时事件**: token → tool_call → tool_result → references → done（token 级流式输出）
- **Markdown**: 使用 `marked` 库渲染，支持标题、代码块、表格、列表、引用等

## 配置

编辑 `src/config.py`：

```python
config = Config(
    model="qwen3.5:4b-mlx",       # Ollama 模型名称
    temperature=0.0,             # 生成温度 (0=确定性)
    max_search_results=5,        # 搜索返回条数
    log_level=logging.INFO,      # 日志级别
)
```

## 工具说明

### get_current_time
获取当前系统日期和时间（备用工具，正常情况下时间已自动注入）。

### web_search
通过百度搜索引擎查询互联网（自动故障转移到搜狗），返回标号列表（标题 + 链接 + 摘要）。适合查找新闻、事实、实时数据。搜索结果摘要不完整时，可进一步调用 `fetch_webpage`。

### fetch_webpage
抓取指定 URL 的文本内容，自动去除 `<script>`、`<style>`、`<nav>` 等噪音标签，默认返回前 3000 字符。

## 添加自定义工具

1. 在 `src/tools/` 下创建新文件，如 `calculator.py`
2. 使用 `@tool` 装饰器定义：

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。参数 expression: 数学表达式字符串。"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"
```

3. 在 `src/tools/__init__.py` 中注册：

```python
from .calculator import calculator
ALL_TOOLS = [get_current_time, web_search, fetch_webpage, calculator]
```

## 依赖

| 包 | 用途 |
|----|------|
| langgraph | Agent 工作流图 + ToolNode |
| langchain-ollama | Ollama 模型集成 |
| langchain-core | 消息、工具抽象 |
| mcpcn-web-search-mcp | 百度/搜狗搜索 |
| beautifulsoup4 | HTML 内容解析 |
| requests | HTTP 请求 |
| httpx | HTTP 客户端（网页获取） |
| lxml | HTML 解析 |
| fastapi | Web 服务器 |
| uvicorn | ASGI 服务器 |

## License

MIT
