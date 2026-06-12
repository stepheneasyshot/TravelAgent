import logging
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .config import config
from .models import create_llm
from .state import AgentState
from .tools import ALL_TOOLS

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是一个具备联网搜索能力的 AI 助手，名叫"小智"。你可以使用以下工具：

- **web_search**：通过百度搜索互联网，获取最新信息
- **fetch_webpage**：获取指定网页的详细文本内容
- **get_current_time**：获取当前系统日期和时间（备用）

## 时间规则（极其重要）

系统信息中已提供当前真实时间。当用户使用相对时间描述时，你必须自行换算为绝对日期：
- "最近两天/这两天" → 从 {date_before_yesterday} 到 {today} 的范围
- "今天" → {today}
- "昨天" → {yesterday}
- "本周" → 本周一到今天
- "最近一周" → 从7天前到今天

构建搜索关键词时，**必须把相对时间替换为具体日期**。例如：
- 用户问"最近两天有什么AI新闻" → 搜索词应为 "AI新闻 {date_before_yesterday} {today}"
- 用户问"今天的股市行情" → 搜索词应为 "股市行情 {today}"
- 用户问"本周科技大事" → 搜索词应为 "科技新闻 {monday} {today}"

## 工作规则
1. 如果问题涉及近期事件或需要最新数据，使用 web_search 搜索。
2. 搜索结果摘要不完整时，用 fetch_webpage 获取页面全文。
3. 基于获取的信息，用中文给出准确、清晰的回答，必要时附上来源链接。
4. 搜索不到相关信息时，如实告知用户。
5. 回答应详细、充分，提供足够的背景信息和细节。可以适度展开说明。
"""


def _build_system_prompt() -> str:
    """每次调用时动态生成系统提示词，注入当前真实时间"""
    now = datetime.now()
    weekday_names = ["一", "二", "三", "四", "五", "六", "日"]
    today = now.strftime("%Y年%m月%d日")

    yesterday = (now - timedelta(days=1)).strftime("%Y年%m月%d日")
    day_before_yesterday = (now - timedelta(days=2)).strftime("%Y年%m月%d日")
    monday = (now - timedelta(days=now.weekday())).strftime("%Y年%m月%d日")

    time_context = (
        f"当前真实时间: {today} {now.hour:02d}:{now.minute:02d}:{now.second:02d}"
        f" 星期{weekday_names[now.weekday()]}\n"
        f"今天是 {today}，昨天是 {yesterday}，前天是 {day_before_yesterday}，"
        f"本周一是 {monday}。"
    )

    return SYSTEM_PROMPT.format(
        today=today,
        yesterday=yesterday,
        date_before_yesterday=day_before_yesterday,
        monday=monday,
    ) + f"\n\n{time_context}"



def create_agent():
    """构建并编译 Agent 工作流图。

    工作流:
        START -> agent (LLM推理) -> [有工具调用?]
                  ^                    |
                  |                    v
                  +--- tools (执行工具) <-
    """
    llm = create_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def call_model(state: AgentState):
        """Agent 节点：流式调用 LLM，返回响应或工具调用请求"""
        messages = state["messages"]
        dynamic_prompt = _build_system_prompt()
        full_messages = [SystemMessage(content=dynamic_prompt)] + messages

        log.debug("call_model: 共 %d 条消息", len(full_messages))

        response = None
        for chunk in llm_with_tools.stream(full_messages):
            if response is None:
                response = chunk
            else:
                response += chunk

        if response is None:
            response = AIMessage(content="")

        if response.tool_calls:
            names = [tc["name"] for tc in response.tool_calls]
            log.info("call_model: 请求 %d 个工具调用 → %s", len(response.tool_calls), names)
        else:
            log.info("call_model: 生成最终回复")

        return {"messages": [response]}

    def should_continue(state: AgentState):
        """路由判断：有待执行的工具调用 → tools，否则 → END"""
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(ALL_TOOLS)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile()
