import json
import logging
from typing import Annotated, Dict, List, Literal, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from ddgs import DDGS
from datetime import datetime

# ==========================================
# 0. 日志配置 — 用于排查 Agent 是否触发联网搜索
# ==========================================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")

# ==========================================
# 1. 初始化本地 Gemma 模型（开启 JSON Mode）
# ==========================================
# 关键改动：添加 format="json" 强制模型输出 JSON
llm = ChatOllama(model="qwen3.5:2b", temperature=0, format="json")


# 获取当前系统时间的工具
@tool
def get_current_time() -> str:
    """获取当前系统的日期和时间，用于判断是否需要联网搜索最新信息。"""
    log.info("[TIME] get_current_time 被调用")
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y年%m月%d日 %H:%M:%S")
    log.info("[TIME] 返回当前时间: %s", formatted_time)
    return f"当前时间: {formatted_time}\n\n注意：你的训练数据截止日期是2025年1月。如果用户的问题涉及此日期之后的事件或数据，你必须使用搜索工具获取最新信息。"  


# 使用最新标准的 @tool 包装原生 DuckDuckGo
@tool
def google_search(query: str) -> str:
    """当你需要网络搜索最新信息、近期新闻、实时股票时，调用此工具。"""
    log.info("[SEARCH] google_search 被调用, query=%r", query)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            log.info("[SEARCH] DuckDuckGo 返回 %d 条结果", len(results))
            if not results:
                log.warning("[SEARCH] 无搜索结果")
                return "没有找到相关的网络搜索结果。"
            formatted = "\n\n".join([f"标题: {r['title']}\n摘要: {r['body']}" for r in results])
            log.debug("[SEARCH] 结果预览: %s", formatted[:300])
            return formatted
    except Exception as e:
        log.error("[SEARCH] 搜索失败: %s", e, exc_info=True)
        return f"网络搜索遇到错误: {str(e)}"


# 工具映射字典
tools_map = {"get_current_time": get_current_time, "google_search": google_search}


# ==========================================
# 2. 定义 Agent 的状态
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ==========================================
# 3. 定义图的节点（Nodes）
# ==========================================

def _summarize_messages(messages: List[BaseMessage]) -> str:
    """简要描述当前消息列表，便于日志排查。"""
    parts = []
    for i, msg in enumerate(messages):
        role = type(msg).__name__
        content = getattr(msg, "content", "")
        preview = (content[:120] + "…") if len(content) > 120 else content
        parts.append(f"  [{i}] {role}: {preview!r}")
    return "\n".join(parts)


def call_model(state: AgentState) -> Dict:
    """Agent 节点：强迫大模型选择行动或直接回答"""
    messages = state["messages"]
    log.info("[MODEL] call_model 开始, 当前消息数=%d", len(messages))
    log.debug("[MODEL] 输入消息:\n%s", _summarize_messages(messages))

    # 全新的系统提示词，规范模型的行为。必须包含 action 和 content 字段。
    system_prompt = (
        "你是一个有联网能力的 AI 助手。你必须严格以下列 JSON 格式之一回答，不得包含任何其他文字：\n\n"
        "情况 A: 首先必须调用 get_current_time 获取当前时间：\n"
        '{"action": "get_time", "query": "", "content": ""}\n\n'
        "情况 B: 如果你已经获取了当前时间，且用户的问题涉及2025年1月之后的事件或需要最新信息（如实时新闻、最新股市）：\n"
        '{"action": "search", "query": "搜索关键词", "content": ""}\n\n'
        "情况 C: 如果你已经获取了时间并且进行了搜索（如果需要），或者可以直接回答用户的问题：\n"
        '{"action": "reply", "query": "", "content": "你的详细中文回答内容"}\n\n'
        "重要规则：\n"
        "1. 每次对话必须首先调用 get_current_time 获取当前时间！\n"
        "2. 你的训练数据截止日期是2025年1月。如果问题涉及此日期之后的内容，必须使用 search 工具！\n"
        "3. 永远不要说'我的知识截止日期是XXX'，必须使用工具获取最新信息！\n"
        "4. 你的输出直接对应程序解析，格式错误程序会崩溃，必须严格遵循JSON格式。"
    )

    formatted_messages = [HumanMessage(content=system_prompt)] + messages
    log.debug("[MODEL] 调用 LLM (model=gemma4:e2b, format=json), 总消息数=%d", len(formatted_messages))
    response = llm.invoke(formatted_messages)

    raw_content = response.content
    log.info("[MODEL] LLM 原始输出 (%d 字符):\n%s", len(raw_content), raw_content)

    try:
        data = json.loads(raw_content.strip())
        action = data.get("action")
        log.info(
            "[MODEL] JSON 解析成功 → action=%r, query=%r, content_len=%d",
            action,
            data.get("query", ""),
            len(data.get("content", "")),
        )
        if action == "reply":
            log.warning(
                "[MODEL] ⚠️  模型选择直接回复 (action=reply)，未触发搜索。"
                "若问题需要实时信息，说明模型未按预期选择 search。"
            )
        elif action == "search":
            log.info("[MODEL] ✓ 模型选择搜索 (action=search), query=%r", data.get("query"))
        else:
            log.warning("[MODEL] ⚠️  未知 action=%r，路由将走向 end", action)
    except json.JSONDecodeError as e:
        log.error("[MODEL] ❌ JSON 解析失败: %s\n原始内容: %r", e, raw_content)
    except Exception as e:
        log.error("[MODEL] ❌ 解析输出时异常: %s", e, exc_info=True)

    return {"messages": [response]}


def call_tool(state: AgentState) -> Dict:
    """Action 节点：根据 action 类型执行对应的工具（获取时间或搜索）"""
    last_message = state["messages"][-1]
    log.info("[TOOL] call_tool 节点被进入")

    try:
        # 解析强约束下的 JSON
        raw = last_message.content.strip()
        log.debug("[TOOL] 待解析内容: %r", raw)
        data = json.loads(raw)
        action = data.get("action", "")
        search_query = data.get("query", "")

        if action == "get_time":
            log.info("[TOOL] ⏰ 触发获取当前时间")
            time_result = get_current_time.invoke({})
            log.info("[TOOL] 📥 获取时间完成, 返回给模型判断是否需要搜索")
            return {"messages": [ToolMessage(content=time_result, tool_call_id="time_call")]}
        
        elif action == "search":
            if not search_query:
                log.warning("[TOOL] ⚠️  action=search 但 query 为空，搜索可能无意义")

            log.info("[TOOL] 🛠️  触发联网搜索, query=%r", search_query)

            # 执行检索
            search_result = google_search.invoke(search_query)

            log.info(
                "[TOOL] 📥 搜索完成, 结果长度=%d 字符, 交还给模型二次推理",
                len(search_result),
            )
            log.debug("[TOOL] 搜索结果预览: %s", search_result[:300])

            # 将结果包装为 ToolMessage 返回给状态
            return {"messages": [ToolMessage(content=search_result, tool_call_id="search_call")]}
        
        else:
            log.warning("[TOOL] ⚠️  未知 action=%r，返回错误消息", action)
            return {"messages": [ToolMessage(content=f"未知的 action 类型: {action}", tool_call_id="unknown")]}
            
    except json.JSONDecodeError as e:
        error_msg = f"解析 JSON 时出错: {str(e)}"
        log.error("[TOOL] ❌ JSON 解析失败: %s, 原始=%r", e, last_message.content)
        return {"messages": [ToolMessage(content=error_msg, tool_call_id="error_call")]}
    except Exception as e:
        error_msg = f"执行工具时出错: {str(e)}"
        log.error("[TOOL] ❌ 执行失败: %s", e, exc_info=True)
        return {"messages": [ToolMessage(content=error_msg, tool_call_id="error_call")]}


# ==========================================
# 4. 定义条件边（Conditional Edges）
# ==========================================

def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """通过解析 JSON 的 action 字段决定去向"""
    last_message = state["messages"][-1]
    log.info("[ROUTER] should_continue 判断路由, 最后消息类型=%s", type(last_message).__name__)

    try:
        raw = last_message.content.strip()
        data = json.loads(raw)
        action = data.get("action")
        log.info("[ROUTER] 解析到 action=%r", action)

        if action == "get_time":
            log.info("[ROUTER] → 路由到 action 节点 (continue)，将获取当前时间")
            return "continue"
        elif action == "search":
            log.info("[ROUTER] → 路由到 action 节点 (continue)，将执行联网搜索")
            return "continue"

        log.info("[ROUTER] → 路由到 END (end), action=%r，不会调用工具", action)
    except json.JSONDecodeError as e:
        log.error(
            "[ROUTER] ❌ JSON 解析失败，默认 end（不调用工具）: %s\n原始内容: %r",
            e,
            last_message.content,
        )
    except Exception as e:
        log.error("[ROUTER] ❌ 路由判断异常，默认 end: %s", e, exc_info=True)

    return "end"


# ==========================================
# 5. 构建 LangGraph 工作流
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)

workflow.add_edge("action", "agent")
app = workflow.compile()

# ==========================================
# 6. 交互运行
# ==========================================
if __name__ == "__main__":
    print("🤖 本地强制 JSON 模式 Agent 已启动！输入 'quit' 退出。")
    log.info("[MAIN] Agent 启动: model=qwen3.5:2b, format=json, 工具=[get_current_time, google_search]")
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == 'quit':
            break

        log.info("[MAIN] ========== 新一轮对话开始 ==========")
        log.info("[MAIN] 用户输入: %r", user_input)
        inputs = {"messages": [HumanMessage(content=user_input)]}

        step = 0
        search_triggered = False
        for output in app.stream(inputs, stream_mode="updates"):
            step += 1
            for key, value in output.items():
                log.info("[MAIN] 图步骤 #%d, 节点=%r", step, key)

                if key == "action":
                    search_triggered = True
                    log.info("[MAIN] ✓ action 节点已执行")

                if key == "agent":
                    raw_content = value["messages"][-1].content
                    try:
                        data = json.loads(raw_content.strip())
                        action = data.get("action")
                        log.info("[MAIN] agent 节点输出 action=%r", action)
                        # 如果是最终回复，打印给用户
                        if action == "reply":
                            print(f"\nAI: {data.get('content')}")
                        elif action == "search":
                            log.info("[MAIN] agent 请求搜索, 等待 action 节点...")
                    except json.JSONDecodeError:
                        log.error("[MAIN] ❌ agent 输出非合法 JSON: %r", raw_content)
                        print(f"\nAI (Raw): {raw_content}")
                    except Exception as e:
                        log.error("[MAIN] 处理 agent 输出异常: %s", e)
                        print(f"\nAI (Raw): {raw_content}")

        if not search_triggered:
            log.warning(
                "[MAIN] ⚠️  本轮对话结束，从未进入 action 节点 — 联网搜索未被调用。"
                "请检查上方 [MODEL] 和 [ROUTER] 日志。"
            )
        else:
            log.info("[MAIN] 本轮对话结束，联网搜索已执行。")
        log.info("[MAIN] ========== 对话结束 (共 %d 步) ==========", step)