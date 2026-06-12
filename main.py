#!/usr/bin/env python3
"""CustomAgent — 基于 LangChain + LangGraph 的本地 AI Agent

使用 Ollama 运行本地大模型，支持联网搜索、网页获取等工具调用。
"""

import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，支持从任意位置运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import HumanMessage

from src.agent import create_agent
from src.config import config, setup_logging

log = logging.getLogger("main")

TOOL_LABELS = {
    "get_current_time": "正在获取当前时间...",
    "web_search": "正在百度搜索...",
    "fetch_webpage": "正在获取网页内容...",
}


def main():
    setup_logging()
    agent = create_agent()

    print("=" * 50)
    print("   CustomAgent 已启动")
    print(f"   模型: {config.model}")
    print(f"   工具: 百度搜索 | 网页获取 | 时间查询")
    print("   输入 'quit' 退出")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        log.info("用户输入: %r", user_input)

        try:
            final_msg = None
            for event in agent.stream(
                {"messages": [HumanMessage(content=user_input)]},
                stream_mode="values",
            ):
                if "messages" not in event:
                    continue
                msg = event["messages"][-1]

                # 显示工具调用状态
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        label = TOOL_LABELS.get(tc["name"], f"正在调用 {tc['name']}...")
                        print(f"\n  {label}")

                final_msg = msg

            # 打印最终回复
            if final_msg and hasattr(final_msg, "content") and final_msg.content:
                if not (hasattr(final_msg, "tool_calls") and final_msg.tool_calls):
                    print(f"\n小智: {final_msg.content}")

        except Exception as e:
            log.error("运行出错: %s", e, exc_info=True)
            print(f"\n[错误] {e}")


if __name__ == "__main__":
    main()
