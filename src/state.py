from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """通用 Q&A Agent 状态"""

    messages: Annotated[list[BaseMessage], add_messages]


class TravelState(TypedDict):
    """旅行规划 Agent 状态

    带 operator.add reducer 的 list 字段支持 Send() 并行合并结果。
    """

    # 用户输入
    city: str
    days: int
    style: str
    budget: str
    start_date: str

    # Phase 1: 研究阶段（列表用 operator.add 累加）
    search_queries: Annotated[list[str], operator.add]
    search_results: Annotated[list[dict[str, Any]], operator.add]
    enriched_results: Annotated[list[dict[str, Any]], operator.add]
    research_data: dict[str, Any]

    # Phase 2: 规划阶段
    travel_plan: dict[str, Any]
