from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Agent 状态：消息列表会自动合并"""

    messages: Annotated[list[BaseMessage], add_messages]
