"""FastAPI 应用 —— SSE 端点 + 通用聊天

POST /api/v1/travel/plan  → SSE 流式旅行规划
POST /api/v1/chat          → 通用问答
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.agent import create_agent
from src.config import config, setup_logging
from src.state import TravelState
from src.travel_agent import create_travel_agent

log = logging.getLogger("api")

setup_logging()

app = FastAPI(
    title="TravelAgent API",
    description="AI 旅行规划 + 通用问答 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 请求模型 ────────────────────────────────────────────────────────

class TravelPlanRequest(BaseModel):
    city: str = Field(description="旅行城市", examples=["上海"])
    days: int = Field(default=3, ge=1, le=14, description="旅行天数")
    style: str = Field(default="休闲", description="旅行风格", examples=["亲子游", "情侣", "美食", "历史文化"])
    budget: str = Field(default="中等", description="预算水平", examples=["经济", "中等", "高端"])
    start_date: str = Field(default="", description="出发日期 YYYY-MM-DD", examples=["2026-07-01"])


class ChatRequest(BaseModel):
    message: str = Field(description="用户消息")


# ─── SSE 辅助 ────────────────────────────────────────────────────────

PHASE_MESSAGES = {
    "query_gen": ("query_gen", "分析需求，生成搜索计划..."),
    "search_node": ("searching", "并行搜索 POI、美食、交通信息..."),
    "enrich_node": ("enriching", "获取重点 POI 详细信息..."),
    "research_summary": ("summarizing", "整理搜索数据，提炼关键信息..."),
    "plan_generate": ("planning", "编排路线，生成结构化攻略..."),
}

SEARCH_DONE = False
ENRICH_DONE = False


async def _stream_travel_plan(request: TravelPlanRequest):
    """SSE 流式返回旅行规划结果"""
    try:
        agent = create_travel_agent()

        initial_state: TravelState = {
            "city": request.city,
            "days": request.days,
            "style": request.style,
            "budget": request.budget,
            "start_date": request.start_date,
            "search_queries": [],
            "search_results": [],
            "enriched_results": [],
            "research_data": {},
            "travel_plan": {},
        }

        seen_phases = set()
        search_count = 0
        enrich_count = 0

        async for event in agent.astream_events(initial_state, version="v2"):
            kind = event.get("event", "")

            if kind == "on_chain_start":
                node_name = event.get("name", "")
                if node_name in PHASE_MESSAGES:
                    phase, message = PHASE_MESSAGES[node_name]

                    # search_node 和 enrich_node 会多次触发，用计数器控制
                    if node_name == "search_node":
                        if search_count == 0:
                            yield _sse_event("progress", {"phase": phase, "message": message})
                            seen_phases.add(phase)
                        search_count += 1
                    elif node_name == "enrich_node":
                        if enrich_count == 0:
                            yield _sse_event("progress", {"phase": phase, "message": message})
                            seen_phases.add(phase)
                        enrich_count += 1
                    elif phase not in seen_phases:
                        yield _sse_event("progress", {"phase": phase, "message": message})
                        seen_phases.add(phase)

            elif kind == "on_chain_end":
                node_name = event.get("name", "")
                output = event.get("data", {}).get("output", {})

                # 捕获最终输出
                if node_name == "plan_generate" and isinstance(output, dict):
                    travel_plan = output.get("travel_plan", {})
                    if travel_plan:
                        yield _sse_event("result", travel_plan)

        yield _sse_event("done", {})

    except Exception as e:
        log.error("旅行规划失败: %s", e, exc_info=True)
        yield _sse_event("error", {"message": str(e)})
        yield _sse_event("done", {})


def _sse_event(event: str, data: dict) -> dict:
    """构建 SSE 事件字典（sse-starlette 格式）"""
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


# ─── 端点 ────────────────────────────────────────────────────────────

@app.post("/api/v1/travel/plan")
async def travel_plan(request: TravelPlanRequest):
    """旅行规划 SSE 端点

    返回 SSE 流：
    - progress: 进度更新 {phase, message}
    - result: 完整 TravelPlan JSON
    - error: 错误信息
    - done: 流结束
    """
    log.info("收到旅行规划请求: city=%s, days=%d, style=%s", request.city, request.days, request.style)
    return EventSourceResponse(_stream_travel_plan(request))


@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    """通用问答端点"""
    log.info("收到聊天请求: %s", request.message[:100])

    try:
        from langchain_core.messages import HumanMessage

        agent = create_agent()
        result = agent.invoke({"messages": [HumanMessage(content=request.message)]})

        last_msg = result["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        return {"response": content}
    except Exception as e:
        log.error("聊天失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "provider": config.provider,
        "model": config.model,
    }


# ─── 启动入口 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
