"""旅行规划 Agent —— 2 阶段 Plan-and-Execute 工作流。

Phase 1: 结构化并行搜索
  1.1 query_gen: 1 次小 LLM 调用 → N 个搜索关键词
  1.2 parallel_search: N 路 web_search 并行执行
  1.3 parallel_enrich: M 路 fetch_webpage 并行补全详情
  1.4 research_summary: 1 次小 LLM 调用 → 结构化 TravelResearchData

Phase 2: 规划生成
  单次 LLM 调用 with_structured_output(TravelPlan) → 完整行程 JSON
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import END, START, Send
from langgraph.graph import StateGraph

from .config import config
from .models import create_llm, create_llm_with_structured_output
from .schemas import TravelPlan
from .state import TravelState
from .tools.web_search import fetch_webpage, web_search

log = logging.getLogger(__name__)

# ─── Phase 1.1: 查询生成 ────────────────────────────────────────────

QUERY_GEN_PROMPT = """\
你是一个旅行规划助手。根据用户的旅行需求，生成一系列百度搜索关键词。

用户需求:
- 城市: {city}
- 天数: {days} 天
- 风格: {style}
- 预算: {budget}
- 出发日期: {start_date}

请生成 8-12 个搜索关键词，覆盖以下类别（每个类别至少 1 个）:
1. 必去景点（含门票、开放时间）
2. 美食推荐（含特色小吃、正餐餐厅）
3. 交通攻略（市内交通、景点间换乘）
4. 行程安排参考（{days}日游路线）
5. 实用贴士（天气、避坑、购物）

要求:
- 每条搜索词用双引号包裹，词与词之间用逗号分隔
- 搜索词要具体、可搜索，包含城市名和关键限定词
- 考虑用户风格偏好
- 考虑出发日期的季节特点

只输出搜索词列表，不要任何解释。"""


def query_gen(state: TravelState) -> dict:
    """Phase 1.1: 根据用户需求生成搜索关键词列表"""
    log.info("query_gen: city=%s, days=%d, style=%s", state["city"], state["days"], state["style"])

    llm = create_llm()
    prompt = QUERY_GEN_PROMPT.format(
        city=state["city"],
        days=state["days"],
        style=state["style"],
        budget=state["budget"],
        start_date=state["start_date"],
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    queries = []
    for m in re.finditer(r'"([^"]+)"', raw):
        queries.append(m.group(1))

    if not queries:
        queries = [q.strip().strip('"') for q in raw.replace("\n", ",").split(",") if q.strip()]

    if not queries:
        city = state["city"]
        style = state["style"]
        days = state["days"]
        queries = [
            f"{city} 必去景点 推荐",
            f"{city} {style} 旅游攻略",
            f"{city} {days}日游 行程安排",
            f"{city} 美食 必吃推荐",
            f"{city} 交通攻略 地铁",
            f"{city} 旅游 注意事项 避坑",
        ]

    log.info("query_gen: 生成 %d 个搜索词", len(queries))
    return {"search_queries": queries}


# ─── Phase 1.2 & 1.3: 并行搜索与补全 ─────────────────────────────────

def continue_to_searches(state: TravelState) -> list[Send]:
    """将每个搜索词分派到 search_node 并行执行"""
    queries = state.get("search_queries", [])
    if not queries:
        return []
    return [Send("search_node", {"query": q, "index": i}) for i, q in enumerate(queries)]


def search_node(state: dict) -> dict:
    """单个搜索执行节点（被 Send() 并行调用）"""
    query = state.get("query", "")
    index = state.get("index", 0)
    log.info("search_node[%d]: %s", index, query)

    try:
        result = web_search.invoke({"query": query, "max_results": config.max_search_results})
    except Exception as e:
        log.error("search_node[%d] 失败: %s", index, e)
        result = f"搜索失败: {e}"

    return {"search_results": [{"query": query, "index": index, "result": result}]}


def extract_urls(text: str) -> list[str]:
    """从搜索文本中提取 URL"""
    urls = []
    for line in text.split("\n"):
        if line.startswith("链接: ") or line.startswith("URL: "):
            url = line.split(": ", 1)[-1].strip()
            if url.startswith("http"):
                urls.append(url)
    return urls


def route_after_search(state: TravelState) -> list[Send] | str:
    """搜索完成后: 有 URL 则并行抓取详情，否则直接进入总结"""
    results = state.get("search_results", [])
    seen = set()
    all_urls = []

    for item in results:
        for url in extract_urls(item.get("result", "")):
            if url not in seen:
                seen.add(url)
                all_urls.append(url)

    urls_to_fetch = all_urls[:6]
    log.info("route_after_search: 发现 %d 个 URL，将抓取 %d 个", len(all_urls), len(urls_to_fetch))

    if not urls_to_fetch:
        return "research_summary"

    return [Send("enrich_node", {"url": u, "index": i}) for i, u in enumerate(urls_to_fetch)]


def enrich_node(state: dict) -> dict:
    """单个网页抓取节点（被 Send() 并行调用）"""
    url = state.get("url", "")
    index = state.get("index", 0)
    log.info("enrich_node[%d]: %s", index, url)

    try:
        content = fetch_webpage.invoke({"url": url, "max_length": 3000})
    except Exception as e:
        log.error("enrich_node[%d] 失败: %s", index, e)
        content = f"抓取失败: {e}"

    return {"enriched_results": [{"url": url, "index": index, "content": content}]}


# ─── Phase 1.4: 研究总结 ────────────────────────────────────────────

RESEARCH_SUMMARY_PROMPT = """\
你是一个旅行研究分析师。根据以下联网搜索结果，整理出结构化的旅行研究数据。

城市: {city}
天数: {days}
风格: {style}
预算: {budget}
出发日期: {start_date}

搜索结果:
{search_results_text}

请以 JSON 格式输出研究总结，包含以下字段:
- attractions: 景点列表，每个包含 name, description, hours, ticket, tips
- foods: 美食列表，每个包含 name, description, address
- transport: 交通信息列表，每个包含 from, to, method, duration
- tips: 旅行贴士列表（字符串数组）
- weather: 预计天气描述

只输出 JSON，不要任何解释。"""


def research_summary(state: TravelState) -> dict:
    """Phase 1.4: 将原始搜索结果总结为结构化数据"""
    log.info("research_summary: 开始总结研究数据")

    parts = []
    for item in state.get("search_results", []):
        parts.append(f"### 搜索词: {item['query']}\n{item['result']}")

    for item in state.get("enriched_results", []):
        parts.append(f"### 详情页: {item['url']}\n{item['content']}")

    search_text = "\n\n---\n\n".join(parts)

    if len(search_text) > 12000:
        search_text = search_text[:12000] + "\n\n... (内容已截断)"

    llm = create_llm()
    prompt = RESEARCH_SUMMARY_PROMPT.format(
        city=state["city"],
        days=state["days"],
        style=state["style"],
        budget=state["budget"],
        start_date=state["start_date"],
        search_results_text=search_text,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        research_data = json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        log.warning("research_summary: JSON 解析失败，使用原始文本")
        research_data = {"raw": raw}

    log.info("research_summary: 完成")
    return {"research_data": research_data}


# ─── Phase 2: 规划生成 ──────────────────────────────────────────────

PLAN_GENERATE_PROMPT = """\
你是一个资深旅行规划师。根据研究数据，为一个{days}天的{city}{style}行程生成完整的旅行计划。

## 用户需求
- 城市: {city}
- 天数: {days} 天
- 风格: {style}
- 预算: {budget}
- 出发日期: {start_date}

## 研究数据
{research_data_text}

请生成一个完整的旅行计划 JSON，严格按照以下结构:
- city: 城市名
- days: 天数
- style: 旅行风格
- overview: 城市概述 + 行程总览（200-300字）
- daily_plans: 每日行程列表，每天包含:
  - day: 第几天
  - theme: 当日主题
  - schedule: 时间轴，每项包含 time, activity, poi(可选，含name/description/address/hours/ticket/tips/coordinates), transport_from_prev
  - meals: 餐食列表，每项包含 meal(早餐/午餐/晚餐), name, description, address
- food_recommendations: 额外的美食推荐（字符串列表）
- transport_tips: 城市交通整体贴士
- budget_estimate: 预算估算
- weather_note: 天气提示

要求:
1. 每天安排 3-5 个景点/活动，时间合理
2. 考虑景点间距离和交通，顺路编排
3. 每天推荐早中晚三餐
4. 结合用户风格（{style}）定制内容
5. coordinates 字段如果不知道可以省略

只输出 JSON。"""


def plan_generate(state: TravelState) -> dict:
    """Phase 2: 单次 LLM 调用生成完整 TravelPlan"""
    log.info("plan_generate: 开始生成旅行计划")

    research_data = state.get("research_data", {})
    research_text = json.dumps(research_data, ensure_ascii=False, indent=2)

    if len(research_text) > 8000:
        research_text = research_text[:8000] + "\n... (已截断)"

    prompt = PLAN_GENERATE_PROMPT.format(
        city=state["city"],
        days=state["days"],
        style=state["style"],
        budget=state["budget"],
        start_date=state["start_date"],
        research_data_text=research_text,
    )

    try:
        structured_llm = create_llm_with_structured_output(TravelPlan)
        plan = structured_llm.invoke([
            SystemMessage(content="你是一个专业的旅行规划师，输出结构化的旅行计划 JSON。"),
            HumanMessage(content=prompt),
        ])

        log.info("plan_generate: 成功生成计划 (days=%d, daily_plans=%d)",
                 plan.days, len(plan.daily_plans))
        return {"travel_plan": plan.model_dump()}

    except Exception as e:
        log.error("plan_generate: 结构化输出失败，尝试原始 JSON 解析: %s", e)
        llm = create_llm()
        response = llm.invoke([
            SystemMessage(content="你是一个专业的旅行规划师，只输出 JSON。"),
            HumanMessage(content=prompt + "\n\n只输出 JSON，不要包裹在 ``` 中。"),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.split("```")[0].strip()

        try:
            plan_dict = json.loads(raw)
            plan = TravelPlan(**plan_dict)
            return {"travel_plan": plan.model_dump()}
        except Exception as e2:
            log.error("plan_generate: JSON 解析也失败: %s", e2)
            return {"travel_plan": {"error": str(e2), "raw": raw}}


# ─── 构建图 ──────────────────────────────────────────────────────────

def create_travel_agent():
    """构建并编译旅行规划 Agent 工作流图。

    工作流:
        START → query_gen → [N × search_node] → route →
            ├─ 有 URL → [M × enrich_node] → research_summary
            └─ 无 URL → research_summary
            → plan_generate → END
    """
    workflow = StateGraph(TravelState)

    workflow.add_node("query_gen", query_gen)
    workflow.add_node("search_node", search_node)
    workflow.add_node("enrich_node", enrich_node)
    workflow.add_node("research_summary", research_summary)
    workflow.add_node("plan_generate", plan_generate)

    # START → query_gen
    workflow.add_edge(START, "query_gen")

    # query_gen → 并行分派搜索
    workflow.add_conditional_edges(
        "query_gen",
        continue_to_searches,
        ["search_node"],
    )

    # 搜索完成后 → 路由：抓取详情 or 直接总结
    workflow.add_conditional_edges(
        "search_node",
        route_after_search,
        {
            "research_summary": "research_summary",
            "enrich_node": "enrich_node",
        },
    )

    # enrich 完成后 → 总结
    workflow.add_edge("enrich_node", "research_summary")

    # 总结 → 规划 → 结束
    workflow.add_edge("research_summary", "plan_generate")
    workflow.add_edge("plan_generate", END)

    return workflow.compile()
