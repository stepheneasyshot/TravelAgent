import logging

from langchain_core.tools import tool

from .web_search import web_search, fetch_webpage

log = logging.getLogger(__name__)

CATEGORY_KEYWORDS = {
    "景点": "必去景点 热门打卡",
    "美食": "必吃美食 网红餐厅 特色小吃",
    "亲子": "亲子游玩 遛娃 儿童乐园 动物园",
    "购物": "购物中心 特色街区 免税店",
    "交通": "交通攻略 地铁 公交",
    "住宿": "住宿推荐 酒店 民宿",
    "攻略": "旅游攻略 行程安排 避坑指南",
}


@tool
def search_city_poi(city: str, category: str = "景点", count: int = 5) -> str:
    """搜索城市热门 POI（景点/美食/购物/亲子/交通/住宿/攻略）。

    参数 city: 城市名称
    参数 category: 类别，可选 "景点"/"美食"/"亲子"/"购物"/"交通"/"住宿"/"攻略"
    参数 count: 返回结果数量，默认 5
    """
    suffix = CATEGORY_KEYWORDS.get(category, "热门推荐")
    query = f"{city} {suffix}"
    log.info("search_city_poi: city=%r, category=%r", city, category)
    return web_search.invoke({"query": query, "max_results": count})


@tool
def get_poi_info(name: str, city: str) -> str:
    """获取单个 POI 详细信息（开放时间/门票/游玩攻略等）。

    参数 name: POI/景点名称
    参数 city: 所在城市
    """
    log.info("get_poi_info: name=%r, city=%r", name, city)

    # 先搜索获取详情
    search_result = web_search.invoke({
        "query": f"{city} {name} 开放时间 门票 攻略",
        "max_results": 3,
    })

    # 尝试提取第一个有效链接获取详情页
    lines = []
    for line in search_result.split("\n"):
        if line.startswith("链接: ") or line.startswith("URL: "):
            url = line.split(": ", 1)[-1].strip()
            if url and url.startswith("http"):
                try:
                    detail = fetch_webpage.invoke({"url": url, "max_length": 2000})
                    return f"=== {name} 搜索概览 ===\n{search_result}\n\n=== 详情页 ({url}) ===\n{detail}"
                except Exception:
                    pass
            break

    return f"=== {name} 搜索概览 ===\n{search_result}"


@tool
def search_transport(from_poi: str, to_poi: str) -> str:
    """搜索两个地点之间的交通方式（地铁/公交/打车/步行）。

    参数 from_poi: 出发地名称
    参数 to_poi: 目的地名称
    """
    log.info("search_transport: %r → %r", from_poi, to_poi)
    return web_search.invoke({
        "query": f"{from_poi} 到 {to_poi} 怎么走 地铁 公交 打车",
        "max_results": 5,
    })
