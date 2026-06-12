from .time_tool import get_current_time
from .web_search import web_search, fetch_webpage
from .travel_search import search_city_poi, get_poi_info, search_transport

# 通用 Q&A Agent 工具
ALL_TOOLS = [get_current_time, web_search, fetch_webpage]

# 旅行规划 Agent 工具
TRAVEL_TOOLS = [search_city_poi, get_poi_info, search_transport, web_search, fetch_webpage]
