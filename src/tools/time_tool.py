from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """获取当前系统时间。当需要知道当前日期时间或判断信息时效性时使用。"""
    now = datetime.now()
    weekday = ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]
    return (
        f"当前时间: {now.year}年{now.month}月{now.day}日 "
        f"{now.hour:02d}:{now.minute:02d}:{now.second:02d} 星期{weekday}"
    )
