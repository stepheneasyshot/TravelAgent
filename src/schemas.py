from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class POIInfo(BaseModel):
    """景点/地点信息"""

    name: str = Field(description="景点名称")
    description: str = Field(description="景点简介")
    address: str = Field(default="", description="地址")
    hours: str = Field(default="", description="开放时间")
    ticket: str = Field(default="", description="门票信息")
    tips: str = Field(default="", description="游玩贴士")
    coordinates: Optional[list[float]] = Field(default=None, description="经纬度 [lat, lng]")


class ScheduleItem(BaseModel):
    """行程中的一个项目"""

    time: str = Field(description="时间段，如 09:00-11:00")
    activity: str = Field(description="活动描述")
    poi: Optional[POIInfo] = Field(default=None, description="关联的景点信息")
    transport_from_prev: str = Field(default="", description="从上个地点到此地的交通方式")


class MealItem(BaseModel):
    """餐食推荐"""

    meal: str = Field(description="餐别: 早餐/午餐/晚餐")
    name: str = Field(description="餐厅/美食名称")
    description: str = Field(description="推荐理由")
    address: str = Field(default="", description="地址")


class DayPlan(BaseModel):
    """单日行程"""

    day: int = Field(description="第几天")
    theme: str = Field(description="当日主题")
    schedule: list[ScheduleItem] = Field(default_factory=list, description="当日行程")
    meals: list[MealItem] = Field(default_factory=list, description="餐食推荐")


class TravelPlan(BaseModel):
    """完整旅行计划 —— 移动端 API 契约"""

    city: str = Field(description="旅行城市")
    days: int = Field(description="旅行天数")
    style: str = Field(description="旅行风格")
    overview: str = Field(description="城市概述 + 行程总览")
    daily_plans: list[DayPlan] = Field(default_factory=list, description="每日行程")
    food_recommendations: list[str] = Field(default_factory=list, description="其他推荐美食")
    transport_tips: str = Field(default="", description="交通贴士")
    budget_estimate: str = Field(default="", description="预算估算")
    weather_note: str = Field(default="", description="天气提示")


class TravelResearchData(BaseModel):
    """Phase 1 研究阶段的结构化输出"""

    city: str = Field(description="城市名")
    attractions: list[dict] = Field(default_factory=list, description="景点列表")
    foods: list[dict] = Field(default_factory=list, description="美食列表")
    transport: list[dict] = Field(default_factory=list, description="交通信息")
    tips: list[str] = Field(default_factory=list, description="旅行贴士")
    weather: str = Field(default="", description="天气信息")
