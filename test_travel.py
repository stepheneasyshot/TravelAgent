#!/usr/bin/env python3
"""旅行规划 API 测试脚本

用法:
    python test_travel.py              # 默认测试（3天上海亲子游）
    python test_travel.py --city 北京 --days 4 --style 历史文化 --budget 高端
    python test_travel.py --host localhost --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import httpx

# ─── 测试参数 ────────────────────────────────────────────────────────

DEFAULTS = {
    "city": "上海",
    "days": 3,
    "style": "亲子游",
    "budget": "中等",
    "start_date": "2026-07-01",
    "host": "127.0.0.1",
    "port": 8000,
}

PHASE_LABELS = {
    "query_gen": "🔍 生成搜索计划",
    "searching": "🌐 并行搜索中",
    "enriching": "📄 获取详情页",
    "summarizing": "📊 整理研究数据",
    "planning": "📝 生成旅行攻略",
}


def parse_sse(line: str) -> tuple[str, dict] | None:
    """解析单行 SSE 数据"""
    if line.startswith("event: "):
        return ("event", line[7:].strip())
    if line.startswith("data: "):
        data_str = line[6:].strip()
        try:
            return ("data", json.loads(data_str))
        except json.JSONDecodeError:
            return ("data", {"raw": data_str})
    return None


async def test_travel_plan(args):
    """测试旅行规划 SSE 端点"""
    base_url = f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/api/v1/travel/plan"

    payload = {
        "city": args.city,
        "days": args.days,
        "style": args.style,
        "budget": args.budget,
        "start_date": args.start_date,
    }

    print("=" * 60)
    print("  TravelAgent API 测试")
    print("=" * 60)
    print(f"  服务地址: {base_url}")
    print(f"  城市: {args.city}")
    print(f"  天数: {args.days}")
    print(f"  风格: {args.style}")
    print(f"  预算: {args.budget}")
    print(f"  出发日期: {args.start_date}")
    print("=" * 60)

    # 1. 健康检查
    print("\n[1/3] 健康检查...", end=" ")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{base_url}/api/v1/health")
            if resp.status_code == 200:
                health = resp.json()
                print(f"✓ OK (provider={health.get('provider')}, model={health.get('model')})")
            else:
                print(f"✗ 失败 (HTTP {resp.status_code})")
                return
    except httpx.ConnectError:
        print(f"✗ 无法连接 {base_url}")
        print("   请先启动服务: python api.py")
        return
    except Exception as e:
        print(f"✗ 异常: {e}")
        return

    # 2. 旅行规划 SSE 请求
    print("\n[2/3] 发送旅行规划请求...")
    print("-" * 40)

    travel_plan = None
    current_phase = None
    start_time = datetime.now()

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", api_url, json=payload) as response:
                if response.status_code != 200:
                    print(f"  ✗ 请求失败 (HTTP {response.status_code})")
                    print(f"  {await response.aread()}")
                    return

                event_type = None
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    parsed = parse_sse(line)
                    if parsed is None:
                        continue

                    key, value = parsed

                    if key == "event":
                        event_type = value

                    elif key == "data" and event_type:
                        if event_type == "progress":
                            phase = value.get("phase", "")
                            message = value.get("message", "")
                            label = PHASE_LABELS.get(phase, phase)
                            if phase != current_phase:
                                current_phase = phase
                                elapsed = (datetime.now() - start_time).total_seconds()
                                print(f"  [{elapsed:5.1f}s] {label}: {message}")

                        elif event_type == "result":
                            elapsed = (datetime.now() - start_time).total_seconds()
                            print(f"\n  [{elapsed:5.1f}s] ✅ 收到旅行计划!")
                            travel_plan = value

                        elif event_type == "error":
                            print(f"\n  ❌ 错误: {value.get('message', value)}")

                        elif event_type == "done":
                            pass  # 流结束

    except httpx.ReadTimeout:
        print("  ⚠️ 请求超时 (5分钟)")
        return
    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  总耗时: {elapsed:.1f}s")

    # 3. 结果展示
    print("\n[3/3] 结果预览")
    print("=" * 60)

    if travel_plan is None:
        print("  ❌ 未收到有效的旅行计划")
        return

    # 概要
    print(f"  城市: {travel_plan.get('city', 'N/A')}")
    print(f"  天数: {travel_plan.get('days', 'N/A')}")
    print(f"  风格: {travel_plan.get('style', 'N/A')}")
    overview = travel_plan.get("overview", "")
    if overview:
        print(f"  概述: {overview[:120]}...")

    # 每日行程
    daily_plans = travel_plan.get("daily_plans", [])
    print(f"\n  每日行程 ({len(daily_plans)} 天):")
    for day_plan in daily_plans:
        day = day_plan.get("day", "?")
        theme = day_plan.get("theme", "")
        print(f"\n  ┌─ 第{day}天: {theme}")

        schedule = day_plan.get("schedule", [])
        for item in schedule:
            time = item.get("time", "")
            activity = item.get("activity", "")
            poi = item.get("poi")
            poi_name = poi.get("name", "") if poi else ""
            transport = item.get("transport_from_prev", "")
            line = f"  │  {time}  {activity}"
            if poi_name:
                line += f" @{poi_name}"
            if transport:
                line += f"  [{transport}]"
            print(line)

        meals = day_plan.get("meals", [])
        if meals:
            print("  │")
            for meal in meals:
                meal_type = meal.get("meal", "")
                name = meal.get("name", "")
                desc = meal.get("description", "")[:40]
                print(f"  │  🍽️ {meal_type}: {name} — {desc}")

        print("  └" + "─" * 38)

    # 其他信息
    food_recs = travel_plan.get("food_recommendations", [])
    if food_recs:
        print(f"\n  美食推荐: {', '.join(food_recs[:5])}")

    transport_tips = travel_plan.get("transport_tips", "")
    if transport_tips:
        print(f"\n  交通贴士: {transport_tips[:200]}")

    budget = travel_plan.get("budget_estimate", "")
    if budget:
        print(f"\n  预算估算: {budget}")

    weather = travel_plan.get("weather_note", "")
    if weather:
        print(f"\n  天气提示: {weather[:200]}")

    # 保存完整结果
    output_file = f"test_result_{args.city}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(travel_plan, f, ensure_ascii=False, indent=2)
    print(f"\n  📁 完整结果已保存至: {output_file}")

    print("\n" + "=" * 60)
    print("  测试完成 ✓")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="旅行规划 API 测试脚本")
    parser.add_argument("--city", default=DEFAULTS["city"], help="旅行城市")
    parser.add_argument("--days", type=int, default=DEFAULTS["days"], help="旅行天数")
    parser.add_argument("--style", default=DEFAULTS["style"], help="旅行风格")
    parser.add_argument("--budget", default=DEFAULTS["budget"], help="预算水平")
    parser.add_argument("--start-date", default=DEFAULTS["start_date"], help="出发日期 YYYY-MM-DD")
    parser.add_argument("--host", default=DEFAULTS["host"], help="API 主机")
    parser.add_argument("--port", type=int, default=DEFAULTS["port"], help="API 端口")
    args = parser.parse_args()

    import asyncio
    asyncio.run(test_travel_plan(args))


if __name__ == "__main__":
    main()
