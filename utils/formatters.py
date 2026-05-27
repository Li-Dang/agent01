def format_itinerary_table(itinerary_list: list) -> str:
    """将行程列表转换成 Markdown 表格字符串"""
    header = "| 时间 | 活动 | 交通方式 | 预估花费 |\n|------|------|----------|----------|\n"
    rows = ""
    for item in itinerary_list:
        rows += f"| {item.get('time','')} | {item.get('activity','')} | {item.get('transport','')} | {item.get('cost','')} |\n"
    return header + rows

def format_food_table(food_list: list) -> str:
    header = "| 店名 | 必点菜 | 人均消费 | 推荐理由 |\n|------|--------|----------|----------|\n"
    rows = ""
    for item in food_list:
        rows += f"| {item.get('name','')} | {item.get('must_try','')} | {item.get('avg_price','')} | {item.get('reason','')} |\n"
    return header + rows

def add_weather_tip(weather_info: dict) -> str:
    if weather_info.get("rain_prob", 0) > 50:
        return "\n> 🌧️ 温馨提示：未来几小时有降雨，建议随身带伞，户外行程可适当调整。"
    return ""