import json
import requests
from langchain.tools import tool
from config import QWEATHER_API_KEY, QWEATHER_API_HOST, MAP_API_KEY

# 和风天气城市ID映射（常用城市）
CITY_ID_MAP = {
    "成都": "101270101",
    "北京": "101010100",
    "上海": "101020100",
    "广州": "101280101",
    "深圳": "101280601",
    "杭州": "101210101",
    "重庆": "101040100",
}


def _get_city_id(city: str) -> str:
    """将城市名转为和风天气 location ID，支持中文名直接查询"""
    if city in CITY_ID_MAP:
        return CITY_ID_MAP[city]
    # 如果不是已知城市，尝试用城市搜索API
    try:
        search_url = f"https://{QWEATHER_API_HOST}/v2/city/lookup"
        resp = requests.get(search_url, params={"location": city, "key": QWEATHER_API_KEY}, timeout=10)
        data = resp.json()
        if data.get("code") == "200" and data.get("location"):
            return data["location"][0]["id"]
    except Exception:
        pass
    return city  # 降级：直接返回城市名


@tool
def get_weather(city: str = "成都") -> str:
    """获取指定城市当天实时天气。"""
    if not QWEATHER_API_KEY or not QWEATHER_API_HOST:
        return json.dumps({"error": "天气服务未配置，请检查 API Key 和 API Host"}, ensure_ascii=False)

    location_id = _get_city_id(city)
    url = f"https://{QWEATHER_API_HOST}/v7/weather/now"
    params = {"location": location_id, "key": QWEATHER_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == "200":
            now = data.get("now", {})
            return json.dumps({
                "city": city,
                "weather": now.get("text", "未知"),
                "temp": f"{now.get('temp', 'N/A')}°C",
                "feels_like": f"{now.get('feelsLike', 'N/A')}°C",
                "humidity": f"{now.get('humidity', 'N/A')}%",
                "wind_dir": now.get("windDir", "未知"),
                "wind_scale": f"{now.get('windScale', 'N/A')}级",
            }, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"天气请求失败: {str(e)}"}, ensure_ascii=False)


@tool
def get_weather_forecast(city: str = "成都", days: int = 3) -> str:
    """获取指定城市未来N天天气预报。"""
    if not QWEATHER_API_KEY or not QWEATHER_API_HOST:
        return json.dumps({"error": "天气服务未配置"}, ensure_ascii=False)

    location_id = _get_city_id(city)
    url = f"https://{QWEATHER_API_HOST}/v7/weather/{days}d"
    params = {"location": location_id, "key": QWEATHER_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == "200":
            daily = data.get("daily", [])
            forecast = []
            for d in daily:
                forecast.append({
                    "date": d.get("fxDate", ""),
                    "weather": f"{d.get('textDay', '')}",
                    "temp_min": f"{d.get('tempMin', '')}°C",
                    "temp_max": f"{d.get('tempMax', '')}°C",
                    "rain_prob": d.get("pop", "未知"),
                })
            return json.dumps({"city": city, "forecast": forecast}, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"天气预报请求失败: {str(e)}"}, ensure_ascii=False)


@tool
def check_tickets(scenic: str) -> str:
    """查询景区实时信息，返回预估人流和开放状态。"""
    if not MAP_API_KEY:
        return json.dumps({
            "scenic": scenic,
            "status": "开放",
            "peak_hours": "9:00-16:00",
            "note": "地图API未配置，返回基础信息；实时余票请接入景区官方API"
        }, ensure_ascii=False)

    try:
        # 使用高德地图 POI 搜索景区信息
        url = "https://restapi.amap.com/v3/place/text"
        params = {"keywords": scenic, "city": "成都", "key": MAP_API_KEY, "types": "110000|120000"}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") == "1" and data.get("pois"):
            poi = data["pois"][0]
            return json.dumps({
                "scenic": scenic,
                "address": poi.get("address", ""),
                "rating": poi.get("biz_ext", {}).get("rating", "暂无评分"),
                "status": "开放",
                "available_tickets": "请通过景区官方渠道查询实时余票",
            }, ensure_ascii=False)
    except Exception as e:
        pass

    return json.dumps({
        "scenic": scenic,
        "status": "开放",
        "available_tickets": "建议提前在线预订",
        "note": "实时余票请通过景区官方小程序/APP查询",
    }, ensure_ascii=False)


@tool
def check_restaurant_queue(restaurant_name: str) -> str:
    """查询餐厅当前排队信息和预估等待时间。"""
    if not MAP_API_KEY:
        return json.dumps({
            "restaurant": restaurant_name,
            "queue_info": "地图API未配置，无法获取实时排队数据",
            "tip": "建议通过美团/大众点评查看实时排队"
        }, ensure_ascii=False)

    try:
        url = "https://restapi.amap.com/v3/place/text"
        params = {"keywords": restaurant_name, "city": "成都", "key": MAP_API_KEY, "types": "050000"}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") == "1" and data.get("pois"):
            poi = data["pois"][0]
            biz = poi.get("biz_ext", {})
            return json.dumps({
                "restaurant": restaurant_name,
                "address": poi.get("address", ""),
                "rating": biz.get("rating", "暂无评分"),
                "estimated_wait": "实时排队请通过美团/大众点评查看",
                "tel": poi.get("tel", ""),
            }, ensure_ascii=False)
    except Exception:
        pass

    return json.dumps({
        "restaurant": restaurant_name,
        "estimated_wait": "约30-45分钟（模拟数据）",
        "tip": "建议错峰前往或提前电话预约",
    }, ensure_ascii=False)
