import json
import requests
from langchain.tools import tool
from config import MAP_API_KEY


@tool
def detect_current_location() -> str:
    """自动检测用户当前所在城市和大致坐标（基于IP）。无需用户输入。"""
    # 优先用高德 IP 定位（国内精度更高）
    if MAP_API_KEY:
        try:
            resp = requests.get(
                "https://restapi.amap.com/v3/ip",
                params={"key": MAP_API_KEY},
                timeout=5,
            )
            data = resp.json()
            if data.get("status") == "1":
                rect = data.get("rectangle", "")
                adcode = data.get("adcode", "")
                province = data.get("province", "")
                city = data.get("city", "")
                lon, lat = 104.06, 30.67
                if rect:
                    parts = rect.split(";")
                    if len(parts) >= 1:
                        coords = parts[0].split(",")
                        if len(coords) >= 2:
                            lon, lat = float(coords[0]), float(coords[1])
                return json.dumps({
                    "city": city or "成都",
                    "province": province or "四川",
                    "latitude": lat,
                    "longitude": lon,
                    "source": "高德IP定位",
                }, ensure_ascii=False)
        except Exception:
            pass

    # 降级：免费国际 IP API
    try:
        resp = requests.get("http://ip-api.com/json/", timeout=5)
        data = resp.json()
        if data.get("status") == "success":
            return json.dumps({
                "city": data.get("city", "未知"),
                "latitude": data.get("lat", 0),
                "longitude": data.get("lon", 0),
                "source": "IP定位",
            }, ensure_ascii=False)
    except Exception:
        pass

    return json.dumps({"city": "成都", "latitude": 30.67, "longitude": 104.06, "source": "默认位置"}, ensure_ascii=False)


@tool
def geocode_address(address: str, city: str = "成都") -> str:
    """将文本地址（如'春熙路'、'天府广场'）转为经纬度坐标。需要高德API Key。"""
    if not MAP_API_KEY:
        return json.dumps({"error": "地理编码需要高德API Key，请在.env中配置MAP_API_KEY"}, ensure_ascii=False)

    try:
        resp = requests.get(
            "https://restapi.amap.com/v3/geocode/geo",
            params={"key": MAP_API_KEY, "address": address, "city": city},
            timeout=5,
        )
        data = resp.json()
        if data.get("status") == "1" and data.get("geocodes"):
            geo = data["geocodes"][0]
            loc = geo.get("location", "0,0")
            lon, lat = loc.split(",")
            return json.dumps({
                "address": geo.get("formatted_address", address),
                "latitude": float(lat),
                "longitude": float(lon),
                "source": "高德地理编码",
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"地理编码失败: {str(e)}"}, ensure_ascii=False)

    return json.dumps({"error": f"无法解析地址: {address}"}, ensure_ascii=False)


@tool
def search_nearby_food(
    keyword: str = "美食",
    latitude: float = 30.67,
    longitude: float = 104.06,
    radius: int = 3000,
) -> str:
    """在指定坐标周边搜索美食/餐厅。radius单位为米，默认3000（3公里）。需要高德API Key。"""
    if not MAP_API_KEY:
        return json.dumps({"error": "周边搜索需要高德API Key，请在.env中配置MAP_API_KEY"}, ensure_ascii=False)

    try:
        resp = requests.get(
            "https://restapi.amap.com/v3/place/around",
            params={
                "key": MAP_API_KEY,
                "location": f"{longitude},{latitude}",
                "keywords": keyword,
                "radius": radius,
                "types": "050000|060000",
                "offset": 20,
                "page": 1,
                "extensions": "all",
            },
            timeout=10,
        )
        data = resp.json()
        if data.get("status") == "1" and data.get("pois"):
            pois = []
            for p in data["pois"][:15]:
                distance = int(p.get("distance", "0"))
                pois.append({
                    "name": p.get("name", ""),
                    "address": p.get("address", ""),
                    "type": p.get("type", ""),
                    "distance": f"{distance}米" if distance < 1000 else f"{distance / 1000:.1f}公里",
                    "rating": (p.get("biz_ext") or {}).get("rating", ""),
                    "cost": (p.get("biz_ext") or {}).get("cost", ""),
                    "tel": p.get("tel", ""),
                })
            return json.dumps({
                "center": f"({longitude}, {latitude})",
                "radius": f"{radius}米",
                "total": len(pois),
                "restaurants": pois,
            }, ensure_ascii=False)
        return json.dumps({"total": 0, "restaurants": [], "message": "该范围内未找到结果"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"周边搜索失败: {str(e)}"}, ensure_ascii=False)


@tool
def update_location(address: str) -> str:
    """用户告知自己在哪儿时，用此工具解析地址并更新定位。如'我在春熙路'→调用此工具。"""
    result = json.loads(geocode_address.invoke({"address": address}))
    if "error" in result:
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({
        "status": "已更新位置",
        "address": result.get("address", address),
        "latitude": result.get("latitude", 30.67),
        "longitude": result.get("longitude", 104.06),
    }, ensure_ascii=False)
