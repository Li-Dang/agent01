from typing import TypedDict, List, Annotated
import json
import re
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from utils.llm_factory import get_llm
from utils.entity_extractor import extract_filters
from tools.rag_tools import semantic_search, filtered_search
from tools.realtime_tools import get_weather, get_weather_forecast, check_tickets, check_restaurant_queue
from tools.location_tools import detect_current_location, geocode_address, search_nearby_food, update_location


class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    user_profile: dict
    plan: str
    time_aware: bool
    intent: str


def _last_user_query(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _get_user_location(state: AgentState) -> dict:
    loc = state.get("user_profile", {}).get("location", {})
    if not loc:
        return {"city": "成都", "latitude": 30.67, "longitude": 104.06, "address": "成都市"}
    return loc


def _auto_locate(state: AgentState):
    profile = state.get("user_profile", {})
    try:
        result = detect_current_location.invoke({})
        loc_data = json.loads(result)
        if "error" not in loc_data:
            profile["location"] = {
                "city": loc_data.get("city", "成都"),
                "latitude": loc_data.get("latitude", 30.67),
                "longitude": loc_data.get("longitude", 104.06),
                "address": loc_data.get("city", "成都"),
            }
            return profile
    except Exception:
        pass
    if "location" not in profile:
        profile["location"] = {"city": "成都", "latitude": 30.67, "longitude": 104.06, "address": "成都市"}
    return profile


def _extract_location_name(query: str) -> str:
    addr = query
    for kw in ["我现在在", "我在的", "位置在", "定位到", "我在"]:
        if kw in addr:
            addr = addr.split(kw, 1)[-1].strip()
            break
    separators = ["，", "。", "？", "？", "!", "！", "附近", "周边", "离我",
                  "几公里", "3公里", "5公里", "1公里", "2公里",
                  "帮我", "给我", "推荐", "有什么", "哪里", "多远"]
    for sep in separators:
        idx = addr.find(sep)
        if idx > 0:
            addr = addr[:idx]
    return addr.rstrip("，。,.'\"!！？?；;：:、 ")


def _parse_radius(query: str) -> int:
    m = re.search(r"(\d+)\s*(公里|km|千米|米|m)", query)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        return val if unit in ("米", "m") else val * 1000
    return 3000


def _parse_food_type(query: str) -> str:
    for ft in ["火锅", "川菜", "小吃", "串串", "烧烤", "面馆", "烤鱼",
               "冒菜", "麻辣烫", "甜品", "奶茶", "咖啡", "日料", "西餐"]:
        if ft in query:
            return ft
    return "美食"


# ---------- 节点函数 ----------
def classify_intent(state: AgentState):
    query = _last_user_query(state)
    llm = get_llm(temperature=0)
    prompt = f"""判断用户意图，只回复以下类别之一（不要解释）：
- 位置查询（附近、周边、我在哪、我在XX、XX公里内、离我多远等）
- 实时查询（天气、排队、余票、路况等动态信息）
- 知识查询（景点介绍、餐厅推荐、美食攻略等）
- 行程规划（多天行程安排）
- 闲聊

用户消息：{query}
意图："""
    intent = llm.invoke(prompt).content.strip()

    loc_triggers = ["我在哪", "我在", "附近", "周边", "公里", "离我", "定位", "多远", "旁边", "就近"]
    if any(kw in query for kw in loc_triggers) or "位置查询" in intent:
        intent = "位置查询"

    time_triggers = ["天气", "下雨", "温度", "排队", "余票", "路况", "等位", "门票", "限流"]
    time_aware = any(kw in query for kw in time_triggers) or "实时查询" in intent

    return {"time_aware": time_aware, "intent": intent}


def handle_location(state: AgentState):
    query = _last_user_query(state)
    profile = state.get("user_profile", {})
    results = []

    if "location" not in profile or not profile["location"].get("latitude"):
        profile = _auto_locate(state)

    loc = profile.get("location", {})

    if any(kw in query for kw in ["我在", "我现在在"]):
        loc_name = _extract_location_name(query)
        if loc_name and len(loc_name) >= 2 and loc_name not in ("哪", "哪儿"):
            result = update_location.invoke({"address": loc_name})
            loc_data = json.loads(result)
            if "error" not in loc_data:
                profile["location"] = {
                    "address": loc_data.get("address", loc_name),
                    "latitude": loc_data.get("latitude", loc.get("latitude")),
                    "longitude": loc_data.get("longitude", loc.get("longitude")),
                }
                loc = profile["location"]
                results.append(f"已定位到：{loc_data.get('address', loc_name)}")
            else:
                results.append(f"未找到'{loc_name}'，使用当前定位")

    if any(kw in query for kw in ["我在哪", "我在哪儿", "当前位置", "定位"]) and not any(
        kw in query for kw in ["附近", "周边", "公里", "有什么"]
    ):
        results.append(
            f"你当前在 **{loc.get('address', loc.get('city', '成都'))}**"
            f"（{loc.get('latitude')}, {loc.get('longitude')}）"
        )

    if any(kw in query for kw in ["附近", "周边", "公里", "离我", "旁边", "就近"]):
        radius = _parse_radius(query)
        food_type = _parse_food_type(query)
        lat, lon = loc.get("latitude", 30.67), loc.get("longitude", 104.06)

        nearby = search_nearby_food.invoke({
            "keyword": food_type,
            "latitude": lat,
            "longitude": lon,
            "radius": radius,
        })

        addr = loc.get("address", loc.get("city", "当前位置"))
        results.append(
            f"以 **{addr}** 为中心，**{radius}米**内的**{food_type}**：\n{nearby}"
        )

    if not results:
        results.append(f"当前定位：{loc.get('address', '成都')}。试试说'附近3公里的火锅'来搜索美食。")

    return {
        "messages": [AIMessage(content="\n\n".join(results))],
        "user_profile": profile,
    }


def retrieve_knowledge(state: AgentState):
    query = _last_user_query(state)
    filters = extract_filters(query)

    if filters.get("location") or filters.get("type"):
        result = filtered_search.invoke({
            "query": query,
            "location": filters.get("location"),
            "type_filter": filters.get("type"),
        })
    else:
        result = semantic_search.invoke({"query": query})

    return {"messages": [AIMessage(content=f"[知识库检索结果]\n{result}")]}


def real_time_check(state: AgentState):
    query = _last_user_query(state)
    loc = _get_user_location(state)
    city = loc.get("city", "成都")
    results = []

    if any(kw in query for kw in ["天气", "下雨", "温度", "穿什么", "带伞", "防晒"]):
        weather = get_weather.invoke({"city": city})
        results.append(f"实时天气：{weather}")
        forecast = get_weather_forecast.invoke({"city": city, "days": 3})
        results.append(f"未来天气：{forecast}")

    if any(kw in query for kw in ["门票", "余票", "排队", "限流", "人多"]):
        ticket_info = check_tickets.invoke({"scenic": "大熊猫繁育研究基地"})
        results.append(f"景区信息：{ticket_info}")

    if any(kw in query for kw in ["等位", "要等多久", "人多不多"]):
        queue_info = check_restaurant_queue.invoke({"restaurant_name": "热门火锅店"})
        results.append(f"餐厅排队：{queue_info}")

    if not results:
        results.append("请更具体地描述你想查什么，比如'今天天气怎么样'")

    return {"messages": [AIMessage(content="\n\n".join(results))]}


def generate_final_answer(state: AgentState):
    query = _last_user_query(state)
    llm = get_llm(temperature=0.7)
    loc = _get_user_location(state)

    context_parts = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            context_parts.append(f"[用户] {msg.content}")
        elif isinstance(msg, AIMessage):
            content = msg.content
            # 截断过长的检索结果，避免上下文爆炸
            if len(content) > 2000:
                content = content[:2000] + "\n...(结果已截断)"
            context_parts.append(f"[参考数据] {content}")

    context = "\n\n".join(context_parts)
    loc_hint = f"用户位置：{loc.get('address', '成都')}。" if loc else ""

    prompt = f"""你是一位热情的旅游美食专家，像本地老朋友一样推荐。

{loc_hint}

参考数据（包含搜索结果/知识库/天气等）：
{context}

原始需求：{query}

请按以下规则回复：

1. **无论如何都要给出推荐**，不要因为没有具体细节就拒绝回答。用户可能对当地不熟，不知道怎么描述偏好。
2. 如果用户需求较为宽泛，提供 2-3 种不同风格/价位/场景的方案，让用户有得选。例如重口味 vs 清淡、网红打卡 vs 地道老店、高端 vs 性价比。
3. 用 Markdown 表格整理关键信息（店名/景点名、特色、人均、距离等）。
4. 如果用户没指定，默认推荐人气高、口碑好的选择，并在推荐理由中说清楚"为什么推荐这个"。
5. 语气热情轻松，像朋友聊天，不要冷冰冰。
6. **最后**，如果确实需要用户补充偏好来精确定制，用一句轻松的话顺带问（不超过25字），但不要以这个问题作为回复主体——推荐才是主体。

请回复："""

    answer = llm.invoke(prompt).content.strip()
    return {"plan": answer, "messages": [AIMessage(content=answer)]}


# ---------- 构建图 ----------
builder = StateGraph(AgentState)

builder.add_node("classify", classify_intent)
builder.add_node("locate", handle_location)
builder.add_node("retrieve", retrieve_knowledge)
builder.add_node("realtime", real_time_check)
builder.add_node("generate", generate_final_answer)

builder.set_entry_point("classify")


def route_after_classify(state: AgentState):
    intent = state.get("intent", "")
    if intent == "位置查询":
        return "locate"
    if state.get("time_aware", False):
        return "realtime"
    return "retrieve"


builder.add_conditional_edges(
    "classify",
    route_after_classify,
    {"locate": "locate", "realtime": "realtime", "retrieve": "retrieve"},
)

# 所有节点都直接进入 generate（不再有澄清拦截）
builder.add_edge("locate", "generate")
builder.add_edge("retrieve", "generate")
builder.add_edge("realtime", "generate")
builder.add_edge("generate", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
