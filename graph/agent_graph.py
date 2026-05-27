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


# ---------- 状态定义 ----------
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    user_profile: dict
    plan: str
    need_clarify: bool
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
    """自动定位并写入profile。"""
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
    """从"我在春熙路，附近3公里的火锅"中提取纯地点名。"""
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
    """从query中提取搜索半径（米），默认3000。"""
    m = re.search(r"(\d+)\s*(公里|km|千米|米|m)", query)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        return val if unit in ("米", "m") else val * 1000
    return 3000


def _parse_food_type(query: str) -> str:
    """从query中提取想吃的类型。"""
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
    """一站式位置处理：自动定位 → 周边搜索 → 位置更新。"""
    query = _last_user_query(state)
    profile = state.get("user_profile", {})
    results = []

    # 确保有定位数据（没有就自动获取）
    if "location" not in profile or not profile["location"].get("latitude"):
        profile = _auto_locate(state)

    loc = profile.get("location", {})

    # 用户明确告知位置 → 更新为精确坐标
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
                loc = profile["location"]  # 更新loc引用
                results.append(f"已定位到：{loc_data.get('address', loc_name)}")
            else:
                results.append(f"未找到'{loc_name}'，使用当前定位")

    # 用户问"我在哪" → 直接返回定位信息
    if any(kw in query for kw in ["我在哪", "我在哪儿", "当前位置", "定位"]) and not any(
        kw in query for kw in ["附近", "周边", "公里", "有什么"]
    ):
        results.append(
            f"你当前在 **{loc.get('address', loc.get('city', '成都'))}**"
            f"（{loc.get('latitude')}, {loc.get('longitude')}）"
        )

    # 周边搜索：附近/周边/XX公里 → 直接用当前定位搜
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

    if any(kw in query for kw in ["排队", "等位", "人多不多", "要等多久"]):
        queue_info = check_restaurant_queue.invoke({"restaurant_name": "热门火锅店"})
        results.append(f"餐厅排队：{queue_info}")

    if not results:
        results.append("请更具体地描述你想查什么，比如'今天天气怎么样'")

    return {"messages": [AIMessage(content="\n\n".join(results))]}


def should_clarify(state: AgentState):
    query = _last_user_query(state)
    intent = state.get("intent", "")
    # 位置查询和实时查询不需要澄清
    if intent in ("位置查询", "实时查询"):
        return {"need_clarify": False}

    llm = get_llm(temperature=0)
    prompt = f"""用户需求："{query}"
如果用户需求中缺少关键信息（如口味、位置、预算、日期、人数等），回答"需要澄清"并说明缺少什么。
如果需求已经很具体，回答"直接回答"。

判断："""
    judge = llm.invoke(prompt).content.strip()
    return {"need_clarify": "需要澄清" in judge}


def clarify_question(state: AgentState):
    query = _last_user_query(state)
    llm = get_llm(temperature=0.7)
    prompt = f"""用户说："{query}"
请提出1-2个友好的追问，帮助用户细化需求（口味偏好、位置区域、预算范围、出行时间等），总共不超过50字。"""
    question = llm.invoke(prompt).content.strip()
    return {"messages": [AIMessage(content=f"🤔 {question}")]}


def generate_final_answer(state: AgentState):
    query = _last_user_query(state)
    llm = get_llm(temperature=0.3)
    loc = _get_user_location(state)

    context_parts = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            context_parts.append(f"[用户] {msg.content}")
        elif isinstance(msg, AIMessage):
            context_parts.append(f"[系统] {msg.content}")

    context = "\n\n".join(context_parts)
    loc_hint = f"用户当前在：{loc.get('address', '成都')}（{loc.get('latitude')}, {loc.get('longitude')}）。" if loc else ""

    prompt = f"""你是一位资深的旅游美食专家。根据以下参考信息，为用户提供专业建议。

{loc_hint}

要求：
1. 使用Markdown格式，关键信息用表格呈现
2. 如果涉及行程，注明时间段和预估花费
3. 如果有天气信息，加入出行提醒
4. 推荐时说明理由和距离，让用户理解为什么值得去
5. 语气热情友好，像本地朋友推荐

参考信息：
{context}

用户需求：{query}

请回答："""

    answer = llm.invoke(prompt).content.strip()
    return {"plan": answer, "messages": [AIMessage(content=answer)]}


# ---------- 构建图 ----------
builder = StateGraph(AgentState)

builder.add_node("classify", classify_intent)
builder.add_node("locate", handle_location)
builder.add_node("retrieve", retrieve_knowledge)
builder.add_node("realtime", real_time_check)
builder.add_node("clarify_check", should_clarify)
builder.add_node("ask_clarify", clarify_question)
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

builder.add_edge("locate", "clarify_check")
builder.add_edge("retrieve", "clarify_check")
builder.add_edge("realtime", "clarify_check")

builder.add_conditional_edges(
    "clarify_check",
    lambda s: "ask_clarify" if s.get("need_clarify", False) else "generate",
    {"ask_clarify": "ask_clarify", "generate": "generate"},
)

builder.add_edge("ask_clarify", END)
builder.add_edge("generate", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
