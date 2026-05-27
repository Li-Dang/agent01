import json
from utils.llm_factory import get_llm

def extract_filters(user_query: str) -> dict:
    """
    从用户自然语言中提取地点、类型、预算等约束条件。
    返回字典：{"location": "...", "type": "...", "ambiance": "...", "budget": "..."}
    """
    llm = get_llm(temperature=0)
    prompt = f"""从以下用户需求中提取结构化信息，输出纯 JSON 格式。
提取字段：location（地点/商圈）、type（景点/川菜/火锅等类型）、ambiance（氛围，如安静/网红）、budget（预算）。
如果某个字段没有提及，值设为 null。
用户需求：{user_query}
JSON:"""
    resp = llm.invoke(prompt).content
    try:
        # 清理可能存在的 markdown 代码块标记
        if "```" in resp:
            resp = resp.split("```")[1]
            if resp.startswith("json"):
                resp = resp[4:]
        return json.loads(resp)
    except:
        return {}