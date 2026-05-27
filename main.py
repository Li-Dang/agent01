import sys
import io
import json
import uuid
from langchain_core.messages import HumanMessage
from graph.agent_graph import graph
from memory.user_profile import load_profile, save_profile
from tools.location_tools import detect_current_location

# 修复 Windows GBK 编码下 emoji 报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def run():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 加载用户偏好
    user_id = "default_user"
    profile = load_profile(user_id)

    # 启动时自动定位
    if "location" not in profile:
        print("正在定位...", end=" ", flush=True)
        try:
            loc_result = detect_current_location.invoke({})
            loc_data = json.loads(loc_result)
            if "error" not in loc_data:
                profile["location"] = {
                    "city": loc_data.get("city", "成都"),
                    "latitude": loc_data.get("latitude", 30.67),
                    "longitude": loc_data.get("longitude", 104.06),
                    "address": loc_data.get("city", "成都"),
                }
                print(f"已定位到 {loc_data.get('city', '成都')}")
            else:
                print("定位失败，默认成都")
        except Exception:
            print("定位失败，默认成都")

    print("=" * 50)
    print("  成都蓉城干饭指北 & 伴游专家")
    print("=" * 50)
    loc = profile.get("location", {})
    if loc:
        print(f"当前位置：{loc.get('address', loc.get('city', '成都'))}")
    print("可以问我：")
    print("  - 我现在在哪？（定位）")
    print("  - 我在春熙路（更新位置）")
    print("  - 附近3公里有什么火锅？（周边搜索）")
    print("  - 帮我规划一个三日游行程")
    print("  - 今天天气怎么样？")
    print("输入 'exit' 退出\n")

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("再见！")
            break

        inputs = {
            "messages": [HumanMessage(content=user_input)],
            "user_profile": profile,
        }

        try:
            for event in graph.stream(inputs, config):
                for node_name, node_output in event.items():
                    # 同步更新 profile
                    if "user_profile" in node_output:
                        profile.update(node_output["user_profile"])
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            if hasattr(msg, "content") and msg.content:
                                content = msg.content
                                # 跳过内部检索数据
                                if content.startswith("[知识库检索结果]") or content.startswith("[系统]"):
                                    continue
                                print(f"\nAgent：{content}")
        except Exception as e:
            print(f"\n[出错了] {e}")
            print("请重试或检查API配置。")

    # 退出时保存偏好
    save_profile(user_id, profile)


if __name__ == "__main__":
    run()
