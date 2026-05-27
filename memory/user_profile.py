import json
from config import USER_PROFILE_PATH

def load_profile(user_id: str) -> dict:
    """从文件加载用户偏好，如果不存在则返回默认值"""
    try:
        with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        return profiles.get(user_id, {"taste":"无偏好", "budget":"中等", "transport":"地铁"})
    except FileNotFoundError:
        return {"taste":"无偏好", "budget":"中等", "transport":"地铁"}

def save_profile(user_id: str, updates: dict):
    """保存或更新用户偏好"""
    try:
        with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
            profiles = json.load(f)
    except FileNotFoundError:
        profiles = {}
    if user_id not in profiles:
        profiles[user_id] = {}
    profiles[user_id].update(updates)
    with open(USER_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)