import os
from dotenv import load_dotenv

load_dotenv()

# ---------- DeepSeek 大模型 ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"

# 统一对外变量（供 llm_factory 使用）
LLM_API_KEY = DEEPSEEK_API_KEY
LLM_MODEL = DEEPSEEK_MODEL
LLM_BASE_URL = DEEPSEEK_BASE_URL

# ---------- Embedding 模型 ----------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")

# HuggingFace 配置（模型已缓存，无需联网验证）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
if HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT

# ---------- 和风天气 API ----------
QWEATHER_API_KEY = os.getenv("QWEATHER_API_KEY")
QWEATHER_API_HOST = os.getenv("QWEATHER_API_HOST")

# ---------- 高德地图 API ----------
MAP_API_KEY = os.getenv("MAP_API_KEY", "191e60318e637d2350de9334cae8b186")

# ---------- 路径 ----------
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./rag_db")
USER_PROFILE_PATH = os.getenv("USER_PROFILE_PATH", "./user_profiles.json")
