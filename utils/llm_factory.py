from langchain_openai import ChatOpenAI
from config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL

def get_llm(temperature=0):
    """返回统一配置的 LLM 实例，兼容 OpenAI / DeepSeek 等接口"""
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
    )