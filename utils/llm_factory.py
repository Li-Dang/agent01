from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL
import sys


class _TokenPrinter(BaseCallbackHandler):
    """流式输出回调：逐 token 打印到终端。"""
    def __init__(self):
        super().__init__()
        self._started = False

    def on_llm_new_token(self, token: str, **kwargs):
        if not self._started:
            print("\nAgent：", end="", flush=True)
            self._started = True
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_llm_end(self, *args, **kwargs):
        print()


def get_llm(temperature=0, streaming=False):
    """返回统一配置的 LLM 实例。streaming=True 时逐 token 输出。"""
    kwargs = dict(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
    )
    if streaming:
        kwargs["streaming"] = True
        kwargs["callbacks"] = [_TokenPrinter()]
    return ChatOpenAI(**kwargs)
