import json
from langchain.tools import tool
from config import VECTOR_DB_PATH, EMBEDDING_MODEL  # 必须在 HuggingFace 导入之前，确保 HF_ENDPOINT 已设置
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

_vectordb = None


def _get_vectordb():
    """延迟初始化向量库，避免导入时因数据库不存在而崩溃。"""
    global _vectordb
    if _vectordb is None:
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectordb = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding)
    return _vectordb


@tool
def semantic_search(query: str) -> str:
    """用于模糊需求检索，如'适合拍照的地方'。返回相关景点或餐厅信息。"""
    try:
        docs = _get_vectordb().similarity_search(query, k=5)
        return json.dumps([{**d.metadata, "content": d.page_content} for d in docs], ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"知识库检索失败: {str(e)}", "hint": "请先运行 data_pipeline/build_rag_db.py 构建知识库"}, ensure_ascii=False)


@tool
def filtered_search(query: str, location: str = None, type_filter: str = None) -> str:
    """用于带明确约束的检索，如'春熙路附近的川菜馆'。支持按地点和类型过滤。"""
    filter_dict = {}
    if location:
        filter_dict["location"] = location
    if type_filter:
        filter_dict["type"] = type_filter
    try:
        docs = _get_vectordb().similarity_search(query, k=5, filter=filter_dict if filter_dict else None)
        return json.dumps([{**d.metadata, "content": d.page_content} for d in docs], ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"过滤检索失败: {str(e)}"}, ensure_ascii=False)
