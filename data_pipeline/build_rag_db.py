import json
from datetime import datetime
from config import VECTOR_DB_PATH, EMBEDDING_MODEL  # 必须在 HuggingFace 导入之前
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def fetch_data():
    """获取成都旅游美食数据。实际可替换为API拉取。"""
    return [
        # ---------- 景点 ----------
        {
            "name": "大熊猫繁育研究基地",
            "type": "景点",
            "location": "成华区",
            "history": "世界著名的大熊猫保护研究机构，可近距离观察大熊猫。建议早上去，熊猫较活跃。",
            "features": "适合亲子游、户外园区、环境优美。门票55元，地铁3号线熊猫大道站。"
        },
        {
            "name": "宽窄巷子",
            "type": "景点",
            "location": "青羊区",
            "history": "由宽巷子、窄巷子、井巷子三条清代古街组成，是成都保存最完好的历史街区。",
            "features": "免费开放、适合拍照、文创店铺多、有茶馆和川剧变脸表演。"
        },
        {
            "name": "锦里古街",
            "type": "景点",
            "location": "武侯区",
            "history": "紧邻武侯祠，三国文化主题商业街，夜晚灯笼亮起很有氛围。",
            "features": "免费、夜景美、小吃集中、适合买伴手礼。"
        },
        {
            "name": "都江堰",
            "type": "景点",
            "location": "都江堰市",
            "history": "世界文化遗产，两千多年历史的水利工程，李冰父子修建。",
            "features": "门票80元、离堆公园、安澜索桥、适合一日游。从成都犀浦站坐城际列车约30分钟。"
        },
        {
            "name": "青城山",
            "type": "景点",
            "location": "都江堰市",
            "history": "道教发源地之一，有'青城天下幽'美誉。分前山（道观为主）和后山（自然风光为主）。",
            "features": "前山门票80元、后山20元、适合徒步、空气清新。与都江堰可串联一日游。"
        },
        {
            "name": "武侯祠",
            "type": "景点",
            "location": "武侯区",
            "history": "纪念刘备和诸葛亮的祠庙，是中国唯一君臣合祀的寺庙。三国迷必去。",
            "features": "门票50元、红墙夹道拍照出片、紧邻锦里。"
        },
        {
            "name": "人民公园",
            "type": "景点",
            "location": "青羊区",
            "history": "成都最接地气的公园，鹤鸣茶社有百年历史。",
            "features": "免费、喝盖碗茶、掏耳朵、感受市井生活、相亲角有趣。"
        },
        {
            "name": "春熙路/太古里",
            "type": "景点",
            "location": "锦江区",
            "history": "成都核心商圈，IFS爬墙熊猫是地标，太古里融合现代设计与传统建筑。",
            "features": "适合购物、街拍、美女多、网红店集中。"
        },
        {
            "name": "九眼桥",
            "type": "景点",
            "location": "武侯区",
            "history": "成都夜生活代表，沿河酒吧一条街，安顺廊桥夜景经典。",
            "features": "适合晚上去、酒吧多、夜景好、年轻人聚集地。"
        },
        {
            "name": "金沙遗址博物馆",
            "type": "景点",
            "location": "青羊区",
            "history": "商周时期古蜀文明遗址，太阳神鸟金饰出土地。",
            "features": "门票70元、适合文博爱好者、室内展馆不受天气影响。"
        },
        # ---------- 美食 ----------
        {
            "name": "小龙坎老火锅（春熙路概念店）",
            "type": "火锅",
            "location": "春熙路",
            "history": "成都老牌火锅连锁，以麻辣鲜香著称，牛油锅底是招牌。",
            "features": "人气火爆建议错峰、人均120元、推荐毛肚鹅肠黄喉、排队较久。"
        },
        {
            "name": "大龙燚火锅（玉林店）",
            "type": "火锅",
            "location": "武侯区玉林",
            "history": "成都本地人偏爱的火锅品牌，锅底香而不燥。",
            "features": "人均100元、营业至凌晨、麻辣牛肉和冰粉必点。"
        },
        {
            "name": "冒椒火辣（奎星楼店）",
            "type": "串串/小吃",
            "location": "青羊区奎星楼街",
            "history": "网红串串店，奎星楼街是成都著名美食街，常年排队。",
            "features": "人均60元、冷锅串串、纠结千层肚和兔头是招牌。"
        },
        {
            "name": "陈麻婆豆腐（总府路店）",
            "type": "川菜",
            "location": "锦江区总府路",
            "history": "始创于1862年的百年老店，麻婆豆腐的发源地。",
            "features": "人均80元、麻婆豆腐必点、回锅肉和宫保鸡丁经典。"
        },
        {
            "name": "明婷小馆",
            "type": "川菜",
            "location": "成华区",
            "history": "成都最有名的苍蝇馆子之一，从曹家巷菜市场起家。",
            "features": "人均70元、脑花豆腐和荷叶酱肉是招牌、环境接地气。"
        },
        {
            "name": "洞子口张老二凉粉",
            "type": "小吃",
            "location": "青羊区文殊院",
            "history": "文殊院附近的几十年老店，成都凉粉的代表。",
            "features": "人均20元、甜水面和黄凉粉必吃、文殊院逛完直接来。"
        },
        {
            "name": "玉林路小酒馆",
            "type": "酒吧",
            "location": "武侯区玉林路",
            "history": "赵雷《成都》唱火的地标，文艺青年聚集地。",
            "features": "人均80元、live house风格、适合晚上小酌。"
        },
        {
            "name": "叶婆婆钵钵鸡（建设路店）",
            "type": "小吃",
            "location": "成华区建设路",
            "history": "乐山风味钵钵鸡，建设路是成都著名美食街。",
            "features": "人均40元、藤椒味和红油味两种、建设路小吃一条街可连吃。"
        },
        {
            "name": "蜀大侠火锅（太古里店）",
            "type": "火锅",
            "location": "锦江区太古里",
            "history": "以武侠主题出名的火锅店，摆盘精致适合拍照。",
            "features": "人均130元、一米牛肉和花千骨是招牌、环境好。"
        },
        {
            "name": "邓氏兔头（文殊院店）",
            "type": "小吃",
            "location": "青羊区文殊院",
            "history": "成都兔头老字号，麻辣兔头和五香兔头是招牌。",
            "features": "人均30元、可真空打包带走、文殊院周边。"
        },
    ]


def create_documents(raw_data):
    """将原始数据转为LangChain Document列表。"""
    docs = []
    for item in raw_data:
        content = f"名称：{item['name']}\n类型：{item['type']}\n位置：{item['location']}\n介绍：{item.get('history', '')}{item.get('features', '')}"
        metadata = {
            "source": "chengdu_guide_v1",
            "update_time": datetime.now().isoformat(),
            "type": item["type"],
            "location": item["location"],
            "name": item["name"],
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def build_vectorstore():
    """构建ChromaDB向量库。"""
    print("正在拉取数据...")
    raw_data = fetch_data()
    print(f"共获取 {len(raw_data)} 条数据。")

    print("正在创建文档...")
    docs = create_documents(raw_data)

    print("正在分割文档...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"分割为 {len(chunks)} 个文档块。")

    print("正在生成向量并存入ChromaDB...")
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = Chroma.from_documents(chunks, embedding, persist_directory=VECTOR_DB_PATH)
    print(f"向量库已构建完成，路径：{VECTOR_DB_PATH}，共 {len(chunks)} 个文档块。")

    # 验证检索
    test_result = vectordb.similarity_search("火锅", k=3)
    print(f"\n验证检索（关键词'火锅'，top3）：")
    for i, doc in enumerate(test_result):
        print(f"  {i+1}. {doc.metadata.get('name', 'N/A')} ({doc.metadata.get('location', 'N/A')})")


if __name__ == "__main__":
    build_vectorstore()
