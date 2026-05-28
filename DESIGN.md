# pj04 成都旅游美食智能体 — 设计说明

> 供专家审核的完整架构与实现思路文档

---

## 一、项目定位

一个基于 LangGraph 的多节点智能体，专注于成都旅游美食推荐。用户可以自然语言对话，Agent 自动调用真实API（天气、地图、定位、周边搜索）结合本地知识库（RAG）生成个性化建议。

**核心理念**：静态知识（RAG）+ 动态数据（实时API）双引擎驱动，让推荐既有文化深度又有实时价值。

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────┐
│                       main.py                           │
│  启动定位 → 加载偏好 → 对话循环 → 流式输出 → 保存偏好      │
└────────────────────┬────────────────────────────────────┘
                     │ graph.stream(inputs)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  agent_graph.py                         │
│                   LangGraph 状态图                       │
│                                                         │
│   classify ──┬── retrieve (知识检索) ──┐                  │
│    (意图分类)  ├── realtime (实时API) ──┼── generate (生成)  │
│              └── locate   (位置服务) ──┘                  │
└──────┬──────────────┬──────────────────┬────────────────┘
       │              │                  │
       ▼              ▼                  ▼
┌────────────┐ ┌──────────────┐ ┌──────────────┐
│rag_tools   │ │realtime_tools│ │location_tools│
│ChromaDB    │ │和风天气 API  │ │高德地图 API  │
│语义检索     │ │景区/排队 API │ │IP定位/周边   │
└────────────┘ └──────────────┘ └──────────────┘
       │              │                  │
       ▼              ▼                  ▼
┌────────────┐ ┌──────────────┐ ┌──────────────┐
│ HuggingFace│ │ QWeather     │ │  Amap (高德) │
│ 本地中文    │ │ 实时天气      │ │  地图服务     │
│ embedding  │ │ 自定义Host    │ │              │
└────────────┘ └──────────────┘ └──────────────┘
```

**技术栈**：
- LLM：DeepSeek-v4-pro（通过 OpenAI 兼容接口）
- 编排框架：LangGraph（StateGraph + MemorySaver）
- 向量检索：ChromaDB + HuggingFace `text2vec-base-chinese`
- 实时数据：和风天气（天气）、高德地图（IP定位/地理编码/周边搜索）

---

## 三、核心设计思路

### 3.1 "静态RAG + 动态API" 双引擎架构

**问题**：传统问答Agent要么只靠RAG（知识陈旧），要么只靠搜索API（缺乏深度），无法满足旅游场景的复合需求。

**方案**：意图分类后，按需调度：

| 意图类型 | 数据来源 | 示例 |
|---------|---------|------|
| 知识查询 | ChromaDB RAG（语义检索 + 元数据过滤） | "宽窄巷子有什么历史？" |
| 实时查询 | 和风天气API + 高德API | "今天天气怎么样？" |
| 位置查询 | 高德IP定位 → 周边POI搜索 | "附近3公里的火锅" |

意图分类由 LLM（temperature=0）一次调用完成，辅助关键词兜底：
- 位置关键词：附近、周边、离我、公里 → 强制路由到 locate 节点
- 时间关键词：天气、排队、余票 → 强制路由到 realtime 节点

### 3.2 多节点 LangGraph 编排

**图结构**（5节点，6条边）：

```
classify (意图分类)
   ├── "位置查询" → locate    → generate
   ├── "实时查询" → realtime  → generate
   └── 其他       → retrieve  → generate
```

**设计决策：取消澄清拦截节点**

原设计有 `should_clarify` → `ask_clarify` 阻塞链，会导致模糊查询（"推荐好吃的"）得不到任何推荐。修改为：
- 所有查询直接到 `generate`
- generate 的 system prompt 规定"无论如何都要给出推荐"
- 模糊时自动分 2-3 种风格方案，追问放到推荐末尾轻描淡写一句

**状态设计**：

```python
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]  # 对话历史（自动累加）
    user_profile: dict                        # 用户偏好 + 位置
    plan: str                                 # 最终回答
    time_aware: bool                          # 是否需要实时数据
    intent: str                               # 意图标签
```

关键点：`messages` 使用 `operator.add` reducer，节点返回 `{"messages": [AIMessage(...)]}` 自动追加而非覆盖。

### 3.3 用户无感的自动定位

**设计目标**："打开就能用"——用户说"附近3公里的火锅"，无需先设位置。

**实现**：
1. 启动时：`main.py` 调用 `detect_current_location()` → 高德IP定位（免费，精度到区/街道）
2. 运行时：`locate` 节点检测到用户无位置数据 → 调用 `_auto_locate()` 补位
3. 用户说"我在春熙路"：`_extract_location_name()` 智能提取地点名 → 高德地理编码 → 精确坐标
4. 后续搜索自动以该坐标为中心

**地址提取算法**：
```
输入："我在春熙路，附近3公里有什么好的火锅？"
步骤1：split("我在") → "春熙路，附近3公里..."
步骤2：查找第一个分隔词（"附近"/"周边"/"推荐"...）→ 截断
步骤3：去掉尾部标点
输出："春熙路"
```

### 3.4 高德IP定位的多级降级策略

```
高德IP API（免费，国内城市级精度，30km范围）
  ↓ 失败或无key
ip-api.com（免费国际IP API，城市级）
  ↓ 失败
默认成都（30.67, 104.06）
```

设计理由：国内用高德（返回adcode+rectangle，精度更好），海外降级到ip-api.com，最终兜底成都。后续部署到微信小程序后，换用`wx.getLocation()`即可获得GPS级精度（共用同一套`handle_location`逻辑）。

### 3.5 周边搜索的距离计算

高德 `place/around` API 天然支持 `radius` 参数（单位：米）。`_parse_radius()` 从用户查询中提取半径：

```python
"附近3公里的火锅" → radius=3000
"周边500米的川菜" → radius=500
"旁边2km的小吃" → radius=2000
# 无指定 → 默认3000米
```

### 3.6 流式输出设计

利用 LangChain `BaseCallbackHandler` 实现逐token打印：

```
LLM生成token → Callback.on_llm_new_token() → sys.stdout.write(token) → flush()
```

`main.py` 不重复打印 generate 节点的结果（流式回调已实时输出）。这样用户看到打字机效果，不会觉得"卡住了"。

### 3.7 RAG知识库设计

**数据**：20条成都景点+美食数据（手动整理），每条包含：
- name（名称）、type（景点/火锅/川菜/小吃等）、location（区/商圈）
- history（历史背景）、features（特色/实用信息）

**向量化**：`shibing624/text2vec-base-chinese`（中文语义优化，400MB，本地运行）

选择理由：DeepSeek API 不提供 `/v1/embeddings` 端点（实测返回404），必须本地模型。选择 `text2vec-base-chinese` 而非通用英文模型，因为知识库全是中文，语义匹配精度更高。

**检索**：双模式——
1. `semantic_search`：纯语义匹配（"适合拍照的地方" → 宽窄巷子、锦里）
2. `filtered_search`：元数据过滤（"春熙路的川菜" → location=春熙路 + type=川菜）

**延迟初始化**：ChromaDB 不在 import 时加载（会导致模块导入崩溃），而是在首次调用时初始化。模型配置 `local_files_only=True` 阻止联网验证。

### 3.8 天气API的城市名标准化

**问题**：高德IP返回"成都市"，CITY_ID_MAP的key是"成都"。匹配失败后中文名被直接作为 QWeather 的 location ID 传入 → API 返回400。

**解决**：
```python
clean = city.replace("市", "").replace("省", "")  # 去掉行政后缀
if clean in CITY_ID_MAP: return CITY_ID_MAP[clean]
# ...降级链：城市查询API → 默认成都ID(101270101)
```

降级返回有效ID而非中文名，保证始终可用。

---

## 四、文件职责

| 文件 | 职责 | 关键设计点 |
|------|------|-----------|
| `main.py` | 入口：定位→对话循环→流式输出 | 最顶部设`HF_HUB_OFFLINE`和`PYTHONWARNINGS`；只显示generate节点输出 |
| `config.py` | 所有配置集中管理 | os.getenv从.env读取，不硬编码key；自动设HF镜像 |
| `graph/agent_graph.py` | LangGraph图定义+5个节点 | classify路由→retrieve/realtime/locate→generate |
| `tools/rag_tools.py` | ChromaDB语义检索 | 延迟初始化；新版langchain_huggingface导入 |
| `tools/realtime_tools.py` | 和风天气+高德景区/排队 | 城市名标准化；多级降级 |
| `tools/location_tools.py` | 高德IP定位+地理编码+周边搜索 | 无需MAP_API_KEY也能工作（降级ip-api.com） |
| `utils/llm_factory.py` | LLM实例工厂+流式回调 | `_TokenPrinter` 实现逐token输出 |
| `utils/entity_extractor.py` | LLM提取结构化过滤条件 | JSON输出+markdown代码块清理 |
| `utils/formatters.py` | Markdown表格格式化 | 行程表/美食表/天气提醒 |
| `data_pipeline/build_rag_db.py` | 一次性构建ChromaDB | 20条种子数据；可替换为API定时拉取 |
| `memory/user_profile.py` | JSON文件持久化用户偏好 | 位置、口味、预算、交通方式 |

---

## 五、关键API依赖

| API | 提供商 | 用途 | 是否需要Key | 降级方案 |
|-----|--------|------|:----------:|---------|
| DeepSeek Chat | DeepSeek | 大模型推理 | 是 | — |
| QWeather v7/now | 和风天气（自定义Host） | 实时天气 | 是 | 默认成都天气 |
| QWeather v7/3d | 和风天气 | 3天预报 | 是 | 返回空 |
| Amap IP | 高德地图 | IP定位 | 是 | ip-api.com |
| Amap Geocode | 高德地图 | 地址→坐标 | 是 | 返回错误提示 |
| Amap Place Around | 高德地图 | 周边POI搜索 | 是 | 返回错误提示 |
| HuggingFace Embeddings | 本地模型 | 文本向量化 | 否 | — |

---

## 六、待扩展方向

1. **微信小程序部署**：替换IP定位为`wx.getLocation()` GPS定位，精度从区级提升到米级
2. **多Agent协作**：行程规划Agent + 美食推荐Agent + 实时信息Agent 通过 LangGraph 子图协作
3. **知识库自动化**：定时任务拉取高德POI + 大众点评数据，替换手动整理的20条种子数据
4. **多模态输出**：地图路线图、餐厅照片通过DALL-E/Stable Diffusion生成
5. **可观测性**：接入LangSmith追踪Agent决策链，RAGAS评估检索质量
