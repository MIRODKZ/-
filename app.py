"""
app.py — 全咨档案智能问答 Agent（Streamlit 前端）
架构：用户提问 → DashScope Embedding → ChromaDB 向量检索（Top-3）
         → 组装 Prompt → 通义千问 LLM → 流式输出回答 + 来源溯源
"""

import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
load_dotenv()

CHROMA_DB_PATH  = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "knowledge_base"
EMBED_MODEL     = "text-embedding-v3"
LLM_MODEL       = "qwen-max"
TOP_K           = 3

# ─────────────────────────────────────────────
# 页面基础设置（必须第一条 st 调用）
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="全咨档案智能问答 Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 全局 CSS 美化
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* 主标题 */
    .main-title {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #1e88e5, #00acc1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    /* 来源卡片 */
    .source-card {
        background: #f0f4ff;
        border-left: 4px solid #1e88e5;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: 0.85rem;
    }
    .source-card .filename { font-weight: 700; color: #1e88e5; }
    .source-card .meta     { color: #555; line-height: 1.7; }
    /* 回答区块 */
    .answer-box {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 18px 22px;
        line-height: 1.8;
    }
    /* 侧边栏标题 */
    section[data-testid="stSidebar"] h2 {
        color: #1e88e5;
    }
    /* 隐藏 Streamlit 默认页脚 */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 获取 API Key（优先 .env，其次侧边栏输入）
# ─────────────────────────────────────────────
def get_api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not key:
        key = st.session_state.get("api_key_input", "").strip()
    return key

# ─────────────────────────────────────────────
# 缓存资源初始化
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        col = client.get_collection(COLLECTION_NAME)
        return col
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_embed_model(api_key: str):
    return DashScopeEmbeddings(model=EMBED_MODEL, dashscope_api_key=api_key)

@st.cache_resource(show_spinner=False)
def load_llm(api_key: str):
    return ChatTongyi(
        model_name=LLM_MODEL,
        dashscope_api_key=api_key,
        streaming=True,
    )

# ─────────────────────────────────────────────
# 核心检索函数
# ─────────────────────────────────────────────
def retrieve(query: str, api_key: str) -> list[dict]:
    """向量相似度检索，返回 Top-K 结果（含文本 + 元数据 + 相似度）"""
    collection = load_collection()
    if collection is None:
        return []
    embed_model = load_embed_model(api_key)
    q_vec = embed_model.embed_query(query)
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    items = []
    for doc, meta, dist in zip(docs, metas, distances):
        similarity = max(0.0, 1.0 - dist)   # cosine distance → similarity
        items.append({"text": doc, "metadata": meta, "similarity": similarity})
    return items

# ─────────────────────────────────────────────
# Prompt 组装
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
你是一位专业的全过程工程咨询档案智能问答助手。
请严格依据下方提供的【参考文档】回答用户问题，不得捏造文档中不存在的内容。
若文档中确无相关信息，请直接说明"暂无相关档案记录"。
回答要求：条理清晰，语言专业，适当分点。
"""

def build_prompt(query: str, docs: list[dict]) -> list:
    context_parts = []
    for i, d in enumerate(docs, 1):
        m = d["metadata"]
        block = (
            f"【参考文档 {i}】\n"
            f"一级类别：{m.get('category_1','')}\n"
            f"二级分类：{m.get('category_2','')}\n"
            f"三级分类：{m.get('category_3','')}\n"
            f"文件名称：{m.get('source_file','')}\n"
            f"片段位置：第 {m.get('chunk_index',0)+1} / {m.get('chunk_total',1)} 段\n\n"
            f"{d['text']}"
        )
        context_parts.append(block)

    context_str = "\n\n" + "─" * 50 + "\n\n".join(context_parts)
    user_content = (
        f"参考文档如下：\n{context_str}\n\n"
        f"{'─' * 50}\n\n"
        f"用户问题：{query}\n\n"
        f"请基于上述文档作答，并在回答末尾用「📎 参考来源」标题列出你引用了哪些文档（文件名 + 片段序号）。"
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

# ─────────────────────────────────────────────
# 流式生成回答
# ─────────────────────────────────────────────
def stream_answer(query: str, docs: list[dict], api_key: str):
    """生成器：逐 token yield LLM 输出"""
    llm = load_llm(api_key)
    messages = build_prompt(query, docs)
    for chunk in llm.stream(messages):
        yield chunk.content

# ─────────────────────────────────────────────
# Session State 初始化
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{"role": "user"/"assistant", "content": str}]
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []      # 最新一次检索的来源列表
if "api_key_input" not in st.session_state:
    st.session_state.api_key_input = ""
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ─────────────────────────────────────────────
# 访问密码保护（黑客松演示用）
# ─────────────────────────────────────────────
# 从环境变量读取密码（本地 .env 或 Streamlit Cloud Secrets 配置）
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "hackathon2026")

if not st.session_state.authenticated:
    st.markdown('<p class="main-title">🔒 全咨档案智能问答 Agent</p>', unsafe_allow_html=True)
    st.caption("请输入访问密码以继续")
    pwd_input = st.text_input("访问密码", type="password", placeholder="请输入密码…")
    if st.button("🔓 验证", use_container_width=True):
        if pwd_input == ACCESS_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("密码错误，请重试")
    st.stop()

# ─────────────────────────────────────────────
# 侧边栏
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 全咨档案智能问答")
    st.caption("Powered by DashScope · ChromaDB · LangChain")
    st.divider()

    # API Key 输入（仅当 .env 中未配置时展示）
    if not os.getenv("DASHSCOPE_API_KEY", "").strip():
        st.markdown("### 🔑 API Key 设置")
        typed_key = st.text_input(
            "DashScope API Key",
            type="password",
            placeholder="sk-xxxxxxxxxxxx",
            key="api_key_input",
            help="前往 https://dashscope.aliyun.com 获取",
        )
        st.divider()

    # 向量库状态
    col = load_collection()
    if col:
        count = col.count()
        st.success(f"✅ 向量库已就绪\n\n共 **{count}** 条记录")
    else:
        st.error("❌ 向量库未初始化\n\n请先运行 `python ingest.py`")

    st.divider()

    # 来源文献展示区
    st.markdown("## 📎 参考文献来源")
    if st.session_state.last_sources:
        for idx, src in enumerate(st.session_state.last_sources, 1):
            meta = src["metadata"]
            sim  = src["similarity"]
            with st.expander(
                f"来源 {idx}｜{meta.get('source_file','未知文件')}",
                expanded=(idx == 1),
            ):
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="filename">📄 {meta.get('source_file','')}</div>
                        <div class="meta">
                            🏷 <b>一级类别：</b>{meta.get('category_1','')}<br>
                            🏷 <b>二级分类：</b>{meta.get('category_2','')}<br>
                            🏷 <b>三级分类：</b>{meta.get('category_3','—')}<br>
                            🔢 <b>片段：</b>第 {meta.get('chunk_index',0)+1} / {meta.get('chunk_total',1)} 段
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(sim, text=f"向量相似度：{sim:.1%}")
                st.markdown("**原文片段预览：**")
                preview = src["text"][:400]
                if len(src["text"]) > 400:
                    preview += "…"
                st.caption(preview)
    else:
        st.info("💡 提问后，本区域将自动展示\n检索到的参考文献及相似度")

    st.divider()
    # 清空对话按钮
    if st.button("🗑️ 清空对话历史", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

# ─────────────────────────────────────────────
# 主内容区：对话界面
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">📚 全咨档案智能问答 Agent</p>', unsafe_allow_html=True)
st.caption("基于全过程工程咨询项目档案库，通过 RAG 技术精准检索并智能作答")
st.divider()

# 展示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 欢迎提示（无历史时显示）
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "你好！我是**全咨档案智能问答 Agent** 👋\n\n"
            "我已接入项目档案知识库，可以回答关于工程管理、施工方案、质量安全等方面的问题。\n\n"
            "**示例问题：**\n"
            "- 临电施工组织设计中，变压器容量是多少？\n"
            "- 项目配电系统的主要设计要求是什么？\n"
            "- 请介绍本工程的安全用电组织措施。"
        )

# 输入框
user_query = st.chat_input("请输入您的问题，我将从档案库中检索并作答…")

if user_query:
    api_key = get_api_key()

    # ── 校验前置条件 ──
    if not api_key:
        st.error("请先在侧边栏输入 DashScope API Key，或在 `.env` 文件中配置 `DASHSCOPE_API_KEY`。")
        st.stop()

    if load_collection() is None:
        st.error("向量库未初始化，请先运行 `python ingest.py` 完成数据入库。")
        st.stop()

    # ── 展示用户消息 ──
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # ── 检索 + 流式回答 ──
    with st.chat_message("assistant"):
        with st.status("🔍 正在检索相关档案…", expanded=False) as status:
            retrieved_docs = retrieve(user_query, api_key)
            st.session_state.last_sources = retrieved_docs
            if retrieved_docs:
                status.update(
                    label=f"✅ 已检索到 {len(retrieved_docs)} 条相关片段",
                    state="complete",
                )
            else:
                status.update(label="⚠️ 未检索到相关文档", state="error")

        if not retrieved_docs:
            answer = "抱歉，未能在档案库中检索到与您问题相关的内容，请尝试换一种提问方式。"
            st.markdown(answer)
        else:
            # 流式输出
            answer_placeholder = st.empty()
            full_answer = ""
            for token in stream_answer(user_query, retrieved_docs, api_key):
                full_answer += token
                answer_placeholder.markdown(full_answer + "▌")
            answer_placeholder.markdown(full_answer)  # 最终版本（去掉光标）
            answer = full_answer

            # ── 来源徽章（内联展示） ──
            st.divider()
            badge_cols = st.columns(len(retrieved_docs))
            for i, (bcol, doc) in enumerate(zip(badge_cols, retrieved_docs), 1):
                meta = doc["metadata"]
                with bcol:
                    st.markdown(
                        f"""<div style="background:#e8f4fd;border-radius:8px;padding:8px 10px;
                                        font-size:0.78rem;line-height:1.6;border:1px solid #b3d9f5">
                            <b>📄 来源 {i}</b><br>
                            {meta.get('source_file','')}<br>
                            <span style="color:#555">{meta.get('category_1','')}</span><br>
                            <span style="color:#1e88e5">{meta.get('category_2','')}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    # 保存到历史
    st.session_state.messages.append({"role": "assistant", "content": answer})
    # 刷新侧边栏来源显示
    st.rerun()
