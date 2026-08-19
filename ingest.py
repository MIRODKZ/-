"""
ingest.py — 全咨档案知识库入库脚本
功能：读取 knowledge_base.jsonl，调用 DashScope Embedding API 向量化，
     连同元数据一起存入本地 ChromaDB 持久化向量库。
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import chromadb
from langchain_community.embeddings import DashScopeEmbeddings

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
load_dotenv()

JSONL_PATH     = Path(__file__).parent / "knowledge_base.jsonl"
CHROMA_DB_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "knowledge_base"
EMBED_MODEL     = "text-embedding-v3"
BATCH_SIZE      = 20        # DashScope 单次请求上限约 25，留余量
RETRY_LIMIT     = 3
RETRY_DELAY     = 5         # 秒


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    """逐行加载 JSONL 文件"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠ 第 {line_no} 行 JSON 解析失败，已跳过：{e}")
    return records


def safe_embed(model: DashScopeEmbeddings, texts: list[str]) -> list[list[float]]:
    """带重试的批量 Embedding 调用"""
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            return model.embed_documents(texts)
        except Exception as e:
            print(f"  ⚠ Embedding 请求失败（第 {attempt}/{RETRY_LIMIT} 次）：{e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
            else:
                raise


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def ingest():
    # 1. 检查 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        print("❌  未找到 DASHSCOPE_API_KEY，请在 .env 文件中设置后重试。")
        sys.exit(1)

    # 2. 加载数据
    print(f"📂  正在加载数据文件：{JSONL_PATH}")
    if not JSONL_PATH.exists():
        print(f"❌  文件不存在：{JSONL_PATH}")
        sys.exit(1)
    records = load_jsonl(JSONL_PATH)
    print(f"✅  共加载 {len(records)} 条记录")

    # 3. 初始化 ChromaDB
    print(f"💾  初始化 ChromaDB（路径：{CHROMA_DB_PATH}）")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # 检查是否已入库
    try:
        existing = client.get_collection(COLLECTION_NAME)
        count = existing.count()
        if count >= len(records):
            print(f"✅  向量库已存在且完整（{count} 条），无需重复入库。")
            print("    如需强制重建，请删除 chroma_db 目录后重新运行。")
            return
        elif count > 0:
            print(f"⚠  向量库已有 {count} 条，将继续补充剩余数据…")
            existing_ids = set(existing.get(include=[])["ids"])
        else:
            existing_ids = set()
    except Exception:
        existing_ids = set()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 4. 过滤已入库的记录
    new_records = [r for r in records if r["id"] not in existing_ids]
    if not new_records:
        print("✅  所有记录已入库，退出。")
        return
    print(f"📝  待入库：{len(new_records)} 条（跳过已有 {len(existing_ids)} 条）")

    # 5. 初始化 Embedding 模型
    print(f"🤖  初始化 DashScope Embedding 模型：{EMBED_MODEL}")
    embed_model = DashScopeEmbeddings(
        model=EMBED_MODEL,
        dashscope_api_key=api_key,
    )

    # 6. 分批向量化并写入
    total_batches = (len(new_records) + BATCH_SIZE - 1) // BATCH_SIZE
    success_count = 0

    for i in tqdm(range(0, len(new_records), BATCH_SIZE),
                  total=total_batches, desc="🔄 入库进度", unit="batch"):
        batch = new_records[i : i + BATCH_SIZE]

        texts     = [r["text"] for r in batch]
        ids       = [r["id"]   for r in batch]
        metadatas = [
            {
                "category_1":  str(r.get("category_1", "")),
                "category_2":  str(r.get("category_2", "")),
                "category_3":  str(r.get("category_3", "")),
                "source_file": str(r.get("source_file", "")),
                "chunk_index": int(r.get("chunk_index", 0)),
                "chunk_total": int(r.get("chunk_total", 0)),
            }
            for r in batch
        ]

        embeddings = safe_embed(embed_model, texts)

        # upsert 确保幂等：重跑不会重复插入
        collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        success_count += len(batch)

    print(f"\n🎉  入库完成！本次写入 {success_count} 条，向量库总计 {collection.count()} 条。")


if __name__ == "__main__":
    ingest()
