"""
retrieval.py
------------
รับ query → embed → search FAISS → rerank ด้วย Claude
คืน top results พร้อม score และ reasoning

ใช้งาน:
  from retrieval import retrieve
  results = retrieve("อนิเมะแนว psychological ที่ทำให้คิดเยอะ", index_type="general")
"""

import json

import numpy as np
from anthropic import Anthropic
from embedder import embed_query
from sentence_transformers import SentenceTransformer
from vector_store import load_indexes, search

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────

FAISS_TOP_K    = 10   # ดึงจาก FAISS ก่อน 10 อัน
RERANK_TOP_K   = 3    # หลัง rerank เหลือ 3 อัน

CLAUDE_MODEL   = "claude-sonnet-4-20250514"


# ───────────────────────────────────────────────
# SINGLETON — โหลดครั้งเดียวตอน import
# ───────────────────────────────────────────────

_indexes  = None   # (gen_index, mus_index, gen_meta, mus_meta)
_st_model = None   # SentenceTransformer


def _get_indexes():
    global _indexes
    if _indexes is None:
        _indexes = load_indexes()
    return _indexes


def _get_st_model():
    global _st_model
    if _st_model is None:
        from embedder import MODEL_NAME
        _st_model = SentenceTransformer(MODEL_NAME)
    return _st_model


# ───────────────────────────────────────────────
# MAIN ENTRY
# ───────────────────────────────────────────────

def retrieve(
    query:      str,
    index_type: str = "general",   # "general" | "music"
    top_k:      int = RERANK_TOP_K,
    rerank:     bool = True,
) -> list[dict]:
    """
    Pipeline หลัก: query → embed → FAISS search → (rerank) → results

    คืน list of dict แต่ละตัวมี:
      chunk_id, title_en, synopsis, rating, score, (rerank_reason ถ้า rerank=True)
    """
    gen_index, mus_index, gen_meta, mus_meta = _get_indexes()
    model = _get_st_model()

    # ── 1. Embed query ──
    print(f"[retrieval] query: '{query}'")
    query_vec = embed_query(query, model)

    # ── 2. FAISS search ──
    if index_type == "music":
        candidates = search(query_vec, mus_index, mus_meta, top_k=FAISS_TOP_K)
    else:
        candidates = search(query_vec, gen_index, gen_meta, top_k=FAISS_TOP_K)

    print(f"[retrieval] FAISS คืน {len(candidates)} candidates")

    # ── 3. Rerank ──
    if rerank and len(candidates) > top_k:
        results = _rerank(query, candidates, top_k)
    else:
        results = candidates[:top_k]

    print(f"[retrieval] คืน {len(results)} results")
    return results


# ───────────────────────────────────────────────
# RERANKING
# ───────────────────────────────────────────────

def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    ส่ง candidates ให้ Claude เรียงลำดับใหม่ตามความเกี่ยวข้องกับ query
    คืน top_k results พร้อม rerank_reason แต่ละตัว
    """
    client = Anthropic()

    # สร้าง numbered list สำหรับ Claude
    candidates_text = "\n\n".join([
        f"[{i+1}] {c['title_en']} (rating: {c['rating']}, score: {c['score']:.3f})\n"
        f"    Tags: {', '.join(c['filter_meta']['tags'])}\n"
        f"    Synopsis: {c['synopsis'][:200]}..."
        for i, c in enumerate(candidates)
    ])

    prompt = f"""You are an anime recommendation expert. Rerank the following anime by relevance to the user's query.

User query: "{query}"

Candidates:
{candidates_text}

Return ONLY a JSON array of objects, ranked best to worst (top {top_k} only).
Each object must have:
  "rank": int (1 = best)
  "index": int (1-based, from the list above)
  "reason": str (one sentence why this fits the query, in Thai)

Example format:
[
  {{"rank": 1, "index": 3, "reason": "ตรงกับ query มากที่สุดเพราะ..."}},
  {{"rank": 2, "index": 1, "reason": "..."}}
]

Return JSON only, no other text."""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    try:
        ranked = json.loads(raw)
    except json.JSONDecodeError:
        # ถ้า Claude ใส่ ```json ``` มาด้วย ให้ strip ออก
        import re
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        ranked = json.loads(match.group()) if match else []

    # map กลับเป็น result objects
    results = []
    for item in ranked[:top_k]:
        idx = item["index"] - 1   # แปลงจาก 1-based → 0-based
        if 0 <= idx < len(candidates):
            result = {**candidates[idx], "rerank_reason": item.get("reason", "")}
            results.append(result)

    # fallback ถ้า rerank ล้มเหลว
    if not results:
        print("[retrieval] rerank ล้มเหลว — ใช้ FAISS score แทน")
        results = candidates[:top_k]

    return results


# ───────────────────────────────────────────────
# CONVENIENCE WRAPPERS — ใช้ใน tools/
# ───────────────────────────────────────────────

def retrieve_general(query: str, top_k: int = RERANK_TOP_K) -> list[dict]:
    """Semantic search ทั่วไป — ใช้ใน semantic_tool.py"""
    return retrieve(query, index_type="general", top_k=top_k)


def retrieve_music(query: str, top_k: int = RERANK_TOP_K) -> list[dict]:
    """Music-focused search — ใช้ใน music_tool.py"""
    return retrieve(query, index_type="music", top_k=top_k, rerank=False)
    # music ไม่ rerank เพราะ query มักตรงไปตรงมา เช่น "jazz anime"


def retrieve_no_rerank(query: str, top_k: int = FAISS_TOP_K) -> list[dict]:
    """Raw FAISS results ไม่ rerank — ใช้ใน compare_tool.py และ filter_tool.py"""
    return retrieve(query, index_type="general", top_k=top_k, rerank=False)


# ───────────────────────────────────────────────
# ENTRY POINT — ทดสอบ
# ───────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "อนิเมะแนว psychological ที่ทำให้คิดเยอะ",
        "anime with beautiful orchestral soundtrack",
        "dark fantasy action แบบ attack on titan",
    ]

    for q in test_queries:
        print("\n" + "═" * 60)
        results = retrieve(q)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['title_en']}  (score={r['score']:.3f})")
            if "rerank_reason" in r:
                print(f"   → {r['rerank_reason']}")
        print()