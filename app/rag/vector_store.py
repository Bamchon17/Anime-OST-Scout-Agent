"""
vector_store.py
---------------
รับ embeddings จาก embedder.py → สร้าง FAISS index 2 ตัว
แล้ว save ลง disk เพื่อโหลดซ้ำได้โดยไม่ต้อง embed ใหม่

ติดตั้ง: pip install faiss-cpu

Output:
  indexes/general.faiss + indexes/general_meta.json
  indexes/music.faiss   + indexes/music_meta.json

การใช้งาน:
  build()   → รันครั้งเดียวตอน setup
  search()  → เรียกใน retrieval.py ทุก query
"""

import json
import os

import faiss
import numpy as np

from .embedder import embed_query, load_embeddings
from sentence_transformers import SentenceTransformer

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────

# ใช้ __file__ เพื่อ construct absolute path จาก script location
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))

PREPARED_JSON   = os.path.join(SCRIPT_DIR, "data/prepared_anime.json")
INDEX_DIR       = os.path.join(SCRIPT_DIR, "indexes")

GENERAL_FAISS   = os.path.join(INDEX_DIR, "general.faiss")
GENERAL_META    = os.path.join(INDEX_DIR, "general_meta.json")
MUSIC_FAISS     = os.path.join(INDEX_DIR, "music.faiss")
MUSIC_META      = os.path.join(INDEX_DIR, "music_meta.json")

TOP_K_DEFAULT   = 10   # ดึงมา 10 ก่อน แล้วค่อย rerank เหลือ 3-5


# ───────────────────────────────────────────────
# BUILD
# ───────────────────────────────────────────────

def build(
    prepared_json: str = None,
    emb_general:   str = None,
    emb_music:     str = None,
) -> None:
    """
    สร้าง FAISS index 2 ตัวและ save ลง disk
    รันครั้งเดียวหลังจาก embedder.py เสร็จ
    """
    # ใช้ default path ถ้าไม่ระบุ
    if prepared_json is None:
        prepared_json = PREPARED_JSON
    if emb_general is None:
        emb_general = os.path.join(SCRIPT_DIR, "data/embeddings_general.npy")
    if emb_music is None:
        emb_music = os.path.join(SCRIPT_DIR, "data/embeddings_music.npy")
    
    # ── โหลด embeddings ──
    general_vecs, music_vecs = load_embeddings(emb_general, emb_music)
    dim = general_vecs.shape[1]   # 384

    # ── โหลด metadata ──
    print(f"[vector_store] โหลด metadata จาก {prepared_json}")
    with open(prepared_json, encoding="utf-8") as f:
        records = json.load(f)

    # metadata ที่เก็บใน index (เฉพาะ field ที่ต้องแสดงผล)
    display_fields = [
        "chunk_id", "id", "title_en", "title", "rating",
        "year", "type", "synopsis", "image_url", "mal_url",
        "filter_meta",
    ]
    meta = [{k: r[k] for k in display_fields} for r in records]

    os.makedirs(INDEX_DIR, exist_ok=True)

    # ── สร้าง general index ──
    print(f"\n[vector_store] สร้าง general index  dim={dim}, n={len(general_vecs)}")
    gen_index = _build_index(general_vecs, dim)
    faiss.write_index(gen_index, GENERAL_FAISS)
    _save_meta(meta, GENERAL_META)
    print(f"  บันทึก → {GENERAL_FAISS}")

    # ── สร้าง music index ──
    print(f"\n[vector_store] สร้าง music index    dim={dim}, n={len(music_vecs)}")
    mus_index = _build_index(music_vecs, dim)
    faiss.write_index(mus_index, MUSIC_FAISS)
    _save_meta(meta, MUSIC_META)   # metadata เหมือนกัน index ต่างกัน
    print(f"  บันทึก → {MUSIC_FAISS}")

    print(f"[vector_store] build เสร็จ [OK]  ({len(records)} records, dim={dim})")


def _build_index(vectors: np.ndarray, dim: int) -> faiss.IndexFlatIP:
    """
    IndexFlatIP = Inner Product (= cosine similarity เมื่อ vectors normalize แล้ว)
    เหมาะกับ 50 records — ไม่ต้องใช้ approximate index (IVF/HNSW)
    """
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)   # vectors ถูก normalize แล้วจาก embedder.py
    return index


def _save_meta(meta: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ───────────────────────────────────────────────
# LOAD
# ───────────────────────────────────────────────

def load_indexes() -> tuple[faiss.IndexFlatIP, faiss.IndexFlatIP, list[dict], list[dict]]:
    """
    โหลด indexes และ metadata จาก disk
    คืน (general_index, music_index, general_meta, music_meta)
    เรียกตอนเริ่ม agent เพื่อ keep in memory ตลอด session
    """
    _check_built()

    gen_index = faiss.read_index(GENERAL_FAISS)
    mus_index = faiss.read_index(MUSIC_FAISS)

    with open(GENERAL_META, encoding="utf-8") as f:
        gen_meta = json.load(f)
    with open(MUSIC_META, encoding="utf-8") as f:
        mus_meta = json.load(f)

    print(f"[vector_store] โหลด indexes [OK]  general={gen_index.ntotal}, music={mus_index.ntotal}")
    return gen_index, mus_index, gen_meta, mus_meta


# ───────────────────────────────────────────────
# SEARCH
# ───────────────────────────────────────────────

def search(
    query_vec:  np.ndarray,
    index:      faiss.IndexFlatIP,
    meta:       list[dict],
    top_k:      int = TOP_K_DEFAULT,
) -> list[dict]:
    """
    รับ query vector shape (384,) → คืน list of results พร้อม score

    ใช้ใน retrieval.py:
      results = search(embed_query("..."), gen_index, gen_meta, top_k=10)
    """
    # FAISS ต้องการ shape (1, dim)
    q = query_vec.reshape(1, -1).astype(np.float32)

    scores, indices = index.search(q, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:   # FAISS คืน -1 ถ้าหาไม่ครบ top_k
            continue
        results.append({
            **meta[idx],
            "score": float(score),   # cosine similarity 0.0–1.0
        })

    return results


def search_by_id(
    chunk_id:  str,
    index:     faiss.IndexFlatIP,
    meta:      list[dict],
    vectors:   np.ndarray,
    top_k:     int = 6,
) -> list[dict]:
    """
    หา anime ที่คล้ายกับ chunk_id ที่ระบุ
    ใช้ใน recommend_tool.py
    """
    # หา index position ของ chunk_id
    positions = [i for i, m in enumerate(meta) if m["chunk_id"] == chunk_id]
    if not positions:
        raise ValueError(f"ไม่พบ chunk_id: {chunk_id}")

    source_vec = vectors[positions[0]]
    results = search(source_vec, index, meta, top_k=top_k + 1)

    # กรองตัวเองออก
    return [r for r in results if r["chunk_id"] != chunk_id][:top_k]


# ───────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────

def _check_built() -> None:
    missing = [p for p in [GENERAL_FAISS, MUSIC_FAISS] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"ยังไม่มี index: {missing}\n"
            f"รัน: python vector_store.py  เพื่อ build ก่อน"
        )


# ───────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────

if __name__ == "__main__":
    build()