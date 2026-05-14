"""
vector_store.py
---------------
รับ embeddings จาก embedder.py → สร้าง FAISS index 2 ตัว
แล้ว save ลง disk เพื่อโหลดซ้ำได้โดยไม่ต้อง embed ใหม่

ติดตั้ง:
    pip install faiss-cpu

run: python -m app.rag.vector_store

Output:
    data/faiss_index/kb.faiss
    data/faiss_index/music.faiss

    data/faiss_index/kb_meta.json
    data/faiss_index/music_meta.json

"""

from pathlib import Path
import json

import faiss
import numpy as np

from .embedder import embed_query, load_embeddings

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────

# vector_store.py อยู่ที่:
# app/rag/vector_store.py

SCRIPT_DIR = Path(__file__).resolve().parent

# Anime-OST-Scout-Agent/
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# data/
DATA_DIR = PROJECT_ROOT / "data"

# data/faiss_index/
INDEX_DIR = DATA_DIR / "faiss_index"

# ─────────────────────────────
# INPUT FILES
# ─────────────────────────────

KB_JSON = DATA_DIR / "knowledge_base.json"
KB_VEC = DATA_DIR / "embeddings_kb.npy"

MUSIC_JSON = DATA_DIR / "music_meta.json"
MUSIC_VEC = DATA_DIR / "embeddings_music.npy"

# ─────────────────────────────
# OUTPUT FILES
# ─────────────────────────────

KB_FAISS = INDEX_DIR / "kb.faiss"
KB_META_OUT = INDEX_DIR / "kb_meta.json"

MUSIC_FAISS = INDEX_DIR / "music.faiss"
MUSIC_META_OUT = INDEX_DIR / "music_meta.json"

TOP_K_DEFAULT = 10

# ───────────────────────────────────────────────
# BUILD
# ───────────────────────────────────────────────

def build() -> None:
    """
    สร้าง FAISS index 2 ตัว:
      1. knowledge base
      2. music metadata
    """

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print(" Vector Store Builder")
    print("==============================")

    # ─────────────────────────────
    # KB INDEX
    # ─────────────────────────────

    print(f"\n[vector_store] สร้าง KB index...")

    kb_vecs = np.load(KB_VEC).astype(np.float32)

    with open(KB_JSON, encoding="utf-8") as f:
        kb_records = json.load(f)

    kb_index = _build_index(
        vectors=kb_vecs,
        dim=kb_vecs.shape[1]
    )

    faiss.write_index(kb_index, str(KB_FAISS))

    _save_meta(
        meta=kb_records,
        path=KB_META_OUT
    )

    print(f"  ✅ KB vectors: {kb_vecs.shape}")
    print(f"  ✅ saved: {KB_FAISS}")

    # ─────────────────────────────
    # MUSIC INDEX
    # ─────────────────────────────

    print(f"\n[vector_store] สร้าง Music index...")

    music_vecs = np.load(MUSIC_VEC).astype(np.float32)

    with open(MUSIC_JSON, encoding="utf-8") as f:
        music_records = json.load(f)

    music_index = _build_index(
        vectors=music_vecs,
        dim=music_vecs.shape[1]
    )

    faiss.write_index(music_index, str(MUSIC_FAISS))

    _save_meta(
        meta=music_records,
        path=MUSIC_META_OUT
    )

    print(f"  ✅ Music vectors: {music_vecs.shape}")
    print(f"  ✅ saved: {MUSIC_FAISS}")

    print("\n[vector_store] build เสร็จ [OK]")


# ───────────────────────────────────────────────
# INDEX HELPERS
# ───────────────────────────────────────────────

def _build_index(
    vectors: np.ndarray,
    dim: int
) -> faiss.IndexFlatIP:
    """
    IndexFlatIP
    = cosine similarity
    (เพราะ embeddings ถูก normalize แล้ว)
    """

    index = faiss.IndexFlatIP(dim)

    index.add(vectors)

    return index


def _save_meta(
    meta: list[dict],
    path: Path
) -> None:

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            meta,
            f,
            ensure_ascii=False,
            indent=2
        )

# ───────────────────────────────────────────────
# LOAD
# ───────────────────────────────────────────────

def load_indexes():
    """
    โหลด indexes + metadata จาก disk
    """

    _check_built()

    kb_index = faiss.read_index(str(KB_FAISS))
    music_index = faiss.read_index(str(MUSIC_FAISS))

    with open(KB_META_OUT, encoding="utf-8") as f:
        kb_meta = json.load(f)

    with open(MUSIC_META_OUT, encoding="utf-8") as f:
        music_meta = json.load(f)

    print(
        f"\n[vector_store] โหลด indexes [OK]"
        f"\n  KB: {kb_index.ntotal}"
        f"\n  Music: {music_index.ntotal}"
    )

    return (
        kb_index,
        music_index,
        kb_meta,
        music_meta
    )

# ───────────────────────────────────────────────
# SEARCH
# ───────────────────────────────────────────────

def search(
    query_vec: np.ndarray,
    index: faiss.IndexFlatIP,
    meta: list[dict],
    top_k: int = TOP_K_DEFAULT,
) -> list[dict]:

    q = query_vec.reshape(1, -1).astype(np.float32)

    scores, indices = index.search(q, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        if idx >= len(meta):
            continue

        result = meta[idx].copy()

        result["score"] = float(score)

        results.append(result)

    return results


# ───────────────────────────────────────────────
# SEARCH BY ID
# ───────────────────────────────────────────────

def search_by_id(
    chunk_id: str,
    index: faiss.IndexFlatIP,
    meta: list[dict],
    vectors: np.ndarray,
    top_k: int = 6,
) -> list[dict]:
    """
    หา item ที่คล้ายกับ chunk_id
    """

    positions = [
        i for i, m in enumerate(meta)
        if m.get("chunk_id") == chunk_id
    ]

    if not positions:
        raise ValueError(f"ไม่พบ chunk_id: {chunk_id}")

    source_vec = vectors[positions[0]]

    results = search(
        query_vec=source_vec,
        index=index,
        meta=meta,
        top_k=top_k + 1
    )

    return [
        r for r in results
        if r.get("chunk_id") != chunk_id
    ][:top_k]

# ───────────────────────────────────────────────
# CHECK
# ───────────────────────────────────────────────

def _check_built() -> None:

    required_files = [
        KB_FAISS,
        MUSIC_FAISS,
        KB_META_OUT,
        MUSIC_META_OUT,
    ]

    missing = [
        str(p)
        for p in required_files
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "\nยังไม่มี index:\n"
            + "\n".join(missing)
            + "\n\nให้รัน:\npython app/rag/vector_store.py"
        )

# ───────────────────────────────────────────────
# ENTRYPOINT
# ───────────────────────────────────────────────

if __name__ == "__main__":
    build()