"""
embedder.py
-----------
อ่าน data/prepared_anime.json แล้วแปลง text → vectors
ผ่าน SentenceTransformer (รันบนเครื่อง — ไม่ต้องเสีย API cost)

ติดตั้ง: pip install sentence-transformers

Output:
  data/embeddings_general.npy   ← vectors สำหรับ general index (50, 384)
  data/embeddings_music.npy     ← vectors สำหรับ music index   (50, 384)

ลำดับ index ตรงกับ prepared_anime.json ทุก position — อย่าสลับลำดับ
"""

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────

# ใช้ __file__ เพื่อ construct absolute path จาก script location
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))

INPUT_JSON     = os.path.join(SCRIPT_DIR, "data/prepared_anime.json")
OUTPUT_GENERAL = os.path.join(SCRIPT_DIR, "data/embeddings_general.npy")
OUTPUT_MUSIC   = os.path.join(SCRIPT_DIR, "data/embeddings_music.npy")

# all-MiniLM-L6-v2: เล็ก เร็ว ดี — 384 dims, ~80MB
# เปลี่ยนเป็น "all-mpnet-base-v2" ได้ถ้าต้องการแม่นขึ้น (768 dims, ~420MB)
MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 32   # ปรับตาม RAM — 32 เหมาะกับเครื่องทั่วไป


# ───────────────────────────────────────────────
# CORE
# ───────────────────────────────────────────────

def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    รับ list of strings → คืน numpy array shape (N, 384)
    normalize_embeddings=True → cosine similarity ใช้ dot product ได้เลย
    """
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,   # สำคัญ: ทำให้ cosine sim = dot product
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


# ───────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────

def embed(
    input_json:     str = None,
    output_general: str = None,
    output_music:   str = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    โหลด prepared_anime.json → embed 2 fields → บันทึก .npy
    คืน (general_vectors, music_vectors)
    """
    # ใช้ default path ถ้าไม่ระบุ
    if input_json is None:
        input_json = INPUT_JSON
    if output_general is None:
        output_general = OUTPUT_GENERAL
    if output_music is None:
        output_music = OUTPUT_MUSIC
    
    # ── โหลดข้อมูล ──
    print(f"[embedder] โหลด {input_json}")
    with open(input_json, encoding="utf-8") as f:
        records = json.load(f)
    print(f"[embedder] พบ {len(records)} records")

    general_texts = [r["general_text"] for r in records]
    music_texts   = [r["music_text"]   for r in records]

    # ── โหลด model (โหลดครั้งแรกจะ download ~80MB อัตโนมัติ) ──
    print(f"\n[embedder] โหลด model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"[embedder] embedding dim: {model.get_sentence_embedding_dimension()}")

    # ── Embed ──
    print(f"\n[embedder] embed general_text ({len(general_texts)} texts)...")
    general_vecs = embed_texts(general_texts, model)

    print(f"\n[embedder] embed music_text ({len(music_texts)} texts)...")
    music_vecs = embed_texts(music_texts, model)

    # ── บันทึก ──
    os.makedirs(os.path.dirname(output_general), exist_ok=True)
    np.save(output_general, general_vecs)
    np.save(output_music,   music_vecs)

    print(f"\n[embedder] บันทึกแล้ว:")
    print(f"  {output_general}  shape={general_vecs.shape}")
    print(f"  {output_music}    shape={music_vecs.shape}")
    _sanity_check(general_vecs, music_vecs)

    return general_vecs, music_vecs


# ───────────────────────────────────────────────
# HELPERS — ใช้ใน vector_store.py และ retrieval.py
# ───────────────────────────────────────────────

def load_embeddings(
    path_general: str = OUTPUT_GENERAL,
    path_music:   str = OUTPUT_MUSIC,
) -> tuple[np.ndarray, np.ndarray]:
    """
    โหลด .npy ที่ embed ไว้แล้ว — ไม่ต้องรัน model ซ้ำ
    """
    general_vecs = np.load(path_general)
    music_vecs   = np.load(path_music)
    print(f"[embedder] โหลด embeddings: general={general_vecs.shape}, music={music_vecs.shape}")
    return general_vecs, music_vecs


def embed_query(query: str, model: SentenceTransformer | None = None) -> np.ndarray:
    """
    แปลง query string เดียว → vector shape (384,)
    ใช้ตอน query-time ใน retrieval.py

    tip: ส่ง model เข้ามาถ้าโหลดไว้แล้ว เพื่อไม่ต้องโหลดซ้ำ
    """
    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector[0].astype(np.float32)   # shape (384,)


def _sanity_check(general: np.ndarray, music: np.ndarray) -> None:
    """ตรวจสอบ shape, NaN, และ norm"""
    dim = general.shape[1]
    assert general.shape == (len(general), dim)
    assert music.shape   == (len(music),   dim)
    assert not np.isnan(general).any(), "พบ NaN ใน general vectors!"
    assert not np.isnan(music).any(),   "พบ NaN ใน music vectors!"

    norms = np.linalg.norm(general, axis=1)
    print(f"\n[embedder] sanity check ผ่าน [OK]")
    print(f"  dim={dim}, norm: min={norms.min():.4f}, max={norms.max():.4f} (ควรใกล้ 1.0)")


# ───────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────

if __name__ == "__main__":
    embed()