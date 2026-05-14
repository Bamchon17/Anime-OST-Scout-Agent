"""
embedder.py
-----------
อ่าน data/knowledge_base.json และ data/music_meta.json แล้วแปลง text → vectors
ผ่าน SentenceTransformer (Jina AI v2 - รองรับ 8192 context length)

Output:
  data/embeddings_kb.npy      ← vectors สำหรับ semantic index
  data/embeddings_music.npy   ← vectors สำหรับ music index
"""


from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────

# app/rag/embedder.py
SCRIPT_DIR = Path(__file__).resolve().parent

# Anime-OST-Scout-Agent/
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# data/
DATA_DIR = PROJECT_ROOT / "data"

# Input files
INPUT_KB = DATA_DIR / "knowledge_base.json"
INPUT_MUSIC = DATA_DIR / "music_meta.json"

# Output files
OUTPUT_KB = DATA_DIR / "embeddings_kb.npy"
OUTPUT_MUSIC = DATA_DIR / "embeddings_music.npy"

# Embedding model
MODEL_NAME = "jinaai/jina-embeddings-v2-base-en"

# Performance
BATCH_SIZE = 32

# ───────────────────────────────────────────────
# CORE
# ───────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    model: SentenceTransformer
) -> np.ndarray:
    """
    แปลง list[str] → embeddings
    """

    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return vectors.astype(np.float32)

# ───────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────

def embed(
    input_kb: Path = INPUT_KB,
    input_music: Path = INPUT_MUSIC,
    output_kb: Path = OUTPUT_KB,
    output_music: Path = OUTPUT_MUSIC,
) -> tuple[np.ndarray, np.ndarray]:

    print("\n==============================")
    print(" Anime OST Scout Embedder")
    print("==============================")

    print(f"\n[embedder] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[embedder] DATA_DIR     = {DATA_DIR}")

    # ─────────────────────────────
    # ตรวจไฟล์
    # ─────────────────────────────

    if not input_kb.exists():
        raise FileNotFoundError(
            f"\nไม่พบไฟล์ KB:\n{input_kb}"
        )

    if not input_music.exists():
        raise FileNotFoundError(
            f"\nไม่พบไฟล์ music_meta:\n{input_music}"
        )

    # ─────────────────────────────
    # โหลดโมเดล
    # ─────────────────────────────

    print(f"\n[embedder] โหลด model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True
    )

    dim = model.get_sentence_embedding_dimension()

    print(f"[embedder] embedding dim: {dim}")

    # ─────────────────────────────
    # LOAD KB
    # ─────────────────────────────

    print(f"\n[embedder] โหลด KB:")
    print(f"  {input_kb}")

    with open(input_kb, encoding="utf-8") as f:
        kb_records = json.load(f)

    # แก้ไข: เปลี่ยนจาก "text" เป็น "embed_text" ให้ตรงกับไฟล์ JSON
    kb_texts = [
        r["embed_text"]
        for r in kb_records
        if "embed_text" in r
    ]

    print(f"[embedder] KB texts = {len(kb_texts)}")


    # ─────────────────────────────
    # EMBED KB
    # ─────────────────────────────

    print(f"\n[embedder] embedding KB...")

    kb_vecs = embed_texts(kb_texts, model)

    # ─────────────────────────────
    # LOAD MUSIC META
    # ─────────────────────────────

    print(f"\n[embedder] โหลด music meta:")
    print(f"  {input_music}")

    with open(input_music, encoding="utf-8") as f:
        music_records = json.load(f)

    # แก้ไข: เช็คทั้ง 'music_search_text' หรือ 'composer' 
    music_texts = []
    for r in music_records:
        if "music_search_text" in r:
            music_texts.append(r["music_search_text"])
        elif "composer" in r: # สำรองไว้เผื่อกรณี music_search_text ไม่มี
            music_texts.append(r["composer"])

    print(f"[embedder] music texts = {len(music_texts)}")
    # ─────────────────────────────
    # EMBED MUSIC
    # ─────────────────────────────

    print(f"\n[embedder] embedding music meta...")

    music_vecs = embed_texts(music_texts, model)

    # ─────────────────────────────
    # SAVE
    # ─────────────────────────────

    output_kb.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(output_kb, kb_vecs)
    np.save(output_music, music_vecs)

    print(f"\n[embedder] บันทึกสำเร็จ")

    print(f"\nKB:")
    print(f"  {output_kb}")
    print(f"  shape = {kb_vecs.shape}")

    print(f"\nMusic:")
    print(f"  {output_music}")
    print(f"  shape = {music_vecs.shape}")

    # sanity check
    _sanity_check(kb_vecs, music_vecs)

    return kb_vecs, music_vecs

# ───────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────

def load_embeddings(
    path_kb: Path = OUTPUT_KB,
    path_music: Path = OUTPUT_MUSIC,
) -> tuple[np.ndarray, np.ndarray]:
    """
    โหลด embeddings จาก .npy
    """

    kb_vecs = np.load(path_kb)
    music_vecs = np.load(path_music)

    print(
        f"[embedder] โหลด embeddings:"
        f" kb={kb_vecs.shape},"
        f" music={music_vecs.shape}"
    )

    return kb_vecs, music_vecs


def embed_query(
    query: str,
    model: SentenceTransformer | None = None
) -> np.ndarray:
    """
    แปลง query → vector
    """

    if model is None:
        model = SentenceTransformer(
            MODEL_NAME,
            trust_remote_code=True
        )

    vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return vector[0].astype(np.float32)


def _sanity_check(
    kb: np.ndarray,
    music: np.ndarray
) -> None:
    """
    ตรวจ embedding
    """

    assert not np.isnan(kb).any(), \
        "พบ NaN ใน KB vectors!"

    assert not np.isnan(music).any(), \
        "พบ NaN ใน music vectors!"

    dim = kb.shape[1]

    norms = np.linalg.norm(kb, axis=1)

    print(f"\n[embedder] sanity check ผ่าน [OK]")

    print(
        f"  dim={dim}"
        f", norm min={norms.min():.4f}"
        f", norm max={norms.max():.4f}"
    )


# ───────────────────────────────────────────────
# ENTRYPOINT
# ───────────────────────────────────────────────

if __name__ == "__main__":
    embed()