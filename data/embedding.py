"""
embedding.py (Robust CPU/GPU Fallback)
════════════════════════════════════════════════════════════════
FAISS index builder with automatic GPU → CPU fallback.
Uses Jina-v3 multilingual embeddings.
"""

from __future__ import annotations

import json
import sys
import torch
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
KB_PATH = Path("data/knowledge_base.json")
FAISS_DIR = Path("data/faiss_index")

EMBED_MODEL = "jinaai/jina-embeddings-v3"
GPU_BATCH_SIZE = 32
CPU_BATCH_SIZE = 8


# ══════════════════════════════════════════════
# LOAD KB
# ══════════════════════════════════════════════

def load_knowledge_base(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[error] {path} not found!")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    print(f"[load]  {len(kb)} records loaded from {path}")
    return kb


# ══════════════════════════════════════════════
# DEVICE SELECTION
# ══════════════════════════════════════════════

def get_best_device() -> str:
    """
    Return:
      - 'cuda' if usable
      - otherwise 'cpu'
    """

    if not torch.cuda.is_available():
        print("[device] CUDA not available → using CPU")
        return "cpu"

    try:
        gpu_name = torch.cuda.get_device_name(0)

        print(f"[device] CUDA detected")
        print(f"[gpu] Device: {gpu_name}")

        # Tiny CUDA test
        x = torch.tensor([1.0]).cuda()
        y = x * 2

        print("[device] GPU test successful")
        return "cuda"

    except Exception as e:
        print(f"[warn] GPU unusable → fallback CPU")
        print(f"[warn] Reason: {e}")
        return "cpu"


# ══════════════════════════════════════════════
# BUILD INDEX
# ══════════════════════════════════════════════

def build_faiss_index(kb_records: list[dict], output_dir: Path) -> None:

    try:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer

    except ImportError:
        print(
            "[faiss] Missing dependencies.\n"
            "Install:\n"
            "pip install sentence-transformers faiss-cpu numpy torch"
        )
        return

    device = get_best_device()

    batch_size = GPU_BATCH_SIZE if device == "cuda" else CPU_BATCH_SIZE

    print(f"\n[embed] Loading model: {EMBED_MODEL}")
    print(f"[embed] Device: {device.upper()}")
    print(f"[embed] Batch size: {batch_size}")

    model = SentenceTransformer(
        EMBED_MODEL,
        trust_remote_code=True,
        device=device
    )

    texts = [r["embed_text"] for r in kb_records]
    ids = [r["main_title"] for r in kb_records]

    print(f"\n[embed] Encoding {len(texts)} documents...")

    try:

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
            task="text-matching"
        )

    except Exception as e:

        # GPU failed during actual inference
        if device == "cuda":

            print("\n[warn] GPU inference failed")
            print(f"[warn] Reason: {e}")

            print("\n[fallback] Switching to CPU...")

            device = "cpu"

            model = SentenceTransformer(
                EMBED_MODEL,
                trust_remote_code=True,
                device=device
            )

            embeddings = model.encode(
                texts,
                batch_size=CPU_BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
                task="text-matching"
            )

        else:
            raise e

    dim = embeddings.shape[1]

    print(f"\n[faiss] Embedding shape: {embeddings.shape}")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    output_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_dir / "index.faiss"))

    with open(output_dir / "id_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {str(i): chunk_id for i, chunk_id in enumerate(ids)},
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\n[faiss] Saved → {output_dir}/index.faiss")
    print(f"[faiss] Total vectors: {index.ntotal}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():

    print("═" * 60)
    print("  Anime OST Scout — Embedding Pipeline")
    print("═" * 60 + "\n")

    kb = load_knowledge_base(KB_PATH)

    if kb:
        build_faiss_index(kb, FAISS_DIR)

    print("\n" + "═" * 60)
    print(" DONE ")
    print("═" * 60)


if __name__ == "__main__":
    main()