"""
retrieval.py
------------
รับ query → embed → search FAISS → rerank ด้วย Ollama

ใช้งาน:
    from retrieval import retrieve
    results = retrieve(
        "อนิเมะแนว psychological ที่ทำให้คิดเยอะ",
        index_type="kb"
    )

Requirements:
    pip install openai sentence-transformers faiss-cpu

Ollama:
    ollama pull supachai/llama-3-typhoon-v1.5:8b-instruct

run:  python -m app.rag.retrieval
"""

import json
import re

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from .embedder import (
    MODEL_NAME,
    embed_query,
)

from .vector_store import (
    load_indexes,
    search,
)

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
FAISS_TOP_K = 10
RERANK_TOP_K = 3
OLLAMA_MODEL = "supachai/llama-3-typhoon-v1.5:8b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# ───────────────────────────────────────────────
# SINGLETONS
# ───────────────────────────────────────────────

_indexes = None
_st_model = None
_client = None

# ───────────────────────────────────────────────
# LOADERS
# ───────────────────────────────────────────────

def _get_indexes():
    """
    โหลด FAISS indexes ครั้งเดียว
    """
    global _indexes
    if _indexes is None:
        _indexes = load_indexes()
    return _indexes


def _get_st_model():
    """
    โหลด embedding model ครั้งเดียว
    """
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(
            MODEL_NAME,
            trust_remote_code=True
        )
    return _st_model


def _get_client():
    """
    OpenAI-compatible client → Ollama
    """
    global _client
    if _client is None:
        _client = OpenAI(
            api_key="ollama",
            base_url=OLLAMA_BASE_URL,
        )
    return _client

# ───────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────
def retrieve(
    query: str,
    index_type: str = "kb",   # "kb" | "music"
    top_k: int = RERANK_TOP_K,
    rerank: bool = True,
) -> list[dict]:
    """
    query → embed → FAISS → rerank

    Returns:
        list[dict]
    """
    kb_index, music_index, kb_meta, music_meta = _get_indexes()
    model = _get_st_model()

    # ─────────────────────────────
    # EMBED QUERY
    # ─────────────────────────────
    print(f"\n[retrieval] query = '{query}'")
    query_vec = embed_query(
        query=query,
        model=model
    )

    # ─────────────────────────────
    # SELECT INDEX
    # ─────────────────────────────
    if index_type == "music":
        candidates = search(
            query_vec=query_vec,
            index=music_index,
            meta=music_meta,
            top_k=FAISS_TOP_K,
        )

    else:
        candidates = search(
            query_vec=query_vec,
            index=kb_index,
            meta=kb_meta,
            top_k=FAISS_TOP_K,
        )
    print( f"[retrieval] FAISS candidates = {len(candidates)}")

    # ─────────────────────────────
    # RERANK
    # ─────────────────────────────
    if rerank and len(candidates) > top_k:
        results = _rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )
    else:
        results = candidates[:top_k]
    print( f"[retrieval] final results = {len(results)}")
    return results

# ───────────────────────────────────────────────
# RERANK
# ───────────────────────────────────────────────
def _rerank(
    query: str,
    candidates: list[dict],
    top_k: int,
) -> list[dict]:
    """
    ใช้ Ollama rerank semantic candidates
    """
    client = _get_client()

    # ─────────────────────────────
    # BUILD CANDIDATES TEXT
    # ─────────────────────────────
    formatted_candidates = []
    for i, c in enumerate(candidates):
        title = (
            c.get("title_en")
            or c.get("title")
            or c.get("anime_title")
            or "Unknown"
        )

        synopsis = c.get("synopsis", "")
        rating = c.get("rating", "N/A")
        tags = []
        filter_meta = c.get("filter_meta")

        if isinstance(filter_meta, dict):
            tags = filter_meta.get("tags", [])

        candidate_text = (
            f"[{i+1}] {title} "
            f"(rating: {rating}, score: {c['score']:.3f})\n"
            f"Tags: {', '.join(tags)}\n"
            f"Synopsis: {synopsis[:250]}"
        )

        formatted_candidates.append(candidate_text)

    candidates_text = "\n\n".join(formatted_candidates)

    # ─────────────────────────────
    # PROMPT
    # ─────────────────────────────

    prompt = f"""
You are a STRICT anime ranking system.

Your job is NOT to be creative.

Your job is ONLY to select the most relevant matches.

IMPORTANT RULES:

1. If relevance is weak → DO NOT include it.
2. Do NOT reinterpret genres loosely.
3. Psychological means:
   - mind games
   - mental conflict
   - philosophy
   - identity crisis
   NOT just "dark tone" or "serious story"

4. Music/OST query means:
   - strong emphasis on soundtrack reputation
   - orchestral composition MUST be explicit or well known
   NOT just "has music"

5. Attack on Titan-like means:
   - survival horror
   - military dystopia
   - existential threat
   NOT just "fantasy + fighting"

User query:
"{query}"

Candidates:
{candidates_text}

Return ONLY JSON array.

Rules:
- rank ONLY truly relevant items
- max {top_k} results
- if unsure → exclude instead of guessing

Each item:
{{
  "rank": int,
  "index": int,
  "reason": "short Thai explanation"
}}

Return JSON only.
"""

    # ─────────────────────────────
    # CALL OLLAMA
    # ─────────────────────────────
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        temperature=0,
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    raw = response.choices[0].message.content.strip()

    # ─────────────────────────────
    # PARSE JSON
    # ─────────────────────────────
    ranked = _parse_json(raw)

    # ─────────────────────────────
    # MAP BACK
    # ─────────────────────────────
    results = []

    for item in ranked[:top_k]:
        idx = item.get("index", 0) - 1
        if 0 <= idx < len(candidates):
            result = {
                **candidates[idx],
                "rerank_reason": item.get("reason", "")
            }
            results.append(result)

    # ─────────────────────────────
    # FALLBACK
    # ─────────────────────────────
    if not results:
        print("[retrieval] rerank failed → fallback FAISS"  )
        results = candidates[:top_k]
    return results

# ───────────────────────────────────────────────
# JSON PARSER
# ───────────────────────────────────────────────
def _parse_json(raw: str) -> list[dict]:
    """
    parse JSON จาก LLM output
    """
    # parse ตรง
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # strip ```json
    stripped = re.sub( r"```(?:json)?","", raw ).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # extract [ ... ]
    match = re.search(
        r"\[.*\]",
        raw,
        re.DOTALL
    )

    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    print(
        f"[retrieval] parse failed:\n"
        f"{raw[:300]}"
    )
    return []

# ───────────────────────────────────────────────
# WRAPPERS
# ───────────────────────────────────────────────

def retrieve_kb(
    query: str,
    top_k: int = RERANK_TOP_K,
) -> list[dict]:

    return retrieve(
        query=query,
        index_type="kb",
        top_k=top_k,
        rerank=True,
    )


def retrieve_music(
    query: str,
    top_k: int = RERANK_TOP_K,
) -> list[dict]:

    return retrieve(
        query=query,
        index_type="music",
        top_k=top_k,
        rerank=False,
    )


def retrieve_no_rerank(
    query: str,
    top_k: int = FAISS_TOP_K,
) -> list[dict]:

    return retrieve(
        query=query,
        index_type="kb",
        top_k=top_k,
        rerank=False,
    )

# ───────────────────────────────────────────────
# TEST
# ───────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "อนิเมะแนว psychological ที่ทำให้คิดเยอะ",
        "anime with beautiful orchestral soundtrack",
        "dark fantasy action แบบ attack on titan",
    ]

    for q in test_queries:
        print("\n" + "=" * 60)
        results = retrieve_kb(q)
        for i, r in enumerate(results, 1):
            title = (
                r.get("title_en")
                or r.get("title")
                or "Unknown"
            )
            print(
                f"\n{i}. {title}"
                f" (score={r['score']:.3f})"
            )
            if "rerank_reason" in r:
                print(
                    f"   → {r['rerank_reason']}"
                )