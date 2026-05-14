"""
data_prep.py
════════════════════════════════════════════════════════════════
Full data preparation pipeline for Anime OST Scout Agentic RAG.

Reads  : General_meta.json  (500 records)
Outputs:
  data/knowledge_base.json      → semantic_tool  (RAG, to embed)
  data/music_meta.json          → music_tool     (keyword/metadata search)
  data/filter_meta.json         → filter_tool    (pandas structured filter)
  data/compare_meta.json        → compare_tool   (title-match + side-by-side)
  data/faiss_index/             → FAISS vector index + id_map.json

Run:
  pip install sentence-transformers faiss-cpu numpy
  python data_prep.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
INPUT_PATH   = "data/General_meta1.json"
OUTPUT_DIR   = Path("data")
FAISS_DIR    = OUTPUT_DIR / "faiss_index"
EMBED_MODEL  = "BAAI/bge-m3"          # swap to any sentence-transformers model
EMBED_DIM    = 1024                    # BGE-M3 output dim (768 for smaller models)
BATCH_SIZE   = 32


# ══════════════════════════════════════════════
#  STEP 1 — LOAD & INSPECT
# ══════════════════════════════════════════════

def load_raw(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[load]  {len(data)} records loaded from {path}")
    return data


# ══════════════════════════════════════════════
#  STEP 2 — CLEAN HELPERS
# ══════════════════════════════════════════════

def _safe_str(val, fallback: str = "") -> str:
    """Coerce None / 'None' / empty → fallback string."""
    if val is None or val == "None" or val == "":
        return fallback
    return str(val).strip()


def _parse_json_field(val) -> dict | list | None:
    """Parse a JSON-encoded string field (Cast, Resources, etc.)."""
    if val is None or val == "None" or val == "":
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_tags(val) -> list[str]:
    """Split pipe-separated Tags / Char Tags into clean list."""
    if not val or val == "None":
        return []
    return [t.strip() for t in str(val).split("|") if t.strip()]


def _parse_year(val) -> int | None:
    """Extract 4-digit year from messy Year field."""
    if not val:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", str(val))
    return int(m.group()) if m else None


def _parse_rating(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _cast_list(cast_raw) -> list[str]:
    """Return ['VA name (role)', ...] from dict or raw string."""
    parsed = _parse_json_field(cast_raw)
    if isinstance(parsed, dict):
        return [f"{va} ({role})" for va, role in parsed.items()]
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return []


def _resources_dict(res_raw) -> dict:
    parsed = _parse_json_field(res_raw)
    if isinstance(parsed, dict):
        return parsed
    return {}


# ══════════════════════════════════════════════
#  STEP 3 — FIELD SELECTION PER TOOL
# ══════════════════════════════════════════════

def build_knowledge_base(raw: list[dict]) -> list[dict]:
    """
    semantic_tool knowledge base.

    Embedded text = Synopsis + Tags + Char Tags.
    Metadata kept alongside for display and citation.
    Drops: synopsis_length, Original Plan, Chief Direction,
           Chief Animation Direction, Animation Character Design,
           Logo Image (72% null), Poster Image (45% null).
    """
    records = []
    for r in raw:
        synopsis  = _safe_str(r.get("Synopsis"))
        tags      = _parse_tags(r.get("Tags"))
        char_tags = _parse_tags(r.get("Char Tags"))

        # Skip records with no embeddable content
        if not synopsis:
            continue

        # Build the text that will be embedded
        embed_text = (
            f"{synopsis}\n"
            f"Genre/Tags: {', '.join(tags[:20]) if tags else 'N/A'}.\n"
            f"Character archetypes: {', '.join(char_tags[:15]) if char_tags else 'N/A'}."
        )

        records.append({
            # ── identity ──
            "id":              _safe_str(r.get("Main Title")),
            "main_title":      _safe_str(r.get("Main Title")),
            "title_en":        _safe_str(r.get("Official Title (en)")),
            "title_ja":        _safe_str(r.get("Official Title (ja)")),
            # ── embed content ──
            "embed_text":      embed_text,
            "synopsis":        synopsis,
            "tags":            tags,
            "char_tags":       char_tags,
            # ── display metadata ──
            "music":           _safe_str(r.get("Music")),
            "direction":       _safe_str(r.get("Direction")),
            "animation_work":  _safe_str(r.get("Animation Work")),
            "max_rating":      _parse_rating(r.get("Max Rating")),
            "filter_year":     r.get("filter_year"),
            "filter_type":     _safe_str(r.get("filter_type")),
            "season":          _safe_str(r.get("Season")),
            "resources":       _resources_dict(r.get("Resources")),
            "backdrop_image":  _safe_str(r.get("Backdrop Image Link Path")),
            "poster_image":    _safe_str(r.get("Poster Image Link Path")),
            "cast":            _cast_list(r.get("Cast"))[:5],  # top 5 VAs
        })

    print(f"[kb]    {len(records)} knowledge base records  "
          f"(dropped {len(raw) - len(records)} with no synopsis)")
    return records


def build_music_meta(raw: list[dict]) -> list[dict]:
    """
    music_tool — keyword search + metadata mapping.
    Only keeps records where Music field is non-null.
    """
    records = []
    for r in raw:
        music = _safe_str(r.get("Music"))
        if not music:
            continue

        # Build searchable music text for keyword matching
        tags  = _parse_tags(r.get("Tags"))
        # Extract mood-flavored tags only for music context
        music_mood_tags = [
            t for t in tags
            if any(kw in t.lower() for kw in [
                "orchestra", "choir", "rock", "electronic", "jazz", "folk",
                "piano", "intense", "calm", "dark", "epic", "sad", "happy",
                "atmospheric", "dramatic", "ambient", "acoustic", "vocal",
                "instrumental", "classical", "hip-hop", "pop", "metal"
            ])
        ]

        series_comp = _safe_str(r.get("Series Composition"))

        records.append({
            "id":               _safe_str(r.get("Main Title")),
            "main_title":       _safe_str(r.get("Main Title")),
            "title_en":         _safe_str(r.get("Official Title (en)")),
            # ── music-specific fields ──
            "composer":         music,                      # primary search key
            "series_composition": series_comp,
            "music_mood_tags":  music_mood_tags,
            # ── search text (for keyword/BM25 matching) ──
            "music_search_text": (
                f"Composer: {music}. "
                f"Series composition: {series_comp or 'N/A'}. "
                f"Music mood: {', '.join(music_mood_tags) or 'N/A'}."
            ),
            # ── context for response generation ──
            "synopsis_short":   _safe_str(r.get("Synopsis"))[:200],
            "tags":             _parse_tags(r.get("Tags"))[:10],
            "max_rating":       _parse_rating(r.get("Max Rating")),
            "filter_year":      r.get("filter_year"),
            "resources":        _resources_dict(r.get("Resources")),
        })

    print(f"[music] {len(records)} music_meta records  "
          f"(dropped {len(raw) - len(records)} with null Music)")
    return records


def build_filter_meta(raw: list[dict]) -> list[dict]:
    """
    filter_tool — pandas-ready structured records.
    All values are clean primitives (no nested dicts/lists).
    filter_year and filter_type are pre-normalized.
    """
    records = []
    for r in raw:
        records.append({
            "main_title":      _safe_str(r.get("Main Title")),
            "title_en":        _safe_str(r.get("Official Title (en)")),
            "filter_type":     _safe_str(r.get("filter_type")),    # "TV Series" | "Movie" | "OVA"
            "filter_year":     r.get("filter_year"),               # int, already clean
            "season":          _safe_str(r.get("Season")),
            "max_rating":      _parse_rating(r.get("Max Rating")),
            "animation_work":  _safe_str(r.get("Animation Work")),
            "direction":       _safe_str(r.get("Direction")),
            "music":           _safe_str(r.get("Music")),
            # Tags as flat string for pandas .str.contains()
            "tags_str":        " | ".join(_parse_tags(r.get("Tags"))[:30]),
            "resources":       json.dumps(_resources_dict(r.get("Resources"))),
            "backdrop_image":  _safe_str(r.get("Backdrop Image Link Path")),
        })

    print(f"[filt]  {len(records)} filter_meta records")
    return records


def build_compare_meta(raw: list[dict]) -> list[dict]:
    """
    compare_tool — rich structured records for side-by-side comparison.
    Keeps all meaningful fields; nulls are filled with empty string/list.
    """
    records = []
    for r in raw:
        records.append({
            "main_title":       _safe_str(r.get("Main Title")),
            "title_en":         _safe_str(r.get("Official Title (en)")),
            "title_ja":         _safe_str(r.get("Official Title (ja)")),
            "synopsis":         _safe_str(r.get("Synopsis"))[:400],
            "type":             _safe_str(r.get("filter_type")),
            "year":             r.get("filter_year"),
            "season":           _safe_str(r.get("Season")),
            "max_rating":       _parse_rating(r.get("Max Rating")),
            # ── creative staff ──
            "direction":        _safe_str(r.get("Direction")),
            "animation_work":   _safe_str(r.get("Animation Work")),
            "series_composition": _safe_str(r.get("Series Composition")),
            "music":            _safe_str(r.get("Music")),
            "character_design": _safe_str(r.get("Character Design")),
            "original_work":    _safe_str(r.get("Original Work")),
            # ── cast (top 4 VAs) ──
            "cast":             _cast_list(r.get("Cast"))[:4],
            # ── genre/theme ──
            "tags":             _parse_tags(r.get("Tags"))[:15],
            "char_tags":        _parse_tags(r.get("Char Tags"))[:10],
            # ── display ──
            "resources":        _resources_dict(r.get("Resources")),
            "backdrop_image":   _safe_str(r.get("Backdrop Image Link Path")),
            "poster_image":     _safe_str(r.get("Poster Image Link Path")),
        })

    print(f"[comp]  {len(records)} compare_meta records")
    return records


# ══════════════════════════════════════════════
#  STEP 4 — SAVE JSON FILES
# ══════════════════════════════════════════════

def save_json(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[save]  {path}  ({len(records)} records, "
          f"{path.stat().st_size / 1024:.1f} KB)")



# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  Anime OST Scout — Data Preparation Pipeline")
    print("═" * 60 + "\n")

    # ── Load ──
    raw = load_raw(INPUT_PATH)

    # ── Build datasets ──
    kb      = build_knowledge_base(raw)
    music   = build_music_meta(raw)
    filters = build_filter_meta(raw)
    compare = build_compare_meta(raw)

    # ── Save JSON ──
    print()
    save_json(kb,      OUTPUT_DIR / "knowledge_base.json")
    save_json(music,   OUTPUT_DIR / "music_meta.json")
    save_json(filters, OUTPUT_DIR / "filter_meta.json")
    save_json(compare, OUTPUT_DIR / "compare_meta.json")


    # ── Summary ──
    print("\n" + "═" * 60)
    print("  DONE — output files:")
    print(f"  data/knowledge_base.json  → semantic_tool  ({len(kb)} records)")
    print(f"  data/music_meta.json      → music_tool     ({len(music)} records)")
    print(f"  data/filter_meta.json     → filter_tool    ({len(filters)} records)")
    print(f"  data/compare_meta.json    → compare_tool   ({len(compare)} records)")
    print(f"  data/faiss_index/         → FAISS index + id_map")
    print(f"  retrieval.py              → two-stage retrieval helper")
    print("═" * 60)


if __name__ == "__main__":
    main()