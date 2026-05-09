"""
tools/compare_tool.py
---------------------
เปรียบเทียบ anime 2-4 ตัวแบบ side-by-side
เหมาะกับ query เช่น:
  "เปรียบเทียบ Naruto กับ Bleach"
  "FMA Brotherhood vs Attack on Titan ต่างกันยังไง"
  "ระหว่าง Ghibli movies ตัวไหนดีสุด"

ค้นหา anime แต่ละตัวด้วย title matching
แล้วจัด structured comparison ให้ agent นำไปสรุป
"""

import json
from difflib import SequenceMatcher

import pandas as pd

# ───────────────────────────────────────────────
# TOOL DEFINITION
# ───────────────────────────────────────────────

TOOL_DEFINITION = {
    "name": "compare_anime",
    "description": (
        "เปรียบเทียบ anime 2-4 ตัวแบบ side-by-side ในทุกมิติ "
        "ใช้เมื่อ user ต้องการรู้ความแตกต่างระหว่าง anime หลายเรื่อง "
        "เช่น 'Naruto vs Bleach' หรือ 'เปรียบเทียบ Ghibli movies'"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "titles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "รายชื่อ anime ที่ต้องการเปรียบเทียบ (2-4 เรื่อง)",
                "minItems": 2,
                "maxItems": 4,
            },
        },
        "required": ["titles"],
    },
}


# ───────────────────────────────────────────────
# DATA LOADER
# ───────────────────────────────────────────────

_records: list[dict] | None = None

def _get_records() -> list[dict]:
    global _records
    if _records is None:
        with open("data/prepared_anime.json", encoding="utf-8") as f:
            _records = json.load(f)
    return _records


# ───────────────────────────────────────────────
# TITLE MATCHING
# ───────────────────────────────────────────────

def _find_anime(query: str) -> dict | None:
    """
    ค้นหา anime จาก title ด้วย fuzzy matching
    รองรับทั้ง title_en และ title_ja
    คืน record ที่ใกล้เคียงที่สุด หรือ None ถ้าไม่เจอ
    """
    records = _get_records()
    query_lower = query.lower().strip()

    best_score = 0.0
    best_match = None

    for r in records:
        candidates = [
            r["title_en"].lower(),
            r["title"].lower(),
        ]
        # เพิ่ม title_ja ถ้ามี
        if r.get("title_ja"):
            candidates.append(r["title_ja"].lower())

        for candidate in candidates:
            # exact match → คืนเลย
            if query_lower == candidate:
                return r

            # partial match — query อยู่ใน title
            if query_lower in candidate or candidate in query_lower:
                score = 0.9
            else:
                score = SequenceMatcher(None, query_lower, candidate).ratio()

            if score > best_score:
                best_score = score
                best_match = r

    # threshold 0.5 — ต่ำกว่านี้ถือว่าไม่เจอ
    return best_match if best_score >= 0.5 else None


# ───────────────────────────────────────────────
# COMPARISON BUILDER
# ───────────────────────────────────────────────

def _build_comparison(records: list[dict]) -> dict:
    """
    สร้าง structured comparison จาก records หลายตัว
    จัดกลุ่มข้อมูลเป็นมิติต่างๆ เพื่อให้ agent สรุปง่าย
    """
    titles = [r["title_en"] for r in records]

    # ── Basic info ──
    basic = {
        title: {
            "year":   r["year"],
            "season": r["filter_meta"]["season"],
            "type":   r["type"],
            "rating": r["rating"],
            "studio": r["filter_meta"]["studio"],
        }
        for title, r in zip(titles, records)
    }

    # ── Themes & genres ──
    themes = {
        title: r["filter_meta"]["tags"]
        for title, r in zip(titles, records)
    }

    # Tags ที่มีร่วมกัน vs ต่างกัน
    all_tag_sets = [set(r["filter_meta"]["tags"]) for r in records]
    shared_tags  = set.intersection(*all_tag_sets) if all_tag_sets else set()
    unique_tags  = {
        title: list(all_tag_sets[i] - shared_tags)
        for i, title in enumerate(titles)
    }

    # ── Music ──
    music = {
        title: {
            "composer":    r["filter_meta"]["music_style"],
            "style":       r["filter_meta"]["music_style"],
            "mood":        r["filter_meta"].get("music_mood_tags", []),
        }
        for title, r in zip(titles, records)
    }

    # ── Synopsis ──
    synopsis = {
        title: r["synopsis"]
        for title, r in zip(titles, records)
    }

    # ── Ratings ranked ──
    rating_rank = sorted(
        [{"title": r["title_en"], "rating": r["rating"]} for r in records],
        key=lambda x: x["rating"],
        reverse=True,
    )

    return {
        "titles":       titles,
        "basic":        basic,
        "themes":       themes,
        "shared_tags":  list(shared_tags),
        "unique_tags":  unique_tags,
        "music":        music,
        "synopsis":     synopsis,
        "rating_rank":  rating_rank,
    }


# ───────────────────────────────────────────────
# HANDLER
# ───────────────────────────────────────────────

def run(titles: list[str]) -> dict:
    """
    รับ list of title strings → หา records → สร้าง comparison

    Return format:
    {
      "tool": "compare_anime",
      "requested": ["Naruto", "Bleach"],
      "found": ["Naruto", "Bleach"],
      "not_found": [],
      "comparison": { ... }   ← structured comparison
      "records": [ ... ]      ← full records สำหรับ agent
    }
    """
    print(f"[compare_tool] ค้นหา: {titles}")

    found_records = []
    not_found     = []

    for title in titles:
        record = _find_anime(title)
        if record:
            # กัน duplicate
            if record["chunk_id"] not in [r["chunk_id"] for r in found_records]:
                found_records.append(record)
            print(f"  ✓ '{title}' → {record['title_en']}")
        else:
            not_found.append(title)
            print(f"  ✗ '{title}' → ไม่พบ")

    if len(found_records) < 2:
        return {
            "tool":      "compare_anime",
            "requested": titles,
            "found":     [r["title_en"] for r in found_records],
            "not_found": not_found,
            "error":     f"พบ anime แค่ {len(found_records)} เรื่อง ต้องการอย่างน้อย 2 เรื่องเพื่อเปรียบเทียบ",
            "comparison": None,
        }

    comparison = _build_comparison(found_records)

    return {
        "tool":       "compare_anime",
        "requested":  titles,
        "found":      [r["title_en"] for r in found_records],
        "not_found":  not_found,
        "comparison": comparison,
    }


# ───────────────────────────────────────────────
# ENTRY POINT — ทดสอบ
# ───────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ["Naruto", "Bleach"],
        ["Attack on Titan", "Fullmetal Alchemist Brotherhood"],
        ["Spirited Away", "Princess Mononoke", "Howl's Moving Castle"],
    ]

    for titles in tests:
        print("\n" + "═" * 60)
        output = run(titles)

        if output.get("error"):
            print(f"Error: {output['error']}")
            continue

        comp = output["comparison"]
        print(f"เปรียบเทียบ: {' vs '.join(comp['titles'])}")
        print()

        print("Basic info:")
        for title, info in comp["basic"].items():
            print(f"  {title}: {info['year']} | ★{info['rating']} | {info['studio']}")

        print(f"\nShared tags: {comp['shared_tags']}")
        print("Unique tags:")
        for title, tags in comp["unique_tags"].items():
            print(f"  {title}: {tags}")

        print("\nRating rank:")
        for r in comp["rating_rank"]:
            print(f"  {r['rating']} → {r['title']}")