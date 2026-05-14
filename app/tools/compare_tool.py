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
import os
from difflib import SequenceMatcher

import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "compare_meta.json")
# ───────────────────────────────────────────────
# TOOL DEFINITION
# ───────────────────────────────────────────────

TOOL_DEFINITION = {
    "name": "compare_anime",
    "description": (
        "เปรียบเทียบ anime 2-4 เรื่องแบบ side-by-side ทั้งด้านเนื้อเรื่อง ธีม เพลง และคะแนน"
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
        with open(DATA_PATH, encoding="utf-8") as f:
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
        # FIX: safe extraction กัน None
        candidates = [
            (r.get("main_title") or "").lower(),
            (r.get("title_en") or "").lower(),
            (r.get("title_ja") or "").lower()
        ]
        candidates = [c for c in candidates if c]

        for candidate in candidates:
            if not candidate:
                continue

            if query_lower == candidate:
                return r

            # Partial match
            if query_lower in candidate or candidate in query_lower:
                score = 0.90
            else:
                score = SequenceMatcher(None, query_lower, candidate).ratio()

            if score > best_score:
                best_score = score
                best_match = r

    return best_match if best_score >= 0.6 else None

# ───────────────────────────────────────────────
# COMPARISON BUILDER
# ───────────────────────────────────────────────
def _build_comparison(records: list[dict]) -> dict:
    """
    สร้าง structured comparison จาก records หลายตัว
    จัดกลุ่มข้อมูลเป็นมิติต่างๆ เพื่อให้ agent สรุปง่าย
    """

    titles = [
        (r.get("title_en") or r.get("main_title") or "Unknown")
        for r in records
    ]

    # ── 1. ข้อมูลพื้นฐาน ──
    basic = {
        title: {
            "year": r.get("year"),
            "season": r.get("season"),
            "type": r.get("type"),
            "rating": r.get("max_rating"),
            "studio": r.get("animation_work"),
        }
        for title, r in zip(titles, records)
    }

    # ── 2. ทีมงานเบื้องหลัง ──
    staff = {
        title: {
            "director": r.get("direction"),
            "music": r.get("music"),
            "series_comp": r.get("series_composition"),
            "original_work": r.get("original_work")
        }
        for title, r in zip(titles, records)
    }

    # ── 3. Themes & Tags ──
    all_tag_sets = [
        set(r.get("tags", []) or [])
        for r in records
    ]

    shared_tags = set.intersection(*all_tag_sets) if all_tag_sets else set()

    unique_tags = {
        title: list(set(r.get("tags", []) or []) - shared_tags)
        for title, r in zip(titles, records)
    }

    # ── 4. นักพากย์ ──
    cast = {
        title: (r.get("cast") or [])[:5]
        for title, r in zip(titles, records)
    }

    # ── 5. สรุปอันดับตาม Rating ──
    rating_rank = sorted(
        [
            {
                "title": (r.get("title_en") or r.get("main_title") or "Unknown"),
                "rating": r.get("max_rating", 0)
            }
            for r in records
        ],
        key=lambda x: x["rating"],
        reverse=True,
    )

    return {
        "titles": titles,
        "basic": basic,
        "staff": staff,
        "shared_tags": list(shared_tags),
        "unique_tags": unique_tags,
        "cast": cast,
        "rating_rank": rating_rank,
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
    print(f"[compare_tool] เปรียบเทียบ: {titles}")

    found_records = []
    not_found = []

    for title in titles:
        record = _find_anime(title)
        if record:
            if record not in found_records:
                found_records.append(record)
        else:
            not_found.append(title)

    if len(found_records) < 2:
        return {
            "tool": "compare_anime",
            "error": f"พบข้อมูลเพียง {len(found_records)} เรื่อง (ต้องการ 2 เรื่องขึ้นไป)",
            "found": [r.get("main_title") for r in found_records],
            "not_found": not_found
        }

    comparison = _build_comparison(found_records)

    return {
        "tool": "compare_anime",
        "requested": titles,
        "found": comparison["titles"],
        "not_found": not_found,
        "comparison": comparison,
        "records": found_records,
    }
# ───────────────────────────────────────────────
# ENTRY POINT — ทดสอบ
# ───────────────────────────────────────────────

if __name__ == "__main__":
    # รายการทดสอบ
    tests = [
        ["Seikai no Monshou", "Ordian"],
        ["Crest of the Stars", "Ginsoukikou Ordian"], # ทดสอบ Title EN/Alternate
        ["Spirited Away", "Naruto"] # ทดสอบกรณีไม่พบข้อมูล (ถ้าไม่มีใน 500 records)
    ]

    for titles in tests:
        print("\n" + "═" * 70)
        output = run(titles)

        if "error" in output:
            print(f"❌ Error: {output['error']}")
            if output.get("not_found"):
                print(f"   ไม่พบ: {output['not_found']}")
            continue

        comp = output["comparison"]
        print(f"📊 เปรียบเทียบ: {' vs '.join(comp['titles'])}")
        print("═" * 70)

        # 1. ข้อมูลพื้นฐาน
        print("\n[ Basic Info ]")
        for title, info in comp["basic"].items():
            print(f"  • {title:25} | {info['year']} | ★{info['rating'] or 'N/A':<4} | {info['studio']}")

        # 2. ทีมงาน (ข้อมูลใหม่จาก compare_meta.json)
        print("\n[ Production Staff ]")
        for title, staff in comp["staff"].items():
            print(f"  • {title}:")
            print(f"    - Director: {staff['director']}")
            print(f"    - Music:    {staff['music']}")
            print(f"    - Original: {staff['original_work']}")

        # 3. นักพากย์ (ข้อมูลใหม่)
        print("\n[ Top Cast ]")
        for title, cast_list in comp["cast"].items():
            cast_str = ", ".join(cast_list) if cast_list else "N/A"
            print(f"  • {title}: {cast_str}")

        # 4. แท็กและความเกี่ยวข้อง
        print(f"\n[ Shared Tags ]: {', '.join(comp['shared_tags']) if comp['shared_tags'] else '-'}")
        print("[ Unique Traits ]:")
        for title, tags in comp["unique_tags"].items():
            print(f"  • {title}: {', '.join(tags[:5])}...")

        # 5. สรุปอันดับ
        print("\n[ Rating Rank ]")
        for i, r in enumerate(comp["rating_rank"], 1):
            print(f"  {i}. {r['rating']} ★ -> {r['title']}")

        print("\n" + "─" * 70)
