"""
tools/filter_tool.py
--------------------
กรอง anime ตาม metadata จริง ไม่ใช้ vector search
เหมาะกับ query ที่ระบุชัดเจน เช่น:
  "อนิเมะ TV ปี 2020 rating มากกว่า 8"
  "anime แนว action ของ studio Mappa"
  "อนิเมะที่ออกช่วง spring"

ใช้ pandas filter ตรงๆ — เร็ว แม่น ไม่ต้องการ embedding
"""

import json

import pandas as pd

# ───────────────────────────────────────────────
# TOOL DEFINITION
# ───────────────────────────────────────────────

TOOL_DEFINITION = {
    "name": "filter_search",
    "description": (
        "กรอง anime ตาม metadata ที่ระบุชัดเจน เช่น ปี, ประเภท, rating, แนว, studio "
        "ใช้เมื่อ user ระบุเงื่อนไขตรงๆ เช่น 'อนิเมะปี 2019' หรือ 'rating มากกว่า 8.5' "
        "สามารถใช้หลายเงื่อนไขพร้อมกันได้"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "year": {
                "type": "integer",
                "description": "ปีที่ออกอากาศ เช่น 2020",
            },
            "year_from": {
                "type": "integer",
                "description": "ปีเริ่มต้น (ใช้คู่กับ year_to)",
            },
            "year_to": {
                "type": "integer",
                "description": "ปีสิ้นสุด (ใช้คู่กับ year_from)",
            },
            "season": {
                "type": "string",
                "description": "ฤดูกาล: spring | summer | fall | winter",
            },
            "type": {
                "type": "string",
                "description": "ประเภท: TV | Movie",
            },
            "rating_min": {
                "type": "number",
                "description": "rating ขั้นต่ำ เช่น 8.0",
            },
            "rating_max": {
                "type": "number",
                "description": "rating สูงสุด เช่น 9.0",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "แนวอนิเมะ เช่น ['action', 'fantasy'] — AND condition",
            },
            "studio": {
                "type": "string",
                "description": "ชื่อ studio เช่น 'Mappa' หรือ 'Studio Ghibli'",
            },
            "music_style": {
                "type": "string",
                "description": "สไตล์ดนตรี เช่น 'orchestral' หรือ 'jazz'",
            },
            "top_k": {
                "type": "integer",
                "description": "จำนวนผลลัพธ์สูงสุด (default 5)",
                "default": 5,
            },
        },
        "required": [],   # ไม่มี required — ใส่อะไรก็ได้อย่างน้อย 1 field
    },
}


# ───────────────────────────────────────────────
# DATA LOADER — โหลดครั้งเดียว
# ───────────────────────────────────────────────

_df: pd.DataFrame | None = None

def _get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        with open("data/prepared_anime.json", encoding="utf-8") as f:
            records = json.load(f)

        # แตก filter_meta ออกมาเป็น columns ตรงๆ
        rows = []
        for r in records:
            row = {
                "chunk_id":    r["chunk_id"],
                "title_en":    r["title_en"],
                "title":       r["title"],
                "rating":      r["rating"],
                "year":        r["year"],
                "type":        r["type"],
                "synopsis":    r["synopsis"],
                "image_url":   r.get("image_url", ""),
                "mal_url":     r.get("mal_url", ""),
                **r["filter_meta"],   # year, season, type, rating, tags, studio, music_style
            }
            rows.append(row)

        _df = pd.DataFrame(rows)
    return _df


# ───────────────────────────────────────────────
# HANDLER
# ───────────────────────────────────────────────

def run(
    year:        int    | None = None,
    year_from:   int    | None = None,
    year_to:     int    | None = None,
    season:      str    | None = None,
    type:        str    | None = None,
    rating_min:  float  | None = None,
    rating_max:  float  | None = None,
    tags:        list   | None = None,
    studio:      str    | None = None,
    music_style: str    | None = None,
    top_k:       int    = 5,
) -> dict:
    """
    กรอง anime ตาม conditions ที่ระบุ
    ทุก condition เป็น AND — ยิ่งใส่มาก ยิ่งกรองเยอะ

    Return format เหมือน semantic_tool เพื่อให้ agent ใช้ร่วมกันได้
    """
    df = _get_df().copy()
    applied_filters = []

    # ── Year ──
    if year is not None:
        df = df[df["year"] == year]
        applied_filters.append(f"year={year}")

    if year_from is not None:
        df = df[df["year"] >= year_from]
        applied_filters.append(f"year≥{year_from}")

    if year_to is not None:
        df = df[df["year"] <= year_to]
        applied_filters.append(f"year≤{year_to}")

    # ── Season ──
    if season is not None:
        df = df[df["season"].str.lower() == season.lower()]
        applied_filters.append(f"season={season}")

    # ── Type ──
    if type is not None:
        df = df[df["type"].str.upper() == type.upper()]
        applied_filters.append(f"type={type}")

    # ── Rating ──
    if rating_min is not None:
        df = df[df["rating"] >= rating_min]
        applied_filters.append(f"rating≥{rating_min}")

    if rating_max is not None:
        df = df[df["rating"] <= rating_max]
        applied_filters.append(f"rating≤{rating_max}")

    # ── Tags (AND — ต้องมีทุก tag ที่ระบุ) ──
    if tags:
        for tag in tags:
            df = df[df["tags"].apply(lambda t: tag.lower() in [x.lower() for x in t])]
        applied_filters.append(f"tags={tags}")

    # ── Studio (partial match) ──
    if studio is not None:
        df = df[df["studio"].str.contains(studio, case=False, na=False)]
        applied_filters.append(f"studio~'{studio}'")

    # ── Music style (partial match) ──
    if music_style is not None:
        df = df[df["music_style"].str.contains(music_style, case=False, na=False)]
        applied_filters.append(f"music_style~'{music_style}'")

    # ── เรียงตาม rating แล้ว top_k ──
    df = df.sort_values("rating", ascending=False).head(top_k)

    print(f"[filter_tool] filters={applied_filters}  พบ {len(df)} results")

    results = [
        {
            "rank":     i + 1,
            "chunk_id": row["chunk_id"],
            "title_en": row["title_en"],
            "title":    row["title"],
            "rating":   row["rating"],
            "year":     row["year"],
            "season":   row["season"],
            "type":     row["type"],
            "tags":     row["tags"],
            "studio":   row["studio"],
            "synopsis": row["synopsis"],
            "image_url": row["image_url"],
            "mal_url":  row["mal_url"],
        }
        for i, (_, row) in enumerate(df.iterrows())
    ]

    return {
        "tool":             "filter_search",
        "applied_filters":  applied_filters,
        "total_found":      len(results),
        "results":          results,
    }


# ───────────────────────────────────────────────
# ENTRY POINT — ทดสอบ
# ───────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        {"rating_min": 9.0},
        {"year_from": 2010, "year_to": 2020, "tags": ["action"]},
        {"type": "Movie", "rating_min": 8.5},
        {"studio": "Mappa"},
        {"season": "spring", "rating_min": 8.0},
    ]

    for kwargs in tests:
        print("\n" + "═" * 60)
        print(f"Filter: {kwargs}")
        output = run(**kwargs)
        print(f"พบ {output['total_found']} results")
        for r in output["results"]:
            print(f"  {r['rank']}. {r['title_en']}  ★{r['rating']}  ({r['year']} {r['season']})")