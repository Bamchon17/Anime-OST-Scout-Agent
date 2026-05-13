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
import os
import json
import pandas as pd

# คำนวณ path ให้ถูกต้อง
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
DATA_PATH = os.path.join(BASE_DIR, "rag", "data", "prepared_anime.json")

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
                "description": "ประเภท: TV | Movie | OVA",
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
        "required": [], 
    },
}

# ───────────────────────────────────────────────
# DATA LOADER — ปรับปรุงให้แปลง Type ข้อมูลให้ถูกต้อง
# ───────────────────────────────────────────────

_df: pd.DataFrame | None = None

def _get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"หาไฟล์ไม่เจอที่: {DATA_PATH}")
            
        with open(DATA_PATH, encoding="utf-8") as f:
            records = json.load(f)

        rows = []
        for r in records:
            # รวม data หลักกับ filter_meta เข้าด้วยกัน
            meta = r.get("filter_meta", {})
            row = {
                "chunk_id":    r.get("chunk_id"),
                "title_en":    r.get("title_en"),
                "title":       r.get("title"),
                "synopsis":    r.get("synopsis", ""),
                "image_url":   r.get("image_url", ""),
                "mal_url":     r.get("mal_url", ""),
                # ดึงจาก root หรือ meta ก็ได้เพื่อความเหนียวแน่น
                "rating":      r.get("rating", meta.get("rating", 0)),
                "year":        r.get("year", meta.get("year")),
                "type":        r.get("type", meta.get("type", "TV")),
                "season":      meta.get("season", "unknown"),
                "tags":        meta.get("tags", []),
                "studio":      meta.get("studio", "unknown"),
                "music_style": meta.get("music_style", "unknown"),
            }
            rows.append(row)

        df_tmp = pd.DataFrame(rows)
        
        # คลีนข้อมูล: แปลงเป็น Numeric และจัดการค่าว่าง
        df_tmp["year"] = pd.to_numeric(df_tmp["year"], errors='coerce')
        df_tmp["rating"] = pd.to_numeric(df_tmp["rating"], errors='coerce').fillna(0)
        
        _df = df_tmp
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
    กรอง anime ตามเงื่อนไขที่ได้รับ
    """
    df = _get_df().copy()
    applied_filters = []

    # Filter: Year (รองรับทั้งปีเดียวและช่วงปี)
    if year is not None:
        df = df[df["year"] == year]
        applied_filters.append(f"year=={year}")
    
    if year_from is not None:
        df = df[df["year"] >= year_from]
        applied_filters.append(f"year>={year_from}")

    if year_to is not None:
        df = df[df["year"] <= year_to]
        applied_filters.append(f"year<={year_to}")

    # Filter: Season
    if season:
        df = df[df["season"].str.lower() == season.lower()]
        applied_filters.append(f"season=={season}")

    # Filter: Type
    if type:
        df = df[df["type"].str.upper() == type.upper()]
        applied_filters.append(f"type=={type}")

    # Filter: Rating
    if rating_min is not None:
        df = df[df["rating"] >= rating_min]
        applied_filters.append(f"rating>={rating_min}")

    if rating_max is not None:
        df = df[df["rating"] <= rating_max]
        applied_filters.append(f"rating<={rating_max}")

    # Filter: Tags (ต้องมีครบทุก Tag ที่ส่งมา)
    if tags and isinstance(tags, list):
        for tag in tags:
            if tag: # กันค่าว่าง
                df = df[df["tags"].apply(lambda t: tag.lower() in [x.lower() for x in t] if isinstance(t, list) else False)]
        applied_filters.append(f"tags_include={tags}")

    # Filter: Studio (ค้นหาแบบคำบางส่วน)
    if studio:
        df = df[df["studio"].str.contains(studio, case=False, na=False)]
        applied_filters.append(f"studio_like='{studio}'")

    # Filter: Music Style
    if music_style:
        df = df[df["music_style"].str.contains(music_style, case=False, na=False)]
        applied_filters.append(f"music_style_like='{music_style}'")

    # เรียงลำดับตามความนิยม (Rating) และตัดเอาเฉพาะที่ต้องการ
    df = df.sort_values("rating", ascending=False)
    final_df = df.head(top_k)

    print(f"[filter_tool] applied_filters={applied_filters} | Found: {len(final_df)}/{len(df)}")

    results = [
        {
            "rank":      i + 1,
            "chunk_id":  row["chunk_id"],
            "title_en":  row["title_en"],
            "title":     row["title"],
            "rating":    row["rating"],
            "year":      int(row["year"]) if pd.notnull(row["year"]) else None,
            "season":    row["season"],
            "type":      row["type"],
            "tags":      row["tags"],
            "studio":    row["studio"],
            "synopsis":  row["synopsis"],
            "image_url": row["image_url"],
            "mal_url":   row["mal_url"],
        }
        for i, (_, row) in enumerate(final_df.iterrows())
    ]

    return {
        "tool":            "filter_search",
        "applied_filters": applied_filters,
        "total_found":      len(results),
        "results":          results,
    }

# ───────────────────────────────────────────────
# TEST BLOCK
# ───────────────────────────────────────────────

if __name__ == "__main__":
    # ทดสอบเคสปี 2021 ที่เคยพัง
    print("\n--- Test Case: Year 2021 ---")
    res = run(year=2021, rating_min=5.0)
    for r in res["results"]:
        print(f"{r['rank']}. {r['title_en']} ({r['year']}) - Star: {r['rating']}")