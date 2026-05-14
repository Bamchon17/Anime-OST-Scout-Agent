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

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "filter_meta.json")

# ───────────────────────────────────────────────
# TOOL DEFINITION
# ───────────────────────────────────────────────

TOOL_DEFINITION = {
    "name": "filter_search",
    "description": (
        "กรอง anime ตาม metadata ที่ระบุชัดเจน เช่น ปี, ประเภท, rating, แนว, studio "
        "ใช้เมื่อ user ระบุเงื่อนไขตรงๆ เช่น 'อนิเมะปี 2019' หรือ 'rating มากกว่า 8.5'"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "year": {"type": "integer", "description": "ปีที่ออกอากาศ เช่น 2020"},
            "year_from": {"type": "integer", "description": "ปีเริ่มต้น"},
            "year_to": {"type": "integer", "description": "ปีสิ้นสุด"},
            "season": {"type": "string", "description": "spring | summer | fall | winter"},
            "type": {"type": "string", "description": "TV Series | Movie | OVA"},
            "rating_min": {"type": "number", "description": "rating ขั้นต่ำ เช่น 8.0"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "แนวอนิเมะ เช่น ['action', 'fantasy']",
            },
            "studio": {"type": "string", "description": "ชื่อ studio เช่น 'Sunrise'"},
            "top_k": {"type": "integer", "default": 5},
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

        # ไฟล์ใหม่เป็น list ของ dict อยู่แล้ว โหลดเข้า DataFrame ได้เลย
        df_tmp = pd.DataFrame(records)
        
        # Mapping ชื่อฟิลด์จาก JSON ให้ตรงกับ Logic ในฟังก์ชัน run
        column_mapping = {
            "filter_year": "year",
            "max_rating": "rating",
            "filter_type": "type",
            "animation_work": "studio",
            "tags_str": "tags_raw"
        }
        df_tmp = df_tmp.rename(columns=column_mapping)

        # แปลงข้อมูลให้เป็นตัวเลขเพื่อความปลอดภัยในการเปรียบเทียบ
        df_tmp["year"] = pd.to_numeric(df_tmp["year"], errors='coerce')
        df_tmp["rating"] = pd.to_numeric(df_tmp["rating"], errors='coerce').fillna(0)
        
        _df = df_tmp
    return _df

# ───────────────────────────────────────────────
# HANDLER
# ───────────────────────────────────────────────
def run(
    year: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    season: str | None = None,
    type: str | None = None,
    rating_min: float | None = None,
    rating_max: float | None = None,
    tags: list | None = None,
    studio: str | None = None,
    top_k: int = 5,
) -> dict:
    df = _get_df().copy()
    applied_filters = []

    # Filter: Year
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
        df = df[df["season"].str.contains(season, case=False, na=False)]
        applied_filters.append(f"season_like='{season}'")

    # Filter: Type
    if type:
        df = df[df["type"].str.contains(type, case=False, na=False)]
        applied_filters.append(f"type_like='{type}'")

    # Filter: Rating
    if rating_min is not None:
        df = df[df["rating"] >= rating_min]
        applied_filters.append(f"rating>={rating_min}")

    if rating_max is not None:
        df = df[df["rating"] <= rating_max]
        applied_filters.append(f"rating<={rating_max}")

    # Filter: Tags (กรองจาก tags_raw ที่เป็น string คั่นด้วย |)
    if tags and isinstance(tags, list):
        for tag in tags:
            if tag:
                df = df[df["tags_raw"].str.contains(tag, case=False, na=False)]
        applied_filters.append(f"tags_include={tags}")

    # Filter: Studio
    if studio:
        df = df[df["studio"].str.contains(studio, case=False, na=False)]
        applied_filters.append(f"studio_like='{studio}'")

    # เรียงลำดับตามความนิยม
    df = df.sort_values("rating", ascending=False)
    final_df = df.head(top_k)

    results = []
    for i, (_, row) in enumerate(final_df.iterrows()):
        res = row.get("resources", {})
        if isinstance(res, str):
            try:
                res = json.loads(res)
            except Exception:
                res = {}
        results.append({
            "rank":       i + 1,
            "main_title": row["main_title"],
            "title_en":   row.get("title_en", ""),
            "rating":     row["rating"],
            "score":      row["rating"],
            "year":       int(row["year"]) if pd.notnull(row["year"]) else None,
            "season":     row.get("season", "N/A"),
            "type":       row["type"],
            "studio":     row["studio"],
            "tags":       row["tags_raw"].split(" | ")[:5] if "tags_raw" in row else [],
            "resources":  res,
            "synopsis":   row.get("synopsis_short", row.get("synopsis", "")),
        })

    return {
        "tool":            "filter_search",
        "data":            results,
        "applied_filters": applied_filters,
        "total_found":     len(df),
        "returned":        len(results),
    }

# ───────────────────────────────────────────────
# TEST BLOCK
# ───────────────────────────────────────────────
if __name__ == "__main__":
    # ทดสอบกรณีปี 2022 หรือข้อมูลอื่นๆ
    print("\n--- Test Filter: Rating >= 8.0 ---")
    res = run(rating_min=8.0)
    for r in res["results"]:
        print(f"{r['rank']}. {r['title']} ({r['year']}) ★{r['rating']}")
