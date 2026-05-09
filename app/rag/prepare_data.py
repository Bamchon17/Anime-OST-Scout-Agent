"""
prepare_data.py
---------------
โหลด anime_50_with_synopsis.csv แล้วสร้าง text fields 3 แบบ:
  - general_text  → สำหรับ Semantic Tool และ Recommend Tool
  - music_text    → สำหรับ Music Tool เท่านั้น
  - filter_meta   → dict พร้อมใช้สำหรับ Filter Tool (ไม่ต้อง embed)

Output: data/prepared_anime.json
"""

import json
import os
import pandas as pd

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────

INPUT_CSV   = "data/anime_50_with_synopsis.csv"
OUTPUT_JSON = "data/prepared_anime.json"


# ───────────────────────────────────────────────
# TEXT BUILDERS
# ───────────────────────────────────────────────

def build_general_text(row: pd.Series) -> str:
    """
    Rich text สำหรับ semantic search และ recommend
    ครอบคลุม title, synopsis, tags, characters, studio, year
    → ใช้สร้าง general FAISS index
    """
    tags      = row["tags"].replace("|", ", ")
    char_tags = row["char_tags"].replace("|", ", ")

    return (
        f"Title: {row['title_en']}\n"
        f"Year: {row['year']} ({row['season']} season)\n"
        f"Type: {row['type']}  Rating: {row['rating']}\n"
        f"Genres and themes: {tags}\n"
        f"Character types: {char_tags}\n"
        f"Studio: {row['studio']}\n"
        f"Synopsis: {row['synopsis']}"
    ).strip()


def build_music_text(row: pd.Series) -> str:
    """
    Text เน้นเฉพาะ music สำหรับ Music Tool
    → ใช้สร้าง music FAISS index แยกต่างหาก
    """
    instruments = row["music_instruments"].replace("|", ", ")
    mood        = row["music_mood_tags"].replace("|", ", ")

    return (
        f"Anime: {row['title_en']}\n"
        f"Composer: {row['music_composer']}\n"
        f"Music style: {row['music_style']}\n"
        f"Mood: {mood}\n"
        f"Instruments: {instruments}"
    ).strip()


def build_filter_meta(row: pd.Series) -> dict:
    """
    Structured metadata สำหรับ Filter Tool
    ใช้ pandas exact/range filter — ไม่ต้องผ่าน embedding
    """
    return {
        "year":         int(row["year"]),
        "season":       row["season"],
        "type":         row["type"],
        "rating":       float(row["rating"]),
        "tags":         row["tags"].split("|"),
        "char_tags":    row["char_tags"].split("|"),
        "studio":       row["studio"],
        "music_style":  row["music_style"],
    }


# ───────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────

def prepare(input_csv: str = INPUT_CSV, output_json: str = OUTPUT_JSON) -> list[dict]:
    """
    อ่าน CSV → สร้าง text fields → บันทึก JSON
    คืน list of records พร้อมใช้งาน
    """
    print(f"[prepare_data] อ่านไฟล์: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"[prepare_data] พบข้อมูล {len(df)} รายการ")

    records = []
    for _, row in df.iterrows():
        record = {
            # ─── IDs & display fields ───
            "chunk_id":    row["chunk_id"],
            "id":          int(row["id"]),
            "title":       row["title"],
            "title_en":    row["title_en"],
            "title_ja":    row["title_ja"],
            "rating":      float(row["rating"]),
            "year":        int(row["year"]),
            "type":        row["type"],
            "synopsis":    row["synopsis"],
            "image_url":   row["image_url"],
            "poster_url":  row["poster_url"],
            "mal_url":     row["mal_url"],

            # ─── Text fields for embedding ───
            "general_text": build_general_text(row),
            "music_text":   build_music_text(row),

            # ─── Structured fields for filtering ───
            "filter_meta":  build_filter_meta(row),
        }
        records.append(record)

    # บันทึก output
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[prepare_data] บันทึกแล้ว → {output_json}")
    _print_sample(records[0])
    return records


def _print_sample(record: dict) -> None:
    """แสดงตัวอย่าง 1 record เพื่อตรวจสอบ"""
    print("\n" + "─" * 60)
    print(f"ตัวอย่าง record: {record['title_en']}")
    print("─" * 60)

    print("\n[general_text]")
    print(record["general_text"])

    print("\n[music_text]")
    print(record["music_text"])

    print("\n[filter_meta]")
    for k, v in record["filter_meta"].items():
        print(f"  {k}: {v}")
    print("─" * 60 + "\n")


# ───────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────

if __name__ == "__main__":
    prepare()