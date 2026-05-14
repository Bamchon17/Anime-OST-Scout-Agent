"""
tools/youtube_search.py
-----------------------
ค้นหา YouTube link จากชื่อเพลง + ศิลปิน (Async Version)
"""

import os
import json
import hashlib
import time
import asyncio
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
YOUTUBE_API_KEY  = os.getenv("YOUTUBE_KEY")
CACHE_DIR        = Path(__file__).parent.parent / "data" / "yt_cache"
CACHE_TTL_DAYS   = 30
MAX_RESULTS      = 1

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────
# CACHE HELPERS
# ───────────────────────────────────────────────

def _cache_key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()

def _cache_path(query: str) -> Path:
    return CACHE_DIR / f"{_cache_key(query)}.json"

def _get_from_cache(query: str) -> Optional[dict]:
    path = _cache_path(query)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        age_days = (time.time() - data["timestamp"]) / (24 * 3600)
        if age_days > CACHE_TTL_DAYS:
            return None
        return data["result"]
    except:
        return None

def _save_to_cache(query: str, result: dict):
    path = _cache_path(query)
    data = {
        "query": query,
        "timestamp": time.time(),
        "result": result
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

# ───────────────────────────────────────────────
# SEARCH CORE (Wrapped in Async)
# ───────────────────────────────────────────────

def _sync_youtube_call(search_query: str) -> Optional[dict]:
    """การเรียก YouTube API แบบปกติ (Synchronous)"""
    if not YOUTUBE_API_KEY:
        return None
    
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(
            q=search_query,
            part="snippet",
            maxResults=MAX_RESULTS,
            type="video"
        )
        response = request.execute()

        if not response.get("items"):
            return None

        item = response["items"][0]
        video_id = item["id"]["videoId"]
        return {
            "title": item["snippet"]["title"],
            "url":   f"https://www.youtube.com/watch?v={video_id}",
            "id":    video_id
        }
    except HttpError:
        return None
    except Exception:
        return None

async def search_youtube(song_title: str, artist: str = "", anime: str = "") -> dict:
    """
    ค้นหา YouTube แบบ Async
    """
    search_query = f"{anime} {song_title} {artist} official music video".strip()
    
    # 1. เช็ค Cache ก่อน
    cached = _get_from_cache(search_query)
    if cached:
        return cached

    # 2. ถ้าไม่มีใน Cache ให้เรียก API (รันใน thread เพื่อไม่ให้บล็อก async loop)
    result = await asyncio.to_thread(_sync_youtube_call, search_query)
    
    if result:
        _save_to_cache(search_query, result)
        return result
    
    return {"title": None, "url": None, "id": None}

# ───────────────────────────────────────────────
# ENRICHMENT (Async Parallel)
# ───────────────────────────────────────────────

async def enrich_music_data(anime_name: str, themes: dict, max_op: int = 1, max_ed: int = 1) -> dict:
    """
    รับข้อมูล themes จาก Jikan และเติมลิงก์ YouTube แบบขนาน (Async)
    """
    def parse_theme(raw: str):
        # ลบเลขลำดับและเครื่องหมายออก เช่น "1: 'Blue Bird' by ..."
        clean = raw.split(":", 1)[-1] if ":" in raw[:3] else raw
        m = __import__("re").search(r"\"(.*)\" by (.*)", clean)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return raw, ""

    async def process_item(raw: str, is_active: bool):
        song, artist = parse_theme(raw)
        yt = None
        if is_active:
            # ค้นหา YouTube แบบ Async
            yt = await search_youtube(song_title=song, artist=artist, anime=anime_name)
        
        return {
            "raw":     raw,
            "song":    song,
            "artist":  artist,
            "youtube": yt,
        }

    # สร้าง Tasks สำหรับการรันแบบขนาน
    op_tasks = [process_item(raw, i < max_op) for i, raw in enumerate(themes.get("openings", []))]
    ed_tasks = [process_item(raw, i < max_ed) for i, raw in enumerate(themes.get("endings", []))]

    # รันทุกอย่างพร้อมกัน (Parallel Execution)
    op_results = await asyncio.gather(*op_tasks)
    ed_results = await asyncio.gather(*ed_tasks)

    return {
        "openings": op_results,
        "endings":  ed_results,
        "trailer":  themes.get("trailer"),
    }

# ───────────────────────────────────────────────
# STANDALONE TEST
# ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    song   = sys.argv[1] if len(sys.argv) > 1 else "ROCKS"
    artist = sys.argv[2] if len(sys.argv) > 2 else "Hound Dog"
    anime  = sys.argv[3] if len(sys.argv) > 3 else "Naruto"

    result = search_youtube(song, artist, anime)
    print(f"Query      : {result['query']}")
    print(f"From cache : {result['from_cache']}")
    print(f"URL        : {result['url']}")
    print(f"YT Title   : {result['title']}")