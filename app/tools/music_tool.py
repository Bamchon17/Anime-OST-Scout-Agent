"""
tools/music_tool.py
-------------------
2 modes:
  1. "ขอเพลง <ชื่อเรื่อง>"  → exact/fuzzy match จาก JSON → YouTube link
  2. แนะนำเพลง / ถามข้อมูล  → FAISS semantic search → ข้อมูลอย่างเดียว
"""

import re
import time
import requests
from functools import lru_cache
from typing import Dict, List, Optional
from urllib.parse import quote_plus

try:
    from ..rag.retrieval import retrieve_music
    from ..rag.youtube_search import search_youtube
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from app.rag.retrieval import retrieve_music
    from app.rag.youtube_search import search_youtube

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
JIKAN_BASE_URL   = "https://api.jikan.moe/v4"
JIKAN_RATE_LIMIT = 0.4
JIKAN_TIMEOUT    = 8
JIKAN_RETRIES    = 2

# คำ trigger ที่บ่งบอกว่า user ขอเพลงจากเรื่องนั้นจริงๆ
REQUEST_SONG_KEYWORDS = ["ขอเพลง", "เปิดเพลง", "หาเพลง", "เพลงจากเรื่อง", "ขอ ost", "หา ost"]


# ───────────────────────────────────────────────
# DETECT MODE
# ───────────────────────────────────────────────

def detect_song_request(query: str) -> Optional[str]:
    """
    ตรวจว่า query คือการ "ขอเพลงจากเรื่อง" หรือเปล่า
    ถ้าใช่ → คืนชื่อเรื่องที่ user พิมพ์มา
    ถ้อไม่ใช่ → คืน None

    ตัวอย่าง:
        "ขอเพลงจากเรื่อง Naruto"     → "Naruto"
        "เปิดเพลง Bleach หน่อย"       → "Bleach"
        "แนะนำเพลงแนว jazz"           → None
        "ใครแต่งเพลง Attack on Titan" → None
    """
    for kw in REQUEST_SONG_KEYWORDS:
        if kw in query:
            # ตัด keyword ออก เหลือแค่ชื่อเรื่อง
            title = re.sub(rf".*{kw}\s*(จากเรื่อง\s*)?", "", query, flags=re.IGNORECASE).strip()
            # ตัด suffix เช่น "หน่อย", "ได้เลย"
            title = re.sub(r"\s*(หน่อยสิ|หน่อยนะ|หน่อย|ได้เลย|ด้วย|นะคะ|นะครับ|นะ|สิ|ครับ|ค่ะ)$", "", title).strip()
            return title if title else None
    return None


# ───────────────────────────────────────────────
# EXACT MATCH จาก JSON
# ───────────────────────────────────────────────

def find_by_title(meta: List[Dict], title_query: str) -> Optional[Dict]:
    """
    หาใน metadata โดย match ชื่อเรื่องตรงๆ (case-insensitive)
    เช็คทั้ง main_title และ title_en
    """
    q = title_query.lower().strip()
    for item in meta:
        if (item.get("main_title", "").lower() == q or
                item.get("title_en", "").lower() == q):
            return item

    # fuzzy fallback: ถ้าชื่อ query เป็น substring ของ title
    for item in meta:
        if (q in item.get("main_title", "").lower() or
                q in item.get("title_en", "").lower()):
            return item

    return None


def find_title_mentioned_in_query(meta: List[Dict], query: str) -> Optional[Dict]:
    q = query.lower()

    for item in meta:
        candidates = [
            item.get("main_title", ""),
            item.get("title_en", ""),
            item.get("title_ja", ""),
        ]

        for candidate in [c for c in candidates if c]:
            candidate_lower = candidate.lower()
            if len(candidate_lower) >= 3 and candidate_lower in q:
                return item

    return None


# ───────────────────────────────────────────────
# JIKAN HELPERS
# ───────────────────────────────────────────────

def _extract_mal_id(resources: Dict) -> Optional[str]:
    mal_url = resources.get("MAL", "")
    match = re.search(r"/anime/(\d+)", mal_url)
    return match.group(1) if match else None


def _jikan_get(path: str) -> Optional[Dict]:
    url = f"{JIKAN_BASE_URL}{path}"
    for attempt in range(JIKAN_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=JIKAN_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(float(resp.headers.get("Retry-After", 1)))
            else:
                return None
        except requests.RequestException:
            if attempt < JIKAN_RETRIES:
                time.sleep(1)
    return None


def _image_from_jikan_anime(anime: Dict) -> str:
    images = anime.get("images", {})
    jpg = images.get("jpg", {}) if isinstance(images, dict) else {}
    webp = images.get("webp", {}) if isinstance(images, dict) else {}
    return (
        jpg.get("large_image_url")
        or jpg.get("image_url")
        or webp.get("large_image_url")
        or webp.get("image_url")
        or ""
    )


@lru_cache(maxsize=512)
def _fetch_cover_url(title: str = "", mal_id: str = "") -> str:
    if title:
        time.sleep(JIKAN_RATE_LIMIT)
        data = _jikan_get(f"/anime?q={quote_plus(title)}&limit=1")
        matches = data.get("data", []) if data else []
        if matches:
            image_url = _image_from_jikan_anime(matches[0])
            if image_url:
                return image_url

    if mal_id:
        time.sleep(JIKAN_RATE_LIMIT)
        data = _jikan_get(f"/anime/{mal_id}")
        anime = data.get("data", {}) if data else {}
        return _image_from_jikan_anime(anime)

    return ""


def _fetch_themes(mal_id: str) -> Dict:
    """ดึง Opening/Ending จาก Jikan — คืนแค่ OP1 + ED1"""
    time.sleep(JIKAN_RATE_LIMIT)
    data = _jikan_get(f"/anime/{mal_id}/themes")
    themes = data.get("data", {}) if data else {}
    return {
        "openings": themes.get("openings", [])[:1],  # แค่ OP1
        "endings":  themes.get("endings",  [])[:1],  # แค่ ED1
    }


def _parse_theme(raw: str) -> tuple[str, str]:
    """'#1: "ROCKS" by Hound Dog (eps 1-25)' → ('ROCKS', 'Hound Dog')"""
    m = re.search(r'"([^"]+)"\s+by\s+([^(]+)', raw)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return raw, ""


# ───────────────────────────────────────────────
# CORE LOGIC
# ───────────────────────────────────────────────
async def run(query: str, meta: List[Dict], include_external: bool = False, top_k: int = 3) -> Dict:
    """
    ปรับปรุง: 
    1. เป็น async def เพื่อรองรับการทำงานแบบ non-blocking
    2. เพิ่ม include_external เพื่อรับค่าจาก Registry
    """

    # ── ตรวจว่า user ขอเพลงจากเรื่องไหนหรือเปล่า
    title_query = detect_song_request(query)

    # ══════════════════════════════════════════
    # MODE 1: ขอเพลงจากเรื่อง (เปิดใช้งานถ้ามี title_query หรือถูกสั่งจาก include_external)
    # ══════════════════════════════════════════
    if title_query or include_external:
        search_target = title_query if title_query else query
        item = find_by_title(meta, search_target)
        if not item and include_external:
            item = find_title_mentioned_in_query(meta, query)

        if not item:
            # ถ้าหาชื่อเรื่องใน JSON ไม่เจอ ให้ fallback ไป semantic mode แทนที่จะ error
            if include_external and not title_query:
                pass # ไหลไปทำ Mode 2 ด้านล่าง
            else:
                return {
                    "mode":  "song_request",
                    "error": f"ไม่พบเรื่อง '{search_target}' ในฐานข้อมูล"
                }
        else:
            mal_id    = _extract_mal_id(item.get("resources", {}))
            composer  = item.get("composer") or item.get("music") or "Unknown"
            image_url = item.get("image_url") or _fetch_cover_url(item.get("main_title", ""), mal_id or "")
            themes_raw = _fetch_themes(mal_id) if mal_id else {"openings": [], "endings": []}

            theme_results = []
            for raw in themes_raw["openings"]:
                song, artist = _parse_theme(raw)
                yt = await search_youtube(song_title=song, artist=artist, anime=item["main_title"])
                theme_results.append({"song": song, "artist": artist, "type": "OP", "youtube_url": yt.get("url")})

            for raw in themes_raw["endings"]:
                song, artist = _parse_theme(raw)
                yt = await search_youtube(song_title=song, artist=artist, anime=item["main_title"])
                theme_results.append({"song": song, "artist": artist, "type": "ED", "youtube_url": yt.get("url")})

            return {
                "mode":        "song_request",
                "main_title":  item["main_title"],
                "title":       item.get("title_en") or item["main_title"],
                "composer":    composer,
                "music":       item.get("music") or composer,
                "themes":      theme_results,
                "youtube_url": theme_results[0]["youtube_url"] if theme_results else None,
                "resources":   item.get("resources", {}),
                "image_url":   image_url,
                "cover_url":   image_url,
                "synopsis":    item.get("synopsis_short") or item.get("synopsis", ""),
                "score":       1.0,
            }

    # ══════════════════════════════════════════
    # MODE 2: semantic search
    # ══════════════════════════════════════════
    local_results = retrieve_music(query, top_k=top_k)
    results = []

    for r in local_results:
        title = r.get("main_title") or r.get("title_en") or r.get("title")
        composer = r.get("composer") or r.get("music") or "Unknown"
        mal_id = _extract_mal_id(r.get("resources", {}))
        image_url = r.get("image_url") or _fetch_cover_url(title or "", mal_id or "")
        youtube_url = None
        youtube_title = None

        if include_external and title:
            yt = await search_youtube(song_title="opening OST", anime=title)
            youtube_url = yt.get("url")
            youtube_title = yt.get("title")

        themes = []
        if youtube_url:
            themes.append(
                {
                    "song": youtube_title or f"{title} OST",
                    "artist": composer,
                    "type": "OST",
                    "youtube_url": youtube_url,
                }
            )

        results.append(
            {
                "main_title": r.get("main_title") or title,
                "title":      title,
                "composer":   composer,
                "music":      r.get("music") or composer,
                "mood":       r.get("music_mood_tags", []),
                "year":       r.get("filter_year") or r.get("year"),
                "rating":     r.get("max_rating") or r.get("rating"),
                "resources":  r.get("resources", {}),
                "image_url":  image_url,
                "cover_url":  image_url,
                "synopsis":   r.get("synopsis_short") or r.get("synopsis", ""),
                "score":      r.get("score", 0),
                "themes":     themes,
                "youtube_url": youtube_url,
            }
        )

    return {
        "mode": "semantic",
        "results": results,
    }

# ───────────────────────────────────────────────
# STANDALONE TEST
# ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json
    from pathlib import Path

    meta_path = Path(__file__).parent.parent.parent / "data" / "music_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "ขอเพลงจากเรื่อง Naruto"
    print(f"🔍 query: {query}\n")

    result = run(query, meta=meta)
    print(json.dumps(result, ensure_ascii=False, indent=2))
