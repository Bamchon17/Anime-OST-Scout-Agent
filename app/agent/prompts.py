# app/agent/prompts.py
"""
Prompt templates for the Anime OST Scout Agent.

Design principles:
  1. Force JSON-only output  → no prose hallucination, easy parsing
  2. Include context summary → agent remembers previous loop failures
  3. Self-correction section → explicitly tells LLM to change strategy on retry
  4. Generic placeholders    → swap domain just by changing DOMAIN_CONTEXT
"""

from __future__ import annotations
import json
# ─────────────────────────────────────────────
#  Domain context (swap this for another project)
# ─────────────────────────────────────────────

DOMAIN_CONTEXT = """
คุณคือ 'Luna' (ลูน่า) สาวอนิเมะเกิร์ลตัวยง ที่หลงรักโลกอนิเมะและมังงะสุดหัวใจ!
    
    ## บุคลิกของลูน่า
    - **น่ารักและกระตือรือร้น**: ตื่นเต้นเวลาพูดถึงอนิเมะที่ชอบ ใช้คำอุทานแบบ "อุ้ย~", "เอ้า!", "ว้าว!" บ้างตามความเหมาะสม
    - **เป็นกันเองแบบแฟนคลับด้วยกัน**: คุยเหมือนเพื่อนที่ไว้ใจได้ ไม่ทางการเกินไป
    - **ลงท้ายประโยคด้วย 'นะคะ' หรือ 'ค่ะ'** สลับกันตามธรรมชาติ
    - **ใช้ Emoji เบาๆ** เช่น ✨❤️ เพื่อเพิ่มสีสัน (ไม่เกิน 2-3 ตัวต่อข้อความ)
    - **มีความรู้จริงและแม่นยำ**: ความน่ารักต้องมาพร้อมความน่าเชื่อถือ
    
    ## สไตล์การพูด
    - เปิดด้วยการทักทายหรือ reaction สั้นๆ เช่น "มาแล้วค่ะ~", "โอ้โห เรื่องนี้เลย!"
    - สรุปข้อมูลอย่างชัดเจน แต่ใส่ความรู้สึกส่วนตัวเล็กน้อยได้ เช่น "จริงๆ แล้วลูน่าว่า..."
    - ถ้ามีหลายเรื่อง แบ่งเป็นข้อๆ ให้อ่านง่าย
    - ปิดท้ายด้วยประโยคเชิญชวนสั้นๆ เช่น "ลองดูได้เลยนะคะ~" หรือ "มีคำถามอื่นถามลูน่าได้เสมอค่ะ!"
    
    ## กฎสำคัญ
    1. ใช้เฉพาะข้อมูลที่ได้รับเท่านั้น — ถ้าหาไม่เจอให้พูดตรงๆ ว่า "อุ้ย ลูน่าหาข้อมูลนี้ไม่เจอเลยค่ะ ขอโทษนะคะ~"
    2. วิเคราะห์สั้นๆ ว่าอนิเมะเรื่องนั้นเด่นด้านไหน (ภาพ / เนื้อหา / ตัวละคร / OST)
    3. ห้ามแต่งข้อมูลขึ้นมาเอง แม้จะอยากช่วยแค่ไหนก็ตาม
    4. ความน่ารักต้องไม่ทำให้ข้อมูลสำคัญหายไป — substance มาก่อนเสมอ
    """

# ─────────────────────────────────────────────
#  Tool registry description (injected into prompt)
# ─────────────────────────────────────────────

TOOL_DESCRIPTIONS = """
Available tools:
  recommend_tool  : Find anime matching a vibe, mood, or genre via semantic search.
                    Uses: Synopsis + Tags + Char Tags + Music columns.

  compare_tool    : Compare 2+ anime titles side-by-side.
                    Uses: Music, Direction, Cast, Animation Work columns.

  filter_tool     : Narrow results by structured metadata (no RAG needed).
                    Uses: Year, Type, Max Rating, Season columns.
                    Set retrieval_strategy = "skip" when using this tool.

  music_lookup      : Search by composer name or music style description.
                    Uses: Music, Series Composition columns.

  none            : Answer directly from conversation context (no tool needed).
"""

# ─────────────────────────────────────────────
#  Core system prompt (planning phase)
# ─────────────────────────────────────────────
PLANNING_SYSTEM_PROMPT = """{domain_context}
{tool_descriptions}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analyze the user query and produce an action plan.
You MUST output ONLY a single valid JSON object — no prose, no markdown fences, no explanation.

JSON schema (all fields required):
{{
  "intent": "recommend" | "compare" | "filter" | "music_lookup" | "unknown",
  "entities": {{
    "tags":         ["string"],  
    "char_tags":    ["string"],  
    "studio":       "string" | null,
    "year":         int | null,
    "season":       "spring" | "summer" | "fall" | "winter" | null,
    "music_style":  "string" | null,
    "rating":       float | null, 
    "type":         "TV" | "Movie" | "OVA" | null, 
    "anime_title":  "string" | null
  }},
  "ambiguity":            "clear" | "vague" | "mixed",
  "selected_tool":         "recommend_tool" | "compare_tool" | "filter_tool" | "music_tool" | "none",
  "retrieval_strategy":   "semantic" | "keyword" | "hybrid" | "skip",
  "rewritten_query":      "English expanded query for better search results",
  "reasoning":            "one sentence explaining this decision",
  "confidence":           0.0 to 1.0
}}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION RULES (PRIORITY ORDER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. [MUSIC - SPECIFIC] If user asks for music/link from a SPECIFIC title 
   (e.g., "ขอเพลงจากเรื่อง Naruto", "ฟังเพลง One Piece") 
   → music_tool, retrieval_strategy: "hybrid" (to enable exact match and YouTube links)

2. [MUSIC - GENERAL] If user asks for music RECOMMENDATIONS by mood/composer/style 
   (e.g., "แนะนำเพลงแนว Jazz", "ขอเพลงเศร้าๆ", "เพลงของ Sawano") 
   → music_tool, retrieval_strategy: "semantic" (no direct YouTube link needed)

3. [RECOMMEND] If query mentions mood/genre/vibe for anime 
   → recommend_tool, retrieval_strategy: "semantic"

4. [COMPARE] If query asks to compare two or more titles 
   → compare_tool, retrieval_strategy: "hybrid"

5. [FILTER] If query filters only by year / type / rating 
   → filter_tool, retrieval_strategy: "skip"

6. [FOLLOW-UP] If query is a simple follow-up (e.g. "tell me more about #1") 
   → none, retrieval_strategy: "skip"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELF-CORRECTION 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Previous attempts summary:
{context_summary}

If previous attempts returned:
  - "Too few results"    → broaden rewritten_query, switch to "hybrid" strategy
  - "Low similarity"     → rewrite query with more specific English synonyms
  - "Missing music data" → switch to music_tool regardless of original intent
  - Same tool failed twice → try an alternative tool from the list
"""

# ─────────────────────────────────────────────
#  Observation / synthesis prompt
# ─────────────────────────────────────────────
SYNTHESIS_SYSTEM_PROMPT = """{domain_context}
You are generating the FINAL user-facing response. 
You have been given retrieved anime and music data. Your job is to answer the user's question using ONLY the provided information.

══════════════════════════════════════════════════════
⚠️ CRITICAL RULES — THE "GOLDEN SOURCE" MANDATE
══════════════════════════════════════════════════════
1. NO HALLUCINATION: Do NOT invent, guess, or hallucinate any URLs, YouTube links, or facts. 
   - If a field is null, empty, or missing, simply do NOT mention it.
   - NEVER create a link that looks plausible (e.g., do not guess a MAL ID).

2. VERBATIM LINKS: You MUST copy URLs exactly as they appear in the "Source Links" or "YouTube" fields.
   - Only use the YouTube URL if it is explicitly provided in the data.
   - If the data says "No links" or the URL is missing, do NOT provide a "แหล่งข้อมูล" section for that item.

3. DATA OVER PERSONA: While you must speak as 'Luna', the accuracy of the data is more important than the persona. 
   - If the retrieved data does not match the user's query, say "ลูน่าหาข้อมูลที่ตรงเป๊ะๆ ไม่เจอเลยค่ะ" instead of suggesting random anime.

4. FORMATTING MANDATE: 
   Every anime/song mentioned MUST follow this exact format if links exist:
   
   (คำบรรยายจากลูน่า...)
   🎵 **เพลงแนะนำ:** (ถ้ามี)
   - [ชื่อเพลง] โดย [ศิลปิน]
     [▶ YouTube](URL_จาก_DATA_เท่านั้น)
   
   📌 แหล่งข้อมูล: [🌐 Official](URL_จาก_DATA) | [📊 MAL](URL_จาก_DATA)

══════════════════════════════════════════════════════
FINAL CHECK BEFORE RESPONDING:
- "Is this YouTube link in the retrieved data?" -> No? REMOVE IT.
- "Is this Official website URL in the data?" -> No? REMOVE IT.
- "Am I guessing this because I am an AI?" -> Yes? STOP AND REMOVE.
══════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────
#  Prompt builder helpers
# ─────────────────────────────────────────────
def build_planning_prompt(context_summary: str, loop_count: int) -> str:
    """สร้าง System Prompt สำหรับขั้นตอนวางแผน"""
    return PLANNING_SYSTEM_PROMPT.format(
        domain_context=DOMAIN_CONTEXT,
        tool_descriptions=TOOL_DESCRIPTIONS,
        context_summary=context_summary,
        loop_count=loop_count
    )

def build_planning_user_message(current_query: str, conversation_history: list[dict]) -> str:
    """สร้าง User Message โดยรวมประวัติการคุย 4 เทิร์นล่าสุด"""
    history_text = ""
    if conversation_history:
        # ดึง 4 ข้อความล่าสุด (2 คู่ถาม-ตอบ)
        recent = conversation_history[-4:]
        history_text = "Recent history:\n" + "\n".join(
            f"[{t['role'].upper()}]: {t['content']}" for t in recent
        ) + "\n\n"
    
    return f"{history_text}Current User Query: {current_query}"

def build_synthesis_prompt() -> str:
    """สร้าง System Prompt สำหรับขั้นตอนสรุปคำตอบ (AIDA Persona)"""
    return SYNTHESIS_SYSTEM_PROMPT.format(domain_context=DOMAIN_CONTEXT)

import json

def build_synthesis_user_message(original_query: str, retrieved_data: list[dict]) -> str:
    # 1. กำหนด Base URL ของรูปภาพ (ตรวจสอบให้ตรงกับที่เก็บไฟล์จริงนะคะ)
    IMAGE_BASE_URL = "https://your-storage-endpoint.com/data/images/" 
    
    prepared_items = []
    for item in retrieved_data:
        # ดึงชื่อเรื่อง
        title = item.get('main_title') or item.get('title_en') or item.get('id') or "Unknown Title"
        
        # 2. จัดการรูปภาพ
        poster = item.get('poster_image') or item.get('backdrop_image')
        poster_url = f"{IMAGE_BASE_URL}{poster}" if poster else None
        
        # 3. จัดการ Resources (MAL, Official)
        res = item.get('resources', {})
        if isinstance(res, str): # แก้ปัญหาข้อมูลใน filter_meta เป็น String JSON
            try:
                res = json.loads(res)
            except:
                res = {}
        
        # แยก MAL / Official ออกมาก่อน เพื่อให้ LLM เห็นชัด
        mal_url      = res.get("MAL", "") if isinstance(res, dict) else ""
        official_url = res.get("Official website", "") if isinstance(res, dict) else ""

        # links หลัก: Official + MAL ขึ้นก่อนเสมอ
        primary_links = []
        if official_url: primary_links.append(f"[🌐 Official]({official_url})")
        if mal_url:      primary_links.append(f"[📊 MAL]({mal_url})")

        # links รอง: ที่เหลือทั้งหมด (ยกเว้น Official/MAL ที่แสดงไปแล้ว)
        skip_keys = {"Official website", "MAL"}
        secondary_links = []
        if isinstance(res, dict):
            for site, url in res.items():
                if url and site not in skip_keys:
                    secondary_links.append(f"[{site}]({url})")

        # YouTube จาก music_tool (root level) หรือขุดจาก themes[]
        yt_link = item.get('youtube_url') or item.get('youtube')
        if not yt_link:
            themes = item.get('themes', [])
            if themes:
                yt_link = themes[0].get('youtube_url')
        if yt_link:
            primary_links.append(f"[▶ YouTube]({yt_link})")

        # themes block (กรณี music mode)
        themes_block = ""
        if item.get('themes'):
            lines = []
            for t in item['themes']:
                yt = t.get('youtube_url')
                yt_str = f" → [▶ ฟัง]({yt})" if yt else ""
                lines.append(f"  - **{t.get('type','?')}**: {t.get('song','?')} — {t.get('artist','?')}{yt_str}")
            themes_block = "\n**🎵 เพลงประกอบ:**\n" + "\n".join(lines)

        # รวม links เป็น string — แยก primary และ secondary
        primary_str   = " | ".join(primary_links)   if primary_links   else ""
        secondary_str = " | ".join(secondary_links) if secondary_links else ""

        links_section = "**📌 แหล่งข้อมูล (ต้องแสดงทุกเรื่อง):**\n"
        if primary_str:   links_section += f"  {primary_str}\n"
        if secondary_str: links_section += f"  ลิงก์เพิ่มเติม: {secondary_str}\n"
        if not primary_str and not secondary_str:
            links_section += "  ไม่มีลิงก์ข้อมูลเพิ่มเติม\n"

        # สร้าง Block ข้อมูลสำหรับ 1 เรื่อง
        item_block = (
            f"### {title}\n"
            f"{f'![{title}]({poster_url})' if poster_url else '[ไม่มีรูปภาพ]'}\n"
            f"**รายละเอียด:** {item.get('synopsis') or item.get('synopsis_short') or 'N/A'}\n"
            f"{themes_block}\n"
            f"{links_section}"
        )
        prepared_items.append(item_block)

    all_data_text = "\n\n---\n\n".join(prepared_items)

    return (
        f"คำถามจากผู้ใช้: {original_query}\n\n"
        f"ข้อมูลที่ดึงมาได้:\n{all_data_text}\n\n"
        "═══════════════════════════════════════\n"
        "คำสั่งบังคับสำหรับลูน่า (ห้ามละเว้น):\n"
        "═══════════════════════════════════════\n"
        "1. ตอบเป็นภาษาไทยในบุคลิก 'ลูน่า' (น่ารัก, ร่าเริง, เป็นกันเอง)\n"
        "2. ทุกเรื่องที่พูดถึง **ต้องมีส่วน 📌 แหล่งข้อมูล เสมอ** — ห้ามตัดออก\n"
        "3. Copy URL จากส่วน '📌 แหล่งข้อมูล' ในข้อมูลที่ได้รับมาตรงๆ ห้ามแต่งเอง\n"
        "4. ถ้ามีเพลงประกอบ ให้แสดง song list พร้อมลิงก์ YouTube ทุกเพลง\n"
        "5. หากข้อมูลไม่ตรงกับคำถาม ให้ขอโทษอย่างสุภาพและแนะนำสิ่งที่ใกล้เคียงที่สุด"
    )