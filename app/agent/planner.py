# Plan task
# app/agent/planner.py
"""
Planner — sends the current AgentState to the LLM and parses
the JSON response into a validated ActionPlan.

Architecture:
  ┌─────────────┐     ┌──────────────┐     ┌────────────┐
  │ AgentState  │────▶│  BaseLLM     │────▶│ ActionPlan │
  │ (context)   │     │  (interface) │     │ (Pydantic) │
  └─────────────┘     └──────────────┘     └────────────┘

Swap BaseLLM implementation for Ollama / OpenAI / Gemini without
touching the controller or any other file.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from app.models.llm import llm_service
from app.agent.prompts import build_planning_prompt, build_planning_user_message
from app.agent.state import (
    ActionPlan,
    ExtractedEntities,
    Intent,
    RetrievalStrategy,
    ToolName,
)

logger = logging.getLogger(__name__)


def _is_out_of_domain(query: str) -> bool:
    q_lower = query.lower()

    anime_terms = [
        "anime",
        "manga",
        "ost",
        "soundtrack",
        "naruto",
        "one piece",
        "bleach",
        "cowboy bebop",
        "อนิเมะ",
        "มังงะ",
        "เพลง",
        "เพลงประกอบ",
    ]
    strong_tech_terms = [
        "iphone",
        "ipad",
        "samsung",
        "galaxy s",
        "s24",
        "s23",
        "s25",
        "smartphone",
        "android",
        "snapdragon",
        "exynos",
        "pro max",
        "มือถือ",
        "โทรศัพท์",
        "กล้อง",
        "แบต",
    ]

    return any(term in q_lower for term in strong_tech_terms) and not any(
        term in q_lower for term in anime_terms
    )


def _extract_metadata_filters(query: str) -> ExtractedEntities:
    q_lower = query.lower()

    season = None
    for candidate in ("spring", "summer", "fall", "autumn", "winter"):
        if candidate in q_lower:
            season = "fall" if candidate == "autumn" else candidate
            break

    year = None
    year_match = re.search(r"\b(19|20)\d{2}\b", query)
    if year_match:
        year = int(year_match.group())

    rating_min = None
    rating_patterns = [
        r"(?:rating|score|rate|คะแนน|เรตติ้ง|เรทติ้ง)\s*(?:>=|>|มากกว่า|เกิน|ไม่ต่ำกว่า|อย่างน้อย)?\s*(\d+(?:\.\d+)?)",
        r"(?:>=|>|มากกว่า|เกิน|ไม่ต่ำกว่า|อย่างน้อย)\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:ขึ้นไป|บวก)",
    ]
    for pattern in rating_patterns:
        match = re.search(pattern, q_lower)
        if match:
            rating_min = float(match.group(1))
            break

    anime_type = None
    if re.search(r"\btv\b|tv series", q_lower):
        anime_type = "TV"
    elif re.search(r"\bmovie\b|ภาพยนตร์", q_lower):
        anime_type = "Movie"
    elif re.search(r"\bova\b", q_lower):
        anime_type = "OVA"

    return ExtractedEntities(
        year=year,
        season=season,
        rating=rating_min,
        rating_min=rating_min,
        type=anime_type,
    )


# ─────────────────────────────────────────────
#  LLM Interface (Abstract Base)
# ─────────────────────────────────────────────

class BaseLLM(ABC):
    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_message:  str,
        temperature:   float = 0.0,   # 0 for deterministic planning
        max_tokens:    int   = 1024,
    ) -> str:
        """Return the raw LLM text response."""
        ...

# ─────────────────────────────────────────────
#   TyphoonLLM — production LLM via llm_service
# ─────────────────────────────────────────────

class TyphoonLLM(BaseLLM):
    """
    ใช้ llm_service (Typhoon-v2.5-30B) เป็นสมองหลักในการวางแผน
    """
    async def complete(
        self,
        system_prompt: str,
        user_message:  str,
        temperature:   float = 0.0,
        max_tokens:    int   = 1024,
    ) -> str:
        # เรียกใช้ chat function จาก llm_service ที่แบมเขียนไว้
        return await llm_service.chat(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature
        )
  

# ─────────────────────────────────────────────
#  OllamaLLM — สำหรับรัน Local (ถ้าต้องการสลับ)
# ─────────────────────────────────────────────
# class OllamaLLM(BaseLLM):
#     def __init__(self, model: str = "typhoon:v1.5-8b-instruct-q4_k_m", base_url: str = "http://localhost:11434"):
#         self.model    = model
#         self.base_url = base_url

#     async def complete(
#         self,
#         system_prompt: str,
#         user_message:  str,
#         temperature:   float = 0.0,
#         max_tokens:    int   = 1024,
#     ) -> str:
#         try:
#             import ollama
#         except ImportError:
#             raise RuntimeError("กรุณาติดตั้ง ollama: pip install ollama")

#         client = ollama.AsyncClient(host=self.base_url)
#         response = await client.chat(
#             model=self.model,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_message},
#             ],
#             options={"temperature": temperature, "num_predict": max_tokens},
#         )
#         return response["message"]["content"]


# ─────────────────────────────────────────────
#  JSON parser — robust against LLM quirks
# ─────────────────────────────────────────────
def _parse_json_safe(raw: str) -> dict[str, Any]:
    """
    Extract JSON from LLM output even when it:
      - wraps it in ```json ... ``` fences
      - adds a preamble sentence before the JSON
      - uses single quotes instead of double quotes
    """
    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM output:\n{raw[:300]}")


def _json_to_action_plan(data: dict[str, Any]) -> ActionPlan:
    entities_raw = data.get("entities", {})

    raw_title = entities_raw.get("anime_title")
    anime_titles = [raw_title] if isinstance(raw_title, str) else raw_title

    entities = ExtractedEntities(
        anime_title = anime_titles,
        tags        = entities_raw.get("tags", []), 
        char_tags   = entities_raw.get("char_tags", []),
        studio      = entities_raw.get("studio"),
        music_style = entities_raw.get("music_style"),
        year        = entities_raw.get("year"),
        year_from   = entities_raw.get("year_from"),
        year_to     = entities_raw.get("year_to"),
        season      = entities_raw.get("season"),
        rating      = entities_raw.get("rating"),
        rating_min  = entities_raw.get("rating_min") or entities_raw.get("rating"),
        rating_max  = entities_raw.get("rating_max"),
        type        = entities_raw.get("type"), 
    )

    # การ Map ToolName ให้ตรงกับ Enum ที่เรานิยามไว้
    raw_tool_val = str(data.get("selected_tool", "")).lower()
    if any(x in raw_tool_val for x in ["music", "song", "ost"]):
        selected_tool = ToolName.MUSIC_TOOL
    elif any(x in raw_tool_val for x in ["compare", "vs"]):
        selected_tool = ToolName.COMPARE_TOOL
    elif any(x in raw_tool_val for x in ["filter", "metadata"]):
        selected_tool = ToolName.FILTER_TOOL
    else:
        selected_tool = ToolName.RECOMMEND_TOOL # Default

    return ActionPlan(
        intent             = Intent(data.get("intent", "recommend")),
        entities           = entities,
        selected_tool=selected_tool,
        retrieval_strategy = RetrievalStrategy(data.get("retrieval_strategy", "semantic")),
        rewritten_query    = data.get("rewritten_query", ""),
        reasoning          = data.get("reasoning", ""),
        confidence         = float(data.get("confidence", 0.0)),
    )

# ─────────────────────────────────────────────
#  Planner class Hybrid rule-base และใช้ Typhoonเป็นหลัก
# ─────────────────────────────────────────────
class Planner:
    def __init__(self, llm: BaseLLM | None = None):
        self.llm: BaseLLM = llm or TyphoonLLM()

    async def plan(
        self,
        current_query:       str,
        context_summary:     str,
        conversation_history: list[dict],
        loop_count:          int, 
    ) -> ActionPlan:
        
        # ────────────────────────────────────────────────────────────
        # STEP 1: Rule-based (Fast track & Token Saving)
        # ────────────────────────────────────────────────────────────
        rule_plan = self._apply_rule_based(current_query, context_summary)
        if rule_plan:
            logger.info("[Planner] Rule-based Hit! | tool=%s", rule_plan.selected_tool)
            return rule_plan

        # ────────────────────────────────────────────────────────────
        # STEP 2: LLM Planning (Typhoon)
        # ────────────────────────────────────────────────────────────
        system_prompt = build_planning_prompt(context_summary, loop_count=loop_count)
        user_message  = build_planning_user_message(current_query, conversation_history)

        logger.info("[Planner] Calling LLM (Typhoon) for Reasoning...")
        
        raw_response = await self.llm.complete(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.0,
        )

        try:
            data = _parse_json_safe(raw_response)
            plan = _json_to_action_plan(data)
        except Exception as exc:
            logger.warning("[Planner] Parse failed: %s", exc)
            # Fallback หาก LLM ตอบผิดรูปแบบ
            plan = self._get_fallback_plan(current_query, raw_response)

        return plan

    def _apply_rule_based(self, query: str, context: str) -> Optional[ActionPlan]:
        """
        ตรวจจับ Intent เบื้องต้น และจัดการ Self-Correction จากประวัติใน context
        """
        q_lower = query.lower()

        if _is_out_of_domain(query):
            return ActionPlan(
                intent=Intent.UNKNOWN,
                selected_tool=ToolName.NONE,
                retrieval_strategy=RetrievalStrategy.SKIP,
                rewritten_query=query,
                reasoning="Out-of-domain: technology/product comparison is outside the anime and OST scope",
                confidence=1.0,
            )

        # ถ้าใน context มีบอกว่า "Tool ก่อนหน้าหาไม่เจอ" ให้เปลี่ยนไปใช้ Semantic Search ทันที
        if "no results found" in context.lower() or "not found" in context.lower():
            return ActionPlan(
                intent=Intent.RECOMMEND,
                selected_tool=ToolName.RECOMMEND_TOOL,
                retrieval_strategy=RetrievalStrategy.SEMANTIC,  # Use the enum directly
                rewritten_query=query,
                reasoning="Self-Correction: Tool ก่อนหน้าไม่พบข้อมูล จึงลองใช้ Semantic Search ทั่วไปแทน",
                confidence=0.9
            )

        # --- [Rule-based: Music] ---
        if any(k in q_lower for k in ["เพลง", "music", "ost", "soundtrack"]):
            # เช็คว่าเป็นการ "ขอลิงก์/ขอฟัง" หรือแค่ "ถามข้อมูลเพลง"
            is_asking_for_link = any(k in q_lower for k in [
                "ขอเพลง",
                "หาเพลง",
                "ขอฟัง",
                "ขอลิงก์",
                "ขอลิงค์",
                "ขอลิ้ง",
                "ขอลิ้งก์",
                "ขอลิ้งค์",
                "เปิดเพลง",
                "ฟังเพลง",
                "youtube",
                "ยูทูป",
                "link",
            ])
            
            return ActionPlan(
                intent=Intent.RECOMMEND,
                selected_tool=ToolName.MUSIC_TOOL, 
                retrieval_strategy=RetrievalStrategy.HYBRID if is_asking_for_link else RetrievalStrategy.SEMANTIC,
                rewritten_query=query,
                # ส่ง flag ไปใน reasoning หรือจะเพิ่ม field ใน ActionPlan ก็ได้
                # เพื่อให้ Controller รู้ว่าต้องเปิด include_external
                reasoning=f"Rule-based: Music Detection | RequestLink: {is_asking_for_link}",
                confidence=1.0
            )

        # --- [Rule-based: Comparison] ---
        if any(k in q_lower for k in ["เทียบ", "ต่างกัน", "vs", "better than", "สูงกว่า", "ยาวกว่า"]):
            english_names = re.findall(r'[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*', query)
            if len(english_names) >= 2:
                clean_titles = [name.strip() for name in english_names]
                return ActionPlan(
                    intent=Intent.COMPARE,
                    selected_tool=ToolName.COMPARE_TOOL,
                    retrieval_strategy=RetrievalStrategy.HYBRID,
                    rewritten_query=query,
                    entities=ExtractedEntities(anime_title=clean_titles),
                    reasoning=f"Extracted explicit English titles {clean_titles} for side-by-side comparison",
                    confidence=1.0
                )

            # 1. ลบคำเชื่อมที่ไม่จำเป็นออกเพื่อให้เหลือแต่ชื่อเรื่อง
            temp_q = query
            stop_words = ["เปรียบเทียบ", "ระหว่าง", "เรตติ้งของ", "เรื่องไหน", "ดีกว่า", "สูงกว่า", "กัน", "ยาวกว่า", "อนิเมะ"]
            for word in stop_words:
                temp_q = temp_q.replace(word, "")
            
            # 2. แยกชื่อเรื่องด้วยตัวคั่นต่างๆ
            # ผลลัพธ์จะเป็น list ของ string ทันที
            raw_titles = re.split(r'\s*กับ\s*|\s*vs\s*|\s*และ\s*|\s*,\s*', temp_q)
            
            # 3. กรองเอาเฉพาะชื่อที่ใช้งานได้ (ตัดค่าว่าง)
            clean_titles = [t.strip() for t in raw_titles if len(t.strip()) > 1]
            
            # --- PERSPECTIVE: คู่หูสายลุย ---
            # ถ้าสกัดได้แค่ชื่อเดียว หรือไม่ได้เลย ให้พยายามหาคำภาษาอังกฤษมาช่วย
            if len(clean_titles) < 2:
                english_names = re.findall(r'[A-Z][A-Za-z0-9\s]*', query)
                if len(english_names) >= 2:
                    clean_titles = [n.strip() for n in english_names]

            return ActionPlan(
                intent=Intent.COMPARE,
                selected_tool=ToolName.COMPARE_TOOL,
                retrieval_strategy=RetrievalStrategy.HYBRID, # ใช้ Hybrid เพราะเรามีชื่อชัดเจน
                rewritten_query=query,
                # แก้ไข: ส่งเป็น list (clean_titles) และใช้ชื่อที่ tool เข้าใจ (titles)
                # หมายเหตุ: ตรวจสอบในคลาส ExtractedEntities ของคุณด้วยว่ามี field ชื่อ 'titles' หรือยัง
                # ถ้ายังไม่มี ให้ใช้ anime_title แต่ต้องส่งเป็น list เท่านั้น
                entities=ExtractedEntities(anime_title=clean_titles), 
                reasoning=f"สกัดชื่อเรื่อง {clean_titles} เพื่อเปรียบเทียบแบบ side-by-side",
                confidence=1.0
            )

        # --- [Rule-based: Specific Filtering] ---
        # ดักจับ ปี (4 หลัก) หรือ เรตติ้ง
        filters = _extract_metadata_filters(query)
        has_metadata_filter = any(
            value is not None and value != []
            for value in [
                filters.year,
                filters.season,
                filters.rating_min,
                filters.type,
                filters.studio,
                filters.tags,
            ]
        )
        if has_metadata_filter:
            return ActionPlan(
                intent=Intent.FILTER,
                selected_tool=ToolName.FILTER_TOOL,
                retrieval_strategy=RetrievalStrategy.METADATA,
                rewritten_query=query,
                # แก้ไข: ใส่ปีที่ตรวจพบลงใน ExtractedEntities
                entities=filters,
                reasoning="Rule-based: detected structured metadata filters such as season, rating, year, or type",
                confidence=1.0
            )

        return None # ถ้าไม่เข้ากฎเลย ให้ไหลไปหา LLM

    def _get_fallback_plan(self, query: str, raw_response: str) -> ActionPlan:
        """
        กรณี LLM หลอน (Hallucinate) หรือ JSON พัง
        """
        detected_tool = ToolName.RECOMMEND_TOOL
        if "เพลง" in raw_response or "music" in raw_response.lower():
            detected_tool = ToolName.MUSIC_TOOL
            
        return ActionPlan(
            intent=Intent.UNKNOWN,
            selected_tool=detected_tool,
            retrieval_strategy=RetrievalStrategy.HYBRID,
            rewritten_query=query,
            reasoning="Fallback: ระบบเข้าสู่โหมดสำรองเนื่องจากข้อผิดพลาดในการประมวลผล",
            confidence=0.3
        )
