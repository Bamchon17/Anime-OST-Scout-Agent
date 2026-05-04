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
from typing import Any

from app.agent.prompts import build_planning_prompt, build_planning_user_message
from app.agent.state import (
    ActionPlan,
    ExtractedEntities,
    Intent,
    RetrievalStrategy,
    ToolName,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  LLM Interface (Abstract Base)
# ─────────────────────────────────────────────

class BaseLLM(ABC):
    """
    Generic LLM interface. Implement this for any provider:
      - OllamaLLM(BaseLLM)  → local Qwen/Gemma/Mistral via Ollama
      - OpenAILLM(BaseLLM)  → GPT-4o-mini via OpenAI API
      - GeminiLLM(BaseLLM)  → Google Gemini
    """

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
#  Mock LLM (for development / testing)
# ─────────────────────────────────────────────

class MockLLM(BaseLLM):
    """
    Hard-coded responses for offline development.
    Replace with OllamaLLM or OpenAILLM when RAG layer is ready.

    Pattern matching is intentionally simple — just enough to demo the loop.
    """

    _DARK_FANTASY_PLAN = {
        "intent": "recommend",
        "entities": {
            "genre": ["dark fantasy", "action"],
            "mood": ["ominous", "epic", "dark"],
            "year_range": None,
            "composer": None,
            "anime_title": None,
            "media_type": None,
            "min_rating": None,
        },
        "ambiguity": "clear",
        "selected_tool": "recommend_tool",
        "retrieval_strategy": "semantic",
        "rewritten_query": (
            "dark fantasy anime with ominous orchestral gothic soundtrack "
            "mature themes medieval supernatural"
        ),
        "reasoning": "Query clearly requests dark fantasy genre with music focus → semantic recommend_tool.",
        "confidence": 0.92,
    }

    _COMPARE_PLAN = {
        "intent": "compare",
        "entities": {
            "genre": [],
            "mood": [],
            "year_range": None,
            "composer": None,
            "anime_title": None,
            "media_type": None,
            "min_rating": None,
        },
        "ambiguity": "clear",
        "selected_tool": "compare_tool",
        "retrieval_strategy": "hybrid",
        "rewritten_query": "compare anime OST music style direction animation",
        "reasoning": "User wants a side-by-side comparison → compare_tool.",
        "confidence": 0.88,
    }

    _FILTER_PLAN = {
        "intent": "filter",
        "entities": {
            "genre": [],
            "mood": [],
            "year_range": [2015, 2024],
            "composer": None,
            "anime_title": None,
            "media_type": "TV",
            "min_rating": 8.0,
        },
        "ambiguity": "clear",
        "selected_tool": "filter_tool",
        "retrieval_strategy": "skip",
        "rewritten_query": "",
        "reasoning": "Query specifies year range + type + rating → structured filter_tool, skip RAG.",
        "confidence": 0.95,
    }

    _MUSIC_PLAN = {
        "intent": "music_lookup",
        "entities": {
            "genre": [],
            "mood": [],
            "year_range": None,
            "composer": "Yuki Kajiura",
            "anime_title": None,
            "media_type": None,
            "min_rating": None,
        },
        "ambiguity": "clear",
        "selected_tool": "music_tool",
        "retrieval_strategy": "keyword",
        "rewritten_query": "Yuki Kajiura anime OST composer",
        "reasoning": "Explicit composer name → music_tool keyword search.",
        "confidence": 0.97,
    }

    _FALLBACK_PLAN = {
        "intent": "recommend",
        "entities": {
            "genre": ["anime"],
            "mood": [],
            "year_range": None,
            "composer": None,
            "anime_title": None,
            "media_type": None,
            "min_rating": None,
        },
        "ambiguity": "vague",
        "selected_tool": "recommend_tool",
        "retrieval_strategy": "hybrid",
        "rewritten_query": "anime recommendation",
        "reasoning": "Vague query — using hybrid strategy for broader recall.",
        "confidence": 0.55,
    }

    async def complete(
        self,
        system_prompt: str,
        user_message:  str,
        temperature:   float = 0.0,
        max_tokens:    int   = 1024,
    ) -> str:
        msg = user_message.lower()

        if any(kw in msg for kw in ["dark", "แฟนตาซีดาร์ก", "fantasy", "ominous", "gothic"]):
            plan = self._DARK_FANTASY_PLAN
        elif any(kw in msg for kw in ["compare", "เปรียบ", "vs", "versus", "difference"]):
            plan = self._COMPARE_PLAN
        elif any(kw in msg for kw in ["year", "ปี", "rating", "tv", "movie", "filter", "กรอง"]):
            plan = self._FILTER_PLAN
        elif any(kw in msg for kw in ["composer", "kajiura", "hirasawa", "music", "เพลง", "แต่ง"]):
            plan = self._MUSIC_PLAN
        else:
            plan = self._FALLBACK_PLAN

        logger.debug("[MockLLM] selected plan: %s", plan["selected_tool"])
        return json.dumps(plan, ensure_ascii=False)


# ─────────────────────────────────────────────
#  Ollama LLM (plug-in when ready)
# ─────────────────────────────────────────────

class OllamaLLM(BaseLLM):
    """
    Real implementation using a local Ollama server.

    Usage:
        llm = OllamaLLM(model="qwen2.5:7b", base_url="http://localhost:11434")
        planner = Planner(llm=llm)

    Install: pip install ollama
    """

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self.model    = model
        self.base_url = base_url

    async def complete(
        self,
        system_prompt: str,
        user_message:  str,
        temperature:   float = 0.0,
        max_tokens:    int   = 1024,
    ) -> str:
        try:
            import ollama  # pip install ollama
        except ImportError:
            raise RuntimeError("Install ollama: pip install ollama")

        client   = ollama.AsyncClient(host=self.base_url)
        response = await client.chat(
            model=self.model,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message},
            ],
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["message"]["content"]


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
    """Validate and coerce raw dict into a typed ActionPlan."""
    entities_raw = data.get("entities", {})
    entities = ExtractedEntities(
        genre       = entities_raw.get("genre", []),
        mood        = entities_raw.get("mood", []),
        year_range  = entities_raw.get("year_range"),
        composer    = entities_raw.get("composer"),
        anime_title = entities_raw.get("anime_title"),
        media_type  = entities_raw.get("media_type"),
        min_rating  = entities_raw.get("min_rating"),
    )

    return ActionPlan(
        intent             = Intent(data.get("intent", "unknown")),
        entities           = entities,
        ambiguity          = data.get("ambiguity", "vague"),
        selected_tool      = ToolName(data.get("selected_tool", "none")),
        retrieval_strategy = RetrievalStrategy(data.get("retrieval_strategy", "semantic")),
        rewritten_query    = data.get("rewritten_query", ""),
        reasoning          = data.get("reasoning", ""),
        confidence         = float(data.get("confidence", 0.5)),
    )


# ─────────────────────────────────────────────
#  Planner class
# ─────────────────────────────────────────────

class Planner:
    """
    Orchestrates the Analyze → Decide phase.

    1. Builds system prompt with self-correction context from AgentState
    2. Calls the LLM backend (injected — any BaseLLM implementation)
    3. Parses + validates the JSON response into ActionPlan
    """

    def __init__(self, llm: BaseLLM | None = None):
        self.llm: BaseLLM = llm or MockLLM()

    async def plan(
        self,
        current_query:       str,
        context_summary:     str,
        conversation_history: list[dict],
    ) -> ActionPlan:
        """
        Main entry point.
        Returns a fully validated ActionPlan or raises on parse failure.
        """
        system_prompt = build_planning_prompt(context_summary)
        user_message  = build_planning_user_message(current_query, conversation_history)

        logger.info("[Planner] calling LLM | query=%r | loop_context=%r",
                    current_query[:60], context_summary[:80])

        raw_response = await self.llm.complete(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.0,   # deterministic for planning
        )

        logger.debug("[Planner] raw LLM response: %s", raw_response[:200])

        try:
            data = _parse_json_safe(raw_response)
            plan = _json_to_action_plan(data)
        except (ValueError, KeyError) as exc:
            logger.warning("[Planner] parse failed (%s) — using fallback plan", exc)
            plan = ActionPlan(
                intent=Intent.UNKNOWN,
                ambiguity="vague",
                selected_tool=ToolName.RECOMMEND,
                retrieval_strategy=RetrievalStrategy.HYBRID,
                rewritten_query=current_query,
                reasoning="Parse failed — defaulting to hybrid recommend.",
                confidence=0.3,
            )

        logger.info(
            "[Planner] → tool=%s strategy=%s confidence=%.2f",
            plan.selected_tool, plan.retrieval_strategy, plan.confidence,
        )
        return plan