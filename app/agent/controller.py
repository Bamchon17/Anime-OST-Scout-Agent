# Main runner 
# app/agent/controller.py
"""
AgentController — the ReAct loop.

  ┌──────────────┐
  │  User Query  │
  └──────┬───────┘
         │
  ┌──────▼────────────────────────────────┐
  │  LOOP (max N iterations)              │
  │                                       │
  │  1. REASON  → Planner.plan()          │
  │  2. DECIDE  → ActionPlan selected     │
  │  3. ACT     → _act() dispatches       │
  │               via tool_registry       │
  │  4. OBSERVE → _observe() evaluates    │
  │               → SATISFIED or RETRY    │
  └──────────────────────────────────────┘
         │
  ┌──────▼───────┐
  │ Final Answer │
  └──────────────┘

tool_registry is injected at construction time.
When the RAG + Tools layer is ready, pass the real registry in.
Until then, MockToolRegistry is used automatically.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.agent.planner import BaseLLM, MockLLM, Planner
from app.agent.prompts import build_synthesis_prompt, build_synthesis_user_message
from app.agent.state import (
    ActionPlan,
    AgentState,
    Observation,
    ObservationStatus,
    RetrievalStrategy,
    ToolName,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Tool Registry Interface
# ─────────────────────────────────────────────

class BaseToolRegistry(ABC):
    """
    Abstract tool registry.

    Implement this in the Tools/RAG layer:
        class AnimeToolRegistry(BaseToolRegistry):
            async def execute(self, tool_name, plan, state): ...
    """

    @abstractmethod
    async def execute(
        self,
        tool_name: ToolName,
        plan:      ActionPlan,
        state:     AgentState,
    ) -> dict[str, Any]:
        """
        Execute the selected tool and return raw results.

        Expected return shape:
        {
            "chunks":   [ { "content": str, "score": float, "metadata": dict } ],
            "filtered": [ { ...row data... } ],   # filter_tool only
            "music":    [ { ...row data... } ],   # music_tool only
        }
        At minimum return {"chunks": [], "filtered": [], "music": []}.
        """
        ...


class MockToolRegistry(BaseToolRegistry):
    """
    Stub registry for offline development.
    Returns realistic-looking fake data so the loop runs end-to-end.

    Replace with AnimeToolRegistry(BaseToolRegistry) once RAG layer is ready.
    """

    _MOCK_CHUNKS = [
        {
            "content": (
                "Berserk (1997) | 剣風伝奇ベルセルク\n"
                "Synopsis: In a dark medieval world, lone mercenary Guts battles demons "
                "and fate in a brutal journey of survival and betrayal.\n"
                "Tags: Dark Fantasy, Action, Seinen, Psychological, Gore, Tragedy\n"
                "Music: Susumu Hirasawa — Electronic, Rock, Ethereal, Folk-influenced\n"
                "Resources: https://myanimelist.net/anime/33/Berserk"
            ),
            "score": 0.94,
            "metadata": {
                "Main Title": "剣風伝奇ベルセルク",
                "Official Title (en)": "Berserk",
                "Year": 1997,
                "Max Rating": 8.6,
                "Type": "TV",
                "Music": "Susumu Hirasawa",
                "Resources": "https://myanimelist.net/anime/33/Berserk",
            },
        },
        {
            "content": (
                "Claymore (2007) | クレイモア\n"
                "Synopsis: Half-human warriors fight Yoma demons while struggling with "
                "their own dark nature in a grim medieval world.\n"
                "Tags: Dark Fantasy, Action, Seinen, Strong Female Lead, Supernatural\n"
                "Music: Masashi Hamauzu — Orchestral, Choir, Gothic, Dramatic\n"
                "Resources: https://myanimelist.net/anime/1818/Claymore"
            ),
            "score": 0.89,
            "metadata": {
                "Main Title": "クレイモア",
                "Official Title (en)": "Claymore",
                "Year": 2007,
                "Max Rating": 8.0,
                "Type": "TV",
                "Music": "Masashi Hamauzu",
                "Resources": "https://myanimelist.net/anime/1818/Claymore",
            },
        },
        {
            "content": (
                "Vinland Saga (2019) | ヴィンランド・サガ\n"
                "Synopsis: A young Viking warrior chases revenge across brutal battlefields, "
                "slowly realising vengeance is destroying him.\n"
                "Tags: Dark Fantasy, Historical, Seinen, Viking, War, Philosophical\n"
                "Music: Yutaka Yamada — Celtic, Nordic Folk, Orchestral, Atmospheric\n"
                "Resources: https://myanimelist.net/anime/37521/Vinland_Saga"
            ),
            "score": 0.86,
            "metadata": {
                "Main Title": "ヴィンランド・サガ",
                "Official Title (en)": "Vinland Saga",
                "Year": 2019,
                "Max Rating": 8.7,
                "Type": "TV",
                "Music": "Yutaka Yamada",
                "Resources": "https://myanimelist.net/anime/37521/Vinland_Saga",
            },
        },
    ]

    async def execute(
        self,
        tool_name: ToolName,
        plan:      ActionPlan,
        state:     AgentState,
    ) -> dict[str, Any]:
        logger.debug("[MockToolRegistry] executing %s", tool_name)

        if tool_name == ToolName.FILTER:
            # Simulate structured DB filter
            yr    = plan.entities.year_range
            mtype = plan.entities.media_type
            rows  = [
                c["metadata"] for c in self._MOCK_CHUNKS
                if (yr is None or yr[0] <= c["metadata"]["Year"] <= yr[1])
                and (mtype is None or c["metadata"]["Type"] == mtype)
            ]
            return {"chunks": [], "filtered": rows, "music": []}

        if tool_name == ToolName.MUSIC:
            composer = (plan.entities.composer or "").lower()
            music_rows = [
                c["metadata"] for c in self._MOCK_CHUNKS
                if composer in c["metadata"].get("Music", "").lower()
            ]
            return {"chunks": self._MOCK_CHUNKS[:1], "filtered": [], "music": music_rows}

        # recommend_tool and compare_tool both use semantic chunks
        return {"chunks": self._MOCK_CHUNKS, "filtered": [], "music": []}


# ─────────────────────────────────────────────
#  Observe thresholds (tune these per project)
# ─────────────────────────────────────────────

OBSERVE_MIN_CHUNKS      = 1      # minimum retrieved chunks to be satisfied
OBSERVE_MIN_SCORE       = 0.70   # minimum top similarity score
OBSERVE_REQUIRE_MUSIC   = False  # set True to require Music metadata always


# ─────────────────────────────────────────────
#  AgentController
# ─────────────────────────────────────────────

class AgentController:
    """
    Runs the full ReAct loop and returns the final AgentState.

    Inject dependencies at construction:
        controller = AgentController(
            planner       = Planner(llm=OllamaLLM(model="qwen2.5:7b")),
            tool_registry = AnimeToolRegistry(),   # real RAG layer
            synthesis_llm = OllamaLLM(model="qwen2.5:7b"),
        )
    """

    def __init__(
        self,
        planner:        Planner | None          = None,
        tool_registry:  BaseToolRegistry | None = None,
        synthesis_llm:  BaseLLM | None          = None,
    ):
        self.planner       = planner       or Planner(llm=MockLLM())
        self.tool_registry = tool_registry or MockToolRegistry()
        self.synthesis_llm = synthesis_llm or MockLLM()

    # ── Public entry point ──────────────────────────────────────────

    async def run(self, query: str, state: AgentState | None = None) -> AgentState:
        """
        Run the full agent loop for a user query.

        Args:
            query: Raw user input string.
            state: Optional existing AgentState (for multi-turn conversations).
                   If None, a fresh state is created.

        Returns:
            Completed AgentState with final_answer populated.
        """
        if state is None:
            state = AgentState(original_query=query, current_query=query)
        else:
            state.original_query = query
            state.current_query  = query

        state.add_turn("user", query)

        logger.info("═" * 60)
        logger.info("[Agent] NEW RUN | session=%s | query=%r", state.session_id, query)
        logger.info("═" * 60)

        while not state.is_done():
            state = await self._loop_once(state)

        # ── Generate final answer ──────────────────────────────────
        if state.last_observation and state.last_observation.raw_results:
            state.final_answer = await self._synthesise(state)
        else:
            state.final_answer = (
                "ขออภัย ไม่พบข้อมูลที่ตรงกับคำถามของคุณ "
                "ลองปรับคำค้นหาหรือระบุ genre / ชื่อ composer เพิ่มเติมได้เลยครับ"
            )

        state.add_turn("assistant", state.final_answer)
        logger.info("[Agent] DONE | loops=%d | answer_length=%d",
                    state.loop_count, len(state.final_answer or ""))
        return state

    # ── Single loop iteration ───────────────────────────────────────

    async def _loop_once(self, state: AgentState) -> AgentState:
        logger.info("─" * 40)
        logger.info("[Loop %d] START | query=%r", state.loop_count, state.current_query[:60])

        # ── 1 + 2: REASON + DECIDE ─────────────────────────────────
        plan = await self.planner.plan(
            current_query        = state.current_query,
            context_summary      = state.build_context_summary(),
            conversation_history = [t.dict() for t in state.conversation],
        )
        state.current_plan = plan

        logger.info(
            "[Loop %d] PLAN → tool=%s strategy=%s rewritten=%r confidence=%.2f",
            state.loop_count, plan.selected_tool, plan.retrieval_strategy,
            plan.rewritten_query[:60], plan.confidence,
        )

        # ── 3: ACT ─────────────────────────────────────────────────
        raw_results = await self._act(plan, state)
        state.tool_calls_made.append(plan.selected_tool.value)

        # ── 4: OBSERVE ─────────────────────────────────────────────
        observation = self._observe(raw_results, state.loop_count)
        state.record_loop(plan, observation)

        logger.info(
            "[Loop %d] OBSERVE → status=%s chunks=%d top_score=%.2f feedback=%s",
            state.loop_count - 1,
            observation.status,
            observation.chunk_count,
            observation.top_score,
            observation.feedback,
        )

        # ── Self-correction: rewrite query if retrying ──────────────
        if observation.status == ObservationStatus.RETRY:
            state.current_query = self._apply_feedback(
                state.current_query, observation.feedback
            )
            logger.info("[Loop %d] RETRY | new_query=%r",
                        state.loop_count, state.current_query[:60])

        return state

    # ── Act: dispatch to tool registry ─────────────────────────────

    async def _act(self, plan: ActionPlan, state: AgentState) -> dict[str, Any]:
        """
        Generic dispatch layer.
        All tool calls go through tool_registry.execute().
        No business logic here — keeps controller decoupled from data layer.
        """
        if plan.selected_tool == ToolName.NONE or plan.retrieval_strategy == RetrievalStrategy.SKIP:
            if plan.selected_tool == ToolName.FILTER:
                # Filter tool can skip RAG but still needs structured query
                return await self.tool_registry.execute(ToolName.FILTER, plan, state)
            # Truly no retrieval needed
            logger.info("[Act] skipping tool execution (none/skip)")
            return {"chunks": [], "filtered": [], "music": []}

        return await self.tool_registry.execute(plan.selected_tool, plan, state)

    # ── Observe: evaluate results quality ─────────────────────────

    def _observe(self, results: dict[str, Any], loop_index: int) -> Observation:
        chunks   = results.get("chunks", [])
        filtered = results.get("filtered", [])
        music    = results.get("music", [])

        chunk_count    = len(chunks)
        filtered_count = len(filtered)
        top_score      = max((c.get("score", 0) for c in chunks), default=0.0)
        has_music      = any(c.get("metadata", {}).get("Music") for c in chunks) or bool(music)

        feedback: list[str] = []

        # ── Satisfaction criteria ────────────────────────────────
        enough_results = (chunk_count >= OBSERVE_MIN_CHUNKS) or (filtered_count > 0)
        good_score     = (top_score >= OBSERVE_MIN_SCORE) or (chunk_count == 0 and filtered_count > 0)
        music_ok       = (not OBSERVE_REQUIRE_MUSIC) or has_music

        if not enough_results:
            feedback.append("Too few results — broaden the query or switch to hybrid strategy")
        if chunk_count > 0 and not good_score:
            feedback.append(f"Low similarity score ({top_score:.2f}) — rewrite query with more specific terms")
        if OBSERVE_REQUIRE_MUSIC and not has_music:
            feedback.append("Missing music data — switch to music_tool")

        if loop_index >= 2:
            # Hard exit after max loops regardless of quality
            status = ObservationStatus.FAILED if not enough_results else ObservationStatus.SATISFIED
        elif enough_results and good_score and music_ok:
            status = ObservationStatus.SATISFIED
        else:
            status = ObservationStatus.RETRY

        return Observation(
            loop_index=loop_index,
            status=status,
            chunk_count=chunk_count + filtered_count,
            top_score=top_score,
            has_music_data=has_music,
            feedback=feedback,
            raw_results=results,
        )

    # ── Self-correction: update query from feedback ────────────────

    @staticmethod
    def _apply_feedback(current_query: str, feedback: list[str]) -> str:
        """
        Append feedback hints to the query so the LLM can adjust its plan.
        This is injected back into the Planner's context_summary on next loop.
        """
        if not feedback:
            return current_query
        hint = "; ".join(feedback)
        return f"{current_query}\n[Self-correction hint: {hint}]"

    # ── Synthesis: generate final answer ──────────────────────────

    async def _synthesise(self, state: AgentState) -> str:
        """
        Call LLM one final time to produce a natural-language answer
        from all retrieved evidence.
        """
        if state.last_observation is None:
            return "ไม่พบข้อมูล"

        results = state.last_observation.raw_results
        chunks  = results.get("chunks", [])
        others  = {k: v for k, v in results.items() if k != "chunks"}

        system_prompt = build_synthesis_prompt()
        user_message  = build_synthesis_user_message(
            original_query   = state.original_query,
            retrieved_chunks = chunks,
            tool_results     = others,
        )

        raw = await self.synthesis_llm.complete(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.4,   # slight creativity for fluent prose
            max_tokens=1200,
        )

        # MockLLM returns JSON, not prose — handle gracefully
        if raw.lstrip().startswith("{"):
            return self._format_chunks_as_text(chunks, state.original_query)
        return raw

    @staticmethod
    def _format_chunks_as_text(chunks: list[dict], query: str) -> str:
        """Fallback formatter when synthesis LLM returns structured data."""
        if not chunks:
            return "ไม่พบอนิเมะที่ตรงกับเงื่อนไข"
        lines = [f"ผลลัพธ์สำหรับ: {query}\n"]
        for i, c in enumerate(chunks, 1):
            meta = c.get("metadata", {})
            lines.append(
                f"{i}. {meta.get('Official Title (en)', 'Unknown')} "
                f"({meta.get('Year', '?')}) — "
                f"Rating: {meta.get('Max Rating', '?')} | "
                f"Music: {meta.get('Music', 'N/A')}\n"
                f"   Source: {meta.get('Resources', 'N/A')}"
            )
        return "\n".join(lines)