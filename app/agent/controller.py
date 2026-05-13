# Main runner 
# app/agent/controller.py
"""
AgentController — the ReAct loop.

  ┌──────────────┐
  │  User Query  │
  └──────┬───────┘
         │
  ┌──────▼────────────────────────────────┐
  │  LOOP (max 3 iterations)              │
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
import dataclasses
from abc import ABC, abstractmethod
from typing import Any, List, Dict

from app.rag.retrieval import retrieve_music
from app.agent.planner import BaseLLM, Planner
from app.agent.prompts import (
    build_planning_prompt, 
    build_planning_user_message,
    build_synthesis_prompt,
    build_synthesis_user_message
)
from app.agent.state import (
    AgentState,
    Observation,
    ObservationStatus,
    ToolName,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Observe Thresholds 
# ─────────────────────────────────────────────
MIN_SCORE_THRESHOLD = 0.65  
MAX_LOOPS = 3

class AgentController:
    """
    ผู้จัดการใหญ่ (Controller) คอยสั่งการ Planner -> Tools -> Responder
    และจัดการวงจร ReAct Loop (Think-Act-Observe)
    """
    def __init__(
        self,
        planner: Planner,
        tool_registry: Any,  # เดี๋ยวจะส่ง AnimeToolRegistry เข้ามา
        responder_llm: Any   # ตัวสรุปคำตอบ (เช่น Typhoon)
    ):
        self.planner = planner
        self.tool_registry = tool_registry
        self.responder_llm = responder_llm

    async def run(self, query: str, state: AgentState) -> AgentState:
        """จุดเริ่มต้นการทำงานของ Agent"""
        state.original_query = query
        state.log_step(f"รับคำถาม: {query}")

        # เริ่มต้น Loop การคิดและทำ
        while not state.is_done():
            state.log_step(f"--- เริ่มต้นรอบที่ {state.loop_count + 1} ---")
            
            # 1. THINK (วางแผน)
            state.log_step("กำลังวางแผนการทำงาน...")
            plan = await self.planner.plan(
                current_query=query,
                context_summary=state.build_context_summary(),
                conversation_history=state.conversation,
                loop_count=state.loop_count # ส่ง loop_count เพิ่ม
            )
            state.current_plan = plan
            state.log_step(f"แผนที่วางไว้: ใช้ {plan.selected_tool.value} เพราะ {plan.reasoning}")

            # 2. ACT (ลงมือทำ)
            state.log_step(f"กำลังเรียกใช้เครื่องมือ: {plan.selected_tool.value}...")
            raw_results = await self._execute_action(plan, state)

            # 3. OBSERVE (สังเกตผล)
            observation = self._observe(raw_results, state.loop_count)
            state.record_result(plan, observation)
            
            if observation.status == ObservationStatus.SATISFIED:
                state.log_step(f"สำเร็จ! พบข้อมูล {len(observation.retrieved_data)} รายการ")
            elif observation.status == ObservationStatus.RETRY:
                state.log_step(f"ยังไม่พอใจ: {observation.feedback} (เตรียมเริ่มรอบใหม่)")

        # 4. RESPOND (สรุปคำตอบ)
        if state.last_observation and state.last_observation.retrieved_data:
            state.log_step("กำลังสรุปคำตอบสุดท้ายจากข้อมูลที่พบ...")
            state.final_answer = await self._synthesize(state)
        else:
            state.final_answer = "ขอโทษน่า ลูน่าพยายามหาข้อมูลแล้วแต่ไม่พบจริงๆ ลองเปลี่ยนคำถามดูไหมคะ?"

        state.log_step("จบการทำงาน")
        return state
    
    async def _execute_action(self, plan: Any, state: AgentState) -> Dict[str, Any]:
        """จัดการการเรียก Tool หรือ Retrieval โดยตรง"""
        tool_name = plan.selected_tool
        
        # กรณีพิเศษ: music_lookup ที่ไม่มีไฟล์ Tool แยก
        if tool_name == ToolName.MUSIC_TOOL:
            state.log_step("กำลังค้นหาจาก music.faiss โดยตรง...")
            # เรียกจาก retrieval.py
            results = retrieve_music(plan.rewritten_query, top_k=3)
            return {"data": results}
        
        try:
            return await self.tool_registry.execute(tool_name, plan, state)
        except Exception as e:
            logger.error(f"Tool Error: {e}")
            return {"data": [], "error": str(e)}

    def _observe(self, raw_results: Dict[str, Any], loop_index: int) -> Observation:
        retrieved_items = raw_results.get("data", raw_results.get("results", []))
        
        # 2. ระบุว่าข้อมูลนี้มาจาก Tool ไหน
        tool_used = raw_results.get("tool", "")
        
        # --- [PERSPECTIVE: คู่หูคิดสายลุย] ---
        is_comparison = tool_used == "compare_anime"
        if is_comparison and "comparison" in raw_results:
            retrieved_items = [raw_results["comparison"]]

        # 3. คำนวณคะแนนสูงสุด
        top_score = max([item.get("score", 0) for item in retrieved_items if isinstance(item, dict)], default=0.0)
        
        status = ObservationStatus.SATISFIED
        feedback = ""

        # --- กรณีไม่พบข้อมูลเลย ---
        if not retrieved_items:
            status = ObservationStatus.RETRY
            feedback = "ไม่พบข้อมูลที่ตรงกับเงื่อนไข"
            
        # --- กรณีพิเศษสำหรับ Filter Search และ Comparison ---
        # ถ้าเจอข้อมูลจากการเปรียบเทียบ หรือการกรอง ให้ผ่านทันที
        elif tool_used in ["filter_search", "compare_anime"]:
            status = ObservationStatus.SATISFIED
            feedback = f"สำเร็จ! พบข้อมูลจาก {tool_used}"
            
        # --- กรณีการค้นหาทั่วไป (Semantic/Music) ---
        elif top_score < MIN_SCORE_THRESHOLD and loop_index < MAX_LOOPS - 1:
            status = ObservationStatus.RETRY
            feedback = f"ความเกี่ยวข้องต่ำ ({top_score:.2f})"

        # --- ตรวจสอบรอบการทำงานสุดท้าย ---
        if loop_index >= MAX_LOOPS - 1:
            status = ObservationStatus.SATISFIED if retrieved_items else ObservationStatus.FAILED

        return Observation(
            loop_index=loop_index,
            status=status,
            retrieved_data=retrieved_items,
            top_score=top_score,
            feedback=feedback
        )
    async def _synthesize(self, state: AgentState) -> str:
        """เรียก LLM สรุปคำตอบสุดท้าย (Synthesis)"""
        system_prompt = build_synthesis_prompt()
        user_message = build_synthesis_user_message(
            original_query=state.original_query,
            retrieved_data=state.last_observation.retrieved_data
        )

        response = await self.responder_llm.complete(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.7 # เพิ่มนิดนึงเพื่อให้ดูเป็นธรรมชาติ
        )
        return response
