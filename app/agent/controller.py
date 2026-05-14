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
    RetrievalStrategy, # เพิ่ม RetrievalStrategy เข้ามาเพื่อเช็คเงื่อนไข
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
        
        # กรณีพิเศษ: music_lookup
        if tool_name == ToolName.MUSIC_TOOL:
            state.log_step("กำลังค้นหาจาก music.faiss โดยตรง...")
            
            # ตรวจสอบว่าต้องการลิงก์ภายนอก (YouTube) หรือไม่จาก Retrieval Strategy
            include_yt = (plan.retrieval_strategy == RetrievalStrategy.HYBRID)
            
            try:
                # พยายามเรียกผ่าน registry โดยส่ง include_external แยกออกไปจาก plan
                return await self.tool_registry.execute(tool_name, plan, state, include_external=include_yt)
            except Exception as e:
                # Fallback: เรียก retrieve_music โดยตรง (ส่งเฉพาะ parameter ที่ฟังก์ชันรองรับ)
                logger.warning(f"Registry Music Error: {e}, falling back to direct retrieval")
                results = retrieve_music(plan.rewritten_query, top_k=3) 
                return {"data": results, "tool": "music_lookup"}
        
        try:
            return await self.tool_registry.execute(tool_name, plan, state)
        except Exception as e:
            logger.error(f"Tool Error: {e}")
            return {"data": [], "error": str(e)}

    def _observe(self, raw_results: Dict[str, Any], loop_index: int) -> Observation:
        retrieved_items = raw_results.get("data", raw_results.get("results", []))
        tool_used = raw_results.get("tool", "")

        #DEbug 
        #print(f"DEBUG: Retrieved items sample: {retrieved_items[0] if retrieved_items else 'None'}")

        # --- [PERSPECTIVE: คู่หูคิดสายลุย] ---
        # รองรับชื่อ tool ที่อาจมาได้หลายรูปแบบ
        if tool_used in ["compare_anime", "compare_tool"] and "comparison" in raw_results:
            retrieved_items = [raw_results["comparison"]]

        # คำนวณคะแนนสูงสุด
        top_score = max([item.get("score", 0) for item in retrieved_items if isinstance(item, dict)], default=0.0)
        
        status = ObservationStatus.SATISFIED
        feedback = ""

        # 1. กรณีไม่พบข้อมูลเลย
        if not retrieved_items:
            status = ObservationStatus.RETRY
            feedback = "ไม่พบข้อมูลที่ตรงกับเงื่อนไข"
            
        # 2. กรณีพิเศษสำหรับ Filter และ Comparison (ถ้ามีข้อมูลให้ผ่านทันที)
        elif tool_used in ["filter_search", "filter_tool", "compare_anime", "compare_tool"]:
            status = ObservationStatus.SATISFIED
            feedback = f"สำเร็จ! พบข้อมูลจาก {tool_used}"
            
        # 3. กรณีการค้นหาทั่วไป (Semantic/Music)
        elif top_score < MIN_SCORE_THRESHOLD:
            if loop_index < MAX_LOOPS - 1:
                status = ObservationStatus.RETRY
                feedback = f"ความเกี่ยวข้องต่ำ ({top_score:.2f})"
            else:
                # รอบสุดท้าย: ผ่อนปรนให้ SATISFIED แม้คะแนนน้อย เพื่อให้มีข้อมูลไปตอบ User
                status = ObservationStatus.SATISFIED
                feedback = f"ยอมรับข้อมูลรอบสุดท้าย (Score: {top_score:.2f})"

        # 4. ตรวจสอบรอบการทำงานสุดท้าย (กรณีบังคับหยุด)
        if loop_index >= MAX_LOOPS - 1:
            # ถ้ามีข้อมูลติดมือมาบ้าง ให้ถือว่าสำเร็จ (Satisfied) เพื่อเข้าขั้นตอนสรุปคำตอบ
            status = ObservationStatus.SATISFIED if retrieved_items else ObservationStatus.FAILED

        return Observation(
            loop_index=loop_index,
            status=status,
            retrieved_data=retrieved_items,
            top_score=top_score,
            feedback=feedback
        )

    async def _synthesize(self, state: AgentState) -> str:
        """
        สรุปคำตอบสุดท้ายจากข้อมูลที่ได้รับจาก Tool
        ใช้ build_synthesis_prompt / build_synthesis_user_message จาก prompts.py
        เพื่อให้ลิงก์ MAL, Official, YouTube แสดงครบทุกครั้ง
        """
        if not state.last_observation or not state.last_observation.retrieved_data:
            return "ขอโทษน่า ลูน่าพยายามหาข้อมูลแล้วแต่ไม่พบจริงๆ ลองเปลี่ยนคำถามดูไหมคะ?"

        retrieved_data = state.last_observation.retrieved_data
        plan = state.current_plan

        # ใช้ prompt จาก prompts.py ที่มี rule บังคับแสดงลิงก์ครบ
        synthesis_prompt  = build_synthesis_prompt()
        synthesis_message = build_synthesis_user_message(state.original_query, retrieved_data)

        try:
            final_answer = await self.responder_llm.complete(
                system_prompt=synthesis_prompt,
                user_message=synthesis_message,
                temperature=0.7
            )
            return final_answer
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return self._format_results(retrieved_data, plan.selected_tool if plan else None)

    def _format_results(self, data: list, tool_name: ToolName = None) -> str:
        """
        จัดเรียงข้อมูลให้อ่านง่าย
        """
        if not data:
            return "ไม่พบข้อมูล"
        
        if tool_name == ToolName.COMPARE_TOOL:
            # สำหรับการเปรียบเทียบ
            if isinstance(data, list) and len(data) > 0:
                comparison = data[0] if isinstance(data[0], dict) else {}
                text = "เปรียบเทียบ:\n"
                for title in comparison.get("titles", []):
                    text += f"- {title}\n"
                return text
        
        # Default: แสดงรายการเป็น bullet points
        text = ""
        for i, item in enumerate(data[:5], 1):
            if isinstance(item, dict):
                title = item.get("title") or item.get("title_en") or item.get("main_title") or "ไม่ระบุ"
                rating = item.get("rating", "N/A")
                text += f"{i}. {title} (★{rating})\n"
            else:
                text += f"{i}. {str(item)}\n"
        return text