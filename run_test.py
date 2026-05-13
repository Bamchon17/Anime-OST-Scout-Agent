import asyncio
import logging
from typing import Any, Dict
import sys
import io
import os

# Fix UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from app.agent.state import AgentState, ToolName
from app.agent.planner import Planner, BaseLLM
from app.agent.controller import AgentController
from app.models.llm import llm_service 

from app.tools import semantic_tools 
from app.tools import filter_tool
from app.tools import compare_tool

class AnimeToolRegistry:
    async def execute(self, tool_name: ToolName, plan: Any, state: Any) -> Dict[str, Any]:
        if tool_name == ToolName.RECOMMEND_TOOL:
            state.log_step(f"[Registry] Running Semantic Search for: {plan.rewritten_query}")
            return semantic_tools.run(query=plan.rewritten_query, top_k=3)
            
        elif tool_name == ToolName.FILTER_TOOL:
            import dataclasses
            # 1. แปลง entities เป็น dict
            all_args = dataclasses.asdict(plan.entities)
            
            valid_keys = [
                'year', 'year_from', 'year_to', 'rating_min', 
                'type', 'season', 'studio', 'tags'
            ]
            
            # กรองเอาเฉพาะที่มีค่า (not None) และอยู่ใน list ที่ tool รองรับ
            clean_args = {
                k: v for k, v in all_args.items() 
                if v is not None and k in valid_keys
            }
            
            state.log_step(f"[Registry] Running Filter with: {clean_args}")
            return filter_tool.run(**clean_args)
            
        elif tool_name == ToolName.COMPARE_TOOL:
            # ตรวจสอบว่าใน entities มี anime_title หรือไม่ (ถ้าไม่มีให้ใช้ rewritten_query)
            # Handle both list (from planner) and string formats
            if isinstance(plan.entities.anime_title, list):
                titles = plan.entities.anime_title
            else:
                titles = plan.entities.anime_title.split(",") if plan.entities.anime_title else []
            state.log_step(f"[Registry] Comparing: {titles}")
            return compare_tool.run(titles=titles)
            
        return {"data": [], "error": "Tool not found"}

# --- Mock LLM สำหรับทดสอบกรณีอยากประหยัด Token (เลือกเปลี่ยนเป็นของจริงได้) ---
class TyphoonLLMWrapper(BaseLLM):
    async def complete(self, system_prompt, user_message, temperature=0.0):
        # ตรงนี้คือการเรียกใช้งาน API ห
        return await llm_service.chat(system_prompt, user_message, temperature)

async def start_demo():
    # 1. เตรียมระบบ
    llm = TyphoonLLMWrapper()
    planner = Planner(llm=llm)
    registry = AnimeToolRegistry()
    controller = AgentController(planner, registry, responder_llm=llm)
    
    # 2. จำลองคำถามจาก User
    test_queries = [
         "แนะนำเรื่องคลาสสิค แนวคล้ายๆcowboy bebopไหม", #semantic_search 
        # "มีเรื่องไหนที่ใช้เพลงแนว Hip-Hop บ้าง?", #music_lookup
        # "เปรียบเทียบ Naruto กับ Bleach", #compare_anime
        # "อนิเมะที่ออกช่วง spring ที่มีคะแนนรีวิวมากกว่า 8" #filter_search
    ]
    
    for query in test_queries:
        print(f"\n{'='*20} TESTING QUERY: {query} {'='*20}")
        state = AgentState()
        final_state = await controller.run(query, state)
        
        print("\n[Luna'S RESPONSE]")
        print(final_state.final_answer)
        print(f"{'='*60}\n")

if __name__ == "__main__":
    # มั่นใจว่าเปิด Ollama หรือเตรียม API Key ไว้แล้วนะ!
    asyncio.run(start_demo())