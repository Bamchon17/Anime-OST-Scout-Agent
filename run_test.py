import asyncio
import logging
from typing import Any, Dict, List
import sys
import io
import os
import json
from pathlib import Path

# Fix UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.agent.state import AgentState, ToolName
from app.agent.planner import Planner, BaseLLM
from app.agent.controller import AgentController
from app.models.llm import llm_service 

# นำเข้าเครื่องมือต่างๆ
from app.tools import semantic_tools 
from app.tools import filter_tool
from app.tools import compare_tool
from app.tools import music_tool
# ลบการเรียก retrieve_music โดยตรงออกเพื่อให้ผ่าน music_tool.run() เท่านั้น

class AnimeToolRegistry:
    async def execute(self, tool_name: ToolName, plan: Any, state: Any, **kwargs) -> Dict[str, Any]:
        """
        Registry ที่แก้ไขให้เรียก music_tool.run() แบบ async 
        เพื่อให้รองรับการดึงข้อมูลจาก Jikan และ YouTube API
        """
        
        # 1. MUSIC TOOL (จุดที่แก้ไขหลัก)
       # run_test.py -> ภายใน AnimeToolRegistry.execute

        # 1. MUSIC TOOL
        if tool_name == ToolName.MUSIC_TOOL:
            include_yt = kwargs.get("include_external", False)
            state.log_step(f"[Registry] 🎵 Music Search: {plan.rewritten_query} (YouTube: {include_yt})")
            
            meta_path = Path(__file__).parent / "data" / "music_meta.json"
            try:
                music_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as e:
                state.log_step(f"Warning: Could not load music_meta.json: {e}")
                music_meta = []
            
            # แก้ไข: เพิ่ม await และส่ง include_external
            result = await music_tool.run(
                plan.rewritten_query, 
                meta=music_meta, 
                include_external=include_yt, 
                top_k=3
            )
            
            if result.get("mode") == "song_request":
                if result.get("error"):
                    return {"data": [], "tool": "music_tool", "score": 0.0, "error": result["error"]}
                data_list = [result]
                score = 0.95
            else:
                # semantic mode: inject score=0.8 ให้ทุก item เพื่อไม่ให้วนลูปเพราะ score=0
                raw_list = result.get("results", [])
                data_list = [{**item, "score": 0.8} for item in raw_list]
                score = 0.8 if data_list else 0.0

            return {"data": data_list, "tool": "music_tool", "score": score}

        # 2. RECOMMEND TOOL
        elif tool_name == ToolName.RECOMMEND_TOOL:
            state.log_step(f"[Registry] 🔍 Semantic Search: {plan.rewritten_query}")
            result = semantic_tools.run(query=plan.rewritten_query, top_k=3)
            return {**result, "score": 0.85}
            
        # 3. FILTER TOOL
        elif tool_name == ToolName.FILTER_TOOL:
            import dataclasses
            all_args = dataclasses.asdict(plan.entities)
            valid_keys = ['year', 'year_from', 'year_to', 'rating_min', 'type', 'season', 'studio', 'tags']
            clean_args = {k: v for k, v in all_args.items() if v is not None and k in valid_keys}

            # ลบ tags=[] (list ว่าง) ออก
            if 'tags' in clean_args and not clean_args['tags']:
                del clean_args['tags']

            # map rating → rating_min ถ้า planner ส่งมาผิด key
            if 'rating' in all_args and all_args['rating'] and 'rating_min' not in clean_args:
                clean_args['rating_min'] = all_args['rating']

            # fallback rating_min จาก query ถ้าไม่มีเลย
            if 'rating_min' not in clean_args:
                q_lower = (state.original_query + " " + plan.rewritten_query).lower()
                import re
                m = re.search(r'(?:rating|คะแนน|score)[^\d]*([0-9]+(?:\.[0-9]+)?)', q_lower)
                if m:
                    clean_args['rating_min'] = float(m.group(1))

            # fallback season จาก query ถ้า planner ไม่ส่งมา
            if 'season' not in clean_args:
                q_lower = (state.original_query + " " + plan.rewritten_query).lower()
                for kw, val in [('spring','spring'),('ใบไม้ผลิ','spring'),
                                ('summer','summer'),('ร้อน','summer'),
                                ('fall','fall'),('autumn','fall'),('ใบไม้ร่วง','fall'),
                                ('winter','winter'),('หนาว','winter')]:
                    if kw in q_lower:
                        clean_args['season'] = val
                        break

            state.log_step(f"[Registry] ⚙️ Filtering with: {clean_args}")
            return filter_tool.run(**clean_args)
            
        # 4. COMPARE TOOL
        elif tool_name == ToolName.COMPARE_TOOL:
            titles = plan.entities.anime_title
            if not isinstance(titles, list):
                titles = titles.split(",") if titles else []
            state.log_step(f"[Registry] 📊 Comparing: {titles}")
            return compare_tool.run(titles=titles)
            
        return {"data": [], "error": "Tool not found"}

# --- LLM Wrapper ---
class TyphoonLLMWrapper(BaseLLM):
    async def complete(self, system_prompt, user_message, temperature=0.0):
        return await llm_service.chat(system_prompt, user_message, temperature)

async def start_demo():
    # 1. เตรียมระบบ
    llm = TyphoonLLMWrapper()
    planner = Planner(llm=llm)
    registry = AnimeToolRegistry()
    controller = AgentController(planner, registry, responder_llm=llm)
    
    # 2. จำลองคำถามทดสอบ
    test_queries = [
        # "แนะนำเรื่องคลาสสิค แนวคล้ายๆ cowboy bebop หน่อยตัวละครด้วย",
        # "ขอเพลงจากเรื่อง Naruto หน่อยสิ",  # ทดสอบการเรียก Jikan + YouTube
        "แนะนำเพลงแนว Heavy Metal ในอนิเมะให้หน่อย",
        "อนิเมะที่ออกช่วง spring ที่มีคะแนนรีวิวมากกว่า 8",
        # "เปรียบเทียบ Seikai no Monshou กับ Ginsoukikou Ordian"
    ]
    
    for query in test_queries:
        print(f"\n{'='*20} TESTING QUERY: {query} {'='*20}")
        state = AgentState()
        # รัน Agent ผ่าน Controller
        final_state = await controller.run(query, state)
        
        print("\n[Luna'S RESPONSE]")
        print(final_state.final_answer)
        print(f"{'='*60}\n")

if __name__ == "__main__":
    asyncio.run(start_demo())