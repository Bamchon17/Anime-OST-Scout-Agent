#!/usr/bin/env python3
# main.py
"""
Demo runner — shows the full ReAct loop end-to-end.
Run: python main.py

Switch to real Ollama by uncommenting OllamaLLM lines below.
"""

import asyncio
import logging
import sys
import os

# Make app importable from project root
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)

from app.agent.controller import AgentController
from app.agent.planner import MockLLM, OllamaLLM, Planner
from app.agent.state import AgentState


async def demo(query: str) -> None:
    print("\n" + "═" * 60)
    print(f"  QUERY: {query}")
    print("═" * 60)

    # ── Wire up dependencies ───────────────────────────────────
    # Mock mode (default — no external services needed)
    controller = AgentController()

    # Real Ollama mode (uncomment when RAG layer is ready):
    # llm = OllamaLLM(model="qwen2.5:7b")
    # controller = AgentController(
    #     planner       = Planner(llm=llm),
    #     synthesis_llm = llm,
    #     tool_registry = AnimeToolRegistry(),   # your implementation
    # )

    state: AgentState = await controller.run(query)

    # ── Print summary ──────────────────────────────────────────
    print("\n┌─ AGENT SUMMARY ─────────────────────────────────────┐")
    print(f"│ Session : {state.session_id[:8]}...")
    print(f"│ Loops   : {state.loop_count}")
    print(f"│ Tools   : {', '.join(state.tool_calls_made) or 'none'}")
    if state.last_observation:
        print(f"│ Status  : {state.last_observation.status.value}")
        print(f"│ Chunks  : {state.last_observation.chunk_count}")
        print(f"│ Score   : {state.last_observation.top_score:.2f}")
    print("└─────────────────────────────────────────────────────┘")

    print("\n┌─ FINAL ANSWER ──────────────────────────────────────┐")
    for line in (state.final_answer or "No answer").splitlines():
        print(f"│ {line}")
    print("└─────────────────────────────────────────────────────┘\n")

    # ── Loop history (observability trace) ────────────────────
    if state.loop_history:
        print("┌─ LOOP TRACE ────────────────────────────────────────┐")
        for rec in state.loop_history:
            print(f"│ Loop {rec.loop_index}: "
                  f"tool={rec.plan.selected_tool.value} | "
                  f"strategy={rec.plan.retrieval_strategy.value} | "
                  f"obs={rec.observation.status.value} | "
                  f"feedback={rec.observation.feedback or 'none'}")
        print("└─────────────────────────────────────────────────────┘")


async def main() -> None:
    test_queries = [
        "หาเพลงอนิเมะแฟนตาซีดาร์กๆ",
        "เปรียบเทียบ Berserk กับ Claymore",
        "อนิเมะ TV ที่ออกหลังปี 2015 rating เกิน 8",
        "หาอนิเมะที่แต่งเพลงโดย Yuki Kajiura",
    ]

    for q in test_queries:
        await demo(q)
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())