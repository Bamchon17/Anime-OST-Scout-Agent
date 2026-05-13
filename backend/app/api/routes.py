from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel

from app.agent.controller import AgentController
from app.agent.planner import BaseLLM, Planner
from app.agent.state import AgentState, ToolName
from app.models.llm import llm_service
from app.tools import compare_tool, filter_tool, semantic_tools


router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    topK: int = 4


class AnimeToolRegistry:
    async def execute(self, tool_name: ToolName, plan: Any, state: AgentState) -> Dict[str, Any]:
        if tool_name == ToolName.RECOMMEND_TOOL:
            exact_record = find_explicit_title_match(state.original_query)
            if exact_record:
                state.log_step(f"[Registry] Exact title match: {exact_record.get('title_en')}")
                return {
                    "tool": "semantic_search",
                    "query": plan.rewritten_query or state.original_query,
                    "results": [record_to_result(exact_record, score=1.0)],
                }

            state.log_step(f"[Registry] Running Semantic Search: {plan.rewritten_query}")
            return semantic_tools.run(query=plan.rewritten_query or state.original_query, top_k=plan.entities.rating or 3)

        if tool_name == ToolName.FILTER_TOOL:
            import dataclasses

            all_args = dataclasses.asdict(plan.entities)
            valid_keys = [
                "year",
                "year_from",
                "year_to",
                "rating_min",
                "rating_max",
                "type",
                "season",
                "studio",
                "tags",
                "music_style",
            ]

            clean_args = {
                key: value
                for key, value in all_args.items()
                if value is not None and value != [] and key in valid_keys
            }

            clean_args["top_k"] = 5
            state.log_step(f"[Registry] Running Filter Tool: {clean_args}")
            return filter_tool.run(**clean_args)

        if tool_name == ToolName.COMPARE_TOOL:
            titles = []

            if isinstance(plan.entities.anime_title, list):
                titles = plan.entities.anime_title
            elif isinstance(plan.entities.anime_title, str):
                titles = [title.strip() for title in plan.entities.anime_title.split(",") if title.strip()]

            state.log_step(f"[Registry] Running Compare Tool: {titles}")
            return compare_tool.run(titles=titles)

        return {
            "tool": "unknown",
            "data": [],
            "error": f"Tool not found: {tool_name}",
        }


def find_explicit_title_match(query: str) -> dict | None:
    q_lower = query.lower()

    related_markers = ["คล้าย", "แนว", "แบบ", "similar", "like", "เหมือน"]
    if any(marker in q_lower for marker in related_markers):
        return None

    for record in compare_tool._get_records():
        candidates = [
            record.get("title_en", ""),
            record.get("title", ""),
            record.get("title_ja", ""),
        ]

        for candidate in [c for c in candidates if c]:
            candidate_lower = candidate.lower()
            if len(candidate_lower) >= 3 and candidate_lower in q_lower:
                return record

    return None


def record_to_result(record: dict, score: float = 1.0) -> dict:
    return {
        "chunk_id": record.get("chunk_id", ""),
        "title_en": record.get("title_en", ""),
        "title": record.get("title", ""),
        "rating": record.get("rating", 0),
        "year": record.get("year", ""),
        "type": record.get("type", ""),
        "tags": record.get("filter_meta", {}).get("tags", []),
        "synopsis": record.get("synopsis", ""),
        "score": score,
        "rerank_reason": "Exact title match from the user query.",
        "image_url": record.get("image_url", ""),
        "mal_url": record.get("mal_url", ""),
    }


class TyphoonLLMWrapper(BaseLLM):
    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        return await llm_service.chat(
            system_prompt=system_prompt,
            user_prompt=user_message,
            temperature=temperature,
        )


def normalize_results(items: List[dict], limit: int) -> List[dict]:
    results = []

    for item in items[:limit]:
        title = (
            item.get("title_en")
            or item.get("title")
            or item.get("name")
            or item.get("anime_title")
            or "Unknown Anime"
        )

        score = item.get("score", item.get("match", 0.8))

        try:
            raw_score = float(score)

            # normalize similarity score
            if raw_score <= 1:
                match = int(raw_score * 100)
            else:
                match = int(raw_score)

            # boost low semantic scores for UI readability
            if match < 40:
                match = 70 + int(match * 0.4)

            # clamp range
            match = max(70, min(98, match))

        except Exception:
            match = 85

        url = (
            item.get("mal_url")
            or item.get("url")
            or item.get("href")
            or f"https://www.youtube.com/results?search_query={title.replace(' ', '+')}+OST"
        )

        results.append(
            {
                "title": title,
                "match": match,
                "url": url,
                "reason": item.get("rerank_reason", item.get("synopsis", "")),
                "image_url": item.get("image_url", ""),
            }
        )

    return results


def normalize_trace(state: AgentState) -> List[dict]:
    trace = []

    for entry in state.loop_history:
        plan = entry["plan"]
        observation = entry["observation"]

        trace.append(
            {
                "step": plan.selected_tool.value,
                "status": observation.status.value,
                "reasoning": plan.reasoning,
                "top_score": observation.top_score,
                "feedback": observation.feedback,
                "duration_ms": observation.duration_ms,
                "duration_breakdown": {
                    "planning_ms": observation.planning_duration_ms,
                    "tool_ms": observation.action_duration_ms,
                    "observe_ms": observation.observe_duration_ms,
                },
            }
        )

    if state.final_answer:
        trace.append(
            {
                "step": "LLM Generation",
                "status": "success",
                "reasoning": "สร้างคำตอบสุดท้ายจากผลลัพธ์ของ Agent",
                "top_score": 1,
                "feedback": "",
                "duration_ms": state.final_response_duration_ms,
                "duration_breakdown": {
                    "generation_ms": state.final_response_duration_ms,
                },
            }
        )

    return trace


@router.post("/chat")
async def chat(req: ChatRequest):
    llm = TyphoonLLMWrapper()
    planner = Planner(llm=llm)
    registry = AnimeToolRegistry()
    controller = AgentController(
        planner=planner,
        tool_registry=registry,
        responder_llm=llm,
    )

    state = AgentState()
    final_state = await controller.run(req.query, state)

    retrieved_data = []
    if final_state.last_observation:
        retrieved_data = final_state.last_observation.retrieved_data

    return {
        "query": req.query,
        "answer": final_state.final_answer,
        "results": normalize_results(retrieved_data, req.topK),
        "trace": normalize_trace(final_state),
        "logs": [
            {
                "type": "agent",
                "message": f"Loop {entry['loop']}: {entry['plan'].selected_tool.value} -> {entry['observation'].status.value}",
            }
            for entry in final_state.loop_history
        ],
    }
