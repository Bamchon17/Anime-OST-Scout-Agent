import asyncio
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter
from starlette.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.controller import AgentController
from app.agent.planner import BaseLLM, Planner
from app.agent.state import AgentState, ToolName
from app.models.llm import llm_service
from app.rag.youtube_search import search_youtube
from app.tools import compare_tool, filter_tool, music_tool, semantic_tools


router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    topK: int = 4


def build_controller() -> AgentController:
    llm = TyphoonLLMWrapper()
    planner = Planner(llm=llm)
    registry = AnimeToolRegistry()
    return AgentController(
        planner=planner,
        tool_registry=registry,
        responder_llm=llm,
    )


class AnimeToolRegistry:
    async def execute(
        self,
        tool_name: ToolName,
        plan: Any,
        state: AgentState,
        include_external: bool = False,
    ) -> Dict[str, Any]:
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
            state.log_step("กำลังค้นหาใน FAISS/Vector Store ด้วย semantic search...")
            result = semantic_tools.run(query=plan.rewritten_query or state.original_query, top_k=plan.entities.rating or 3)
            state.log_step(f"Semantic search ส่งผลลัพธ์กลับมา {len(result.get('results', []))} รายการ")
            return result

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
            result = filter_tool.run(**clean_args)
            state.log_step(f"Filter tool พบทั้งหมด {result.get('total_found', 0)} รายการ และส่งกลับ {result.get('returned', 0)} รายการ")
            return result

        if tool_name == ToolName.COMPARE_TOOL:
            titles = []

            if isinstance(plan.entities.anime_title, list):
                titles = plan.entities.anime_title
            elif isinstance(plan.entities.anime_title, str):
                titles = [title.strip() for title in plan.entities.anime_title.split(",") if title.strip()]

            state.log_step(f"[Registry] Running Compare Tool: {titles}")
            result = compare_tool.run(titles=titles)
            state.log_step("Compare tool สร้างผลเปรียบเทียบเสร็จแล้ว")
            return result

        if tool_name == ToolName.MUSIC_TOOL:
            query = plan.rewritten_query or state.original_query
            state.log_step(f"[Registry] Running Music Tool: {query}")
            result = await music_tool.run(
                query=query,
                meta=get_music_meta(),
                include_external=include_external,
                top_k=5,
            )
            state.log_step(f"Music tool ทำงานเสร็จ โหมด {result.get('mode', '-')} พร้อมผลลัพธ์ {len(result.get('results', [])) if result.get('results') else int(bool(result.get('main_title')))} รายการ")

            if result.get("results"):
                return {
                    "tool": "music_lookup",
                    "query": query,
                    "results": result["results"],
                    "mode": result.get("mode", ""),
                }

            if result.get("main_title"):
                return {
                    "tool": "music_lookup",
                    "query": query,
                    "results": [result],
                    "mode": result.get("mode", ""),
                }

            return {
                "tool": "music_lookup",
                "query": query,
                "results": [],
                "error": result.get("error", "No music results found"),
                "mode": result.get("mode", ""),
            }

        return {
            "tool": "unknown",
            "data": [],
            "error": f"Tool not found: {tool_name}",
        }


@lru_cache(maxsize=1)
def get_music_meta() -> list[dict]:
    meta_path = Path(__file__).resolve().parents[2] / "data" / "music_meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


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
    youtube_url = get_resource_url(record, "YouTube")

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
        "youtube_url": youtube_url,
    }


def get_resource_url(item: dict, key: str) -> str:
    resources = item.get("resources", {})

    if isinstance(resources, str):
        try:
            resources = json.loads(resources)
        except json.JSONDecodeError:
            resources = {}

    if not isinstance(resources, dict):
        return ""

    return resources.get(key, "")


def get_mal_id_from_url(mal_url: str) -> str:
    match = re.search(r"/anime/(\d+)", mal_url or "")
    return match.group(1) if match else ""


@lru_cache(maxsize=512)
def find_cover_image_url(title: str, mal_url: str) -> str:
    mal_id = get_mal_id_from_url(mal_url)

    if mal_id:
        cover_url = music_tool._fetch_cover_url("", mal_id)
        if cover_url:
            return cover_url

    if title:
        return music_tool._fetch_cover_url(title, "")

    return ""


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
            max_tokens=max_tokens,
        )


async def normalize_results(items: List[dict], limit: int, include_youtube: bool = False) -> List[dict]:
    if len(items) == 1 and isinstance(items[0], dict) and items[0].get("records"):
        items = items[0]["records"]

    results = []

    for item in items[:limit]:
        title = (
            item.get("title_en")
            or item.get("title")
            or item.get("name")
            or item.get("anime_title")
            or "Unknown Anime"
        )

        score = item.get("score", item.get("match", item.get("max_rating", item.get("rating", 0.8))))

        try:
            raw_score = float(score)

            # normalize similarity score
            if raw_score <= 1:
                match = int(raw_score * 100)
            elif raw_score <= 10:
                match = int(raw_score * 10)
            else:
                match = int(raw_score)

            # boost low semantic scores for UI readability
            if match < 40:
                match = 70 + int(match * 0.4)

            # clamp range
            match = max(70, min(98, match))

        except Exception:
            match = 85

        youtube_url = ""
        if include_youtube:
            youtube_url = (
                item.get("youtube_url")
                or item.get("playable_url")
                or get_resource_url(item, "YouTube")
            )

        if include_youtube and not youtube_url:
            youtube_url = await find_playable_youtube_url(title)

        url = (
            item.get("mal_url")
            or get_resource_url(item, "MAL")
            or item.get("url")
            or item.get("href")
            or ""
        )

        cover_url = (
            item.get("cover_url")
            or item.get("image_url")
            or find_cover_image_url(title, get_resource_url(item, "MAL") or item.get("mal_url", ""))
        )

        results.append(
            {
                "title": title,
                "match": match,
                "url": url,
                "playable_url": youtube_url,
                "reason": item.get("rerank_reason", item.get("synopsis", "")),
                "image_url": cover_url,
                "cover_url": cover_url,
            }
        )

    return results


async def find_playable_youtube_url(title: str) -> str:
    for query in ("opening", "OP", "OST", "anime opening"):
        youtube_result = await search_youtube(
            song_title=query,
            anime=title,
        )
        youtube_url = youtube_result.get("url")
        if youtube_url:
            return youtube_url

    return ""


def normalize_trace(state: AgentState) -> List[dict]:
    trace = []

    if getattr(state, "event_log", None):
        events = state.event_log
        for index, event in enumerate(events):
            next_event = events[index + 1] if index + 1 < len(events) else None
            duration_ms = 0
            if next_event:
                duration_ms = max(0, next_event.get("elapsed_ms", 0) - event.get("elapsed_ms", 0))

            message = event.get("message", "")
            status = "processing"
            if any(keyword in message for keyword in ["สำเร็จ", "เสร็จ", "จบการทำงาน"]):
                status = "success"
            elif any(keyword in message.lower() for keyword in ["error", "failed", "ไม่พบ", "ยังไม่พอใจ"]):
                status = "retry"

            trace.append(
                {
                    "step": f"{event.get('timestamp', '')} ขั้นตอนที่ {index + 1}",
                    "status": status,
                    "reasoning": message,
                    "top_score": None,
                    "feedback": "",
                    "duration_ms": duration_ms,
                    "duration_breakdown": {
                        "elapsed_ms": event.get("elapsed_ms", 0),
                    },
                }
            )

        return trace

    for entry in state.loop_history:
        plan = entry["plan"]
        observation = entry["observation"]
        planning_ms = getattr(observation, "planning_duration_ms", 0)
        tool_ms = getattr(observation, "action_duration_ms", 0)
        observe_ms = getattr(observation, "observe_duration_ms", 0)
        duration_ms = getattr(observation, "duration_ms", planning_ms + tool_ms + observe_ms)

        trace.append(
            {
                "step": plan.selected_tool.value,
                "status": observation.status.value,
                "reasoning": plan.reasoning,
                "top_score": observation.top_score,
                "feedback": observation.feedback,
                "duration_ms": duration_ms,
                "duration_breakdown": {
                    "planning_ms": planning_ms,
                    "tool_ms": tool_ms,
                    "observe_ms": observe_ms,
                },
            }
        )

    if state.final_answer:
        final_response_duration_ms = getattr(state, "final_response_duration_ms", 0)
        trace.append(
            {
                "step": "LLM Generation",
                "status": "success",
                "reasoning": "สร้างคำตอบสุดท้ายจากผลลัพธ์ของ Agent",
                "top_score": 1,
                "feedback": "",
                "duration_ms": final_response_duration_ms,
                "duration_breakdown": {
                    "generation_ms": final_response_duration_ms,
                },
            }
        )

    return trace


def event_to_trace_step(event: dict, index: int) -> dict:
    message = event.get("message", "")
    status = "processing"
    if any(keyword in message for keyword in ["สำเร็จ", "เสร็จ", "จบการทำงาน"]):
        status = "success"
    elif any(keyword in message.lower() for keyword in ["error", "failed", "ไม่พบ", "ยังไม่พอใจ"]):
        status = "retry"

    return {
        "step": f"{event.get('timestamp', '')} ขั้นตอนที่ {index}",
        "status": status,
        "reasoning": message,
        "top_score": None,
        "feedback": "",
        "duration_ms": 0,
        "duration_breakdown": {
            "elapsed_ms": event.get("elapsed_ms", 0),
        },
    }


def sse_payload(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def build_chat_response(req: ChatRequest, final_state: AgentState) -> dict:
    retrieved_data = []
    if final_state.last_observation:
        retrieved_data = final_state.last_observation.retrieved_data

    include_youtube = (
        final_state.current_plan is not None
        and final_state.current_plan.selected_tool == ToolName.MUSIC_TOOL
    )

    return {
        "query": req.query,
        "answer": final_state.final_answer,
        "results": await normalize_results(retrieved_data, req.topK, include_youtube=include_youtube),
        "trace": normalize_trace(final_state),
        "logs": [
            {
                "type": "agent",
                "message": f"[{event.get('timestamp', '')}] {event.get('message', '')}",
            }
            for event in final_state.event_log
        ],
    }


@router.post("/chat")
async def chat(req: ChatRequest):
    controller = build_controller()
    state = AgentState()
    final_state = await controller.run(req.query, state)
    return await build_chat_response(req, final_state)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def stream():
        queue: asyncio.Queue[dict] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        state = AgentState()

        def enqueue_event(event: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        state.event_callback = enqueue_event
        controller = build_controller()

        async def run_agent() -> AgentState:
            return await controller.run(req.query, state)

        task = asyncio.create_task(run_agent())
        event_index = 0

        try:
            while True:
                if task.done() and queue.empty():
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    yield sse_payload("ping", {"ok": True})
                    continue

                event_index += 1
                yield sse_payload("trace", event_to_trace_step(event, event_index))

            final_state = await task
            yield sse_payload("final", await build_chat_response(req, final_state))
        except Exception as exc:
            yield sse_payload("error", {"message": str(exc)})

    return StreamingResponse(stream(), media_type="text/event-stream")
