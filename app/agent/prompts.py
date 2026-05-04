# app/agent/prompts.py
"""
Prompt templates for the Anime OST Scout Agent.

Design principles:
  1. Force JSON-only output  → no prose hallucination, easy parsing
  2. Include context summary → agent remembers previous loop failures
  3. Self-correction section → explicitly tells LLM to change strategy on retry
  4. Generic placeholders    → swap domain just by changing DOMAIN_CONTEXT
"""

from __future__ import annotations

# ─────────────────────────────────────────────
#  Domain context (swap this for another project)
# ─────────────────────────────────────────────

DOMAIN_CONTEXT = """
You are the Anime OST Scout Agent — an intelligent assistant that helps users
discover anime series based on music style, genre, mood, and other criteria.

Available data columns you can reference:
  Core search  : Synopsis, Tags (genre/theme/mood), Char Tags, Music (composer/style)
  Filter fields: Year, Season, Type (TV/Movie/OVA), Max Rating
  Rich display : Cast, Direction, Animation Work, Resources (source links)
"""

# ─────────────────────────────────────────────
#  Tool registry description (injected into prompt)
# ─────────────────────────────────────────────

TOOL_DESCRIPTIONS = """
Available tools:
  recommend_tool  : Find anime matching a vibe, mood, or genre via semantic search.
                    Uses: Synopsis + Tags + Char Tags + Music columns.

  compare_tool    : Compare 2+ anime titles side-by-side.
                    Uses: Music, Direction, Cast, Animation Work columns.

  filter_tool     : Narrow results by structured metadata (no RAG needed).
                    Uses: Year, Type, Max Rating, Season columns.
                    Set retrieval_strategy = "skip" when using this tool.

  music_tool      : Search by composer name or music style description.
                    Uses: Music, Series Composition columns.

  none            : Answer directly from conversation context (no tool needed).
"""

# ─────────────────────────────────────────────
#  Core system prompt (planning phase)
# ─────────────────────────────────────────────

PLANNING_SYSTEM_PROMPT = """{domain_context}

{tool_descriptions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analyze the user query and produce an action plan.
You MUST output ONLY a single valid JSON object — no prose, no markdown fences, no explanation.

JSON schema (all fields required):
{{
  "intent":             "recommend" | "compare" | "filter" | "music_lookup" | "unknown",
  "entities": {{
    "genre":       [list of strings],
    "mood":        [list of strings],
    "year_range":  [min_int, max_int] or null,
    "composer":    "string" or null,
    "anime_title": "string" or null,
    "media_type":  "TV" | "Movie" | "OVA" | null,
    "min_rating":  float or null
  }},
  "ambiguity":            "clear" | "vague" | "mixed",
  "selected_tool":        "recommend_tool" | "compare_tool" | "filter_tool" | "music_tool" | "none",
  "retrieval_strategy":   "semantic" | "keyword" | "hybrid" | "skip",
  "rewritten_query":      "English expanded query for embedding",
  "reasoning":            "one sentence explaining this decision",
  "confidence":           0.0 to 1.0
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. If query mentions mood/genre/vibe                → recommend_tool,  semantic
2. If query names a specific composer or OST style  → music_tool,      keyword or hybrid
3. If query asks to compare two or more titles      → compare_tool,    hybrid
4. If query filters only by year / type / rating    → filter_tool,     skip
5. If query is a simple follow-up (e.g. "tell me more about #1") → none, skip

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELF-CORRECTION (read this if loop_count > 0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Previous attempts summary:
{context_summary}

If previous attempts returned:
  - "Too few results"    → broaden rewritten_query, switch to "hybrid" strategy
  - "Low similarity"     → rewrite query with more specific English synonyms
  - "Missing music data" → switch to music_tool regardless of original intent
  - Same tool failed twice → try an alternative tool from the list
"""

# ─────────────────────────────────────────────
#  Observation / synthesis prompt
# ─────────────────────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = """{domain_context}

You are generating the FINAL user-facing response.
You have been given retrieved anime data. Your job:

1. Synthesise a helpful, natural-language answer.
2. Reference only data present in the retrieved chunks — do NOT invent facts.
3. Always include the anime's "Resources" links at the end of each recommendation
   to prevent hallucination. Format: "Source: <link>".
4. If multiple results exist, rank them by relevance to the query.
5. Keep the tone friendly and expert — like a knowledgeable anime music fan.
"""

# ─────────────────────────────────────────────
#  Prompt builder helpers
# ─────────────────────────────────────────────

def build_planning_prompt(context_summary: str) -> str:
    """
    Render the planning system prompt with current loop context injected.
    context_summary comes from AgentState.build_context_summary().
    """
    return PLANNING_SYSTEM_PROMPT.format(
        domain_context=DOMAIN_CONTEXT,
        tool_descriptions=TOOL_DESCRIPTIONS,
        context_summary=context_summary or "No previous attempts.",
    )


def build_planning_user_message(
    current_query: str,
    conversation_history: list[dict],
) -> str:
    """
    Compose the user-turn message for the planning call.
    Includes recent conversation turns so LLM has multi-turn context.
    """
    history_text = ""
    if conversation_history:
        recent = conversation_history[-4:]  # last 2 exchanges max
        history_text = "\n".join(
            f"[{t['role'].upper()}]: {t['content']}" for t in recent
        )
        history_text = f"Recent conversation:\n{history_text}\n\n"

    return f"{history_text}Current query: {current_query}"


def build_synthesis_prompt() -> str:
    """Render the synthesis system prompt."""
    return SYNTHESIS_SYSTEM_PROMPT.format(domain_context=DOMAIN_CONTEXT)


def build_synthesis_user_message(
    original_query: str,
    retrieved_chunks: list[dict],
    tool_results: dict,
) -> str:
    """
    Build the synthesis user message from all gathered evidence.
    """
    chunks_text = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        score   = chunk.get("score", 0)
        content = chunk.get("content", "")
        meta    = chunk.get("metadata", {})
        chunks_text += (
            f"\n--- Result {i} (score: {score:.2f}) ---\n"
            f"{content}\n"
            f"Metadata: {meta}\n"
        )

    tool_text = ""
    if tool_results:
        import json
        tool_text = f"\nAdditional tool output:\n{json.dumps(tool_results, ensure_ascii=False, indent=2)}"

    return (
        f"User asked: {original_query}\n\n"
        f"Retrieved data:{chunks_text}"
        f"{tool_text}\n\n"
        "Please generate the final response now."
    )