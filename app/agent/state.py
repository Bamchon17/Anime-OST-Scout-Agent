# memory for agent
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

class Intent(str, Enum):
    RECOMMEND    = "recommend"
    COMPARE      = "compare"
    FILTER       = "filter"
    MUSIC_LOOKUP = "music_lookup"
    UNKNOWN      = "unknown"

class ToolName(str, Enum):
    RECOMMEND = "recommend_tool"
    COMPARE   = "compare_tool"
    FILTER    = "filter_tool"
    MUSIC     = "music_tool"
    NONE      = "none"

class RetrievalStrategy(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD  = "keyword"
    HYBRID   = "hybrid"
    SKIP     = "skip"

class ObservationStatus(str, Enum):
    SATISFIED = "satisfied"
    RETRY     = "retry"
    FAILED    = "failed"
    ERROR     = "error"

@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    def dict(self): return {"role": self.role, "content": self.content}

@dataclass
class ExtractedEntities:
    genre: list = field(default_factory=list)
    mood: list = field(default_factory=list)
    year_range: Any = None
    composer: Any = None
    anime_title: Any = None
    media_type: Any = None
    min_rating: Any = None

@dataclass
class ActionPlan:
    intent: Intent = Intent.UNKNOWN
    entities: ExtractedEntities = field(default_factory=ExtractedEntities)
    ambiguity: str = "clear"
    selected_tool: ToolName = ToolName.NONE
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC
    rewritten_query: str = ""
    reasoning: str = ""
    confidence: float = 1.0

@dataclass
class Observation:
    loop_index: int
    status: ObservationStatus
    chunk_count: int = 0
    top_score: float = 0.0
    has_music_data: bool = False
    feedback: list = field(default_factory=list)
    raw_results: dict = field(default_factory=dict)

@dataclass
class LoopRecord:
    loop_index: int
    plan: ActionPlan
    observation: Observation
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AgentState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    original_query: str = ""
    current_query: str = ""
    conversation: list = field(default_factory=list)
    current_plan: Any = None
    loop_history: list = field(default_factory=list)
    last_observation: Any = None
    final_answer: Any = None
    loop_count: int = 0
    max_loops: int = 3
    total_tokens_used: int = 0
    tool_calls_made: list = field(default_factory=list)

    def is_done(self):
        if self.loop_count >= self.max_loops: return True
        if self.last_observation and self.last_observation.status in (
            ObservationStatus.SATISFIED, ObservationStatus.FAILED): return True
        return False

    def add_turn(self, role, content):
        self.conversation.append(ConversationTurn(role=role, content=content))
        return self

    def record_loop(self, plan, observation):
        self.loop_history.append(LoopRecord(loop_index=self.loop_count, plan=plan, observation=observation))
        self.last_observation = observation
        self.loop_count += 1
        return self

    def build_context_summary(self):
        if not self.loop_history: return "No previous attempts."
        return "\n".join(
            f"Loop {r.loop_index}: tool={r.plan.selected_tool.value} | "
            f"strategy={r.plan.retrieval_strategy.value} | "
            f"status={r.observation.status.value} | feedback={r.observation.feedback}"
            for r in self.loop_history
        )