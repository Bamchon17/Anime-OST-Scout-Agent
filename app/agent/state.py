# app/agent/state.py
from __future__ import annotations
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Enums - แก้ไขให้ตรงกับ Prompt
# ─────────────────────────────────────────────
class Intent(str, Enum):
    RECOMMEND    = "recommend"
    COMPARE      = "compare"
    FILTER       = "filter"
    MUSIC_LOOKUP = "music_lookup"
    UNKNOWN      = "unknown"

class ToolName(str, Enum):
    RECOMMEND_TOOL = "recommend_tool" 
    MUSIC_TOOL     = "music_tool"     
    FILTER_TOOL    = "filter_tool"
    COMPARE_TOOL   = "compare_tool"
    NONE           = "none"

class RetrievalStrategy(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD  = "keyword"
    METADATA = "metadata" # เพิ่มสำหรับ Filter Tool
    HYBRID   = "hybrid"
    SKIP     = "skip"

class ObservationStatus(str, Enum):
    SATISFIED = "satisfied" # เจอข้อมูลเพียงพอ
    RETRY     = "retry"     # ไม่เจอข้อมูล ต้องให้ Planner คิดใหม่ (Self-Correction)
    FAILED    = "failed"    # พยายามแล้วแต่หาไม่ได้จริงๆ
    ERROR     = "error"

# ─────────────────────────────────────────────
#  Data Components
# ─────────────────────────────────────────────
@dataclass
class ExtractedEntities:
    """แกะจาก Query เพื่อใช้ใน Filter Tool หรือ Music Tool"""
    tags: list[str] = field(default_factory=list)      
    char_tags: list[str] = field(default_factory=list)
    studio: Optional[str] = None
    year: Optional[int] = None
    season: Optional[str] = None                     
    music_style: Optional[str] = None
    rating: Optional[float] = None                     
    type: Optional[str] = None                         
    anime_title: Optional[List[str]] = None #

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
    """ผลลัพธ์ที่ได้จากการใช้ Tool ในแต่ละรอบ"""
    loop_index: int
    status: ObservationStatus
    chunk_count: int = 0
    top_score: float = 0.0
    retrieved_data: List[dict] = field(default_factory=list) # เก็บรายการอนิเมะที่เจอ
    feedback: str = "" # บอกเหตุผลว่าทำไมถึงต้อง Retry หรือ Failed

# ─────────────────────────────────────────────
#  Main Agent State
# ─────────────────────────────────────────────
@dataclass
class AgentState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str = ""
    
    # ประวัติการคุย (Chat History)
    conversation: list[dict] = field(default_factory=list)
    
    # สถานะปัจจุบัน
    current_plan: Optional[ActionPlan] = None
    last_observation: Optional[Observation] = None
    
    # ระบบบันทึก Loop (สำคัญสำหรับ Self-Correction & Log)
    loop_history: list[dict] = field(default_factory=list)
    loop_count: int = 0
    max_loops: int = 3
    
    final_answer: Optional[str] = None

    def log_step(self, message: str):
        """Helper สำหรับแสดงการทำงานหน้าจอ (Observability)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}]  {message}")
        logger.info(message)

    def record_result(self, plan: ActionPlan, observation: Observation):
        """บันทึกผลลัพธ์การทำงานในรอบนั้นๆ"""
        self.loop_history.append({
            "loop": self.loop_count,
            "plan": plan,
            "observation": observation
        })
        self.last_observation = observation
        self.loop_count += 1

    def build_context_summary(self) -> str:
        """ส่งให้ Planner ดูเพื่อทำ Self-Correction"""
        if not self.loop_history:
            return "No previous attempts."
        
        summary = []
        for entry in self.loop_history:
            loop = entry['loop']
            tool = entry['plan'].selected_tool.value
            status = entry['observation'].status.value
            fb = entry['observation'].feedback
            summary.append(f"Round {loop}: Used {tool}, Result: {status}. Feedback: {fb}")
        
        return "\n".join(summary)

    def is_done(self) -> bool:
        """เช็คว่าควรหยุดรัน Agent หรือยัง"""
        if self.loop_count >= self.max_loops:
            return True
        if self.last_observation and self.last_observation.status == ObservationStatus.SATISFIED:
            return True
        return False