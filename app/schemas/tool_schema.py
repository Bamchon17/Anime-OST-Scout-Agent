from pydantic import BaseModel
from typing import Dict, Any, Optional

class ToolCall(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

class ToolResponse(BaseModel):
    status: str # "success" or "error"
    data: Any   # ข้อมูลที่ดึงมาได้
    message: Optional[str] = None


# เพื่อนแค่เขียนคลาสนี้แล้วส่งมาให้
class AnimeToolRegistry(BaseToolRegistry):
    async def execute(self, tool_name, plan, state) -> dict:
        # เชื่อม ChromaDB / semantic search ที่นี่
        ...

# แล้ว wire เข้ามาตรงนี้ใน main.py
controller = AgentController(
    planner       = Planner(llm=OllamaLLM(model="qwen2.5:7b")),
    tool_registry = AnimeToolRegistry(),
    synthesis_llm = OllamaLLM(model="qwen2.5:7b"),
)