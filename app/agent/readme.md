## LLM Agent Overview
อะไรที่ทำให้ LLM ตัดสินใจเองได้ขอแบ่งเป็นองค์ประกอบหลักในการทำ ดังนี้:

### 1.Cognitive Memory (ความจำ)
เป็นส่วนความทรงจำ และตัวจัดการ “state (สถานะ)” แบ่งออกเป็น 2 memories ทำให้ Agent รู้ว่า ตอนนี้ทำอะไรไปแล้ว และต้องดำเนินอะไรต่อ
* **Short-term Memory:** คือการเก็บ Context ของการสนทนาปัจจุบัน (Conversation Buffer)
* **Long-term Memory** การดึงข้อมูลจาก Vector Database (RAG) กลับมาใช้

### 2. Task Decomposition & Reasoning (ความคิด)
ส่วนวางแผน หรือสมองส่วนหน้าที่ทำหน้าที่คิดล่วงหน้าว่าต้องทำอะไร แบ่งออกเป็น 2 ส่วน
* **Task Decomposition** การย่อยคำถามยากๆ จาก User ให้กลายเป็นขั้นตอนย่อยๆ (Sub-tasks)
* **Reasoning Strategies:** เช่น Chain-of-Thought (CoT) หรือ ReAct (Reason + Act) คือการให้ AI "คิดดังๆ" ก่อนจะตัดสินใจเลือกเครื่องมือ

### 3. Action Executor & Orchestration (ร่างกาย /การกระทำ)
ส่วนการควบคุมการทำงาน หรือเรียกว่า  "Agent Executor" หรือ "Orchestrator" เราจะใช้ในการเรียก TOOL ที่เราพัฒตาขึ้นเพื่อเป็น External data ให้ใช้ได้ในกรณีที่ knowledgebaseเราข้อมูลไม่มี
* **Tool Use / Function Calling:** คือความสามารถในการ "กดปุ่ม" เรียกใช้โค้ดภายนอก (เช่น API หรือ Database)
* **Self-Correction / Evaluation:** การที่ Controller ตรวจสอบผลลัพธ์ (Observe) แล้วตัดสินใจว่า "ข้อมูลนี้แย่เกินไป ต้องหาใหม่" หรือ "พอแล้ว สรุปคำตอบได้"

### 4.  LLM หรือ The Core Cognitive Engine (สติปัญญา)
 "หน่วยประมวลผลกลาง" ของระบบ
* **Zero-shot / Few-shot Prompting:** เป็นเทคนิคที่ใช้ใน prompts.py เพื่อสั่งให้ LLM ปรับตัวเข้ากับงานเฉพาะทาง  โดยไม่ต้องไปเทรน Model ใหม่
---

## Main Core
### 1. Custom agent loop
* **Decision Loop** Observe and Evaluate 
* **Multi-Tool Habdling** Mixed Signals 
* **Fallback Strategy** 




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