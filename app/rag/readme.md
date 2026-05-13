## ฐานข้อมูลที่จะนำมาใช้
### 1. KnowledgeBase 
* **general_meta.json** เป็นเนื้อหาฐานข้อมูลหลักของ RAG ในการใช้ตอบคำถามเชิงเนื้อหา และข้อมูลทั่วไปของอนิเมะ เพราะมีเรื่องย่อ Synopsis อยู่   

### 2. Music Scout Tool
เป็น Keyword Search or Metadata Mapping ใช้เมื่อ User ถามเจาะจงเรื่องเพลง, อารมณ์เพลง หรือคนแต่งเพลง
* **music_meta.json** จะใช้เป็น Tool เมื่อผู้ใช้ถามเรื่องเพลงโดยจะเชื่อมต่อ Planner เพราะมันมี Keyword เช่น orchestra, choir, intense 


## The 3 Intelligence Tools
### 1. semantic_tool 
เป็น RAG แบบ Two-stage Retrieval ที่ไปเรียก  retrieval.py เพื่อหาข้อมูลจากความหมาย ใช้ 2 วิธีในการดึงคำตอบที่แม่นยำ
* **Consine Simirality** ค้นหาความคล้ายคลึงใน faiss vector ได้ค่าเป็น TOP 10
* **Ollama handling LLM (Typhoon-v1.5-8B-instruct)** Re-Ranking คำตอบจาก TOPS 10 ให้เหลือ TOP3 ที่แม่นและตรงที่สุด

### 2. filter_tool
ใช้ Pandas ในการกรองข้อมูล เช่น กรอง anime ตาม metadata จริง ไม่ใช้ vector search
เหมาะกับ query ที่ระบุชัดเจน เช่น:
  "อนิเมะ TV ปี 2020 rating มากกว่า 8"
  "anime แนว action ของ studio Mappa"
  "อนิเมะที่ออกช่วง spring"

### 3. compare_tool
 เอาไว้เปรียบเทียบระหว่างเรื่อง Anime แบบ side-by-side
เหมาะกับ query เช่น:
  "เปรียบเทียบ Naruto กับ Bleach"
  "FMA Brotherhood vs Attack on Titan ต่างกันยังไง"
  "ระหว่าง Ghibli movies ตัวไหนดีสุด"
ค้นหา anime แต่ละตัวด้วย title matching
แล้วจัด structured comparison ให้ agent นำไปสรุป