<p align="center">
  <img src="assets/images/download.jpg" width="100%" alt="Project Banner">
</p>


#  Anime OST Scout Agent RAG

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)


## 📌 Project Overview
Luna Anime Agent กูรูคลังแสงอนิเมะที่ช่วยแนะนำเรื่องที่ใช่ และตามหาเพลง OST ที่ชอบ โดยเปลี่ยน 'มูด' และ 'ความต้องการ' ของคุณให้เป็นลิสต์อนิเมะที่ตรงใจที่สุด ด้วยระบบวิเคราะห์ความหมายและเครื่องมือเปรียบเทียบข้อมูลที่แม่นยำ
---

## 🛠 Project Structure & Responsibilities

โปรเจกต์นี้พัฒนาด้วยภาษา **Python** เป็นหลัก โดยแบ่งการทำงานออกเป็น 3 ส่วนหลัก ดังนี้:

### 1. Agent Brain & Workflow (แบม - Bam)
รับผิดชอบ “สมองของระบบ” และการควบคุมการทำงานของ Agent
* **LLM Selection:** เลือกโมเดล typhoon-v2.5-30b-a3b-instruct
* **Planner:** 
* **State:** 
* **Controller** 
* **Prompts:** 

### 2. RAG & Tools Layer (คิว - Kew)
Data Preparetion และ ระบบสืบค้นข้อมูล
* **Data** นำเข้าข้อมูลจากKaggle 
* **Create Tools**  Semantic Tool, Filter Tool, Compare Tool, Music Tool 
* **Embedding:** นำข้อมูลมาแปลงเป็น vector เพื่อเป็นฐานข้อมูล
* **Vector Database:** เลือกใช้ FAISS 
* **retrieval + reranking logic** 

### 3. API, Observability & Integration (มาร์ค - Mark)
รับผิดชอบ การเชื่อมระบบ + การแสดง process
* **Fast API:** พัฒนา FastAPI Backend Endpoint และความแม่นยำของโมเดล ออกแบบ API endpoints (/chat)
* **Observability system:** reasoning trace
tool call logs
retrieval logs
loop tracking
---

## 🗂 สำหรับ Dataset

โปรเจกต์นี้ใช้ **Kaggle Data** [text](https://www.kaggle.com/datasets/divyanshusingh369/aimi-anime-rag-and-receipts-dataset-sample/data)


```
### Git Anime-OST-Scout-Agent 

```bash
# 1. Clone 
git clone https://github.com/Bamchon17/Anime-OST-Scout-Agent.git

# 2. CD Anime-OST-Scout-Agent 

# 3. อย่าลืมแตก branch ไปทำกันต่อนะ 
```

---

## 📁 Directory Structure (Proposed)
```text
.
anime-agentic-rag/
│
├── app/
│   ├── main.py                  # FastAPI entry
│   ├── api/
│   │   └── routes.py           # endpoint /chat
│   │
│   ├── agent/                  # Bam 
│   │   ├── controller.py       # main agent loop (Reason→Act→Observe)
│   │   ├── planner.py          # intent + tool selection
│   │   ├── state.py            # agent state (loop, memory)
│   │   └── prompts.py          # system prompt / JSON schema
│   │
│   ├── tools/                  #  Kew 
│   │   ├── recommend.py
│   │   ├── filter.py
│   │   ├── compare.py
│   │   ├── music.py
│   │   └── registry.py         # register tools
│   │
│   ├── rag/                    #  Kew 
│   │   ├── embedder.py         # embedding model
│   │   ├── vectorstore.py      # FAISS / Chroma
│   │   ├── retriever.py        # semantic search
│   │   ├── reranker.py         # optional
│   │   └── query_refiner.py
│   │
│   ├── models/                 # LLM loader
│   │   └── llm.py
│   │
│   ├── schemas/                # Mark ดูแล
│   │   ├── request.py
│   │   ├── response.py
│   │   └── tool_schema.py      # JSON tool call format
│   │
│   ├── observability/          # Mark ดูแล
│   │   ├── logger.py
│   │   ├── tracer.py
│   │   └── formatter.py
│   │
│   └── utils/
│       └── helpers.py
│
├── data/
│   ├── raw/                    # CSV จาก Kaggle
│   └── processed/              # cleaned + chunked
│
├── scripts/
│   └── ingest.py               # build vector DB
│
├── tests/
│
├── requirements.txt
└── README.md