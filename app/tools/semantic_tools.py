"""
tools/semantic_tool.py
----------------------
ค้นหา anime ตามความหมายของ query
เหมาะกับ query แบบ free-text เช่น:
  "อนิเมะที่ทำให้ร้องไห้"
  "anime about friendship and sacrifice"
  "อนิเมะแนว dark ที่มีพล็อตซับซ้อน"

ใช้: retrieve_general() → FAISS top-10 → Claude rerank top-3
"""

from app.rag.retrieval import retrieve_general

# ───────────────────────────────────────────────
# TOOL DEFINITION — ส่งให้ agent.py ลงทะเบียน
# ───────────────────────────────────────────────

TOOL_DEFINITION = {
    "name": "semantic_search",
    "description": (
        "ค้นหา anime จาก query ภาษาธรรมชาติ "
        "ใช้เมื่อ user อธิบายสิ่งที่ต้องการโดยไม่ระบุ title, ปี, หรือ rating ชัดเจน "
        "เช่น 'อนิเมะที่ทำให้คิดเยอะ' หรือ 'anime with great plot twist'"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "คำอธิบายอนิเมะที่ต้องการ เป็นภาษาไทยหรืออังกฤษก็ได้",
            },
            "top_k": {
                "type": "integer",
                "description": "จำนวนผลลัพธ์ที่ต้องการ (default 3, max 5)",
                "default": 3,
            },
        },
        "required": ["query"],
    },
}


# ───────────────────────────────────────────────
# HANDLER
# ───────────────────────────────────────────────

def run(query: str, top_k: int = 3) -> dict:
    """
    รับ query → คืน dict พร้อม results และ metadata สำหรับ agent

    Return format:
    {
      "tool": "semantic_search",
      "query": "...",
      "results": [
        {
          "rank": 1,
          "chunk_id": "ani_003",
          "title_en": "Attack on Titan",
          "rating": 9.0,
          "year": 2013,
          "synopsis": "...",
          "score": 0.872,
          "rerank_reason": "ตรงกับ query เพราะ..."
        },
        ...
      ]
    }
    """
    top_k = min(top_k, 5)   # cap ที่ 5

    print(f"[semantic_tool] query='{query}', top_k={top_k}")
    raw_results = retrieve_general(query, top_k=top_k)

    results = [
        {
            "rank":          i + 1,
            "chunk_id":      r["chunk_id"],
            "title_en":      r["title_en"],
            "title":         r["title"],
            "rating":        r["rating"],
            "year":          r["year"],
            "type":          r["type"],
            "tags":          r["filter_meta"]["tags"],
            "synopsis":      r["synopsis"],
            "score":         round(r["score"], 4),
            "rerank_reason": r.get("rerank_reason", ""),
            "image_url":     r.get("image_url", ""),
            "mal_url":       r.get("mal_url", ""),
        }
        for i, r in enumerate(raw_results)
    ]

    return {
        "tool":    "semantic_search",
        "query":   query,
        "results": results,
    }


# ───────────────────────────────────────────────
# ENTRY POINT — ทดสอบ
# ───────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "อนิเมะที่ทำให้ร้องไห้และรู้สึกอิ่มใจ",
        "dark psychological anime with complex characters",
        "anime about found family and friendship",
    ]

    for q in test_queries:
        print("\n" + "═" * 60)
        output = run(q)
        for r in output["results"]:
            print(f"\n{r['rank']}. {r['title_en']}  ★{r['rating']}  (score={r['score']})")
            print(f"   Tags: {', '.join(r['tags'])}")
            if r["rerank_reason"]:
                print(f"   → {r['rerank_reason']}")