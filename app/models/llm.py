import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("TYPHOON_KEY")
        self.base_url = "https://api.opentyphoon.ai/v1"
        self.model = "typhoon-v2.5-30b-a3b-instruct"

        if not self.api_key:
            raise ValueError("หา API ไม่เจอ หรือ KEY มีปัญหาลองเช็คดูนะ")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.4,
        max_tokens: int | None = None,
    ):
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,    
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"เกิดข้อผิดพลาดในการเรียก LLM: {str(e)}"

llm_service = LLMService()     
