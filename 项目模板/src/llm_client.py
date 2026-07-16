# llm_client.py
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL_NAME")

        if not self.api_key:
            raise ValueError("Missing LLM_API_KEY")
        if not self.model:
            raise ValueError("Missing LLM_MODEL_NAME")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content

    def chat_with_messages(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )

        return response.choices[0].message.content


llm = LLMClient()


if __name__ == "__main__":
    answer = llm.chat("请用一句话解释 RAG 是什么")
    print(answer)