import os
import logging
import time
from typing import Optional

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


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

        logger.info("LLMClient 初始化成功，model=%s，base_url=%s", self.model, self.base_url)

    def _handle_openai_error(self, error: Exception) -> str:
        if isinstance(error, AuthenticationError):
            return "API Key 错误，请检查配置后再试。"

        if isinstance(error, APITimeoutError):
            return "模型调用超时，请稍后再试。"

        if isinstance(error, APIConnectionError):
            return "网络异常，无法连接模型服务，请稍后再试。"

        if isinstance(error, OpenAIError):
            return "模型调用失败，请稍后再试。"

        return "系统异常，请稍后再试。"

    def _call_model(self, messages: list[dict]) -> str:
        start_time = time.perf_counter()
        logger.info("开始调用模型，model=%s，message_count=%s", self.model, len(messages))

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )

            elapsed = time.perf_counter() - start_time
            logger.info("模型调用成功，耗时 %.2f 秒", elapsed)
            return response.choices[0].message.content
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.exception("模型调用失败，耗时 %.2f 秒，错误：%s", elapsed, e)
            return self._handle_openai_error(e)

    def chat(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._call_model(messages)

    def chat_with_messages(self, messages: list[dict]) -> str:
        return self._call_model(messages)


llm = LLMClient()


if __name__ == "__main__":
    answer = llm.chat("请用一句话解释 RAG 是什么")
    print(answer)
