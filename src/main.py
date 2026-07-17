import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.llm_client import LLMClient


app = FastAPI(
    title="AI Practice API",
    description="FastAPI service for calling the configured LLM client.",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt")
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        min_length=1,
        description="System instruction for the assistant",
    )


class ChatResponse(BaseModel):
    answer: str


class ConfigResponse(BaseModel):
    model: str
    base_url: str
    api_key_configured: bool


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "AI Practice API is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    return ConfigResponse(
        model=os.getenv("LLM_MODEL_NAME", ""),
        base_url=os.getenv("LLM_BASE_URL", ""),
        api_key_configured=bool(os.getenv("LLM_API_KEY")),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        llm = LLMClient()
        answer = llm.chat(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
        )
        return ChatResponse(answer=answer)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="LLM call failed") from exc
