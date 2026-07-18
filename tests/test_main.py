from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_read_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "AI Practice API is running"}


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_config(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_NAME", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    response = client.get("/config")

    assert response.status_code == 200
    assert response.json() == {
        "model": "test-model",
        "base_url": "https://example.com/v1",
        "api_key_configured": True,
    }


def test_chat_success(monkeypatch):
    class FakeLLMClient:
        def chat(self, prompt: str, system_prompt: str) -> str:
            assert prompt == "Hello"
            assert system_prompt == "You are a helpful assistant."
            return "Hi there!"

    monkeypatch.setattr("src.main.LLMClient", FakeLLMClient)

    response = client.post("/chat", json={"prompt": "Hello"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Hi there!"}


def test_chat_rejects_empty_prompt():
    response = client.post("/chat", json={"prompt": ""})

    assert response.status_code == 422


def test_chat_returns_500_when_llm_config_is_invalid(monkeypatch):
    class FakeLLMClient:
        def __init__(self):
            raise ValueError("Missing LLM_API_KEY")

    monkeypatch.setattr("src.main.LLMClient", FakeLLMClient)

    response = client.post("/chat", json={"prompt": "Hello"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Missing LLM_API_KEY"}
