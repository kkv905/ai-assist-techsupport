import os
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("PHOENIX_TRACING_ENABLED", "0")

from app.main import app
from app.schemas.chat import ChatDelta, ChatResponse, Usage


class FakeLLMService:
    """Имитирует сервисный слой для интеграционных тестов FastAPI."""

    async def complete(self, request) -> ChatResponse:
        """Возвращает предсказуемый полный ответ."""

        return ChatResponse(
            content=f"Ответ: {request.messages[0].content}",
            model=request.model or "gpt-4o-mini",
            usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
            finish_reason="stop",
            cached=False,
        )

    async def stream(self, request) -> AsyncIterator[ChatDelta]:
        """Отдает две текстовые дельты и финальную статистику usage."""

        yield ChatDelta(content="Первая часть")
        yield ChatDelta(content=" и вторая часть")
        yield ChatDelta(usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8))


def test_health_endpoint_returns_ok() -> None:
    """Проверяет доступность healthcheck без внешних зависимостей."""

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_returns_chat_response() -> None:
    """Проверяет JSON-ответ нового endpoint `/chat`."""

    app.dependency_overrides.clear()
    from app.deps.providers import get_llm_service

    app.dependency_overrides[get_llm_service] = lambda: FakeLLMService()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "Проверка"}]},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["content"] == "Ответ: Проверка"
    assert response.json()["cached"] is False
    assert len(response.headers["X-Request-ID"]) == 12


def test_chat_endpoint_echoes_request_id_header() -> None:
    """Проверяет возврат клиентского идентификатора запроса в заголовке ответа."""

    app.dependency_overrides.clear()
    from app.deps.providers import get_llm_service

    app.dependency_overrides[get_llm_service] = lambda: FakeLLMService()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                headers={"X-Request-ID": "abc123request"},
                json={"messages": [{"role": "user", "content": "Проверка"}]},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "abc123request"


def test_stream_endpoint_returns_sse_payload() -> None:
    """Проверяет формат SSE для потокового endpoint `/chat/stream`."""

    app.dependency_overrides.clear()
    from app.deps.providers import get_llm_service

    app.dependency_overrides[get_llm_service] = lambda: FakeLLMService()
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/chat/stream",
                json={"messages": [{"role": "user", "content": "Стрим"}]},
            ) as response:
                body = "".join(response.iter_text())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "data: Первая часть" in body
    assert "data:  и вторая часть" in body
    assert '"usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}' in body
    assert "data: [DONE]" in body


def test_validation_error_has_unified_format() -> None:
    """Проверяет единый формат ошибки валидации."""

    with TestClient(app) as client:
        response = client.post("/chat", json={"messages": []})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert response.json()["error"]["details"]
    assert response.headers["X-Request-ID"]
