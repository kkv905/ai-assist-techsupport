"""Тесты HTTP-маршрутов модуля чата."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.chat.deps import get_chat_service
from app.chat.domain import Chat, ChatMessage
from app.main import app


class FakeChatService:
    """Имитирует сервис чата для тестирования FastAPI-роутов."""

    def __init__(self) -> None:
        """Подготавливает предсказуемые данные чата и истории."""

        self.chat = Chat(
            id=uuid4(),
            owner_external_id="tg-1",
            interface="telegram",
            system_prompt="Ты помощник",
            created_at=datetime.now(UTC),
        )
        self.messages = [
            ChatMessage(chat_id=self.chat.id, role="user", content="Привет"),
            ChatMessage(chat_id=self.chat.id, role="assistant", content="Здравствуйте"),
        ]
        self.cleared = False

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        """Возвращает заранее подготовленный чат."""

        self.chat = Chat(
            id=self.chat.id,
            owner_external_id=owner_external_id,
            interface=interface,
            system_prompt=system_prompt,
            created_at=self.chat.created_at,
        )
        return self.chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        """Возвращает чат по совпадающему идентификатору."""

        if chat_id == self.chat.id:
            return self.chat
        return None

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        """Возвращает историю сообщений для тестового чата."""

        return self.messages[-limit:]

    async def send_message(self, chat_id: UUID, user_content: str) -> AsyncIterator[str]:
        """Отдает потоковый ответ из двух частей."""

        yield "Часть 1"
        yield " и часть 2"

    async def clear_history(self, chat_id: UUID) -> None:
        """Помечает историю как очищенную."""

        self.cleared = True


def test_create_chat_route_returns_chat_id() -> None:
    """Проверяет создание чата через POST /chats."""

    fake_service = FakeChatService()
    app.dependency_overrides[get_chat_service] = lambda: fake_service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/chats",
                json={
                    "owner_external_id": "web-user",
                    "interface": "web",
                    "system_prompt": "Ты помощник",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json() == {"chat_id": str(fake_service.chat.id)}


def test_get_chat_route_returns_metadata() -> None:
    """Проверяет чтение метаданных чата."""

    fake_service = FakeChatService()
    app.dependency_overrides[get_chat_service] = lambda: fake_service
    try:
        with TestClient(app) as client:
            response = client.get(f"/chats/{fake_service.chat.id}")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["owner_external_id"] == "tg-1"
    assert response.json()["interface"] == "telegram"


def test_list_messages_route_returns_chronological_messages() -> None:
    """Проверяет выдачу истории сообщений чата."""

    fake_service = FakeChatService()
    app.dependency_overrides[get_chat_service] = lambda: fake_service
    try:
        with TestClient(app) as client:
            response = client.get(f"/chats/{fake_service.chat.id}/messages?limit=2")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert [message["content"] for message in response.json()] == ["Привет", "Здравствуйте"]


def test_send_message_route_returns_sse_stream() -> None:
    """Проверяет SSE-формат ответа у POST /chats/{id}/messages."""

    fake_service = FakeChatService()
    app.dependency_overrides[get_chat_service] = lambda: fake_service
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                f"/chats/{fake_service.chat.id}/messages",
                json={"content": "Привет"},
            ) as response:
                body = "".join(response.iter_text())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "data: Часть 1" in body
    assert "data:  и часть 2" in body
    assert "data: [DONE]" in body


def test_delete_messages_route_returns_ok() -> None:
    """Проверяет мягкую очистку истории чата."""

    fake_service = FakeChatService()
    app.dependency_overrides[get_chat_service] = lambda: fake_service
    try:
        with TestClient(app) as client:
            response = client.delete(f"/chats/{fake_service.chat.id}/messages")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert fake_service.cleared is True


def test_get_unknown_chat_returns_404() -> None:
    """Проверяет возврат 404 для отсутствующего чата."""

    fake_service = FakeChatService()
    app.dependency_overrides[get_chat_service] = lambda: fake_service
    try:
        with TestClient(app) as client:
            response = client.get(f"/chats/{uuid4()}")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json()["detail"] == "Чат не найден."
