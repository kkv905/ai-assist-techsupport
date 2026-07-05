"""HTTP-клиент для работы Telegram-бота с chat-backend."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import httpx


class BackendClient:
    """Инкапсулирует HTTP-вызовы к backend сервиса чатов."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Создает `httpx.AsyncClient` с базовым адресом backend."""

        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    async def get_or_create_chat(self, owner_external_id: str, interface: str) -> UUID:
        """Создает чат через backend и возвращает его идентификатор."""

        response = await self._client.post(
            "/chats",
            json={
                "owner_external_id": owner_external_id,
                "interface": interface,
            },
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return UUID(payload["chat_id"])

    async def send_message(self, chat_id: UUID, content: str) -> AsyncIterator[str]:
        """Отправляет сообщение и по мере поступления отдает SSE-токены ответа."""

        async with self._client.stream(
            "POST",
            f"/chats/{chat_id}/messages",
            json={"content": content},
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line[5:]
                if chunk.startswith(" "):
                    chunk = chunk[1:]
                if chunk == "[DONE]":
                    break
                if chunk:
                    yield chunk

    async def clear_messages(self, chat_id: UUID) -> None:
        """Очищает историю сообщений чата на стороне backend."""

        response = await self._client.delete(f"/chats/{chat_id}/messages")
        response.raise_for_status()

    async def aclose(self) -> None:
        """Закрывает underlying HTTP-клиент."""

        await self._client.aclose()

    async def __aenter__(self) -> BackendClient:
        """Поддерживает использование клиента в `async with`."""

        return self

    async def __aexit__(self, *args: object) -> None:
        """Автоматически закрывает клиент при выходе из контекста."""

        await self.aclose()
