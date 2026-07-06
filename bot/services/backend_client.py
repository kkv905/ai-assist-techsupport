"""HTTP-клиент для работы Telegram-бота с chat-backend."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx

_DEFAULT_TIMEOUT = httpx.Timeout(connect=3.0, read=60.0, write=10.0, pool=5.0)
_STREAM_TIMEOUT = httpx.Timeout(connect=3.0, read=120.0, write=10.0, pool=5.0)
_RETRYABLE_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)
_MAX_ATTEMPTS = 3


@dataclass(slots=True)
class BackendStreamEvent:
    """Описывает одно SSE-событие от backend при ответе модели."""

    type: str
    delta: str | None = None
    message_id: UUID | None = None


class BackendClient:
    """Инкапсулирует HTTP-вызовы к backend сервиса чатов."""

    def __init__(
        self,
        base_url: str,
        *,
        admin_token: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Создает или принимает готовый `httpx.AsyncClient` с базовым адресом backend."""

        self._owns_client = http_client is None
        self._admin_token = admin_token
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=_DEFAULT_TIMEOUT,
            transport=transport,
        )

    async def get_or_create_chat(self, owner_external_id: str, interface: str) -> UUID:
        """Создает чат через backend и возвращает его идентификатор."""

        response = await self._request_with_retry(
            "POST",
            "/chats",
            json={
                "owner_external_id": owner_external_id,
                "interface": interface,
            },
        )
        payload: dict[str, Any] = response.json()
        return UUID(payload["chat_id"])

    async def send_message(
        self,
        chat_id: UUID,
        content: str,
        media: bytes | None = None,
        mime: str | None = None,
    ) -> AsyncIterator[BackendStreamEvent]:
        """Отправляет сообщение и отдает SSE-события ответа backend."""

        files = None
        if media is not None:
            files = {"media": (self._guess_filename(mime), media, mime or "application/octet-stream")}

        request = self._client.build_request(
            "POST",
            f"/chats/{chat_id}/messages",
            data={"content": content},
            files=files,
            headers={"Accept": "text/event-stream"},
            timeout=_STREAM_TIMEOUT,
        )
        response = await self._send_stream_request(request)

        try:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line.removeprefix("data: "))
                message_id = payload.get("message_id")
                yield BackendStreamEvent(
                    type=payload.get("type", "token"),
                    delta=payload.get("delta"),
                    message_id=UUID(message_id) if message_id else None,
                )
        finally:
            await response.aclose()

    async def clear_messages(self, chat_id: UUID) -> None:
        """Очищает историю сообщений чата на стороне backend."""

        await self._request_with_retry("DELETE", f"/chats/{chat_id}/messages")

    async def submit_feedback(self, chat_id: UUID, message_id: UUID, value: str) -> None:
        """Отправляет голос пользователя за сообщение ассистента."""

        await self._request_with_retry(
            "POST",
            f"/chats/{chat_id}/messages/{message_id}/feedback",
            json={"value": value},
        )

    async def get_admin_stats(self) -> dict[str, Any]:
        """Получает агрегированную статистику из admin API."""

        response = await self._request_with_retry("GET", "/chats/admin/stats", headers=self._admin_headers())
        return response.json()

    async def get_admin_users(self, limit: int = 10) -> list[dict[str, Any]]:
        """Получает список последних пользователей из admin API."""

        response = await self._request_with_retry(
            "GET",
            "/chats/admin/users",
            params={"limit": limit},
            headers=self._admin_headers(),
        )
        return response.json()

    async def create_broadcast(self, message: str, interface_filter: str) -> dict[str, Any]:
        """Создает задачу рассылки в admin API."""

        response = await self._request_with_retry(
            "POST",
            "/chats/admin/broadcast",
            json={"message": message, "interface_filter": interface_filter},
            headers=self._admin_headers(),
        )
        return response.json()

    async def claim_broadcasts(self, interface_filter: str, limit: int = 10) -> list[dict[str, Any]]:
        """Забирает задания очереди рассылок для фонового обработчика бота."""

        response = await self._request_with_retry(
            "POST",
            "/chats/admin/broadcasts/claim",
            json={"interface_filter": interface_filter, "limit": limit},
            headers=self._admin_headers(),
        )
        return response.json()

    async def complete_broadcast(self, broadcast_id: UUID, status: str, error: str | None = None) -> None:
        """Подтверждает завершение или провал задачи массовой рассылки."""

        await self._request_with_retry(
            "POST",
            f"/chats/admin/broadcasts/{broadcast_id}/complete",
            json={"status": status, "error": error},
            headers=self._admin_headers(),
        )

    async def _request_with_retry(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Повторяет только сетевые ошибки подключения без ретрая HTTP-статусов."""

        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = await self._client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except _RETRYABLE_ERRORS as error:
                last_error = error
                if attempt == _MAX_ATTEMPTS:
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("HTTP-запрос завершился без результата.")

    async def _send_stream_request(self, request: httpx.Request) -> httpx.Response:
        """Открывает streaming-запрос и ретраит только ошибки подключения до первых байтов."""

        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return await self._client.send(request, stream=True)
            except _RETRYABLE_ERRORS as error:
                last_error = error
                if attempt == _MAX_ATTEMPTS:
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Streaming-запрос завершился без результата.")

    def _admin_headers(self) -> dict[str, str]:
        """Возвращает заголовки административного доступа к backend."""

        if not self._admin_token:
            raise RuntimeError("Для admin API не настроен X-Admin-Token.")
        return {"X-Admin-Token": self._admin_token}

    def _guess_filename(self, mime: str | None) -> str:
        """Подбирает расширение файла по MIME, чтобы backend корректно распознал формат."""

        mapping = {
            "audio/ogg": "voice.ogg",
            "audio/mpeg": "audio.mp3",
            "audio/mp4": "audio.m4a",
            "audio/x-m4a": "audio.m4a",
            "audio/wav": "audio.wav",
            "image/jpeg": "image.jpg",
            "image/png": "image.png",
            "application/pdf": "document.pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document.docx",
        }
        return mapping.get(mime or "", "file.bin")

    async def aclose(self) -> None:
        """Закрывает underlying HTTP-клиент, если он был создан внутри клиента."""

        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> BackendClient:
        """Поддерживает использование клиента в `async with`."""

        return self

    async def __aexit__(self, *args: object) -> None:
        """Автоматически закрывает клиент при выходе из контекста."""

        await self.aclose()
