"""HTTP-клиент для работы Telegram-бота с chat-backend."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import httpx

_DEFAULT_TIMEOUT = httpx.Timeout(connect=3.0, read=60.0, write=10.0, pool=5.0)
_STREAM_TIMEOUT = httpx.Timeout(connect=3.0, read=120.0, write=10.0, pool=5.0)
_RETRYABLE_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)
_MAX_ATTEMPTS = 3


class BackendClient:
    """Инкапсулирует HTTP-вызовы к backend сервиса чатов."""

    def __init__(
        self,
        base_url: str,
        *,
        http_client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Создает или принимает готовый `httpx.AsyncClient` с базовым адресом backend."""

        self._owns_client = http_client is None
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
    ) -> AsyncIterator[str]:
        """Отправляет сообщение и по мере поступления отдает SSE-токены ответа."""

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
                if payload.get("type") == "token":
                    yield payload.get("delta", "")
                elif payload.get("type") == "done":
                    return
        finally:
            await response.aclose()

    async def clear_messages(self, chat_id: UUID) -> None:
        """Очищает историю сообщений чата на стороне backend."""

        await self._request_with_retry("DELETE", f"/chats/{chat_id}/messages")

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
