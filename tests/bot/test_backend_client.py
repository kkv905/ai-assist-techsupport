"""Тесты backend-клиента Telegram-бота."""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest

from bot.services.backend_client import BackendClient


@pytest.mark.asyncio
async def test_get_or_create_chat_returns_uuid() -> None:
    """Проверяет разбор идентификатора чата из ответа backend."""

    expected_chat_id = uuid4()

    async def handler(request: httpx.Request) -> httpx.Response:
        """Возвращает фиктивный ответ на создание чата."""

        assert request.method == "POST"
        assert request.url.path == "/chats"
        return httpx.Response(201, json={"chat_id": str(expected_chat_id)})

    client = BackendClient("http://testserver", transport=httpx.MockTransport(handler))
    try:
        chat_id = await client.get_or_create_chat("123", "telegram")
    finally:
        await client.aclose()

    assert chat_id == expected_chat_id


@pytest.mark.asyncio
async def test_send_message_parses_sse_stream() -> None:
    """Проверяет разбор SSE-ответа в поток событий."""

    chat_id = uuid4()
    assistant_message_id = uuid4()

    async def handler(request: httpx.Request) -> httpx.Response:
        """Возвращает поток в формате SSE с JSON-полезной нагрузкой."""

        assert request.method == "POST"
        assert request.url.path == f"/chats/{chat_id}/messages"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                'data: {"type":"token","delta":"Часть 1"}\n\n'
                'data: {"type":"token","delta":" и часть 2"}\n\n'
                f'data: {{"type":"done","message_id":"{assistant_message_id}"}}\n\n'
            ),
        )

    client = BackendClient("http://testserver", transport=httpx.MockTransport(handler))
    try:
        events = [event async for event in client.send_message(chat_id, "Привет")]
    finally:
        await client.aclose()

    assert [event.delta for event in events if event.type == "token"] == ["Часть 1", " и часть 2"]
    assert events[-1].type == "done"
    assert events[-1].message_id == assistant_message_id


@pytest.mark.asyncio
async def test_send_message_sends_multipart_when_media_present() -> None:
    """Проверяет, что медиа отправляется в backend как multipart/form-data."""

    chat_id = uuid4()

    async def handler(request: httpx.Request) -> httpx.Response:
        """Проверяет multipart-тело запроса с бинарным вложением."""

        body = await request.aread()
        assert request.method == "POST"
        assert request.url.path == f"/chats/{chat_id}/messages"
        assert "multipart/form-data" in request.headers["content-type"]
        assert b'name="content"' in body
        assert b'name="media"; filename="document.pdf"' in body
        assert b"%PDF-test" in body
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text='data: {"type":"done"}\n\n',
        )

    client = BackendClient("http://testserver", transport=httpx.MockTransport(handler))
    try:
        events = [
            event
            async for event in client.send_message(
                chat_id,
                "Проверь документ",
                media=b"%PDF-test",
                mime="application/pdf",
            )
        ]
    finally:
        await client.aclose()

    assert events[-1].type == "done"


@pytest.mark.asyncio
async def test_clear_messages_sends_delete_to_expected_url() -> None:
    """Проверяет корректный HTTP-метод и URL очистки истории."""

    chat_id = uuid4()

    async def handler(request: httpx.Request) -> httpx.Response:
        """Проверяет параметры DELETE-запроса."""

        assert request.method == "DELETE"
        assert request.url.path == f"/chats/{chat_id}/messages"
        return httpx.Response(200, json={"status": "ok"})

    client = BackendClient("http://testserver", transport=httpx.MockTransport(handler))
    try:
        await client.clear_messages(chat_id)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_admin_stats_uses_admin_header() -> None:
    """Проверяет, что admin API вызывается с заголовком X-Admin-Token."""

    async def handler(request: httpx.Request) -> httpx.Response:
        """Проверяет заголовки запроса к admin API."""

        assert request.headers["X-Admin-Token"] == "secret"
        return httpx.Response(200, json={"total_messages": 1, "active_users": 1, "avg_latency_ms": 1, "moderation_block_rate": 0, "feedback_up_ratio": 1})

    client = BackendClient("http://testserver", admin_token="secret", transport=httpx.MockTransport(handler))
    try:
        payload = await client.get_admin_stats()
    finally:
        await client.aclose()

    assert payload["total_messages"] == 1
