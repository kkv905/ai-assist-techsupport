"""Тесты потокового рендера Telegram-ответов."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from aiogram.exceptions import TelegramRetryAfter

from bot.handlers.common import format_sources, stream_to_chat
from bot.services.backend_client import BackendStreamEvent


async def _stream_chunks() -> object:
    """Возвращает тестовый асинхронный поток из двух чанков и финального done."""

    for event in (
        BackendStreamEvent(type="token", delta="Привет"),
        BackendStreamEvent(type="token", delta=", мир"),
        BackendStreamEvent(type="done", message_id=uuid4()),
    ):
        yield event


async def _stream_with_sources(sources: list[dict[str, object]], answer: str = "Ответ") -> object:
    """Возвращает поток ответа, источников и завершающего события."""

    for event in (
        BackendStreamEvent(type="token", delta=answer),
        BackendStreamEvent(type="sources", sources=sources),
        BackendStreamEvent(type="done", message_id=uuid4()),
    ):
        yield event


@pytest.mark.asyncio
async def test_stream_to_chat_uses_draft_and_sends_final_message() -> None:
    """Проверяет, что ответ стримится через draft и затем фиксируется обычным сообщением."""

    bot = AsyncMock()
    message = SimpleNamespace(
        chat=SimpleNamespace(id=100),
        bot=bot,
    )
    bot.send_message_draft = AsyncMock()
    bot.send_message = AsyncMock()
    bot.send_chat_action = AsyncMock()

    result = await stream_to_chat(message, _stream_chunks())

    assert result.text == "Привет, мир"
    assert result.message_id is not None
    assert bot.send_message_draft.await_count >= 2
    assert bot.send_message.await_count == 1
    assert bot.send_message.await_args.kwargs["chat_id"] == 100
    assert bot.send_message.await_args.kwargs["text"] == "Привет, мир"
    assert bot.send_message.await_args.kwargs["reply_markup"] is not None


@pytest.mark.asyncio
async def test_stream_to_chat_retries_after_flood_control_on_draft() -> None:
    """Проверяет повтор обновления draft после TelegramRetryAfter."""

    bot = AsyncMock()
    message = SimpleNamespace(
        chat=SimpleNamespace(id=100),
        bot=bot,
    )
    bot.send_message_draft = AsyncMock(
        side_effect=[
            TelegramRetryAfter(method=AsyncMock(), message="retry later", retry_after=0),
            None,
            None,
            None,
        ]
    )
    bot.send_message = AsyncMock()
    bot.send_chat_action = AsyncMock()

    result = await stream_to_chat(message, _stream_chunks())

    assert result.text == "Привет, мир"
    assert bot.send_message_draft.await_count >= 3
    assert bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_stream_to_chat_adds_sources_only_to_final_message() -> None:
    """Финальное сообщение содержит очищенные источники в порядке SSE, draft — нет."""

    bot = AsyncMock()
    message = SimpleNamespace(chat=SimpleNamespace(id=100), bot=bot)
    sources = [
        {"file_name": "04-printer.txt", "snippet": "Пустые\n страницы   удалите очередь."},
        {"file_name": "05-network.txt", "snippet": "Проверьте подключение."},
    ]

    result = await stream_to_chat(message, _stream_with_sources(sources))

    final_text = bot.send_message.await_args.kwargs["text"]
    assert result.text == final_text
    assert final_text == (
        "Ответ\n\nИсточники:\n"
        "1. 04-printer.txt — Пустые страницы удалите очередь.\n"
        "2. 05-network.txt — Проверьте подключение."
    )
    assert bot.send_message.await_args.kwargs["reply_markup"] is not None
    assert all("Источники:" not in call.kwargs["text"] for call in bot.send_message_draft.await_args_list)


def test_format_sources_omits_incomplete_data_and_internal_fields() -> None:
    """Неполный источник не отображается и служебные поля не попадают в текст."""

    assert format_sources([{"file_name": "04-printer.txt", "id": "secret", "score": 0.99}]) == ""
    assert format_sources([{"file_name": "", "snippet": "Фрагмент", "path": "/internal/doc"}]) == ""


@pytest.mark.asyncio
async def test_stream_to_chat_limits_long_answer_and_snippet() -> None:
    """Итоговое сообщение соответствует лимиту Telegram и сохраняет блок источников."""

    bot = AsyncMock()
    message = SimpleNamespace(chat=SimpleNamespace(id=100), bot=bot)
    long_answer = "А" * 5000
    long_snippet = "фрагмент " * 100

    await stream_to_chat(
        message,
        _stream_with_sources([{"file_name": "data/rag/04-printer.txt", "snippet": long_snippet}], long_answer),
    )

    final_text = bot.send_message.await_args.kwargs["text"]
    assert len(final_text) <= 4096
    assert "Источники:\n1. 04-printer.txt" in final_text
    assert "…" in final_text
