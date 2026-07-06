"""Тесты потокового рендера Telegram-ответов."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramRetryAfter

from bot.handlers.common import stream_to_chat


async def _stream_chunks() -> object:
    """Возвращает тестовый асинхронный поток из двух чанков."""

    for chunk in ("Привет", ", мир"):
        yield chunk


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

    assert result == "Привет, мир"
    assert bot.send_message_draft.await_count >= 2
    bot.send_message.assert_awaited_once_with(chat_id=100, text="Привет, мир")


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

    assert result == "Привет, мир"
    assert bot.send_message_draft.await_count >= 3
    bot.send_message.assert_awaited_once_with(chat_id=100, text="Привет, мир")
