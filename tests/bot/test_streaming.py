"""Тесты потокового рендера Telegram-ответов."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramRetryAfter

from bot.handlers.common import stream_text_answer


async def _stream_chunks() -> object:
    """Возвращает тестовый асинхронный поток из двух чанков."""

    for chunk in ("Привет", ", мир"):
        yield chunk


@pytest.mark.asyncio
async def test_stream_text_answer_retries_after_flood_control() -> None:
    """Проверяет повторное обновление сообщения после `retry_after`."""

    message = AsyncMock()
    message.edit_text = AsyncMock(
        side_effect=[
            TelegramRetryAfter(method=AsyncMock(), message="retry later", retry_after=0),
            None,
            None,
        ]
    )

    result = await stream_text_answer(message, _stream_chunks())

    assert result == "Привет, мир"
    assert message.edit_text.await_count == 3
