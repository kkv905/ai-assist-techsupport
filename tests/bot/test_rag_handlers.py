"""Проверки выбора RAG-потока текстовыми Telegram-хендлерами."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

from bot.handlers.fsm import ask_question_handler
from bot.handlers.text import text_message_handler
from bot.states import AskFlow


@pytest.mark.asyncio
async def test_text_handler_uses_rag_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Обычное текстовое сообщение направляется в RAG, а не в общий чат."""

    chat_id = uuid4()
    backend = SimpleNamespace(
        get_or_create_chat=AsyncMock(return_value=chat_id),
        send_rag_message=Mock(return_value=object()),
        send_message=Mock(),
    )
    stream = AsyncMock()
    monkeypatch.setattr("bot.handlers.text.stream_to_chat", stream)
    message = SimpleNamespace(from_user=SimpleNamespace(id=100), text="Как войти в VPN?", answer=AsyncMock())

    await text_message_handler(message, backend)

    backend.send_rag_message.assert_called_once_with(chat_id, "Как войти в VPN?")
    backend.send_message.assert_not_called()
    stream.assert_awaited_once()


@pytest.mark.asyncio
async def test_fsm_question_handler_uses_rag_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Вопрос после выбора темы также направляется в RAG-поток."""

    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=StorageKey(bot_id=1, chat_id=100, user_id=100))
    await state.set_state(AskFlow.waiting_for_question)
    await state.update_data(topic="billing")
    chat_id = uuid4()
    backend = SimpleNamespace(
        get_or_create_chat=AsyncMock(return_value=chat_id),
        send_rag_message=Mock(return_value=object()),
        send_message=Mock(),
    )
    stream = AsyncMock()
    monkeypatch.setattr("bot.handlers.fsm.stream_to_chat", stream)
    message = SimpleNamespace(from_user=SimpleNamespace(id=100), text="Где счёт?", answer=AsyncMock())

    await ask_question_handler(message, state, backend)

    backend.send_rag_message.assert_called_once()
    backend.send_message.assert_not_called()
    assert await state.get_state() is None
