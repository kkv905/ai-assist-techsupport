"""Тесты FSM-сценария Telegram-бота."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

from bot.handlers.fsm import ask_command, topic_callback_handler
from bot.states import AskFlow


@pytest.mark.asyncio
async def test_ask_flow_moves_to_waiting_for_question_and_stores_topic() -> None:
    """Проверяет переход сценария `/ask` к ожиданию вопроса после выбора темы."""

    storage = MemoryStorage()
    state = FSMContext(
        storage=storage,
        key=StorageKey(bot_id=1, chat_id=100, user_id=100),
    )
    message = AsyncMock()
    message.answer = AsyncMock()

    await ask_command(message, state)

    assert await state.get_state() == AskFlow.waiting_for_topic.state
    message.answer.assert_awaited_once()

    callback_message = AsyncMock()
    callback_message.edit_text = AsyncMock()
    callback = SimpleNamespace(
        data="topic:billing",
        message=callback_message,
        answer=AsyncMock(),
    )

    await topic_callback_handler(callback, state)

    assert await state.get_state() == AskFlow.waiting_for_question.state
    assert await state.get_data() == {"topic": "billing"}
    callback.answer.assert_awaited_once()
