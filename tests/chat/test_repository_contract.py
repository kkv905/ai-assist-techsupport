"""Контрактные тесты для репозиториев чата."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.chat.domain import ChatMessage


@pytest.mark.asyncio
async def test_create_chat_and_read_back(chat_repository) -> None:
    """Проверяет создание чата и последующее чтение его метаданных."""

    chat = await chat_repository.create_chat("user-1", "web", "Системный промпт")
    loaded = await chat_repository.get_chat(chat.id)

    assert loaded is not None
    assert loaded.id == chat.id
    assert loaded.owner_external_id == "user-1"
    assert loaded.system_prompt == "Системный промпт"


@pytest.mark.asyncio
async def test_append_and_list_messages_in_chronological_order(chat_repository) -> None:
    """Проверяет, что сообщения возвращаются от старых к новым."""

    chat = await chat_repository.create_chat("user-1", "web")
    first = ChatMessage(
        chat_id=chat.id,
        role="user",
        content="Первое",
        created_at=datetime.now(UTC),
    )
    second = ChatMessage(
        chat_id=chat.id,
        role="assistant",
        content="Второе",
        created_at=datetime.now(UTC) + timedelta(seconds=1),
    )

    await chat_repository.append_message(chat.id, second)
    await chat_repository.append_message(chat.id, first)

    messages = await chat_repository.list_messages(chat.id)

    assert [message.content for message in messages] == ["Первое", "Второе"]


@pytest.mark.asyncio
async def test_list_messages_returns_last_n_messages(chat_repository) -> None:
    """Проверяет, что limit отдает последние сообщения, а не первые."""

    chat = await chat_repository.create_chat("user-1", "web")
    for index in range(5):
        message = ChatMessage(
            chat_id=chat.id,
            role="user",
            content=f"Сообщение {index}",
            created_at=datetime.now(UTC) + timedelta(seconds=index),
        )
        await chat_repository.append_message(chat.id, message)

    messages = await chat_repository.list_messages(chat.id, limit=2)

    assert [message.content for message in messages] == ["Сообщение 3", "Сообщение 4"]


@pytest.mark.asyncio
async def test_soft_delete_hides_old_messages_but_keeps_new_ones(chat_repository) -> None:
    """Проверяет мягкую очистку истории и сохранение новых сообщений после нее."""

    chat = await chat_repository.create_chat("user-1", "web")
    await chat_repository.append_message(chat.id, ChatMessage(chat_id=chat.id, role="user", content="Старое"))
    await chat_repository.soft_delete_messages(chat.id)

    cleared = await chat_repository.list_messages(chat.id)
    assert cleared == []

    await chat_repository.append_message(chat.id, ChatMessage(chat_id=chat.id, role="user", content="Новое"))
    restored = await chat_repository.list_messages(chat.id)

    assert [message.content for message in restored] == ["Новое"]


@pytest.mark.asyncio
async def test_get_unknown_chat_returns_none(chat_repository) -> None:
    """Проверяет безопасное поведение при чтении неизвестного чата."""

    loaded = await chat_repository.get_chat(uuid4())

    assert loaded is None
