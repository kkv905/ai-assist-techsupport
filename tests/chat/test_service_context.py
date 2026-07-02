"""Тесты сервисного слоя чата."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.chat.domain import Chat, ChatMessage
from app.chat.service import ChatNotFoundError, ChatService, fit_to_budget


class InMemoryChatRepository:
    """Простая in-memory реализация репозитория для unit-тестов сервиса."""

    def __init__(self) -> None:
        """Инициализирует словари чатов и сообщений."""

        self.chats: dict[UUID, Chat] = {}
        self.messages: dict[UUID, list[ChatMessage]] = {}

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        """Создает чат и сохраняет его в памяти."""

        chat = Chat(owner_external_id=owner_external_id, interface=interface, system_prompt=system_prompt)
        self.chats[chat.id] = chat
        self.messages[chat.id] = []
        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        """Возвращает чат из in-memory хранилища."""

        return self.chats.get(chat_id)

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage:
        """Добавляет сообщение в память."""

        self.messages.setdefault(chat_id, []).append(message)
        self.messages[chat_id].sort(key=lambda item: item.created_at)
        return message

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        """Возвращает последние сообщения в хронологическом порядке."""

        return self.messages.get(chat_id, [])[-limit:]

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        """Очищает историю чата в памяти."""

        self.messages[chat_id] = []


class FakeStream:
    """Имитирует поток событий OpenAI SDK."""

    def __init__(self, chunks: list[str], *, fail_after_first: bool = False) -> None:
        """Запоминает чанки и режим аварийного завершения потока."""

        self._chunks = chunks
        self._fail_after_first = fail_after_first
        self._closed = False

    def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        """Возвращает асинхронный итератор по чанкам."""

        return self._iterate()

    async def _iterate(self) -> AsyncIterator[SimpleNamespace]:
        """Отдает чанки как объекты, похожие на события OpenAI SDK."""

        for index, chunk in enumerate(self._chunks):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=chunk))])
            if self._fail_after_first and index == 0:
                raise RuntimeError("Поток оборвался")

    async def close(self) -> None:
        """Помечает поток как закрытый."""

        self._closed = True


class FakeChatCompletions:
    """Имитирует API `chat.completions` для тестов сервиса."""

    def __init__(self, stream_chunks: list[str], *, fail_after_first: bool = False) -> None:
        """Подготавливает историю вызовов и сценарии ответов."""

        self.calls: list[dict[str, object]] = []
        self._stream_chunks = stream_chunks
        self._fail_after_first = fail_after_first

    async def create(self, *, model: str, messages: list[dict[str, str]], stream: bool):
        """Возвращает либо поток, либо синтетический summary-ответ."""

        self.calls.append({"model": model, "messages": messages, "stream": stream})
        if stream:
            return FakeStream(self._stream_chunks, fail_after_first=self._fail_after_first)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Сводка"))])


class FakeLLMClient:
    """Имитирует минимальную структуру клиента OpenAI для сервиса чата."""

    def __init__(self, stream_chunks: list[str], *, fail_after_first: bool = False) -> None:
        """Создает вложенную структуру `chat.completions`."""

        self.chat = SimpleNamespace(
            completions=FakeChatCompletions(stream_chunks, fail_after_first=fail_after_first),
        )


@pytest.mark.asyncio
async def test_send_message_builds_sliding_window_context() -> None:
    """Проверяет сборку контекста по стратегии sliding window."""

    repository = InMemoryChatRepository()
    chat = await repository.create_chat("user-1", "web", "Ты помощник поддержки")
    await repository.append_message(
        chat.id,
        ChatMessage(
            chat_id=chat.id,
            role="user",
            content="Первый вопрос",
            created_at=datetime.now(UTC),
        ),
    )
    await repository.append_message(
        chat.id,
        ChatMessage(
            chat_id=chat.id,
            role="assistant",
            content="Первый ответ",
            created_at=datetime.now(UTC) + timedelta(seconds=1),
        ),
    )
    llm_client = FakeLLMClient(["Готов", "о"])
    service = ChatService(
        repository=repository,
        llm_client=llm_client,
        context_window=3,
        context_window_tokens=10000,
        response_tokens=16,
        safety_margin=0,
    )

    chunks = [chunk async for chunk in service.send_message(chat.id, "Второй вопрос")]

    assert chunks == ["Готов", "о"]
    payload = llm_client.chat.completions.calls[0]["messages"]
    assert payload[0] == {"role": "system", "content": "Ты помощник поддержки"}
    assert {"role": "user", "content": "Второй вопрос"} in payload
    stored_messages = await repository.list_messages(chat.id)
    assert any(message.role == "assistant" and message.content == "Готово" for message in stored_messages)


@pytest.mark.asyncio
async def test_send_message_saves_partial_answer_when_stream_breaks() -> None:
    """Проверяет сохранение частичного ответа при аварийном завершении стрима."""

    repository = InMemoryChatRepository()
    chat = await repository.create_chat("user-1", "web")
    llm_client = FakeLLMClient(["Часть ответа"], fail_after_first=True)
    service = ChatService(repository=repository, llm_client=llm_client)

    with pytest.raises(RuntimeError):
        async for _ in service.send_message(chat.id, "Проблема"):
            pass

    messages = await repository.list_messages(chat.id)
    assert messages[-1].role == "assistant"
    assert messages[-1].content == "Часть ответа"


@pytest.mark.asyncio
async def test_clear_history_delegates_to_repository() -> None:
    """Проверяет очистку истории через сервисный слой."""

    repository = InMemoryChatRepository()
    chat = await repository.create_chat("user-1", "web")
    await repository.append_message(chat.id, ChatMessage(chat_id=chat.id, role="user", content="До очистки"))
    service = ChatService(repository=repository, llm_client=FakeLLMClient(["ok"]))

    await service.clear_history(chat.id)

    assert await repository.list_messages(chat.id) == []


@pytest.mark.asyncio
async def test_send_message_raises_for_unknown_chat() -> None:
    """Проверяет доменную ошибку при отправке сообщения в неизвестный чат."""

    service = ChatService(repository=InMemoryChatRepository(), llm_client=FakeLLMClient(["ok"]))

    with pytest.raises(ChatNotFoundError):
        async for _ in service.send_message(uuid4(), "Привет"):
            pass


def test_fit_to_budget_keeps_system_message() -> None:
    """Проверяет, что при обрезке контекста сохраняется системное сообщение."""

    messages = [
        {"role": "system", "content": "Системные правила"},
        {"role": "user", "content": "Очень длинное сообщение " * 20},
        {"role": "assistant", "content": "Еще один длинный ответ " * 20},
    ]

    trimmed = fit_to_budget(messages, budget=20)

    assert trimmed == [{"role": "system", "content": "Системные правила"}] or trimmed == []

