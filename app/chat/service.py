"""Сервисный слой для работы с чатами."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import tiktoken

from app.chat.domain import Chat, ChatMessage
from app.chat.repository import ChatRepository

logger = logging.getLogger(__name__)

CONTEXT_WINDOW_TOKENS = 8192
RESPONSE_TOKENS = 1024
SAFETY_MARGIN = 256
DEFAULT_CONTEXT_WINDOW = 10
DEFAULT_MODEL = "gpt-4o-mini"
TOKEN_ENCODING = tiktoken.get_encoding("o200k_base")


class ChatNotFoundError(Exception):
    """Сигнализирует, что запрошенный чат отсутствует в хранилище."""


class ChatService:
    """Оркестрирует работу репозитория, контекста и LLM-клиента."""

    def __init__(
        self,
        repository: ChatRepository,
        llm_client: Any,
        *,
        model: str = DEFAULT_MODEL,
        context_strategy: str = "sliding",
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        context_window_tokens: int = CONTEXT_WINDOW_TOKENS,
        response_tokens: int = RESPONSE_TOKENS,
        safety_margin: int = SAFETY_MARGIN,
    ) -> None:
        """Сохраняет зависимости и параметры формирования контекста."""

        self._repository = repository
        self._llm_client = llm_client
        self._model = model
        self._context_strategy = context_strategy
        self._context_window = context_window
        self._context_window_tokens = context_window_tokens
        self._response_tokens = response_tokens
        self._safety_margin = safety_margin

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        """Создает чат через репозиторий."""

        return await self._repository.create_chat(owner_external_id, interface, system_prompt)

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        """Возвращает метаданные чата по идентификатору."""

        return await self._repository.get_chat(chat_id)

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        """Возвращает историю чата в хронологическом порядке."""

        return await self._repository.list_messages(chat_id, limit=limit)

    async def clear_history(self, chat_id: UUID) -> None:
        """Запускает мягкую очистку истории сообщений."""

        await self._ensure_chat_exists(chat_id)
        await self._repository.soft_delete_messages(chat_id)

    async def send_message(self, chat_id: UUID, user_content: str) -> AsyncIterator[str]:
        """Сохраняет запрос пользователя, стримит ответ и пишет его в историю."""

        await self._ensure_chat_exists(chat_id)

        user_message = ChatMessage(chat_id=chat_id, role="user", content=user_content)
        await self._repository.append_message(chat_id, user_message)

        chat = await self._ensure_chat_exists(chat_id)
        history = await self._load_history(chat_id)
        llm_messages = await self._build_context(chat, history)
        budget = self._context_window_tokens - self._response_tokens - self._safety_margin
        llm_messages = fit_to_budget(llm_messages, budget)

        answer_parts: list[str] = []
        stream = await self._llm_client.chat.completions.create(
            model=self._model,
            messages=llm_messages,
            stream=True,
        )
        stream_interrupted = False

        try:
            async for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                answer_parts.append(delta)
                yield delta
        except Exception:
            stream_interrupted = True
            logger.warning("Потоковый ответ LLM прерван, сохраняю накопленный ответ.")
            raise
        finally:
            await stream.close()
            assistant_content = "".join(answer_parts)
            if assistant_content:
                assistant_message = ChatMessage(
                    chat_id=chat_id,
                    role="assistant",
                    content=assistant_content,
                )
                await self._repository.append_message(chat_id, assistant_message)
            elif stream_interrupted:
                logger.warning("Поток был прерван до получения текстовых токенов.")

    async def _load_history(self, chat_id: UUID) -> list[ChatMessage]:
        """Загружает историю чата по выбранной стратегии."""

        if self._context_strategy == "sliding":
            return await self._repository.list_messages(chat_id, limit=self._context_window)
        if self._context_strategy == "hybrid":
            return await self._repository.list_messages(chat_id, limit=200)
        raise ValueError(
            f"Неизвестная стратегия контекста: {self._context_strategy}. Ожидалось 'sliding' или 'hybrid'."
        )

    async def _build_context(
        self,
        chat: Chat,
        history: list[ChatMessage],
    ) -> list[dict[str, str]]:
        """Преобразует чат и историю в формат OpenAI messages."""

        messages: list[dict[str, str]] = []
        if chat.system_prompt:
            messages.append({"role": "system", "content": chat.system_prompt})

        if self._context_strategy == "sliding":
            messages.extend(_to_openai_messages(history))
            return messages

        summary_messages = await self._build_hybrid_context(history)
        messages.extend(summary_messages)
        return messages

    async def _build_hybrid_context(self, history: list[ChatMessage]) -> list[dict[str, str]]:
        """Строит гибридный контекст из summary и последних сообщений."""

        keep_recent = min(self._context_window, len(history))
        old_messages = history[:-keep_recent]
        recent_messages = history[-keep_recent:]
        if not old_messages:
            return _to_openai_messages(recent_messages)

        summarize_prompt = (
            "Сожми историю диалога. Обязательно перечисли темы, имена, числа, решения "
            "и нерешенные вопросы. Сохрани факты без домыслов."
        )
        summary_request = [
            {"role": "system", "content": summarize_prompt},
            {
                "role": "user",
                "content": "\n".join(f"{message.role}: {message.content}" for message in old_messages),
            },
        ]
        response = await self._llm_client.chat.completions.create(
            model=self._model,
            messages=summary_request,
            stream=False,
        )
        summary_text = response.choices[0].message.content or ""
        summary_message = {"role": "system", "content": f"Краткое содержание предыдущего диалога: {summary_text}"}
        return [summary_message, *_to_openai_messages(recent_messages)]

    async def _ensure_chat_exists(self, chat_id: UUID) -> Chat:
        """Проверяет наличие чата и выбрасывает доменную ошибку при отсутствии."""

        chat = await self._repository.get_chat(chat_id)
        if chat is None:
            raise ChatNotFoundError(f"Чат {chat_id} не найден.")
        return chat


def count_tokens(messages: list[dict[str, str]]) -> int:
    """Подсчитывает примерное число токенов с учетом ChatML overhead."""

    total = 2
    for message in messages:
        total += 4
        total += len(TOKEN_ENCODING.encode(message["role"]))
        total += len(TOKEN_ENCODING.encode(message["content"]))
    return total


def fit_to_budget(messages: list[dict[str, str]], budget: int) -> list[dict[str, str]]:
    """Обрезает сообщения с начала, сохраняя системное сообщение, если оно есть."""

    if count_tokens(messages) <= budget:
        return messages

    system_message: dict[str, str] | None = None
    history = messages
    if messages and messages[0]["role"] == "system":
        system_message = messages[0]
        history = messages[1:]

    trimmed_history = history[:]
    while trimmed_history:
        candidate = [system_message, *trimmed_history] if system_message else trimmed_history
        candidate = [message for message in candidate if message is not None]
        if count_tokens(candidate) <= budget:
            return candidate
        trimmed_history = trimmed_history[1:]

    return [system_message] if system_message and count_tokens([system_message]) <= budget else []


def _to_openai_messages(history: list[ChatMessage]) -> list[dict[str, str]]:
    """Преобразует доменные сообщения в формат OpenAI API."""

    return [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in history
    ]
