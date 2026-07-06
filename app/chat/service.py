"""Сервисный слой для работы с чатами."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import structlog
import tiktoken
from fastapi import HTTPException

from app.chat.domain import Chat, ChatMessage, ChatStreamEvent, FeedbackValue, MediaRef, MessageFeedback
from app.chat.repository import ChatRepository
from app.moderation.domain import ModerationResult
from app.moderation.service import ModerationService
from app.observability.pii import prompt_hash, redact_pii

logger = structlog.get_logger("app.chat")

CONTEXT_WINDOW_TOKENS = 8192
RESPONSE_TOKENS = 1024
SAFETY_MARGIN = 256
DEFAULT_CONTEXT_WINDOW = 10
DEFAULT_MODEL = "gpt-5.2"
TOKEN_ENCODING = tiktoken.get_encoding("o200k_base")
BLOCKED_OUTPUT_TEXT = "Не могу показать ответ — он мог нарушить правила"


class ChatNotFoundError(Exception):
    """Сигнализирует, что запрошенный чат отсутствует в хранилище."""


class ChatService:
    """Оркестрирует работу репозитория, контекста, модерации и LLM-клиента."""

    def __init__(
        self,
        repository: ChatRepository,
        llm_client: Any,
        moderation_service: ModerationService,
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
        self._moderation_service = moderation_service
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

    async def append_message(
        self,
        chat_id: UUID,
        role: str,
        content: str,
        media_refs: MediaRef | None = None,
    ) -> ChatMessage:
        """Сохраняет сообщение в историю без запуска модели."""

        await self._ensure_chat_exists(chat_id)
        message = ChatMessage(
            chat_id=chat_id,
            role=role,
            content=content,
            media_refs=media_refs,
        )
        return await self._repository.append_message(chat_id, message)

    async def clear_history(self, chat_id: UUID) -> None:
        """Запускает мягкую очистку истории сообщений."""

        await self._ensure_chat_exists(chat_id)
        await self._repository.soft_delete_messages(chat_id)

    async def save_feedback(
        self,
        chat_id: UUID,
        message_id: UUID,
        value: FeedbackValue,
    ) -> MessageFeedback:
        """Сохраняет feedback за сообщение внутри существующего чата."""

        chat = await self._ensure_chat_exists(chat_id)
        return await self._repository.save_feedback(message_id, chat.owner_external_id, value)

    async def send_message(
        self,
        chat_id: UUID,
        user_content: str,
        media_refs: MediaRef | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Сохраняет запрос пользователя, проверяет модерацию и отдает итоговый поток SSE-событий."""

        chat = await self._ensure_chat_exists(chat_id)

        input_result = await self._moderation_service.check_input(user_content)
        await self._audit_moderation(
            chat_id=chat_id,
            owner_external_id=chat.owner_external_id,
            direction="input",
            content=user_content,
            result=input_result,
        )
        if not input_result.allowed:
            raise HTTPException(
                status_code=403,
                detail={"code": "moderation_blocked", "categories": input_result.categories},
            )

        user_message = ChatMessage(
            chat_id=chat_id,
            role="user",
            content=user_content,
            media_refs=media_refs,
        )
        await self._repository.append_message(chat_id, user_message)

        history = await self._load_history(chat_id)
        llm_messages = await self._build_context(chat, history)
        budget = self._context_window_tokens - self._response_tokens - self._safety_margin
        llm_messages = fit_to_budget(llm_messages, budget)

        answer_parts: list[str] = []
        stream = await self._llm_client.chat.completions.create(
            model=self._model,
            messages=llm_messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        stream_error: Exception | None = None
        started_at = time.perf_counter()

        try:
            async for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                answer_parts.append(delta)
        except Exception as error:
            stream_error = error
            logger.warning("llm_stream_interrupted")
        finally:
            await stream.close()

        latency_ms = round((time.perf_counter() - started_at) * 1000)
        assistant_content = "".join(answer_parts)
        if assistant_content:
            output_result = await self._moderation_service.check_output(assistant_content)
            await self._audit_moderation(
                chat_id=chat_id,
                owner_external_id=chat.owner_external_id,
                direction="output",
                content=assistant_content,
                result=output_result,
            )
            if not output_result.allowed:
                assistant_content = BLOCKED_OUTPUT_TEXT
                answer_parts = [assistant_content]

        assistant_message_id: UUID | None = None
        if assistant_content:
            assistant_message = ChatMessage(
                chat_id=chat_id,
                role="assistant",
                content=assistant_content,
                latency_ms=latency_ms,
            )
            stored_message = await self._repository.append_message(chat_id, assistant_message)
            assistant_message_id = stored_message.id
        elif stream_error is not None:
            logger.warning("llm_stream_interrupted_before_text")

        if stream_error is not None:
            raise stream_error

        for chunk in answer_parts:
            yield ChatStreamEvent(type="token", delta=chunk)
        yield ChatStreamEvent(type="done", message_id=assistant_message_id)

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
    ) -> list[dict[str, Any]]:
        """Преобразует чат и историю в формат OpenAI messages."""

        messages: list[dict[str, Any]] = []
        if chat.system_prompt:
            messages.append({"role": "system", "content": chat.system_prompt})

        if self._context_strategy == "sliding":
            messages.extend(_to_openai_messages(history))
            return messages

        summary_messages = await self._build_hybrid_context(history)
        messages.extend(summary_messages)
        return messages

    async def _build_hybrid_context(self, history: list[ChatMessage]) -> list[dict[str, Any]]:
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

    async def _audit_moderation(
        self,
        *,
        chat_id: UUID,
        owner_external_id: str,
        direction: str,
        content: str,
        result: ModerationResult,
    ) -> None:
        """Логирует и сохраняет результат проверки модерации без сырого текста."""

        text_hash = prompt_hash(content)
        logger.info(
            "moderation_checked",
            direction=direction,
            allowed=result.allowed,
            categories=result.categories,
            reasons=result.reasons,
            blocked_by=result.blocked_by,
            text_hash=text_hash,
            text_preview=redact_pii(content)[:120],
        )
        await self._repository.record_moderation_event(
            chat_id=chat_id,
            owner_external_id=owner_external_id,
            direction=direction,
            allowed=result.allowed,
            categories=result.categories,
            reasons=result.reasons,
            blocked_by=result.blocked_by,
            text_hash=text_hash,
        )


def count_tokens(messages: list[dict[str, Any]]) -> int:
    """Подсчитывает примерное число токенов с учетом ChatML overhead."""

    total = 2
    for message in messages:
        total += 4
        total += len(TOKEN_ENCODING.encode(message["role"]))
        total += len(TOKEN_ENCODING.encode(_flatten_content_for_tokens(message["content"])))
    return total



def fit_to_budget(messages: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Обрезает сообщения с начала, сохраняя системное сообщение, если оно есть."""

    if count_tokens(messages) <= budget:
        return messages

    system_message: dict[str, Any] | None = None
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



def _to_openai_messages(history: list[ChatMessage]) -> list[dict[str, Any]]:
    """Преобразует доменные сообщения в формат OpenAI API."""

    messages: list[dict[str, Any]] = []
    for message in history:
        if message.media_refs is None:
            messages.append({"role": message.role, "content": message.content})
            continue

        parts: list[dict[str, Any]] = []
        if message.content:
            parts.append({"type": "text", "text": message.content})
        parts.append(message.media_refs.part)
        messages.append({"role": message.role, "content": parts})
    return messages



def _flatten_content_for_tokens(content: Any) -> str:
    """Преобразует мультимодальный content в текст для грубой оценки токенов."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                parts.append("[изображение]")
        return "\n".join(parts)
    return str(content)
