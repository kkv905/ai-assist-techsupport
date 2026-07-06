"""Общие вспомогательные функции для Telegram-хендлеров."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from time import monotonic
from uuid import UUID

import httpx
from aiogram.enums.chat_action import ChatAction
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message

from bot.keyboards.inline import feedback_kb
from bot.services.backend_client import BackendClient

_DRAFT_UPDATE_INTERVAL_SECONDS = 1.0


@dataclass(slots=True)
class StreamResult:
    """Содержит финальный текст ответа и идентификатор backend-сообщения."""

    text: str
    message_id: UUID | None


async def resolve_chat_id(backend: BackendClient, user_id: int) -> UUID:
    """Возвращает идентификатор чата для Telegram-пользователя."""

    return await backend.get_or_create_chat(
        owner_external_id=str(user_id),
        interface="telegram",
    )



def format_backend_error(error: Exception) -> str:
    """Преобразует ошибки backend-клиента в понятный ответ пользователю."""

    if isinstance(error, httpx.ConnectError):
        return "Сервис недоступен, попробуйте позже."
    if isinstance(error, httpx.ReadTimeout):
        return "Ответ занимает слишком долго."
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        if status_code == 403:
            return "Сообщение заблокировано модерацией. Попробуйте переформулировать запрос."
        if status_code == 404:
            return "Чат не найден. Попробуйте снова начать диалог командой /start."
        if status_code == 409:
            return "Такой голос уже был сохранен раньше."
        if status_code == 429:
            return "Слишком много запросов, подождите минуту."
        if status_code >= 500:
            return "Внутренняя ошибка сервиса."
        return "Сервис чата отклонил запрос. Проверьте данные и попробуйте еще раз."
    return "Во время обращения к сервису чата произошла ошибка."


async def stream_to_chat(message: Message, events) -> StreamResult:
    """Показывает поток ответа через Telegram draft и фиксирует итог обычным сообщением."""

    draft_id = uuid.uuid4().int & 0xFFFFFFFF
    buffer = ""
    rendered_text = ""
    last_update_at = 0.0
    first_token = asyncio.Event()
    typing_task = asyncio.create_task(_send_typing_until_first_token(message, first_token))
    backend_message_id: UUID | None = None

    try:
        await _safe_send_message_draft(message, "", draft_id)
        async for event in events:
            if event.type == "done":
                backend_message_id = event.message_id
                continue
            delta = event.delta or ""
            if not first_token.is_set() and delta:
                first_token.set()
            buffer += delta
            now = monotonic()
            if not buffer.strip():
                continue
            if buffer == rendered_text:
                continue
            if now - last_update_at < _DRAFT_UPDATE_INTERVAL_SECONDS:
                continue

            await _safe_send_message_draft(message, buffer, draft_id)
            rendered_text = buffer
            last_update_at = monotonic()
    finally:
        first_token.set()
        await typing_task

    if not buffer:
        buffer = "Сервис вернул пустой ответ."

    if buffer != rendered_text:
        await _safe_send_message_draft(message, buffer, draft_id)

    await _safe_send_message(message, buffer, backend_message_id)
    return StreamResult(text=buffer, message_id=backend_message_id)


async def _safe_send_message_draft(message: Message, text: str, draft_id: int) -> None:
    """Безопасно обновляет Telegram draft с учетом flood control и повторов."""

    try:
        await message.bot.send_message_draft(
            chat_id=message.chat.id,
            text=text,
            draft_id=draft_id,
        )
    except TelegramRetryAfter as error:
        await asyncio.sleep(error.retry_after)
        await message.bot.send_message_draft(
            chat_id=message.chat.id,
            text=text,
            draft_id=draft_id,
        )
    except TelegramBadRequest as error:
        if "message is not modified" not in str(error).lower():
            raise


async def _safe_send_message(message: Message, text: str, backend_message_id: UUID | None) -> None:
    """Отправляет финальное сообщение с учетом flood control и feedback-кнопок."""

    reply_markup = feedback_kb(backend_message_id) if backend_message_id else None
    try:
        await message.bot.send_message(chat_id=message.chat.id, text=text, reply_markup=reply_markup)
    except TelegramRetryAfter as error:
        await asyncio.sleep(error.retry_after)
        await message.bot.send_message(chat_id=message.chat.id, text=text, reply_markup=reply_markup)


async def _send_typing_until_first_token(message: Message, first_token: asyncio.Event) -> None:
    """Периодически отправляет chat action, пока backend не начал стримить токены."""

    while not first_token.is_set():
        await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
        try:
            await asyncio.wait_for(first_token.wait(), timeout=4.0)
        except TimeoutError:
            continue

