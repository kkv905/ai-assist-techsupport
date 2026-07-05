"""Общие вспомогательные функции для Telegram-хендлеров."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from time import monotonic
from uuid import UUID

import httpx
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message

from bot.services.backend_client import BackendClient

STREAM_EDIT_INTERVAL_SECONDS = 0.75


async def resolve_chat_id(backend: BackendClient, user_id: int) -> UUID:
    """Возвращает идентификатор чата для Telegram-пользователя."""

    return await backend.get_or_create_chat(
        owner_external_id=str(user_id),
        interface="telegram",
    )


def format_backend_error(error: Exception) -> str:
    """Преобразует ошибки backend-клиента в понятный ответ пользователю."""

    if isinstance(error, httpx.ConnectError):
        return "Не удалось подключиться к сервису чата. Попробуйте позже."
    if isinstance(error, httpx.ReadTimeout):
        return "Сервис чата отвечает слишком долго. Попробуйте повторить запрос."
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        if status_code == 404:
            return "Чат не найден. Попробуйте снова начать диалог командой /start."
        if status_code >= 500:
            return "Сервис чата временно недоступен. Попробуйте позже."
        return "Сервис чата отклонил запрос. Проверьте команду и попробуйте еще раз."
    return "Во время обращения к сервису чата произошла ошибка."


async def _safe_edit_text(answer_message: Message, text: str) -> None:
    """Обновляет сообщение с обработкой flood control и безопасных ошибок Telegram."""

    try:
        await answer_message.edit_text(text)
    except TelegramRetryAfter as error:
        await asyncio.sleep(error.retry_after)
        await answer_message.edit_text(text)
    except TelegramBadRequest as error:
        if "message is not modified" not in str(error).lower():
            raise


async def stream_text_answer(answer_message: Message, stream: AsyncIterator[str]) -> str:
    """Постепенно обновляет сообщение, рендеря потоковый ответ backend."""

    buffer = ""
    rendered_text = ""
    last_edit_at = 0.0

    async for chunk in stream:
        buffer += chunk
        now = monotonic()
        if now - last_edit_at < STREAM_EDIT_INTERVAL_SECONDS:
            continue
        if buffer == rendered_text:
            continue

        await _safe_edit_text(answer_message, buffer)
        rendered_text = buffer
        last_edit_at = monotonic()

    if not buffer:
        buffer = "Сервис вернул пустой ответ."

    if buffer != rendered_text:
        await _safe_edit_text(answer_message, buffer)
    return buffer
