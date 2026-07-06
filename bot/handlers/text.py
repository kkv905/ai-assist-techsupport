"""Обработчики обычных текстовых сообщений Telegram-бота."""

from __future__ import annotations

import httpx
from aiogram import F, Router
from aiogram.types import Message

from bot.handlers.common import format_backend_error, resolve_chat_id, stream_to_chat
from bot.services.backend_client import BackendClient

router = Router(name="text")


@router.message(F.text & ~F.text.startswith("/"))
async def text_message_handler(message: Message, backend: BackendClient) -> None:
    """Отправляет пользовательский текст в backend и стримит ответ обратно."""

    if message.from_user is None or message.text is None:
        return

    try:
        chat_id = await resolve_chat_id(backend, message.from_user.id)
        await stream_to_chat(message, backend.send_message(chat_id, message.text))
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as error:
        await message.answer(format_backend_error(error))
