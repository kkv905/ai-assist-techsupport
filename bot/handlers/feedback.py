"""Обработчики feedback-кнопок Telegram-бота."""

from __future__ import annotations

from uuid import UUID

import httpx
from aiogram import F, Router
from aiogram.types import CallbackQuery

from bot.handlers.common import format_backend_error, resolve_chat_id
from bot.services.backend_client import BackendClient

router = Router(name="feedback")


@router.callback_query(F.data.startswith("fb:"))
async def feedback_callback_handler(callback: CallbackQuery, backend: BackendClient) -> None:
    """Сохраняет пользовательский голос за ответ и убирает inline-клавиатуру."""

    if callback.data is None or callback.from_user is None:
        await callback.answer()
        return

    try:
        _, vote, raw_message_id = callback.data.split(":", maxsplit=2)
        message_id = UUID(raw_message_id)
        chat_id = await resolve_chat_id(backend, callback.from_user.id)
        await backend.submit_feedback(chat_id, message_id, vote)
        if callback.message is not None:
            await callback.message.edit_reply_markup(reply_markup=None)
        await callback.answer("Спасибо за оценку.")
    except ValueError:
        await callback.answer("Некорректные данные feedback.", show_alert=True)
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as error:
        await callback.answer(format_backend_error(error), show_alert=True)
