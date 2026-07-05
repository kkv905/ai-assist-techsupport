"""Командные хендлеры Telegram-бота."""

from __future__ import annotations

import httpx
from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.handlers.common import format_backend_error, resolve_chat_id
from bot.services.backend_client import BackendClient

router = Router(name="commands")


def _help_text() -> str:
    """Возвращает текст справки по доступным командам."""

    return (
        "Доступные команды:\n"
        "/start — создать или открыть диалог\n"
        "/help — показать эту справку\n"
        "/ask — запустить сценарий с выбором темы\n"
        "/clear — очистить историю чата\n"
        "/cancel — отменить активный сценарий"
    )


@router.message(Command("start"))
async def start_command(message: Message, backend: BackendClient) -> None:
    """Создает чат для пользователя и показывает краткую инструкцию."""

    if message.from_user is None:
        return

    try:
        await resolve_chat_id(backend, message.from_user.id)
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as error:
        await message.answer(format_backend_error(error))
        return

    await message.answer(
        "Привет! Я Telegram-клиент сервиса поддержки.\n"
        "Напишите вопрос текстом или используйте /ask для пошагового сценария.\n\n"
        f"{_help_text()}"
    )


@router.message(Command("help"))
async def help_command(message: Message) -> None:
    """Показывает список доступных команд."""

    await message.answer(_help_text())


@router.message(Command("clear"))
async def clear_command(message: Message, backend: BackendClient) -> None:
    """Очищает историю диалога пользователя на стороне backend."""

    if message.from_user is None:
        return

    try:
        chat_id = await resolve_chat_id(backend, message.from_user.id)
        await backend.clear_messages(chat_id)
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as error:
        await message.answer(format_backend_error(error))
        return

    await message.answer("История очищена.")


@router.message(Command("cancel"))
async def cancel_command(message: Message, state: FSMContext) -> None:
    """Сбрасывает активный FSM-сценарий пользователя."""

    current_state = await state.get_state()
    if current_state is None:
        await message.answer("Сейчас нет активного сценария.")
        return

    await state.clear()
    await message.answer("Текущее действие отменено.")
