"""FSM-хендлеры Telegram-бота."""

from __future__ import annotations

import httpx
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.handlers.common import format_backend_error, resolve_chat_id, stream_to_chat
from bot.keyboards.inline import get_topic_title, topics_kb
from bot.services.backend_client import BackendClient
from bot.states import AskFlow

router = Router(name="fsm")


@router.message(Command("ask"))
async def ask_command(message: Message, state: FSMContext) -> None:
    """Запускает FSM-сценарий выбора темы и вопроса."""

    await state.set_state(AskFlow.waiting_for_topic)
    await message.answer(
        "Выберите тему вопроса:",
        reply_markup=topics_kb(),
    )


@router.callback_query(AskFlow.waiting_for_topic, F.data.startswith("topic:"))
async def topic_callback_handler(callback: CallbackQuery, state: FSMContext) -> None:
    """Сохраняет выбранную тему и переводит сценарий к вводу вопроса."""

    if callback.data is None:
        await callback.answer()
        return

    _, topic_slug = callback.data.split(":", maxsplit=1)
    if topic_slug == "cancel":
        await state.clear()
        if callback.message is not None:
            await callback.message.edit_text("Сценарий отменен.")
        await callback.answer()
        return

    await state.update_data(topic=topic_slug)
    await state.set_state(AskFlow.waiting_for_question)
    if callback.message is not None:
        await callback.message.edit_text(
            f"Тема выбрана: {get_topic_title(topic_slug)}.\nТеперь напишите ваш вопрос."
        )
    await callback.answer()


@router.message(AskFlow.waiting_for_question, F.text & ~F.text.startswith("/"))
async def ask_question_handler(
    message: Message,
    state: FSMContext,
    backend: BackendClient,
) -> None:
    """Собирает prompt по выбранной теме и отправляет его в backend."""

    if message.from_user is None or message.text is None:
        return

    data = await state.get_data()
    topic_slug = data.get("topic", "")
    topic_title = get_topic_title(topic_slug)
    prompt = f"Тема: {topic_title}. Вопрос: {message.text}"

    try:
        chat_id = await resolve_chat_id(backend, message.from_user.id)
        await stream_to_chat(message, backend.send_message(chat_id, prompt))
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as error:
        await message.answer(format_backend_error(error))
        return

    await state.clear()
