"""Административные хендлеры Telegram-бота."""

from __future__ import annotations

import html

import httpx
from aiogram import Router
from aiogram.filters import BaseFilter, Command
from aiogram.types import Message

from bot.services.backend_client import BackendClient

router = Router(name="admin")


class IsAdmin(BaseFilter):
    """Проверяет, что команду выполняет Telegram-пользователь из списка администраторов."""

    async def __call__(self, message: Message, admin_ids: set[int]) -> bool:
        """Сравнивает `from_user.id` со списком разрешенных администраторов."""

        return message.from_user is not None and message.from_user.id in admin_ids


router.message.filter(IsAdmin())


@router.message(Command("stats"))
async def stats_command(message: Message, backend: BackendClient) -> None:
    """Запрашивает агрегированную статистику backend и отправляет ее администратору."""

    try:
        stats = await backend.get_admin_stats()
    except httpx.HTTPStatusError as error:
        await message.answer(_format_admin_error(error))
        return
    except (httpx.ConnectError, httpx.ReadTimeout) as error:
        await message.answer(_format_admin_error(error))
        return

    await message.answer(
        "<b>Статистика за 24 часа</b>\n"
        f"Сообщений: <code>{stats['total_messages']}</code>\n"
        f"DAU: <code>{stats['active_users']}</code>\n"
        f"Средняя задержка: <code>{stats['avg_latency_ms']}</code> мс\n"
        f"Доля блокировок модерации: <code>{stats['moderation_block_rate']}</code>\n"
        f"Доля 👍: <code>{stats['feedback_up_ratio']}</code>",
        parse_mode="HTML",
    )


@router.message(Command("users"))
async def users_command(message: Message, backend: BackendClient) -> None:
    """Показывает администратору первые десять последних пользователей."""

    try:
        users = await backend.get_admin_users(limit=10)
    except httpx.HTTPStatusError as error:
        await message.answer(_format_admin_error(error))
        return
    except (httpx.ConnectError, httpx.ReadTimeout) as error:
        await message.answer(_format_admin_error(error))
        return

    if not users:
        await message.answer("Пользователи пока не найдены.")
        return

    lines = ["<b>Последние пользователи</b>", "<pre>ID | Интерфейс | Чатов | Последняя активность</pre>"]
    for user in users[:10]:
        lines.append(
            "<pre>"
            f"{html.escape(user['owner_external_id'])} | {html.escape(user['interface'])} | "
            f"{user['chats_count']} | {html.escape(user['last_seen_at'])}"
            "</pre>"
        )
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("broadcast"))
async def broadcast_command(message: Message, backend: BackendClient) -> None:
    """Ставит текстовую рассылку по Telegram-пользователям в очередь backend."""

    text = (message.text or "").split(maxsplit=1)
    if len(text) < 2 or not text[1].strip():
        await message.answer("Использование: /broadcast <текст>")
        return

    try:
        result = await backend.create_broadcast(text[1].strip(), interface_filter="telegram")
    except httpx.HTTPStatusError as error:
        await message.answer(_format_admin_error(error))
        return
    except (httpx.ConnectError, httpx.ReadTimeout) as error:
        await message.answer(_format_admin_error(error))
        return

    await message.answer(
        f"Рассылка поставлена в очередь. ID: <code>{result['broadcast_id']}</code>",
        parse_mode="HTML",
    )



def _format_admin_error(error: Exception) -> str:
    """Преобразует ошибки admin API в понятный ответ для администратора."""

    if isinstance(error, httpx.ConnectError):
        return "Backend недоступен. Проверьте сервисы."
    if isinstance(error, httpx.ReadTimeout):
        return "Backend не ответил вовремя."
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        if status_code == 401:
            return "Admin API отклонил запрос: неверный X-Admin-Token."
        if status_code >= 500:
            return "Backend вернул внутреннюю ошибку для admin API."
        return f"Admin API вернул ошибку HTTP {status_code}."
    return "Во время обращения к admin API произошла ошибка."
