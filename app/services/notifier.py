"""Сервис отправки внутренних уведомлений в Telegram-бота."""

from __future__ import annotations

import httpx


async def notify_user(chat_id_tg: int, text: str, bot_url: str, internal_token: str) -> None:
    """Отправляет текстовое уведомление во внутренний HTTP API бота."""

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{bot_url.rstrip('/')}/notify",
            json={"chat_id": chat_id_tg, "text": text},
            headers={"X-Internal-Token": internal_token},
        )
        response.raise_for_status()
