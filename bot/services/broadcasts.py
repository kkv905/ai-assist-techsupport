"""Фоновая доставка массовых рассылок из очереди backend."""

from __future__ import annotations

import asyncio
import contextlib
from uuid import UUID

from aiogram import Bot

from bot.services.backend_client import BackendClient


async def run_broadcast_worker(bot: Bot, backend: BackendClient, poll_interval_seconds: float) -> None:
    """Периодически забирает pending-рассылки и доставляет их Telegram-пользователям."""

    while True:
        deliveries = await backend.claim_broadcasts(interface_filter="telegram", limit=10)
        for delivery in deliveries:
            broadcast_id = UUID(delivery["broadcast_id"])
            try:
                await _deliver_broadcast(bot, delivery["message"], delivery["recipient_ids"])
            except Exception as error:
                await backend.complete_broadcast(broadcast_id, status="failed", error=str(error))
                continue
            await backend.complete_broadcast(broadcast_id, status="sent")
        await asyncio.sleep(poll_interval_seconds)


async def _deliver_broadcast(bot: Bot, text: str, recipient_ids: list[str]) -> None:
    """Отправляет одно сообщение всем получателям из задания рассылки."""

    for recipient_id in recipient_ids:
        with contextlib.suppress(Exception):
            await bot.send_message(chat_id=int(recipient_id), text=text)
