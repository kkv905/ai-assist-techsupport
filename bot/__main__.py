"""Точка входа для запуска Telegram-бота."""

from __future__ import annotations

import asyncio
from contextlib import suppress

import httpx
import uvicorn
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from bot.config import get_settings
from bot.handlers import get_root_router
from bot.services.backend_client import BackendClient
from bot.web import build_api


async def main() -> None:
    """Создает объекты бота и запускает polling вместе с внутренним HTTP API."""

    settings = get_settings()
    bot = Bot(token=settings.bot_token.get_secret_value())
    dispatcher = Dispatcher(storage=MemoryStorage())
    http_client = httpx.AsyncClient(
        base_url=settings.backend_url.rstrip("/"),
        timeout=httpx.Timeout(connect=3.0, read=60.0, write=10.0, pool=5.0),
    )
    backend = BackendClient(settings.backend_url, http_client=http_client)

    dispatcher["backend"] = backend
    dispatcher["admin_ids"] = set(settings.bot_admin_ids)
    dispatcher.include_router(get_root_router())

    api = build_api(bot, settings.internal_token.get_secret_value())
    config = uvicorn.Config(api, host="0.0.0.0", port=settings.bot_api_port, log_level="info")
    server = uvicorn.Server(config)

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        with suppress(asyncio.CancelledError):
            await asyncio.gather(
                dispatcher.start_polling(bot, handle_signals=False),
                server.serve(),
            )
    finally:
        await http_client.aclose()
        await backend.aclose()
        await bot.session.close()


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        asyncio.run(main())
