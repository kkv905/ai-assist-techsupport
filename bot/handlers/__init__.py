"""Сборка роутеров Telegram-бота."""

from __future__ import annotations

from aiogram import Router

from bot.handlers import commands, fsm, media, text


def get_root_router() -> Router:
    """Собирает корневой роутер из всех обработчиков бота."""

    router = Router(name="bot-root")
    router.include_router(commands.router)
    router.include_router(fsm.router)
    router.include_router(media.router)
    router.include_router(text.router)
    return router
