"""Внутренний HTTP API Telegram-бота для уведомлений от backend."""

from __future__ import annotations

from aiogram import Bot
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel


class NotifyRequest(BaseModel):
    """Описывает payload внутреннего уведомления для Telegram-пользователя."""

    chat_id: int
    text: str


def build_api(bot: Bot, internal_token: str) -> FastAPI:
    """Создает FastAPI-приложение для приема внутренних уведомлений от backend."""

    api = FastAPI()

    @api.post("/notify")
    async def notify(req: NotifyRequest, x_internal_token: str = Header(...)) -> dict[str, bool]:
        """Проверяет внутренний токен и пересылает уведомление в Telegram."""

        if x_internal_token != internal_token:
            raise HTTPException(status_code=401)
        await bot.send_message(chat_id=req.chat_id, text=req.text)
        return {"ok": True}

    return api
