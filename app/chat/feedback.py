"""Схемы и сервисные типы для пользовательского feedback."""

from __future__ import annotations

from pydantic import BaseModel

from app.chat.domain import FeedbackValue


class FeedbackIn(BaseModel):
    """Описывает payload голосования пользователя по сообщению."""

    value: FeedbackValue


class FeedbackOut(BaseModel):
    """Возвращает статус сохранения feedback."""

    status: str
