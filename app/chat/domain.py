"""Доменные модели чата."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

ChatRole = Literal["user", "assistant", "system"]
FeedbackValue = Literal["up", "down"]


class MediaRef(BaseModel):
    """Описывает сохраненную ссылку на мультимодальную часть сообщения."""

    mime: str
    size: int | None = None
    filename: str | None = None
    part: dict[str, Any]


class ChatMessage(BaseModel):
    """Описывает одно сообщение внутри чата."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    chat_id: UUID
    role: ChatRole
    content: str
    media_refs: MediaRef | None = None
    tokens: int | None = None
    latency_ms: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Chat(BaseModel):
    """Описывает метаданные одного диалога."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    owner_external_id: str
    interface: str
    system_prompt: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatStreamEvent(BaseModel):
    """Описывает одно событие потокового ответа backend."""

    type: Literal["token", "done"]
    delta: str | None = None
    message_id: UUID | None = None


class MessageFeedback(BaseModel):
    """Описывает голос пользователя за сообщение ассистента."""

    message_id: UUID
    owner_external_id: str
    value: FeedbackValue
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
