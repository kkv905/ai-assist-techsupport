"""Доменные модели чата."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

ChatRole = Literal["user", "assistant", "system"]


class ChatMessage(BaseModel):
    """Описывает одно сообщение внутри чата."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    chat_id: UUID
    role: ChatRole
    content: str
    tokens: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Chat(BaseModel):
    """Описывает метаданные одного диалога."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    owner_external_id: str
    interface: str
    system_prompt: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
