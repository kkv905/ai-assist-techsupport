"""Pydantic-схемы для административного API."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class StatsOut(BaseModel):
    """Описывает агрегированную статистику сервиса за последние 24 часа."""

    total_messages: int
    active_users: int
    avg_latency_ms: float
    moderation_block_rate: float
    feedback_up_ratio: float


class AdminUserOut(BaseModel):
    """Описывает краткую карточку пользователя для админского списка."""

    owner_external_id: str
    interface: str
    chats_count: int
    last_seen_at: datetime


class BroadcastIn(BaseModel):
    """Описывает запрос на создание рассылки."""

    message: str = Field(min_length=1)
    interface_filter: str


class BroadcastOut(BaseModel):
    """Возвращает результат постановки рассылки в очередь."""

    broadcast_id: UUID
    status: str


class BroadcastClaimIn(BaseModel):
    """Описывает параметры выборки заданий для фоновой рассылки."""

    interface_filter: str
    limit: int = 10


class BroadcastDeliveryOut(BaseModel):
    """Описывает одно задание рассылки с уже подобранными получателями."""

    broadcast_id: UUID
    message: str
    interface: str
    recipient_ids: list[str]


class BroadcastCompleteIn(BaseModel):
    """Описывает завершение обработки задания рассылки."""

    status: str
    error: str | None = None
