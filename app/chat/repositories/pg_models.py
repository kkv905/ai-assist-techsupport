"""ORM-модели для PostgreSQL-хранилища чатов."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, Text, UniqueConstraint, Uuid, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Базовый класс ORM-моделей чата."""


class ChatRow(Base):
    """Описывает строку таблицы `chats`."""

    __tablename__ = "chats"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    owner_external_id: Mapped[str] = mapped_column(Text, nullable=False)
    interface: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ChatMessageRow(Base):
    """Описывает строку таблицы `chat_messages`."""

    __tablename__ = "chat_messages"
    __table_args__ = (
        Index(
            "ix_chat_messages_chat_created",
            "chat_id",
            text("created_at DESC"),
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    chat_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    media_refs: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MessageFeedbackRow(Base):
    """Описывает строку таблицы `message_feedback`."""

    __tablename__ = "message_feedback"
    __table_args__ = (
        UniqueConstraint("owner_external_id", "message_id", name="uq_message_feedback_owner_message"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    message_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    owner_external_id: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ModerationEventRow(Base):
    """Описывает строку таблицы `moderation_events`."""

    __tablename__ = "moderation_events"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    chat_id: Mapped[UUID | None] = mapped_column(
        Uuid,
        ForeignKey("chats.id", ondelete="SET NULL"),
        nullable=True,
    )
    owner_external_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    direction: Mapped[str] = mapped_column(Text, nullable=False)
    allowed: Mapped[bool] = mapped_column(nullable=False)
    categories: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    blocked_by: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class BroadcastQueueRow(Base):
    """Описывает строку таблицы `broadcast_queue`."""

    __tablename__ = "broadcast_queue"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    interface: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'pending'"))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
