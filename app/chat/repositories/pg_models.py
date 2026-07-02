"""ORM-модели для PostgreSQL-хранилища чатов."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Text, Uuid, func, text
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
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
