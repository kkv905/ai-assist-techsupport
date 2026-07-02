"""Контракт хранилища чатов."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.chat.domain import Chat, ChatMessage


class ChatRepository(Protocol):
    """Описывает минимальный контракт репозитория чатов."""

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        """Создает чат и возвращает его метаданные."""

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        """Возвращает чат по идентификатору или `None`, если он отсутствует."""

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage:
        """Добавляет сообщение в историю чата."""

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        """Возвращает последние сообщения в хронологическом порядке."""

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        """Мягко очищает историю чата без удаления самого чата."""
