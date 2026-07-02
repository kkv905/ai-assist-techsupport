"""JSONL-реализация репозитория чатов."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import aiofiles
from pydantic import BaseModel, Field

from app.chat.domain import Chat, ChatMessage


class SoftDeleteMarker(BaseModel):
    """Описывает служебную запись мягкой очистки истории."""

    type: str = "soft_delete"
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class JsonChatRepository:
    """Хранит чаты на файловой системе в виде JSON и JSONL."""

    def __init__(self, base_dir: Path) -> None:
        """Сохраняет базовую директорию файлового хранилища."""

        self._base_dir = Path(base_dir)

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        """Создает каталог чата и записывает его метаданные."""

        chat = Chat(
            owner_external_id=owner_external_id,
            interface=interface,
            system_prompt=system_prompt,
        )
        chat_dir = self._chat_dir(chat.id)
        chat_dir.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(self._chat_path(chat.id), "w", encoding="utf-8") as file:
            await file.write(chat.model_dump_json())

        async with aiofiles.open(self._messages_path(chat.id), "a", encoding="utf-8"):
            pass

        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        """Читает метаданные чата, если каталог уже существует."""

        path = self._chat_path(chat_id)
        if not path.exists():
            return None

        async with aiofiles.open(path, encoding="utf-8") as file:
            payload = await file.read()

        return Chat.model_validate_json(payload)

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage:
        """Дописывает сообщение в конец JSONL-файла без переписывания истории."""

        if await self.get_chat(chat_id) is None:
            raise LookupError(f"Чат {chat_id} не найден.")

        async with aiofiles.open(self._messages_path(chat_id), "a", encoding="utf-8") as file:
            await file.write(f"{message.model_dump_json()}\n")

        return message

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        """Возвращает последние сообщения после последнего soft delete-маркера."""

        path = self._messages_path(chat_id)
        if not path.exists():
            return []

        async with aiofiles.open(path, encoding="utf-8") as file:
            lines = await file.readlines()

        messages: list[ChatMessage] = []
        for line in lines:
            payload = json.loads(line)
            if payload.get("type") == "soft_delete":
                messages.clear()
                continue
            messages.append(ChatMessage.model_validate_json(line))

        messages.sort(key=lambda item: item.created_at)
        if limit <= 0:
            return []
        return messages[-limit:]

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        """Добавляет служебный маркер мягкой очистки истории."""

        if await self.get_chat(chat_id) is None:
            raise LookupError(f"Чат {chat_id} не найден.")

        marker = SoftDeleteMarker()
        async with aiofiles.open(self._messages_path(chat_id), "a", encoding="utf-8") as file:
            await file.write(f"{marker.model_dump_json()}\n")

    def _chat_dir(self, chat_id: UUID) -> Path:
        """Строит путь до каталога конкретного чата."""

        return self._base_dir / "chats" / str(chat_id)

    def _chat_path(self, chat_id: UUID) -> Path:
        """Строит путь до файла метаданных чата."""

        return self._chat_dir(chat_id) / "chat.json"

    def _messages_path(self, chat_id: UUID) -> Path:
        """Строит путь до файла с историей сообщений чата."""

        return self._chat_dir(chat_id) / "messages.jsonl"
