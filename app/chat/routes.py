"""HTTP-маршруты для управления чатами."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage

router = APIRouter(prefix="/chats", tags=["chat-history"])


class CreateChatIn(BaseModel):
    """Описывает входные данные для создания чата."""

    owner_external_id: str
    interface: str
    system_prompt: str | None = None


class CreateChatOut(BaseModel):
    """Возвращает идентификатор только что созданного чата."""

    chat_id: UUID


class MessageIn(BaseModel):
    """Описывает входное сообщение пользователя."""

    content: str = Field(min_length=1)


class StatusOut(BaseModel):
    """Возвращает простой статус выполненной операции."""

    status: str


@router.post("", response_model=CreateChatOut, status_code=status.HTTP_201_CREATED)
async def create_chat(payload: CreateChatIn, chat_service: ChatServiceDep) -> CreateChatOut:
    """Создает новый чат и возвращает его идентификатор."""

    chat = await chat_service.create_chat(
        owner_external_id=payload.owner_external_id,
        interface=payload.interface,
        system_prompt=payload.system_prompt,
    )
    return CreateChatOut(chat_id=chat.id)


@router.get("/{chat_id}", response_model=Chat)
async def get_chat(chat_id: UUID, chat_service: ChatServiceDep) -> Chat:
    """Возвращает метаданные чата по идентификатору."""

    chat = await chat_service.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Чат не найден.")
    return chat


@router.get("/{chat_id}/messages", response_model=list[ChatMessage])
async def list_messages(
    chat_id: UUID,
    chat_service: ChatServiceDep,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[ChatMessage]:
    """Возвращает историю сообщений чата в хронологическом порядке."""

    chat = await chat_service.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Чат не найден.")
    return await chat_service.list_messages(chat_id, limit=limit)


@router.post("/{chat_id}/messages")
async def send_message(
    chat_id: UUID,
    payload: MessageIn,
    chat_service: ChatServiceDep,
) -> StreamingResponse:
    """Стримит ответ модели в формате SSE и завершает поток событием `[DONE]`."""

    chat = await chat_service.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Чат не найден.")

    async def event_stream() -> AsyncIterator[str]:
        """Сериализует чанки модели в SSE-формат."""

        async for chunk in chat_service.send_message(chat_id, payload.content):
            yield f"data: {chunk}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.delete("/{chat_id}/messages", response_model=StatusOut)
async def clear_messages(chat_id: UUID, chat_service: ChatServiceDep) -> StatusOut:
    """Очищает историю чата мягким удалением."""

    chat = await chat_service.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Чат не найден.")
    await chat_service.clear_history(chat_id)
    return StatusOut(status="ok")
