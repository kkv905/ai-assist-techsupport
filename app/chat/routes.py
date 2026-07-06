"""HTTP-маршруты для управления чатами."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage, MediaRef
from app.chat.media import media_to_part
from app.core.config import get_settings
from app.services.notifier import notify_user

router = APIRouter(prefix="/chats", tags=["chat-history"])


class CreateChatIn(BaseModel):
    """Описывает входные данные для создания чата."""

    owner_external_id: str
    interface: str
    system_prompt: str | None = None


class CreateChatOut(BaseModel):
    """Возвращает идентификатор только что созданного чата."""

    chat_id: UUID


class StatusOut(BaseModel):
    """Возвращает простой статус выполненной операции."""

    status: str


class SystemMessageIn(BaseModel):
    """Описывает системное сообщение, которое backend дописывает в чат."""

    text: str
    notify: bool = False


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
    content: str = Form(...),
    media: UploadFile | None = File(None),
    chat_service: ChatServiceDep = ...,  # type: ignore[assignment]
) -> StreamingResponse:
    """Стримит ответ модели в формате SSE и завершает поток событием done."""

    chat = await chat_service.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Чат не найден.")

    media_refs: MediaRef | None = None
    if media is not None:
        try:
            part = await media_to_part(media)
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(error)) from error

        size = getattr(media, "size", None)
        if size is None:
            await media.seek(0)
            size = len(await media.read())
        media_refs = MediaRef(
            mime=media.content_type or "application/octet-stream",
            size=size,
            filename=media.filename,
            part=part,
        )

    async def event_stream() -> AsyncIterator[str]:
        """Сериализует чанки модели в SSE-формат с JSON-полезной нагрузкой."""

        async for delta in chat_service.send_message(chat_id, content, media_refs=media_refs):
            payload = json.dumps({"type": "token", "delta": delta}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

        done_payload = json.dumps({"type": "done"}, ensure_ascii=False)
        yield f"data: {done_payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/{chat_id}/system-message", response_model=StatusOut)
async def post_system_message(
    chat_id: UUID,
    payload: SystemMessageIn,
    chat_service: ChatServiceDep,
) -> StatusOut:
    """Добавляет системное сообщение и при необходимости отправляет уведомление в бот."""

    chat = await chat_service.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Чат не найден.")

    await chat_service.append_message(chat_id=chat_id, role="assistant", content=payload.text)

    if payload.notify:
        try:
            chat_id_tg = int(chat.owner_external_id)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Для уведомлений owner_external_id должен быть Telegram chat_id.",
            ) from error
        settings = get_settings()
        await notify_user(
            chat_id_tg=chat_id_tg,
            text=payload.text,
            bot_url=settings.bot_url,
            internal_token=settings.internal_token.get_secret_value(),
        )

    return StatusOut(status="ok")


@router.delete("/{chat_id}/messages", response_model=StatusOut)
async def clear_messages(chat_id: UUID, chat_service: ChatServiceDep) -> StatusOut:
    """Очищает историю чата мягким удалением."""

    chat = await chat_service.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Чат не найден.")
    await chat_service.clear_history(chat_id)
    return StatusOut(status="ok")
