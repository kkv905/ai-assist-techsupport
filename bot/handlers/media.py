"""Медиа-хендлеры Telegram-бота."""

from __future__ import annotations

from io import BytesIO

import httpx
from aiogram import F, Router
from aiogram.types import Message, PhotoSize

from bot.handlers.common import format_backend_error, resolve_chat_id, stream_to_chat
from bot.services.backend_client import BackendClient

router = Router(name="media")

_MAX_PHOTO_SIZE = 2 * 1024 * 1024
_MAX_DOCUMENT_SIZE = 10 * 1024 * 1024
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@router.message(F.photo)
async def photo_message_handler(message: Message, backend: BackendClient) -> None:
    """Скачивает фото пользователя и отправляет его в backend как изображение."""

    if message.from_user is None or not message.photo:
        return

    photo = _select_photo(message.photo)
    if photo is None:
        await message.answer("Не удалось выбрать фото до 2 МБ. Отправьте изображение поменьше.")
        return

    try:
        media = await _download_file(message, photo.file_id)
        chat_id = await resolve_chat_id(backend, message.from_user.id)
        content = message.caption or "[фото]"
        await stream_to_chat(message, backend.send_message(chat_id, content, media=media, mime="image/jpeg"))
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as error:
        await message.answer(format_backend_error(error))


@router.message(F.voice)
async def voice_message_handler(message: Message, backend: BackendClient) -> None:
    """Скачивает голосовое сообщение и отправляет его в backend без конвертации."""

    if message.from_user is None or message.voice is None:
        return

    await _send_binary_message(
        message=message,
        backend=backend,
        file_id=message.voice.file_id,
        content=message.caption or "[голосовое сообщение]",
        mime="audio/ogg",
    )


@router.message(F.audio)
async def audio_message_handler(message: Message, backend: BackendClient) -> None:
    """Скачивает обычный аудиофайл и отправляет его в backend."""

    if message.from_user is None or message.audio is None:
        return

    mime = message.audio.mime_type or "audio/mpeg"
    await _send_binary_message(
        message=message,
        backend=backend,
        file_id=message.audio.file_id,
        content=message.caption or "[аудио]",
        mime=mime,
    )


@router.message(F.document)
async def document_message_handler(message: Message, backend: BackendClient) -> None:
    """Скачивает PDF или DOCX и проксирует его в backend при допустимом размере."""

    if message.from_user is None or message.document is None:
        return

    document = message.document
    filename = (document.file_name or "").lower()
    if not filename.endswith((".pdf", ".docx")):
        await message.answer("Поддерживаются только документы PDF и DOCX.")
        return
    if document.file_size is not None and document.file_size > _MAX_DOCUMENT_SIZE:
        await message.answer("Документ слишком большой. Отправьте файл до 10 МБ.")
        return

    mime = document.mime_type or ("application/pdf" if filename.endswith(".pdf") else _DOCX_MIME)
    await _send_binary_message(
        message=message,
        backend=backend,
        file_id=document.file_id,
        content=message.caption or "[документ]",
        mime=mime,
    )


async def _send_binary_message(
    message: Message,
    backend: BackendClient,
    file_id: str,
    content: str,
    mime: str,
) -> None:
    """Скачивает бинарное вложение и отправляет его в backend единым методом."""

    try:
        media = await _download_file(message, file_id)
        chat_id = await resolve_chat_id(backend, message.from_user.id)
        await stream_to_chat(message, backend.send_message(chat_id, content, media=media, mime=mime))
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as error:
        await message.answer(format_backend_error(error))


async def _download_file(message: Message, file_id: str) -> bytes:
    """Скачивает файл из Telegram в память и возвращает сырые байты."""

    file = await message.bot.get_file(file_id)
    destination = BytesIO()
    await message.bot.download_file(file.file_path, destination=destination)
    return destination.getvalue()


def _select_photo(photos: list[PhotoSize]) -> PhotoSize | None:
    """Выбирает самое крупное фото, которое укладывается в ограничение 2 МБ."""

    eligible = [photo for photo in photos if (photo.file_size or 0) <= _MAX_PHOTO_SIZE]
    if not eligible:
        return None
    return max(eligible, key=lambda photo: photo.file_size or 0)
