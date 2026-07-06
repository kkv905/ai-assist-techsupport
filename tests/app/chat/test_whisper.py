"""Тесты расшифровки голосовых сообщений в chat-backend."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from app.chat import media as media_module


@pytest.mark.asyncio
async def test_media_to_part_transcribes_audio_with_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверяет, что для audio/ogg вызывается Whisper и возвращается text-part."""

    create_mock = AsyncMock(return_value=SimpleNamespace(text="Тестовая расшифровка"))
    fake_client = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(create=create_mock),
        )
    )
    monkeypatch.setattr(media_module, "_get_openai_client", lambda: fake_client)

    upload = UploadFile(
        file=BytesIO(b"OggS-test"),
        filename="voice.ogg",
        headers=Headers({"content-type": "audio/ogg"}),
    )

    part = await media_module.media_to_part(upload)

    assert part == {
        "type": "text",
        "text": "[пользователь сказал голосом]:\nТестовая расшифровка",
    }
    create_mock.assert_awaited_once()
