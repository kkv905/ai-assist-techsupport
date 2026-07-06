"""Тесты обработки медиа во внутреннем chat-backend."""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi import UploadFile
from pypdf import PdfWriter
from starlette.datastructures import Headers

from app.chat.media import media_to_part


@pytest.mark.asyncio
async def test_media_to_part_returns_text_part_for_pdf() -> None:
    """Проверяет преобразование PDF в текстовую часть контента."""

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buffer = BytesIO()
    writer.write(buffer)
    upload = UploadFile(
        file=BytesIO(buffer.getvalue()),
        filename="test.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )

    part = await media_to_part(upload)

    assert part["type"] == "text"
    assert part["text"].startswith("[документ PDF]:\n")


@pytest.mark.asyncio
async def test_media_to_part_returns_image_url_for_png() -> None:
    """Проверяет преобразование изображения в data URI для OpenAI."""

    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    upload = UploadFile(
        file=BytesIO(png_bytes),
        filename="image.png",
        headers=Headers({"content-type": "image/png"}),
    )

    part = await media_to_part(upload)

    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
