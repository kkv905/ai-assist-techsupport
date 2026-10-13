"""Изоляция HTTP-тестов между наборами."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI


@asynccontextmanager
async def _test_lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Предоставляет состояние, необходимое DI, без запуска внешних сервисов."""

    from app.main import app

    # Валидация разрешает зависимости `/chat` до отклонения некорректного
    # запроса, поэтому в app.state нужны эти безопасные заглушки.
    app.state.openai = object()
    app.state.cache = object()
    yield


@pytest.fixture(autouse=True)
def disable_external_app_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Сохраняет каждый тест TestClient изолированным и быстрым.

    Тесты маршрутов сами подменяют сервисные зависимости. Иначе lifespan для
    каждого TestClient загружал бы BAAI/bge-m3 и выполнял сетевые запросы к
    Qdrant и Hugging Face.
    """

    from app.main import app

    monkeypatch.setattr(app.router, "lifespan_context", _test_lifespan)
