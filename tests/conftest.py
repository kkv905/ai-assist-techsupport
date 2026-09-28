"""Cross-suite isolation for HTTP tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI


@asynccontextmanager
async def _test_lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Provide state required by DI without external service startup."""

    from app.main import app

    # Validation still resolves `/chat` dependencies before it rejects an
    # invalid request, so these harmless sentinels must exist on app.state.
    app.state.openai = object()
    app.state.cache = object()
    yield


@pytest.fixture(autouse=True)
def disable_external_app_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep each TestClient test hermetic and fast.

    Route tests inject their service dependencies themselves. Starting the
    production lifespan for each TestClient would otherwise load BAAI/bge-m3
    and make network requests to Qdrant and Hugging Face.
    """

    from app.main import app

    monkeypatch.setattr(app.router, "lifespan_context", _test_lifespan)
