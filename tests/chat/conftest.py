"""Общие фикстуры для тестов модуля чата."""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.repositories.pg_models import Base, ChatMessageRow, ChatRow
from app.chat.repositories.pg_repo import PostgresChatRepository
from postgres_test_database import ensure_safe_test_database_url, get_configured_test_database_url


@pytest_asyncio.fixture(params=["json", "postgres"])
async def chat_repository(request, tmp_path):
    """Возвращает реализацию репозитория для контрактных тестов."""

    if request.param == "json":
        yield JsonChatRepository(tmp_path)
        return

    database_url = get_configured_test_database_url(os.environ)
    if database_url is None:
        pytest.skip("PostgreSQL-контрактные тесты skipped: TEST_DATABASE_URL не задан.")

    engine = create_async_engine(database_url, future=True)
    try:
        try:
            ensure_safe_test_database_url(database_url)
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
        except Exception:
            pytest.skip("PostgreSQL-контрактные тесты skipped: тестовая БД недоступна.")

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            try:
                ensure_safe_test_database_url(database_url)
                await session.execute(delete(ChatMessageRow))
                await session.execute(delete(ChatRow))
                await session.commit()
                yield PostgresChatRepository(session)
            finally:
                ensure_safe_test_database_url(database_url)
                async with engine.begin() as connection:
                    await connection.run_sync(Base.metadata.drop_all)
    finally:
        await engine.dispose()
