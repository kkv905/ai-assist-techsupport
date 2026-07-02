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


@pytest_asyncio.fixture(params=["json", "postgres"])
async def chat_repository(request, tmp_path):
    """Возвращает реализацию репозитория для контрактных тестов."""

    if request.param == "json":
        yield JsonChatRepository(tmp_path)
        return

    database_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://chat:chat@localhost:5432/chat")
    engine = create_async_engine(database_url, future=True)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    except Exception as error:
        await engine.dispose()
        pytest.skip(f"PostgreSQL недоступен для контрактного теста: {error}")

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        await session.execute(delete(ChatMessageRow))
        await session.execute(delete(ChatRow))
        await session.commit()
        yield PostgresChatRepository(session)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()
