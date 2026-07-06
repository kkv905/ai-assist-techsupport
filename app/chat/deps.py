"""Зависимости FastAPI для модуля чата."""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.repositories.pg_repo import PostgresChatRepository
from app.chat.repository import ChatRepository
from app.chat.service import ChatService
from app.core.config import Settings, get_settings
from app.moderation.service import ModerationService


@lru_cache
def _get_engine(database_url: str):
    """Создает и кеширует async engine для PostgreSQL."""

    return create_async_engine(database_url, future=True)


@lru_cache
def _get_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Создает и кеширует фабрику SQLAlchemy-сессий."""

    return async_sessionmaker(_get_engine(database_url), expire_on_commit=False)


async def get_db_session(settings: Settings = Depends(get_settings)) -> AsyncIterator[AsyncSession | None]:
    """Открывает SQLAlchemy-сессию только для PostgreSQL-репозитория."""

    if settings.chat_repository != "postgres":
        yield None
        return

    session_factory = _get_session_factory(settings.database_url)
    async with session_factory() as session:
        yield session


def get_llm_client(request: Request) -> Any:
    """Достает клиент OpenAI из состояния приложения."""

    return request.app.state.openai


def get_repository(
    settings: Settings = Depends(get_settings),
    session: AsyncSession | None = Depends(get_db_session),
) -> ChatRepository:
    """Собирает нужную реализацию репозитория на основе настроек."""

    if settings.chat_repository == "json":
        return JsonChatRepository(base_dir=Path(settings.chat_storage_dir))
    if settings.chat_repository == "postgres":
        if session is None:
            raise ValueError("Для PostgreSQL-репозитория не была создана сессия базы данных.")
        return PostgresChatRepository(session=session)
    raise ValueError(
        f"Неизвестное значение CHAT_REPOSITORY: {settings.chat_repository}. Ожидалось 'json' или 'postgres'."
    )


def get_moderation_service(
    settings: Settings = Depends(get_settings),
    llm_client: Any = Depends(get_llm_client),
) -> ModerationService:
    """Собирает сервис модерации с локальными и внешними правилами."""

    return ModerationService(
        keywords_path=settings.moderation_keywords_path,
        openai_client=llm_client,
        openai_enabled=settings.moderation_openai_enabled,
        category_thresholds=settings.moderation_category_thresholds,
    )


def get_chat_service(
    repository: ChatRepository = Depends(get_repository),
    llm_client: Any = Depends(get_llm_client),
    moderation_service: ModerationService = Depends(get_moderation_service),
    settings: Settings = Depends(get_settings),
) -> ChatService:
    """Собирает сервисный слой модуля чата."""

    return ChatService(
        repository=repository,
        llm_client=llm_client,
        moderation_service=moderation_service,
        model=settings.llm.default_model,
        context_strategy=settings.chat_context_strategy,
        context_window=settings.chat_context_window,
        context_window_tokens=settings.chat_context_window_tokens,
        response_tokens=settings.chat_response_tokens,
        safety_margin=settings.chat_safety_margin,
    )


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
DbSessionDep = Annotated[AsyncSession | None, Depends(get_db_session)]
