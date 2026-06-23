from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request
from openai import AsyncOpenAI
from redis.asyncio import Redis

from app.core.config import Settings, get_settings as load_settings
from app.services.llm import LLMService


@lru_cache
def get_settings() -> Settings:
    """Возвращает настройки приложения для DI."""

    return load_settings()


def get_openai(request: Request) -> AsyncOpenAI:
    """Достает клиент OpenAI из состояния приложения."""

    return request.app.state.openai


def get_cache(request: Request) -> Redis:
    """Достает Redis-клиент из состояния приложения."""

    return request.app.state.cache


def get_llm_service(
    openai: AsyncOpenAI = Depends(get_openai),
    cache: Redis = Depends(get_cache),
    settings: Settings = Depends(get_settings),
) -> LLMService:
    """Собирает сервисный слой для работы с LLM."""

    return LLMService(openai=openai, cache=cache, settings=settings)


SettingsDep = Annotated[Settings, Depends(get_settings)]
CacheDep = Annotated[Redis, Depends(get_cache)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
