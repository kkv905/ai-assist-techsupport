"""Конфигурация Telegram-бота."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    """Описывает настройки Telegram-бота из переменных окружения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: SecretStr
    backend_url: str
    bot_admin_ids: list[int] = Field(default_factory=list)


@lru_cache
def get_settings() -> BotSettings:
    """Возвращает кешированный объект настроек Telegram-бота."""

    return BotSettings()
