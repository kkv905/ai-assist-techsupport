"""Конфигурация Telegram-бота."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, SecretStr, field_validator
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
    internal_token: SecretStr = SecretStr("change-me")
    admin_token: SecretStr = SecretStr("change-me-admin")
    bot_api_port: int = 9000
    bot_admin_ids: list[int] = Field(default_factory=list)
    broadcast_poll_interval_seconds: float = 5.0

    @field_validator("bot_admin_ids", mode="before")
    @classmethod
    def parse_admin_ids(cls, value: Any) -> list[int]:
        """Поддерживает JSON-массив и список ID через запятую в env."""

        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [int(item.strip()) for item in value.split(",") if item.strip()]
        return list(value)


@lru_cache
def get_settings() -> BotSettings:
    """Возвращает кешированный объект настроек Telegram-бота."""

    return BotSettings()
