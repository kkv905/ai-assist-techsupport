from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Хранит настройки подключения к языковой модели."""

    model_config = SettingsConfigDict(
        env_prefix="LLM__",
        extra="ignore",
    )

    openai_api_key: SecretStr = Field(
        validation_alias=AliasChoices("LLM__OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    default_model: str = "gpt-5.2"
    request_timeout: float = 30.0
    max_retries: int = 3


class Settings(BaseSettings):
    """Описывает корневые настройки HTTP-сервиса."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "ai-assist-techsupport"
    debug: bool = False
    redis_url: str = "redis://localhost:6379/0"
    redis_password: SecretStr | None = None
    cache_ttl_seconds: int = 3600
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    phoenix_tracing_enabled: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006"
    database_url: str = "postgresql+asyncpg://chat:chat@localhost:5432/chat"
    postgres_password: SecretStr | None = None
    chat_repository: Literal["json", "postgres"] = "json"
    chat_storage_dir: Path = Path("./var/chats")
    chat_context_strategy: Literal["sliding", "hybrid"] = "sliding"
    chat_context_window: int = 10
    chat_context_window_tokens: int = 8192
    chat_response_tokens: int = 1024
    chat_safety_margin: int = 256
    bot_url: str = "http://bot:9000"
    internal_token: SecretStr = SecretStr("change-me")
    bot_api_port: int = 9000
    llm: LLMSettings = Field(default_factory=LLMSettings)


@lru_cache
def get_settings() -> Settings:
    """Возвращает кешированный экземпляр настроек приложения."""

    return Settings()
