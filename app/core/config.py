from functools import lru_cache

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
    default_model: str = "gpt-4o-mini"
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
    llm: LLMSettings = Field(default_factory=LLMSettings)


@lru_cache
def get_settings() -> Settings:
    """Возвращает кешированный экземпляр настроек приложения."""

    return Settings()
