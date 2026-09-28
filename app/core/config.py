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
    admin_token: SecretStr = SecretStr("change-me-admin")
    bot_api_port: int = 9000
    moderation_keywords_path: Path = Path("app/moderation/moderation_keywords.yaml")
    moderation_openai_enabled: bool = False
    moderation_category_thresholds: dict[str, float] = Field(default_factory=dict)
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 16
    embedding_cache_dir: Path = Path("./var/embeddings")
    embedding_max_retries: int = 3
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"
    embedding_dim: int = 1024
    rag_collection: str = "rag_block_03"
    rag_baremetal_collection: str = "rag_block_03_baremetal"
    rag_data_dir: Path = Path("data")
    rag_docstore_dir: Path = Path("var/rag_docstore")
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    # M5B6: selected after isolated A/B evaluation (512/64, top-K 5).
    rag_similarity_top_k: int = 5
    rag_chunking_strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    rag_reranker_enabled: bool = False
    rag_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rag_score_threshold: float = 0.3
    # Judge is deliberately independent from the model serving end users.
    rag_eval_judge_model: str = "deepseek-chat"
    rag_eval_judge_base_url: str = "https://api.deepseek.com"
    rag_eval_judge_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("RAG_EVAL_JUDGE_API_KEY", "DEEPSEEK_API_KEY"),
    )
    rag_eval_embedding_model: str = "text-embedding-3-small"
    llm: LLMSettings = Field(default_factory=LLMSettings)


@lru_cache
def get_settings() -> Settings:
    """Возвращает кешированный экземпляр настроек приложения."""

    return Settings()
