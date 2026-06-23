import asyncio
import json
from types import SimpleNamespace

import pytest

from app.core.config import LLMSettings, Settings
from app.core.exceptions import LLMAuthError, LLMRateLimitError, LLMTimeoutError
from app.schemas.chat import ChatRequest
from app.services.llm import LLMService


class FakeRedis:
    """Имитирует Redis-клиент для проверки кеширования."""

    def __init__(self) -> None:
        """Подготавливает in-memory хранилище ключей."""

        self.storage: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        """Возвращает значение по ключу или `None`."""

        return self.storage.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        """Сохраняет значение, игнорируя TTL в рамках теста."""

        self.storage[key] = value


class FakeAsyncStream:
    """Имитирует асинхронный поток OpenAI для streaming-ответов."""

    def __init__(self, chunks: list[object]) -> None:
        """Сохраняет последовательность чанков."""

        self._chunks = chunks

    def __aiter__(self):
        """Возвращает асинхронный итератор по чанкам."""

        return self._iterate()

    async def _iterate(self):
        """По очереди отдает подготовленные чанки."""

        for chunk in self._chunks:
            yield chunk

    async def close(self) -> None:
        """Закрывает поток без побочных эффектов."""


class FakeCompletionsAPI:
    """Имитирует метод `chat.completions.create` у OpenAI SDK."""

    def __init__(self) -> None:
        """Подготавливает счетчик вызовов и режим ответа."""

        self.calls = 0
        self.mode = "success"

    async def create(self, **kwargs):
        """Возвращает обычный ответ, поток или поднимает ошибку."""

        self.calls += 1

        if kwargs.get("stream"):
            return FakeAsyncStream(
                [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="Первая "))],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="часть"))],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[],
                        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8),
                    ),
                ]
            )

        if self.mode == "rate_limit":
            from openai import RateLimitError

            response = SimpleNamespace(request=None, status_code=429, headers={})
            raise RateLimitError("rate", response=response, body=None)

        if self.mode == "timeout":
            from openai import APITimeoutError

            raise APITimeoutError(request=SimpleNamespace())

        if self.mode == "auth":
            from openai import AuthenticationError

            response = SimpleNamespace(request=None, status_code=401, headers={})
            raise AuthenticationError("auth", response=response, body=None)

        prompt = kwargs["messages"][0]["content"]
        return SimpleNamespace(
            model=kwargs["model"],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=f"Ответ: {prompt}"),
                )
            ],
        )


class FakeAsyncOpenAI:
    """Имитирует минимальную структуру клиента `AsyncOpenAI`."""

    def __init__(self) -> None:
        """Создает объект `chat.completions`."""

        self.completions = FakeCompletionsAPI()
        self.chat = SimpleNamespace(completions=self.completions)


def build_settings() -> Settings:
    """Создает тестовые настройки приложения."""

    return Settings.model_construct(
        app_name="test",
        debug=False,
        redis_url="redis://localhost:6379/0",
        cache_ttl_seconds=60,
        cors_origins=["*"],
        llm=LLMSettings.model_construct(
            openai_api_key="sk-test",
            default_model="gpt-4o-mini",
            request_timeout=30.0,
            max_retries=3,
        ),
    )


def build_request() -> ChatRequest:
    """Создает типовой запрос к языковой модели."""

    return ChatRequest(messages=[{"role": "user", "content": "Проверка"}])


def test_complete_uses_cache_for_repeated_request() -> None:
    """Проверяет, что второй одинаковый запрос приходит из кеша."""

    openai = FakeAsyncOpenAI()
    redis = FakeRedis()
    service = LLMService(openai=openai, cache=redis, settings=build_settings())

    first = asyncio.run(service.complete(build_request()))
    second = asyncio.run(service.complete(build_request()))

    assert first.content == "Ответ: Проверка"
    assert second.content == first.content
    assert second.cached is True
    assert openai.completions.calls == 1


def test_complete_stores_serialized_response_in_cache() -> None:
    """Проверяет сохранение сериализованного ответа в Redis."""

    openai = FakeAsyncOpenAI()
    redis = FakeRedis()
    service = LLMService(openai=openai, cache=redis, settings=build_settings())

    asyncio.run(service.complete(build_request()))

    assert len(redis.storage) == 1
    cached_payload = json.loads(next(iter(redis.storage.values())))
    assert cached_payload["content"] == "Ответ: Проверка"


def test_stream_returns_content_and_usage_events() -> None:
    """Проверяет состав событий, которые возвращает стриминговый метод."""

    openai = FakeAsyncOpenAI()
    service = LLMService(openai=openai, cache=FakeRedis(), settings=build_settings())

    async def collect() -> list[tuple[str | None, int | None]]:
        """Собирает дельты в список для проверки."""

        result: list[tuple[str | None, int | None]] = []
        async for chunk in service.stream(build_request()):
            result.append((chunk.content, chunk.usage.total_tokens if chunk.usage else None))
        return result

    assert asyncio.run(collect()) == [("Первая ", None), ("часть", None), (None, 8)]


@pytest.mark.parametrize(
    ("mode", "error_type"),
    [("rate_limit", LLMRateLimitError), ("timeout", LLMTimeoutError), ("auth", LLMAuthError)],
)
def test_complete_maps_openai_errors_to_domain_exceptions(mode: str, error_type: type[Exception]) -> None:
    """Проверяет преобразование ошибок провайдера в доменные исключения."""

    openai = FakeAsyncOpenAI()
    openai.completions.mode = mode
    service = LLMService(openai=openai, cache=FakeRedis(), settings=build_settings())

    with pytest.raises(error_type):
        asyncio.run(service.complete(build_request()))
