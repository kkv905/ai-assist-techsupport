import asyncio
from types import SimpleNamespace

import pytest
from openai import RateLimitError

from app.services.llm_client import AsyncLLMClient


class DummyAsyncOpenAI:
    """Имитирует минимальный AsyncOpenAI для unit-тестов клиента."""

    def __init__(self, create) -> None:
        """Сохраняет мок функции create в структуру SDK."""

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.close = create


def make_rate_limit_error() -> RateLimitError:
    """Создает экземпляр ошибки rate limit для unit-тестов."""

    response = SimpleNamespace(request=None, status_code=429, headers={})
    return RateLimitError("rate", response=response, body=None)


def make_response(content: str) -> SimpleNamespace:
    """Создает упрощенный ответ chat.completions.create."""

    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


def test_complete_returns_cached_value_without_provider_call(mocker) -> None:
    """Проверяет, что cache hit не приводит к вызову провайдера."""

    cache = mocker.Mock()
    cache.get = mocker.AsyncMock(return_value="ответ из кеша")
    cache.set = mocker.AsyncMock()
    create_mock = mocker.AsyncMock()
    client = AsyncLLMClient(
        api_key="sk-test",
        model="gpt-4o-mini",
        cache=cache,
        client=DummyAsyncOpenAI(create_mock),
    )

    result = asyncio.run(client.complete("Проверка кеша"))

    assert result == "ответ из кеша"
    create_mock.assert_not_called()
    cache.set.assert_not_called()


def test_complete_saves_response_after_cache_miss(mocker) -> None:
    """Проверяет, что при cache miss ответ сохраняется после вызова модели."""

    cache = mocker.Mock()
    cache.get = mocker.AsyncMock(return_value=None)
    cache.set = mocker.AsyncMock()
    create_mock = mocker.AsyncMock(return_value=make_response("свежий ответ"))
    client = AsyncLLMClient(
        api_key="sk-test",
        model="gpt-4o-mini",
        cache=cache,
        client=DummyAsyncOpenAI(create_mock),
    )

    result = asyncio.run(client.complete("Проверка кеша"))

    assert result == "свежий ответ"
    cache.set.assert_awaited_once()
    create_mock.assert_awaited_once()


def test_complete_retries_on_rate_limit_and_returns_answer(mocker) -> None:
    """Проверяет retry при 429 до успешного ответа провайдера."""

    create_mock = mocker.AsyncMock(
        side_effect=[
            make_rate_limit_error(),
            make_response("ответ после ретрая"),
        ]
    )
    sleep_mock = mocker.AsyncMock()
    mocker.patch("app.services.llm_client.asyncio.sleep", sleep_mock)
    client = AsyncLLMClient(
        api_key="sk-test",
        model="gpt-4o-mini",
        client=DummyAsyncOpenAI(create_mock),
        max_retries=2,
    )

    result = asyncio.run(client.complete("Повтори запрос"))

    assert result == "ответ после ретрая"
    assert create_mock.await_count == 2
    sleep_mock.assert_awaited_once()


def test_complete_raises_after_exhausting_rate_limit_retries(mocker) -> None:
    """Проверяет, что после исчерпания retry ошибка не скрывается."""

    create_mock = mocker.AsyncMock(
        side_effect=[
            make_rate_limit_error(),
            make_rate_limit_error(),
        ]
    )
    sleep_mock = mocker.AsyncMock()
    mocker.patch("app.services.llm_client.asyncio.sleep", sleep_mock)
    client = AsyncLLMClient(
        api_key="sk-test",
        model="gpt-4o-mini",
        client=DummyAsyncOpenAI(create_mock),
        max_retries=1,
    )

    with pytest.raises(Exception, match="Превышен лимит запросов к LLM"):
        asyncio.run(client.complete("Повтори запрос"))

    assert create_mock.await_count == 2
    sleep_mock.assert_awaited_once()
