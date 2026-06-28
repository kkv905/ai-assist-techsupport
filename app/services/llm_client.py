import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from app.core.exceptions import (
    LLMAuthError,
    LLMContentFilterError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)


@dataclass(slots=True)
class CacheEntry:
    """Хранит значение кеша и время его истечения."""

    value: str
    expires_at: float


class AsyncLLMCacheProtocol(Protocol):
    """Описывает минимальный контракт кеша для LLM-клиента."""

    async def get(self, model: str, messages: list[dict[str, str]], temperature: float) -> str | None:
        """Возвращает сохраненный ответ или `None`, если записи нет."""

    async def set(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        response: str,
    ) -> None:
        """Сохраняет ответ в кеше."""


class InMemoryLLMCache:
    """Простой in-memory кеш для запросов к LLM."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        """Инициализирует хранилище и TTL для записей кеша."""
        self._ttl_seconds = ttl_seconds
        self._storage: dict[str, CacheEntry] = {}

    async def get(self, model: str, messages: list[dict[str, str]], temperature: float) -> str | None:
        """Возвращает ответ из кеша, если запись не истекла."""
        key = self._make_key(model=model, messages=messages, temperature=temperature)
        entry = self._storage.get(key)

        if entry is None:
            return None

        if entry.expires_at <= time.monotonic():
            self._storage.pop(key, None)
            return None

        return entry.value

    async def set(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        response: str,
    ) -> None:
        """Сохраняет ответ в кеше с учетом TTL."""
        key = self._make_key(model=model, messages=messages, temperature=temperature)
        self._storage[key] = CacheEntry(
            value=response,
            expires_at=time.monotonic() + self._ttl_seconds,
        )

    @staticmethod
    def _make_key(model: str, messages: list[dict[str, str]], temperature: float) -> str:
        """Строит стабильный ключ кеша по модели и телу запроса."""
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return payload


class AsyncLLMClient:
    """Асинхронный клиент для работы с OpenAI с кешем, fallback и batch-режимами."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        concurrency: int = 5,
        fallback_models: list[str] | None = None,
        cache: AsyncLLMCacheProtocol | None = None,
        logger: logging.Logger | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """Создает экземпляр клиента и подготавливает ограничение конкурентности."""
        self._default_model = model
        self._fallback_models = fallback_models or []
        self._cache = cache
        self._logger = logger or logging.getLogger("llm.async")
        self._max_retries = max_retries
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._semaphore_limit = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)

    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Возвращает полный текстовый ответ модели на один промпт."""
        selected_model = model or self._default_model
        messages = self._build_messages(prompt)
        prompt_chars = len(prompt)
        start_time = time.perf_counter()
        log_model = selected_model
        status = "ошибка"

        try:
            cached_response = await self._get_from_cache(
                model=selected_model,
                messages=messages,
                temperature=temperature,
            )
            if cached_response is not None:
                status = "кеш"
                return cached_response

            async with self._semaphore:
                async with asyncio.timeout(15):
                    response, used_model = await self._request_with_fallback(
                        messages=messages,
                        model=selected_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

            log_model = used_model
            answer = response.choices[0].message.content or ""
            await self._save_to_cache(
                model=used_model,
                messages=messages,
                temperature=temperature,
                response=answer,
            )
            status = "успех" if used_model == selected_model else "fallback"
            return answer
        except TimeoutError as error:
            status = "таймаут"
            raise LLMTimeoutError("Превышено время ожидания ответа LLM.") from error
        except RateLimitError as error:
            status = "rate_limit"
            raise LLMRateLimitError("Превышен лимит запросов к LLM.") from error
        except AuthenticationError as error:
            status = "auth_error"
            raise LLMAuthError("Ошибка авторизации при обращении к LLM.") from error
        except BadRequestError as error:
            status = "bad_request"
            raise self._map_bad_request_error(error) from error
        except (APITimeoutError, APIConnectionError) as error:
            status = "transport_error"
            raise LLMTimeoutError("Сервис LLM недоступен или не ответил вовремя.") from error
        except APIStatusError as error:
            status = f"http_{error.status_code}"
            raise LLMError(f"LLM вернул ошибку статуса {error.status_code}.") from error
        except LLMError:
            raise
        except Exception as error:
            status = "unexpected_error"
            raise LLMError("Неожиданная ошибка при обращении к LLM.") from error
        finally:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            self._logger.info(
                "llm.call",
                extra={
                    "duration_ms": duration_ms,
                    "model": log_model,
                    "prompt_chars": prompt_chars,
                    "status": status,
                },
            )

    async def batch_chat(
        self,
        prompts: list[str],
        concurrency: int = 5,
        *,
        model_overrides: Mapping[int, str] | None = None,
    ) -> list[str | Exception]:
        """Возвращает результаты батча и сохраняет исключения на своих позициях."""
        self._set_concurrency_limit(concurrency)
        coroutines = [
            self.complete(
                prompt,
                model=model_overrides.get(index) if model_overrides else None,
            )
            for index, prompt in enumerate(prompts)
        ]
        return await asyncio.gather(*coroutines, return_exceptions=True)

    async def batch_chat_strict(
        self,
        prompts: list[str],
        concurrency: int = 5,
        *,
        model_overrides: Mapping[int, str] | None = None,
    ) -> list[str]:
        """Возвращает все ответы батча или пробрасывает `ExceptionGroup`."""
        self._set_concurrency_limit(concurrency)
        results: list[str | None] = [None] * len(prompts)

        async def run_single(index: int, prompt: str) -> None:
            """Обрабатывает один промпт и кладет ответ в результирующий список."""
            results[index] = await self.complete(
                prompt,
                model=model_overrides.get(index) if model_overrides else None,
            )

        try:
            async with asyncio.TaskGroup() as task_group:
                for index, prompt in enumerate(prompts):
                    task_group.create_task(run_single(index=index, prompt=prompt))
        except* Exception:
            raise

        if any(result is None for result in results):
            raise LLMError("Строгий батч завершился без полного набора ответов.")
        return cast(list[str], results)

    async def stream_chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Построчно отдает дельты ответа по мере получения событий стрима."""
        selected_model = model or self._default_model
        usage_total_tokens: int | None = None

        async with self._semaphore:
            stream = await self._client.chat.completions.create(
                model=selected_model,
                messages=self._build_messages(prompt),
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            try:
                async for chunk in stream:
                    if getattr(chunk, "usage", None) and chunk.usage.total_tokens is not None:
                        usage_total_tokens = chunk.usage.total_tokens

                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            finally:
                await stream.close()
                self._logger.info(
                    "llm.stream.usage",
                    extra={
                        "model": selected_model,
                        "prompt_chars": len(prompt),
                        "total_tokens": usage_total_tokens,
                    },
                )

    async def aclose(self) -> None:
        """Корректно закрывает HTTP-клиент OpenAI."""
        await self._client.close()

    async def _request_with_fallback(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int | None,
    ) -> tuple[Any, str]:
        """Пытается получить ответ сначала на основной модели, затем на fallback-моделях."""
        last_error: Exception | None = None

        for candidate_model in self._iter_candidate_models(model):
            try:
                response = await self._create_completion_with_retry(
                    model=candidate_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response, candidate_model
            except (BadRequestError, AuthenticationError):
                raise
            except Exception as error:
                last_error = error

        if last_error is None:
            raise LLMError("Не удалось выбрать модель для выполнения запроса.")

        raise last_error

    async def _create_completion_with_retry(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
    ) -> Any:
        """Повторяет запрос при `429`, чтобы переживать кратковременные rate limit."""

        attempt = 0
        while True:
            try:
                return await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except RateLimitError:
                if attempt >= self._max_retries:
                    raise
                await asyncio.sleep(min(0.25 * (2**attempt), 1.0))
                attempt += 1

    async def _get_from_cache(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str | None:
        """Читает кеш только для детерминированных запросов."""
        if self._cache is None or temperature != 0:
            return None
        return await self._cache.get(model, messages, temperature)

    async def _save_to_cache(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        response: str,
    ) -> None:
        """Сохраняет ответ в кеш только для детерминированных запросов."""
        if self._cache is None or temperature != 0:
            return
        await self._cache.set(model, messages, temperature, response)

    def _iter_candidate_models(self, model: str) -> list[str]:
        """Возвращает список моделей для основного запроса и fallback-пути."""
        ordered_models = [model, *self._fallback_models]
        return list(dict.fromkeys(ordered_models))

    def _set_concurrency_limit(self, concurrency: int) -> None:
        """Обновляет ограничение конкурентности для текущего экземпляра клиента."""
        if concurrency <= 0:
            raise ValueError("Параметр concurrency должен быть больше нуля.")
        if concurrency == self._semaphore_limit:
            return

        self._semaphore_limit = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)

    @staticmethod
    def _build_messages(prompt: str) -> list[dict[str, str]]:
        """Формирует минимальный набор сообщений для chat completion."""
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _map_bad_request_error(error: BadRequestError) -> LLMError:
        """Преобразует ошибки 400 в доменные исключения."""
        message = str(error).lower()
        if "content_filter" in message or "safety" in message:
            return LLMContentFilterError("Запрос отклонен политиками безопасности.")
        return LLMError("Некорректный запрос к LLM.")
