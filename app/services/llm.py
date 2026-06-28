from __future__ import annotations

import hashlib
import json
import time
from collections.abc import AsyncIterator

import structlog
from openai import APITimeoutError, AsyncOpenAI, AuthenticationError, RateLimitError
from redis.asyncio import Redis

from app.core.config import Settings
from app.core.exceptions import LLMAuthError, LLMError, LLMRateLimitError, LLMTimeoutError
from app.observability.pii import prompt_hash, redact_pii
from app.schemas.chat import ChatDelta, ChatRequest, ChatResponse, Usage


class LLMService:
    """Инкапсулирует обращения к OpenAI и работу с Redis-кешем."""

    def __init__(self, openai: AsyncOpenAI, cache: Redis, settings: Settings) -> None:
        """Сохраняет зависимости сервисного слоя."""

        self.openai = openai
        self.cache = cache
        self.settings = settings
        self.logger = structlog.get_logger("app.llm")

    async def complete(self, req: ChatRequest) -> ChatResponse:
        """Возвращает полный ответ модели и использует кеш для повторных запросов."""

        cache_key = self._build_cache_key(req)
        cached_payload = await self.cache.get(cache_key)
        if cached_payload is not None:
            cached_response = ChatResponse.model_validate_json(cached_payload)
            return cached_response.model_copy(update={"cached": True})

        started_at = time.perf_counter()
        model = self._resolve_model(req)
        try:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[message.model_dump() for message in req.messages],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                user=req.user_id,
            )
        except RateLimitError as error:
            raise LLMRateLimitError("Превышен лимит запросов к языковой модели.") from error
        except APITimeoutError as error:
            raise LLMTimeoutError("Превышено время ожидания ответа от языковой модели.") from error
        except AuthenticationError as error:
            raise LLMAuthError("Ошибка авторизации при обращении к языковой модели.") from error
        except LLMError:
            raise
        except Exception as error:
            raise LLMError("Не удалось получить ответ от языковой модели.") from error

        chat_response = ChatResponse.from_openai(response)
        self._log_llm_request_completed(
            req=req,
            model=chat_response.model or model,
            usage=chat_response.usage,
            finish_reason=chat_response.finish_reason,
            latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        await self.cache.setex(
            cache_key,
            self.settings.cache_ttl_seconds,
            chat_response.model_dump_json(),
        )
        return chat_response

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        """Отдает текстовые дельты и финальный usage в режиме стрима."""

        started_at = time.perf_counter()
        model = self._resolve_model(req)
        finish_reason: str | None = None
        try:
            stream = await self.openai.chat.completions.create(
                model=model,
                messages=[message.model_dump() for message in req.messages],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                user=req.user_id,
                stream=True,
                stream_options={"include_usage": True},
            )
        except RateLimitError as error:
            raise LLMRateLimitError("Превышен лимит запросов к языковой модели.") from error
        except APITimeoutError as error:
            raise LLMTimeoutError("Превышено время ожидания ответа от языковой модели.") from error
        except AuthenticationError as error:
            raise LLMAuthError("Ошибка авторизации при обращении к языковой модели.") from error
        except Exception as error:
            raise LLMError("Не удалось открыть поток ответа языковой модели.") from error

        try:
            async for chunk in stream:
                usage_data = getattr(chunk, "usage", None)
                if usage_data is not None:
                    usage = Usage(
                        prompt_tokens=getattr(usage_data, "prompt_tokens", None),
                        completion_tokens=getattr(usage_data, "completion_tokens", None),
                        total_tokens=getattr(usage_data, "total_tokens", None),
                    )
                    self._log_llm_request_completed(
                        req=req,
                        model=model,
                        usage=usage,
                        finish_reason=finish_reason,
                        latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
                    )
                    yield ChatDelta(usage=usage)

                if not getattr(chunk, "choices", None):
                    continue

                choice = chunk.choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                delta = getattr(choice.delta, "content", None)
                if delta:
                    yield ChatDelta(content=delta)
        finally:
            await stream.close()

    def _resolve_model(self, req: ChatRequest) -> str:
        """Возвращает модель запроса с учетом значения по умолчанию."""

        return req.model or self.settings.llm.default_model

    def _build_prompt_text(self, req: ChatRequest) -> str:
        """Собирает единый текст промпта для безопасного логирования."""

        return "\n".join(f"{message.role}: {message.content}" for message in req.messages)

    def _log_llm_request_completed(
        self,
        req: ChatRequest,
        model: str,
        usage: Usage,
        finish_reason: str | None,
        latency_ms: float,
    ) -> None:
        """Пишет структурированный лог завершения LLM-запроса без сырых PII."""

        raw_prompt = self._build_prompt_text(req)
        self.logger.info(
            "llm_request_completed",
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            prompt_hash=prompt_hash(raw_prompt),
            prompt_preview=redact_pii(raw_prompt)[:120],
        )

    def _build_cache_key(self, req: ChatRequest) -> str:
        """Строит стабильный ключ кеша без пользовательских метаданных."""

        normalized_request = req.model_copy(update={"model": self._resolve_model(req)})
        payload = normalized_request.model_dump(
            exclude={"user_id", "session_id", "stream"},
            exclude_none=True,
        )
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"chat:{digest}"
