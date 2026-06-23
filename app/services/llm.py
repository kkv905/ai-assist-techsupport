from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncIterator

from openai import APITimeoutError, AsyncOpenAI, AuthenticationError, RateLimitError
from redis.asyncio import Redis

from app.core.config import Settings
from app.core.exceptions import LLMAuthError, LLMError, LLMRateLimitError, LLMTimeoutError
from app.schemas.chat import ChatDelta, ChatRequest, ChatResponse, Usage


class LLMService:
    """Инкапсулирует обращения к OpenAI и работу с Redis-кешем."""

    def __init__(self, openai: AsyncOpenAI, cache: Redis, settings: Settings) -> None:
        """Сохраняет зависимости сервисного слоя."""

        self.openai = openai
        self.cache = cache
        self.settings = settings
        self.logger = logging.getLogger("llm-service")

    async def complete(self, req: ChatRequest) -> ChatResponse:
        """Возвращает полный ответ модели и использует кеш для повторных запросов."""

        cache_key = self._build_cache_key(req)
        cached_payload = await self.cache.get(cache_key)
        if cached_payload is not None:
            cached_response = ChatResponse.model_validate_json(cached_payload)
            return cached_response.model_copy(update={"cached": True})

        try:
            response = await self.openai.chat.completions.create(
                model=req.model or self.settings.llm.default_model,
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
        await self.cache.setex(
            cache_key,
            self.settings.cache_ttl_seconds,
            chat_response.model_dump_json(),
        )
        return chat_response

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        """Отдает текстовые дельты и финальный usage в режиме стрима."""

        try:
            stream = await self.openai.chat.completions.create(
                model=req.model or self.settings.llm.default_model,
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
                    self.logger.info(
                        "Получена итоговая статистика токенов из стрима.",
                        extra={"usage": usage.model_dump()},
                    )
                    yield ChatDelta(usage=usage)

                if not getattr(chunk, "choices", None):
                    continue

                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield ChatDelta(content=delta)
        finally:
            await stream.close()

    def _build_cache_key(self, req: ChatRequest) -> str:
        """Строит стабильный ключ кеша без пользовательских метаданных."""

        normalized_request = req.model_copy(update={"model": req.model or self.settings.llm.default_model})
        payload = normalized_request.model_dump(
            exclude={"user_id", "session_id", "stream"},
            exclude_none=True,
        )
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"chat:{digest}"
