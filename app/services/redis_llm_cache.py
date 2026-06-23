import hashlib
import json
from typing import Any

from redis.asyncio import Redis


class RedisLLMCache:
    """Асинхронный Redis-кеш для ответов LLM."""

    def __init__(self, redis_url: str, ttl: int = 3600) -> None:
        """Создает Redis-клиент и сохраняет TTL кеша."""
        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl

    async def get(self, model: str, messages: list[dict[str, str]], temperature: float) -> str | None:
        """Возвращает ответ из Redis по вычисленному ключу."""
        key = self._make_key(model=model, messages=messages, temperature=temperature)
        return await self._client.get(key)

    async def set(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        response: str,
    ) -> None:
        """Сохраняет ответ в Redis с TTL."""
        key = self._make_key(model=model, messages=messages, temperature=temperature)
        await self._client.setex(key, self._ttl, response)

    async def stats(self) -> dict[str, Any]:
        """Возвращает базовую статистику по Redis-кешу."""
        info = await self._client.info("stats")
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": f"{hits / total * 100:.1f}%" if total else "N/A",
            "keys": await self._client.dbsize(),
        }

    async def aclose(self) -> None:
        """Закрывает соединение с Redis."""
        await self._client.aclose()

    @staticmethod
    def _make_key(model: str, messages: list[dict[str, str]], temperature: float) -> str:
        """Строит стабильный ключ кеша по параметрам запроса."""
        data = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature},
            ensure_ascii=False,
            sort_keys=True,
        )
        return f"llm:{hashlib.sha256(data.encode('utf-8')).hexdigest()}"
