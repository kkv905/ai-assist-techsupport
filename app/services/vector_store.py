"""Асинхронный шлюз к Qdrant для семантического поиска.

Контракт этого модуля намеренно невелик: в следующем блоке его можно заменить
адаптером LlamaIndex, не раскрывая ``qdrant-client`` вызывающему коду.
"""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
    VectorParams,
)


class VectorStore:
    """Один переиспользуемый async-клиент и рабочая коллекция Qdrant."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str | None,
        collection: str,
        dim: int,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        if dim < 1:
            raise ValueError("EMBEDDING_DIM должен быть положительным числом.")
        # Совместимость версий проверяется при развёртывании образов; отключаем
        # лишний сетевой запрос в lifespan (особенно важен для локального dev).
        self.client = client or AsyncQdrantClient(
            url=url,
            api_key=api_key,
            timeout=1.0,
            check_compatibility=False,
        )
        self.collection = collection
        self.dim = dim

    async def ensure_collection(self) -> None:
        """Создаёт коллекцию и индексы payload; повторный вызов безопасен."""
        collections = await self.client.get_collections()
        existing = {item.name for item in collections.collections}
        if self.collection not in existing:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                # Явно закрепляем production-default Qdrant для предсказуемой индексации.
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )
        else:
            info = await self.client.get_collection(self.collection)
            size = self._vector_size(info.config.params.vectors)
            if size != self.dim:
                raise ValueError(
                    f"Коллекция {self.collection!r} имеет размерность {size}, "
                    f"но EMBEDDING_DIM={self.dim}. Создайте отдельную коллекцию или "
                    "согласуйте настройки."
                )

        for field, schema in (
            ("source", PayloadSchemaType.KEYWORD),
            ("created_at", PayloadSchemaType.DATETIME),
            ("tenant_id", PayloadSchemaType.KEYWORD),
            ("status", PayloadSchemaType.KEYWORD),
        ):
            await self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=schema,
                wait=True,
            )

    async def upsert(self, points: list[PointStruct], batch_size: int = 256) -> None:
        """Записывает точки батчами; последний батч синхронно фиксируется."""
        if batch_size < 1:
            raise ValueError("batch_size должен быть положительным числом.")
        self._validate_point_dimensions(points)
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            await self.client.upsert(
                collection_name=self.collection,
                points=batch,
                wait=start + batch_size >= len(points),
            )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        query_filter: Filter | None = None,
    ) -> list[ScoredPoint]:
        """Возвращает типизированные ``ScoredPoint``, а не внутренний dict клиента."""
        if len(query_vector) != self.dim:
            raise ValueError(
                f"Размерность query vector ({len(query_vector)}) не совпадает с "
                f"EMBEDDING_DIM ({self.dim})."
            )
        if top_k < 1:
            raise ValueError("top_k должен быть положительным числом.")
        result = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return list(result.points)

    async def close(self) -> None:
        """Закрывает HTTP-сессию клиента при завершении FastAPI."""
        await self.client.close()

    @staticmethod
    def _vector_size(vectors: object) -> int:
        if isinstance(vectors, VectorParams):
            return vectors.size
        if isinstance(vectors, dict):
            if len(vectors) != 1:
                raise ValueError("Коллекция использует именованные векторы, что не поддержано.")
            only_config = next(iter(vectors.values()))
            return only_config.size
        raise ValueError("Не удалось определить размерность существующей коллекции.")

    def _validate_point_dimensions(self, points: Sequence[PointStruct]) -> None:
        for point in points:
            vector = point.vector
            if not isinstance(vector, list) or len(vector) != self.dim:
                size = len(vector) if isinstance(vector, list) else "именованный"
                raise ValueError(
                    f"Точка {point.id!r} имеет размерность {size}, ожидалось {self.dim}."
                )
