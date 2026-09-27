"""Сравнить COSINE и DOT на пяти доменных запросах и удалить учебную коллекцию."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.embeddings import embed_texts  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402
from scripts.load_to_qdrant import load_documents  # noqa: E402

QUERIES = [
    "Не удаётся войти в АИС Правоохрана, access denied",
    "Почему XML файл не импортируется в ЦРСВЭД",
    "В Постконтроле при открытии карточки ошибка 500",
    "Как восстановить доступ сотруднику после блокировки пароля",
    "Не подписывается документ электронной подписью",
]


async def main() -> None:
    settings = get_settings()
    base = VectorStore(
        url=settings.qdrant_url,
        api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
        collection=settings.qdrant_collection,
        dim=settings.embedding_dim,
    )
    cosine = VectorStore(
        url=settings.qdrant_url,
        api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
        collection="documents_cosine",
        dim=settings.embedding_dim,
        client=base.client,
    )
    dot = VectorStore(
        url=settings.qdrant_url,
        api_key=None,
        collection="documents_dot",
        dim=settings.embedding_dim,
        client=base.client,
    )
    try:
        documents = load_documents()
        vectors = embed_texts([document["text"] for document in documents])
        if any(len(vector) != settings.embedding_dim for vector in vectors):
            raise ValueError("Размерность bge-m3 не соответствует EMBEDDING_DIM.")
        points = [
            PointStruct(id=document["id"], vector=vector, payload=document)
            for document, vector in zip(documents, vectors, strict=True)
        ]
        await cosine.ensure_collection()
        if await base.client.collection_exists(dot.collection):
            await base.client.delete_collection(dot.collection)
        await base.client.create_collection(
            collection_name=dot.collection,
            vectors_config=VectorParams(size=dot.dim, distance=Distance.DOT),
        )
        await cosine.upsert(points)
        await dot.upsert(points)
        for query, vector in zip(QUERIES, embed_texts(QUERIES), strict=True):
            cosine_ids = [str(point.id) for point in await cosine.search(vector)]
            dot_ids = [str(point.id) for point in await dot.search(vector)]
            print(f"| {query} | {', '.join(cosine_ids)} | {', '.join(dot_ids)} | {cosine_ids == dot_ids} |")
    finally:
        await base.client.delete_collection(dot.collection)
        # documents_cosine — учебный дубль, в production не хранится.
        await base.client.delete_collection(cosine.collection)
        await base.close()


if __name__ == "__main__":
    asyncio.run(main())
