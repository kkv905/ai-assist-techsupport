"""Идемпотентно загрузить 120 смысловых фрагментов базы знаний в Qdrant.

Запуск: ``uv run python scripts/load_to_qdrant.py``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client.models import PointStruct  # noqa: E402
from tqdm import tqdm  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.embeddings import embed_texts  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "support_articles.json"
CHUNKS_PER_ARTICLE = 10


def load_documents(path: Path = DATA_PATH) -> list[dict[str, Any]]:
    """Строит 10 самостоятельных retrieval-чанков из каждой статьи предметной базы."""
    articles = json.loads(path.read_text(encoding="utf-8"))
    now = datetime.now(UTC)
    documents: list[dict[str, Any]] = []
    for article_number, article in enumerate(articles):
        for chunk_index in range(CHUNKS_PER_ARTICLE):
            created_at = now - timedelta(days=article_number * 18 + chunk_index * 3)
            status = "archived" if article["source"] == "policy_archived_cases.md" else "active"
            text = _chunk_text(article, chunk_index)
            documents.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{article['source']}:{chunk_index}")),
                    "text": text,
                    "source": article["source"],
                    "created_at": created_at.isoformat(),
                    "tenant_id": article["tenant_id"],
                    "category": article["category"],
                    "status": status,
                }
            )
    if len(documents) < 100:
        raise RuntimeError("Для загрузки требуется не менее 100 документов.")
    return documents


def _chunk_text(article: dict[str, str], chunk_index: int) -> str:
    variants = (
        article["title"],
        article["problem"],
        article["solution"],
        f"{article['title']}. Симптом: {article['problem']}",
        f"{article['title']}. Решение: {article['solution']}",
        f"Категория {article['category']}. {article['problem']}",
        f"Инструкция для {article['tenant_id']}: {article['solution']}",
        f"Когда обращаться: {article['problem']} Действие: {article['solution']}",
        f"Источник {article['source']}: {article['title']}. {article['solution']}",
        f"Техническая поддержка. {article['problem']} Рекомендуется: {article['solution']}",
    )
    return variants[chunk_index]


async def main() -> None:
    settings = get_settings()
    store = VectorStore(
        url=settings.qdrant_url,
        api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
        collection=settings.qdrant_collection,
        dim=settings.embedding_dim,
    )
    try:
        await store.ensure_collection()
        documents = load_documents()
        texts = [document["text"] for document in documents]
        vectors = embed_texts(texts)
        if any(len(vector) != settings.embedding_dim for vector in vectors):
            sizes = sorted({len(vector) for vector in vectors})
            raise ValueError(
                f"Embedding-модель вернула размерности {sizes}, ожидался EMBEDDING_DIM="
                f"{settings.embedding_dim}. Измените конфигурацию или коллекцию."
            )
        points = [
            PointStruct(id=document["id"], vector=vector, payload={key: value for key, value in document.items() if key != "id"})
            for document, vector in zip(documents, vectors, strict=True)
        ]
        for start in tqdm(range(0, len(points), 128), desc="Uploading to Qdrant", unit="batch"):
            await store.upsert(points[start : start + 128], batch_size=128)
        info = await store.client.get_collection(settings.qdrant_collection)
        print(f"Loaded {len(points)} deterministic points; points_count={info.points_count}")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
