"""Тот же RAG-путь без LlamaIndex — для сравнения с ``rag.py``."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import Settings
from app.services.embeddings import embed_texts
from app.services.rag import NOT_FOUND_ANSWER


@dataclass
class BaremetalRAGService:
    """Чтение, чанкинг, embedding, Qdrant query_points и completion вручную."""

    data_dir: Path
    collection: str
    qdrant_url: str
    qdrant_api_key: str | None
    embedding_dim: int
    llm_model: str
    openai_api_key: str
    chunk_size: int
    chunk_overlap: int
    similarity_top_k: int
    score_threshold: float
    _client: QdrantClient | None = field(default=None, init=False, repr=False)
    _openai: OpenAI | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_settings(cls, settings: Settings) -> "BaremetalRAGService":
        return cls(
            data_dir=settings.rag_data_dir,
            collection=settings.rag_baremetal_collection,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
            embedding_dim=settings.embedding_dim,
            llm_model=settings.llm.default_model,
            openai_api_key=settings.llm.openai_api_key.get_secret_value(),
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            similarity_top_k=settings.rag_similarity_top_k,
            score_threshold=settings.rag_score_threshold,
        )

    def build(self) -> None:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"RAG corpus directory is missing: {self.data_dir}")
        self._client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=1)
        self._openai = OpenAI(api_key=self.openai_api_key)
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )
        if self._client.count(self.collection, exact=True).count:
            return
        chunks = self._chunks()
        vectors = embed_texts([item["text"] for item in chunks])
        if any(len(vector) != self.embedding_dim for vector in vectors):
            raise ValueError("Размерность embedding не совпадает с EMBEDDING_DIM.")
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{item['source']}:{item['index']}")),
                vector=vector,
                payload={"text": item["text"], "source": item["source"], "chunk_index": item["index"]},
            )
            for item, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(self.collection, points=points, wait=True)

    def answer(self, question: str) -> dict[str, Any]:
        if self._client is None or self._openai is None:
            raise RuntimeError("Bare-metal RAG has not been built.")
        results = self._client.query_points(
            collection_name=self.collection,
            query=embed_texts([question])[0],
            limit=self.similarity_top_k,
            with_payload=True,
        ).points
        sources = [
            {
                "text": str(point.payload.get("text", ""))[:300],
                "source": point.payload.get("source"),
                "score": round(float(point.score), 3),
            }
            for point in results
        ]
        top_score = sources[0]["score"] if sources else 0.0
        if top_score < self.score_threshold:
            return {"answer": NOT_FOUND_ANSWER, "top_score": top_score, "sources": sources}
        context = "\n\n".join(f"[{item['source']}] {item['text']}" for item in sources)
        completion = self._openai.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Отвечай только по предоставленному контексту. Если сведений нет, так и скажи.",
                },
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
            ],
        )
        return {
            "answer": completion.choices[0].message.content or NOT_FOUND_ANSWER,
            "top_score": top_score,
            "sources": sources,
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def _chunks(self) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        step = self.chunk_size - self.chunk_overlap
        if step < 1:
            raise ValueError("RAG_CHUNK_OVERLAP должен быть меньше RAG_CHUNK_SIZE.")
        for path in sorted(self.data_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8").strip()
            for index, start in enumerate(range(0, len(text), step)):
                part = text[start : start + self.chunk_size].strip()
                if part:
                    chunks.append({"text": part, "source": path.name, "index": index})
        if not chunks:
            raise ValueError("В bare-metal RAG не найдено .md или .txt документов.")
        return chunks


if __name__ == "__main__":  # pragma: no cover - ручной smoke test
    from app.core.config import get_settings

    service = BaremetalRAGService.from_settings(get_settings())
    service.build()
    print(service.answer("Как сбросить пароль?"))
    service.close()
