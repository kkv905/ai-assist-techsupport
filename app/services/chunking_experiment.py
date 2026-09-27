"""CLI для воспроизводимого прогона M5B4 against Qdrant.

Запуск: ``uv run python -m app.services.chunking_experiment``. Скрипт
пересоздаёт только экспериментальные collections ``docs_*``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import get_settings
from app.services.chunking import STRATEGIES
from app.services.reranker import CrossEncoderReranker
from app.services.retrieval_eval import evaluate_retrieval

EXPERIMENT_COLLECTIONS = {"fixed": "docs_fixed", "recursive": "docs_recursive", "semantic": "docs_semantic"}


def _node_source(node: Any) -> str:
    metadata = node.metadata or {}
    return str(metadata.get("file_name") or metadata.get("file_path") or "")


def run() -> dict[str, dict[str, float]]:
    """Переиндексирует три варианта и печатает их retrieval-метрики JSON."""
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    settings = get_settings()
    docs = SimpleDirectoryReader(input_dir=str(settings.rag_data_dir), recursive=True).load_data()
    dataset_path = Path("tests/eval/retrieval_dataset.json")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
    client = QdrantClient(url=settings.qdrant_url, api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None))
    results: dict[str, dict[str, float]] = {}
    try:
        for name, collection in EXPERIMENT_COLLECTIONS.items():
            if client.collection_exists(collection):
                client.delete_collection(collection)
            client.create_collection(collection, vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE))
            kwargs: dict[str, Any] = {"chunk_size": settings.rag_chunk_size, "chunk_overlap": settings.rag_chunk_overlap}
            if name == "semantic":
                kwargs = {"embed_model": embed_model}
            nodes = STRATEGIES[name](docs, **kwargs)
            store = QdrantVectorStore(client=client, collection_name=collection)
            index = VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=store), embed_model=embed_model)

            def nodes_for(question: str) -> list[Any]:
                return index.as_retriever(similarity_top_k=settings.rag_similarity_top_k).retrieve(question)

            def retrieve(question: str) -> list[str]:
                return [_node_source(item) for item in nodes_for(question)]

            started = time.perf_counter()
            metrics = evaluate_retrieval(dataset, retrieve)
            elapsed_ms = (time.perf_counter() - started) * 1000 / len(dataset)
            lengths = [len(node.text) for node in nodes]
            results[name] = metrics | {
                "chunks_total": float(len(nodes)),
                "chunks_per_document": len(nodes) / len(docs),
                "mean_chunk_characters": sum(lengths) / len(lengths),
                "mean_retrieval_ms": elapsed_ms,
            }
            if name == "recursive" and os.getenv("RAG_EXPERIMENT_SKIP_RERANK") != "true":
                reranker = CrossEncoderReranker(settings.rag_reranker_model)

                def reranked(question: str) -> list[str]:
                    candidates = nodes_for(question)
                    return [_node_source(item) for item in reranker.rerank(question, candidates, top_n=3)]

                started = time.perf_counter()
                reranked_metrics = evaluate_retrieval(dataset, reranked)
                results["recursive_reranked"] = reranked_metrics | {
                    "mean_retrieval_ms": (time.perf_counter() - started) * 1000 / len(dataset)
                }
    finally:
        client.close()
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if output_path := os.getenv("RAG_EXPERIMENT_OUTPUT"):
        Path(output_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":  # pragma: no cover
    run()
