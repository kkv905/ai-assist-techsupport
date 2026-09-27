"""Независимые от векторного хранилища метрики качества retrieval."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


def _identifier(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        return str(item.get("source") or item.get("file_name") or item.get("file_path") or "")
    metadata = getattr(item, "metadata", {}) or {}
    return str(metadata.get("file_name") or metadata.get("file_path") or "")


def _matches(retrieved: Any, relevant: set[str]) -> bool:
    identifier = _identifier(retrieved).replace("\\", "/")
    return identifier in relevant or identifier.rsplit("/", 1)[-1] in relevant


def hit_rate_at_k(retrieved: Sequence[Any], relevant_doc_ids: Iterable[str], k: int = 5) -> float:
    """1, если хотя бы один эталонный документ попал в первые ``k``."""
    relevant = set(relevant_doc_ids)
    return float(any(_matches(item, relevant) for item in retrieved[:k]))


def mrr_at_k(retrieved: Sequence[Any], relevant_doc_ids: Iterable[str], k: int = 10) -> float:
    """Reciprocal rank первого релевантного документа в первых ``k``."""
    relevant = set(relevant_doc_ids)
    for rank, item in enumerate(retrieved[:k], start=1):
        if _matches(item, relevant):
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: Sequence[Any], relevant_doc_ids: Iterable[str], k: int = 10) -> float:
    """Доля уникальных эталонных документов, найденных в первых ``k``."""
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0
    found = {_identifier(item).replace("\\", "/").rsplit("/", 1)[-1] for item in retrieved[:k]}
    return len(found & relevant) / len(relevant)


def evaluate_retrieval(
    dataset: Sequence[Mapping[str, Any]], retrieve: Callable[[str], Sequence[Any]]
) -> dict[str, float]:
    """Возвращает средние Hit Rate@5, MRR@10 и Recall@10 (от 0 до 1)."""
    if not dataset:
        raise ValueError("Golden dataset must not be empty.")
    hit = mrr = recall = 0.0
    for example in dataset:
        relevant = example["relevant_doc_ids"]
        retrieved = retrieve(str(example["question"]))
        hit += hit_rate_at_k(retrieved, relevant)
        mrr += mrr_at_k(retrieved, relevant)
        recall += recall_at_k(retrieved, relevant)
    count = len(dataset)
    return {"hit_rate_at_5": hit / count, "mrr_at_10": mrr / count, "recall_at_10": recall / count}
