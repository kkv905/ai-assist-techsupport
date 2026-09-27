"""Lazy wrapper around multilingual BGE cross-encoder reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CrossEncoderReranker:
    """Пересортировывает кандидатов, не меняя их тип и metadata."""

    model_name: str = "BAAI/bge-reranker-v2-m3"
    _model: Any = field(default=None, init=False, repr=False)

    def rerank(self, query: str, candidates: list[Any], top_n: int = 10) -> list[Any]:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        if not candidates:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        texts = [self._text(candidate) for candidate in candidates]
        scores = self._model.predict([(query, text) for text in texts])
        ranked = sorted(zip(scores, candidates, strict=True), key=lambda item: item[0], reverse=True)
        return [candidate for _, candidate in ranked[:top_n]]

    @staticmethod
    def _text(candidate: Any) -> str:
        return str(getattr(candidate, "text", candidate))
