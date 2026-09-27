"""Получение и persistent-кеширование embedding-векторов для RAG.

Модуль намеренно синхронный: индексация документов является фоновой/batch-задачей.
Тяжелая зависимость ``sentence-transformers`` загружается только при первом cache miss.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from app.core.config import Settings, get_settings


class EmbeddingModel(Protocol):
    """Минимальный контракт модели, удобный и для production, и для тестов."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        """Возвращает векторы для списка текстов."""


class EmbeddingError(RuntimeError):
    """Ошибка получения embedding-векторов после повторных попыток."""


class EmbeddingService:
    """Вычисляет нормализованные dense-векторы и кеширует их на диске."""

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int = 16,
        cache_dir: Path = Path("./var/embeddings"),
        max_retries: int = 3,
        model: EmbeddingModel | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("embedding batch_size должен быть положительным.")
        if max_retries < 0:
            raise ValueError("embedding max_retries не может быть отрицательным.")
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Возвращает нормализованные векторы в исходном порядке.

        Каждый текст кешируется отдельно, поэтому частично пересекающиеся батчи не
        вызывают повторный инференс. Ключ содержит имя модели: смена
        ``EMBEDDING_MODEL`` автоматически создаёт независимый набор векторов.
        """
        self._validate_texts(texts)
        if not texts:
            return []

        vectors_by_text: dict[str, list[float]] = {}
        missing: list[str] = []
        for text in dict.fromkeys(texts):
            cached = self._read_cache(text)
            if cached is None:
                missing.append(text)
            else:
                vectors_by_text[text] = cached

        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self._encode_with_retry(batch)
            if len(vectors) != len(batch):
                raise EmbeddingError("Модель вернула число векторов, не совпадающее с размером батча.")
            for text, vector in zip(batch, vectors, strict=True):
                normalized = self._validate_vector(vector)
                self._write_cache(text, normalized)
                vectors_by_text[text] = normalized

        return [vectors_by_text[text] for text in texts]

    def _encode_with_retry(self, batch: list[str]) -> list[Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                encoded = self._get_model().encode(
                    batch,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return list(encoded)
            except (ConnectionError, OSError, TimeoutError) as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(min(0.25 * (2**attempt), 2.0))

        raise EmbeddingError("Не удалось получить embeddings после повторных попыток.") from last_error

    def _get_model(self) -> EmbeddingModel:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except (ConnectionError, OSError, TimeoutError) as error:
                raise EmbeddingError("Не удалось загрузить embedding-модель.") from error
            except ImportError as error:
                raise EmbeddingError(
                    "Не установлена зависимость sentence-transformers для embedding-модели."
                ) from error
        return self._model

    def _cache_path(self, text: str) -> Path:
        payload = f"{self.model_name}\0{text}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return self.cache_dir / self.model_name.replace("/", "__") / f"{digest}.json"

    def _read_cache(self, text: str) -> list[float] | None:
        path = self._cache_path(text)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return self._validate_vector(payload["vector"])
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _write_cache(self, text: str, vector: list[float]) -> None:
        path = self._cache_path(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"vector": vector}), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _validate_texts(texts: list[str]) -> None:
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Каждый текст для embedding должен быть строкой.")

    @staticmethod
    def _validate_vector(vector: Any) -> list[float]:
        result = [float(value) for value in vector.tolist()] if hasattr(vector, "tolist") else [float(value) for value in vector]
        if not result or not all(math.isfinite(value) for value in result):
            raise EmbeddingError("Embedding-вектор должен быть непустым и состоять из конечных чисел.")
        norm = math.sqrt(sum(value * value for value in result))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise EmbeddingError("Провайдер вернул ненормализованный embedding-вектор.")
        return result


@lru_cache
def _service_for_settings(
    model_name: str, batch_size: int, cache_dir: str, max_retries: int
) -> EmbeddingService:
    return EmbeddingService(
        model_name=model_name,
        batch_size=batch_size,
        cache_dir=Path(cache_dir),
        max_retries=max_retries,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Встраивает тексты с настройками текущего приложения."""
    settings: Settings = get_settings()
    return _service_for_settings(
        settings.embedding_model,
        settings.embedding_batch_size,
        str(settings.embedding_cache_dir),
        settings.embedding_max_retries,
    ).embed_texts(texts)
