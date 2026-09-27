import math

import pytest

from app.services.embeddings import EmbeddingError, EmbeddingService


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
        self.calls.append(sentences)
        assert kwargs["normalize_embeddings"] is True
        return [[0.6, 0.8] for _ in sentences]


def test_embed_texts_batches_normalizes_and_uses_persistent_cache(tmp_path) -> None:
    model = FakeEmbeddingModel()
    service = EmbeddingService(
        model_name="BAAI/bge-m3", batch_size=2, cache_dir=tmp_path, model=model
    )

    first = service.embed_texts(["первый", "второй", "третий", "первый"])
    second = EmbeddingService(
        model_name="BAAI/bge-m3", cache_dir=tmp_path, model=FakeEmbeddingModel()
    ).embed_texts(["первый", "третий"])

    assert model.calls == [["первый", "второй"], ["третий"]]
    assert first[0] == first[3] == [0.6, 0.8]
    assert second == [[0.6, 0.8], [0.6, 0.8]]
    assert math.isclose(math.sqrt(sum(value * value for value in first[0])), 1.0)


def test_embed_texts_separates_cache_when_model_changes(tmp_path) -> None:
    first_model = FakeEmbeddingModel()
    second_model = FakeEmbeddingModel()
    EmbeddingService(model_name="model-a", cache_dir=tmp_path, model=first_model).embed_texts(["текст"])
    EmbeddingService(model_name="model-b", cache_dir=tmp_path, model=second_model).embed_texts(["текст"])

    assert first_model.calls == [["текст"]]
    assert second_model.calls == [["текст"]]


def test_embed_texts_rejects_non_normalized_provider_result(tmp_path) -> None:
    class UnnormalizedModel:
        def encode(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
            return [[1.0, 1.0] for _ in sentences]

    service = EmbeddingService(model_name="test", cache_dir=tmp_path, model=UnnormalizedModel())

    with pytest.raises(EmbeddingError, match="ненормализованный"):
        service.embed_texts(["текст"])
