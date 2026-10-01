from types import SimpleNamespace
from pathlib import Path

from app.services.rag import NOT_FOUND_ANSWER, RAGService
from app.services.rag_baremetal import BaremetalRAGService


def test_rag_returns_fallback_without_calling_llm() -> None:
    service = RAGService(
        data_dir=Path("data/rag-block-03"),
        collection="test", qdrant_url="http://unused", qdrant_api_key=None,
        embedding_model="test", embedding_dim=2, llm_model="test", openai_api_key="test",
        chunk_size=512, chunk_overlap=64, similarity_top_k=3, score_threshold=0.35,
    )

    class Index:
        def as_retriever(self, **kwargs):
            return SimpleNamespace(retrieve=lambda question: [
                SimpleNamespace(text="нерелевантный фрагмент", metadata={"file_name": "x.txt"}, score=0.1)
            ])

    service._index = Index()

    result = service.answer("Вопрос вне базы")

    assert result["answer"] == NOT_FOUND_ANSWER
    assert result["top_score"] == 0.1
    assert result["sources"] == [
        {
            "id": "1",
            "file_name": "x.txt",
            "page": None,
            "score": 0.1,
            "snippet": "нерелевантный фрагмент",
        }
    ]
    assert result["confident"] is False
    assert result["is_fallback"] is True


def test_rag_marks_llm_fallback_even_when_score_is_confident() -> None:
    service = RAGService(
        data_dir=Path("data/rag-block-03"),
        collection="test", qdrant_url="http://unused", qdrant_api_key=None,
        embedding_model="test", embedding_dim=2, llm_model="test", openai_api_key="test",
        chunk_size=512, chunk_overlap=64, similarity_top_k=3, score_threshold=0.35,
    )

    class Index:
        def as_retriever(self, **kwargs):
            return SimpleNamespace(retrieve=lambda question: [
                SimpleNamespace(text="похожий фрагмент", metadata={"file_name": "x.txt"}, score=0.9)
            ])

    service._index = Index()
    service._synthesize = lambda question, nodes: NOT_FOUND_ANSWER  # type: ignore[method-assign]

    result = service.answer("Вопрос вне базы")

    assert result["confident"] is True
    assert result["is_fallback"] is True


def test_rag_marks_substantive_answer_as_not_fallback() -> None:
    service = RAGService(
        data_dir=Path("data/rag-block-03"),
        collection="test", qdrant_url="http://unused", qdrant_api_key=None,
        embedding_model="test", embedding_dim=2, llm_model="test", openai_api_key="test",
        chunk_size=512, chunk_overlap=64, similarity_top_k=3, score_threshold=0.35,
    )

    class Index:
        def as_retriever(self, **kwargs):
            return SimpleNamespace(retrieve=lambda question: [
                SimpleNamespace(text="инструкция VPN", metadata={"file_name": "vpn.txt"}, score=0.9)
            ])

    service._index = Index()
    service._synthesize = lambda question, nodes: "Проверьте настройки VPN [1]."  # type: ignore[method-assign]

    result = service.answer("Не подключается VPN")

    assert result["confident"] is True
    assert result["is_fallback"] is False


def test_baremetal_reader_chunks_markdown_and_text(tmp_path) -> None:
    (tmp_path / "guide.md").write_text("abcdef", encoding="utf-8")
    (tmp_path / "skip.pdf").write_text("not a PDF", encoding="utf-8")
    service = BaremetalRAGService(
        data_dir=tmp_path, collection="test", qdrant_url="http://unused", qdrant_api_key=None,
        embedding_dim=2, llm_model="test", openai_api_key="test", chunk_size=4,
        chunk_overlap=1, similarity_top_k=3, score_threshold=0.35,
    )

    assert service._chunks() == [
        {"text": "abcd", "source": "guide.md", "index": 0},
        {"text": "def", "source": "guide.md", "index": 1},
    ]
