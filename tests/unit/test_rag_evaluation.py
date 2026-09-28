from pathlib import Path
from types import SimpleNamespace

from app.services.rag import NOT_FOUND_ANSWER, RAGService


def test_evaluate_inputs_returns_full_context_without_llm_for_weak_match() -> None:
    service = RAGService(
        data_dir=Path("data"), collection="test", qdrant_url="http://unused", qdrant_api_key=None,
        embedding_model="test", embedding_dim=2, llm_model="test", openai_api_key="test",
        chunk_size=512, chunk_overlap=64, similarity_top_k=3, score_threshold=0.35,
    )

    class Index:
        def as_retriever(self, **kwargs):
            return SimpleNamespace(retrieve=lambda question: [
                SimpleNamespace(text="полный контекст без обрезки", metadata={}, score=0.1)
            ])

    service._index = Index()
    result = service.evaluate_inputs("Вопрос вне базы")

    assert result == {
        "answer": NOT_FOUND_ANSWER,
        "retrieved_contexts": ["полный контекст без обрезки"],
    }
