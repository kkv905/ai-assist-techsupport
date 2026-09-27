"""Минимальный RAG pipeline на LlamaIndex и Qdrant.

Коллекция намеренно отделена от ``documents`` из M5B2: LlamaIndex сохраняет
сериализованные Node в ``_node_content``, которые нужны для цитат source_nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import Settings

NOT_FOUND_ANSWER = "В загруженном корпусе нет достаточно релевантной информации для ответа."


@dataclass
class RAGService:
    """Строит один LlamaIndex и отвечает на запросы к нему."""

    data_dir: Path
    collection: str
    qdrant_url: str
    qdrant_api_key: str | None
    embedding_model: str
    embedding_dim: int
    llm_model: str
    openai_api_key: str
    chunk_size: int
    chunk_overlap: int
    similarity_top_k: int
    score_threshold: float
    _client: QdrantClient | None = field(default=None, init=False, repr=False)
    _index: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGService":
        return cls(
            data_dir=settings.rag_data_dir,
            collection=settings.rag_collection,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            llm_model=settings.llm.default_model,
            openai_api_key=settings.llm.openai_api_key.get_secret_value(),
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            similarity_top_k=settings.rag_similarity_top_k,
            score_threshold=settings.rag_score_threshold,
        )

    @property
    def ready(self) -> bool:
        return self._index is not None

    def build(self) -> None:
        """Индексирует корпус при пустой коллекции, иначе подключается к ней."""
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"RAG corpus directory is missing: {self.data_dir}")
        if self.chunk_size < 1 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("RAG_CHUNK_SIZE и RAG_CHUNK_OVERLAP заданы некорректно.")
        if self.similarity_top_k < 3:
            raise ValueError("RAG_SIMILARITY_TOP_K должен быть не меньше 3.")

        from llama_index.core import Settings as LlamaSettings
        from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.openai import OpenAI
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        self._client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=1)
        self._ensure_collection()
        LlamaSettings.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)
        LlamaSettings.llm = OpenAI(model=self.llm_model, api_key=self.openai_api_key)
        LlamaSettings.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        vector_store = QdrantVectorStore(client=self._client, collection_name=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        if self._client.count(self.collection, exact=True).count:
            self._index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            return

        documents = SimpleDirectoryReader(input_dir=str(self.data_dir), recursive=True).load_data()
        if not documents:
            raise ValueError(f"RAG corpus {self.data_dir} does not contain readable documents.")
        self._index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    def answer(self, question: str) -> dict[str, Any]:
        """Возвращает унифицированный ответ и до трёх лучших source nodes."""
        if not self.ready:
            raise RuntimeError("RAG index has not been built.")
        question = question.strip()
        if not question:
            raise ValueError("Вопрос не должен быть пустым.")
        retriever = self._index.as_retriever(similarity_top_k=self.similarity_top_k)
        nodes = retriever.retrieve(question)
        sources = [self._source(node) for node in nodes]
        top_score = sources[0]["score"] if sources else 0.0
        if top_score < self.score_threshold:
            return {"answer": NOT_FOUND_ANSWER, "top_score": top_score, "sources": sources}

        from llama_index.core import PromptTemplate

        prompt = PromptTemplate(
            "Ты ассистент технической поддержки. Отвечай только по контексту ниже. "
            "Если контекста недостаточно, скажи, что сведений нет.\n"
            "Контекст:\n{context_str}\nВопрос: {query_str}\nОтвет:"
        )
        response = self._index.as_query_engine(
            similarity_top_k=self.similarity_top_k, text_qa_template=prompt
        ).query(question)
        response_sources = [self._source(node) for node in response.source_nodes]
        return {
            "answer": str(response),
            "top_score": top_score,
            "sources": response_sources or sources,
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def _ensure_collection(self) -> None:
        assert self._client is not None
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )
            return
        info = self._client.get_collection(self.collection)
        vectors = info.config.params.vectors
        size = vectors.size if isinstance(vectors, VectorParams) else None
        if size != self.embedding_dim:
            raise ValueError(
                f"Коллекция {self.collection!r} имеет размерность {size}, "
                f"ожидалось {self.embedding_dim}."
            )

    @staticmethod
    def _source(node: Any) -> dict[str, Any]:
        score = float(node.score or 0.0)
        metadata = node.metadata or {}
        return {
            "text": node.text[:300],
            "source": metadata.get("file_name") or metadata.get("file_path"),
            "score": round(score, 3),
        }


if __name__ == "__main__":  # pragma: no cover - ручной smoke test
    from app.core.config import get_settings

    service = RAGService.from_settings(get_settings())
    service.build()
    print(service.answer("Как сбросить пароль?"))
    service.close()
