"""Corporate RAG retrieval, refusal guard and source citations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import Settings
from app.observability.tracing import rag_span
from app.services.ingestion import SUPPORTED_SUFFIXES, load_documents

logger = logging.getLogger(__name__)
NOT_FOUND_ANSWER = "По базе не нашёл, могу эскалировать."
RAG_PROMPT = """Ты корпоративный ассистент технической поддержки. Отвечай только фактами из контекста.
После каждого утверждения ставь номера источников в формате [1], [2]. Если в контексте нет ответа,
ответь ровно: «По базе не нашёл, могу эскалировать.»

Контекст:
{context_str}

Вопрос: {query_str}
Ответ:"""


@dataclass
class RAGService:
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
    chunking_strategy: str = "recursive"
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    docstore_dir: Path = Path("var/rag_docstore")
    _client: QdrantClient | None = field(default=None, init=False, repr=False)
    _index: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGService":
        return cls(
            data_dir=settings.rag_data_dir, collection=settings.rag_collection,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            embedding_model=settings.embedding_model, embedding_dim=settings.embedding_dim,
            llm_model=settings.llm.default_model, openai_api_key=settings.llm.openai_api_key.get_secret_value(),
            chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap,
            similarity_top_k=settings.rag_similarity_top_k, score_threshold=settings.rag_score_threshold,
            chunking_strategy=settings.rag_chunking_strategy, reranker_enabled=settings.rag_reranker_enabled,
            reranker_model=settings.rag_reranker_model, docstore_dir=settings.rag_docstore_dir,
        )

    @property
    def ready(self) -> bool:
        return self._index is not None

    def build(self) -> None:
        self.ingest()

    def ingest(self, paths: list[Path] | None = None) -> dict[str, int]:
        """Run LlamaIndex IngestionPipeline with UPSERTS; unchanged inputs are skipped."""
        self._validate()
        self._configure_llama()
        assert self._client is not None
        self._ensure_collection()
        from llama_index.core import VectorStoreIndex
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        vector_store = QdrantVectorStore(client=self._client, collection_name=self.collection)
        self._index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        candidates = paths or [path for path in self.data_dir.rglob("*") if path.is_file()]
        candidates = [path for path in candidates if path.suffix.lower() in SUPPORTED_SUFFIXES]
        if not candidates:
            return {"changed": 0, "unchanged": 0}
        documents = load_documents(candidates, self.data_dir)
        return self._run_pipeline(documents, vector_store)

    def answer(self, question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Retrieve top-10, refuse before LLM when weak, otherwise cite sources."""
        if not self.ready:
            raise RuntimeError("RAG index has not been built.")
        question = question.strip()
        if not question:
            raise ValueError("Вопрос не должен быть пустым.")
        with rag_span(
            "RAG query",
            {
                "openinference.span.kind": "CHAIN",
                "input.value": question,
                "rag.score_threshold": self.score_threshold,
            },
        ):
            retrieval_question, nodes = self._retrieve(question, history or [])
            sources = [self._source(node, number) for number, node in enumerate(nodes[:5], start=1)]
            top_score = sources[0]["score"] if sources else 0.0
            if top_score < self.score_threshold:
                logger.info("rag_score_guard", extra={"top_score": top_score, "threshold": self.score_threshold})
                return {
                    "answer": NOT_FOUND_ANSWER,
                    "top_score": top_score,
                    "sources": sources,
                    "confident": False,
                    "is_fallback": True,
                }

            answer = self._synthesize(retrieval_question, nodes)
            return {
                "answer": answer,
                "top_score": top_score,
                "sources": sources,
                "confident": True,
                "is_fallback": answer.strip() == NOT_FOUND_ANSWER,
            }

    def evaluate_inputs(self, question: str) -> dict[str, Any]:
        """Return one RAG answer and the untruncated chunks used to produce it.

        This is intentionally separate from the HTTP schema: evaluators need the
        complete node text, whereas clients should receive short safe snippets.
        """
        if not self.ready:
            raise RuntimeError("RAG index has not been built.")
        question = question.strip()
        if not question:
            raise ValueError("Вопрос не должен быть пустым.")
        retrieval_question, nodes = self._retrieve(question, [])
        contexts = [self._node_text(node) for node in nodes]
        top_score = float(nodes[0].score or 0.0) if nodes else 0.0
        if top_score < self.score_threshold:
            return {"answer": NOT_FOUND_ANSWER, "retrieved_contexts": contexts}
        return {"answer": self._synthesize(retrieval_question, nodes), "retrieved_contexts": contexts}

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def _configure_llama(self) -> None:
        from llama_index.core import Settings as LlamaSettings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.openai import OpenAI
        self._client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=10)
        LlamaSettings.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)
        LlamaSettings.llm = OpenAI(model=self.llm_model, api_key=self.openai_api_key)

    def _run_pipeline(self, documents: list[Any], vector_store: Any) -> dict[str, int]:
        from llama_index.core import Settings as LlamaSettings
        from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.storage.docstore import SimpleDocumentStore
        self.docstore_dir.mkdir(parents=True, exist_ok=True)
        persist_path = self.docstore_dir / f"{self.collection}.json"
        docstore = SimpleDocumentStore.from_persist_path(str(persist_path)) if persist_path.exists() else SimpleDocumentStore()
        before = len(docstore.docs)
        pipeline = IngestionPipeline(
            transformations=[SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap), LlamaSettings.embed_model],
            vector_store=vector_store, docstore=docstore, docstore_strategy=DocstoreStrategy.UPSERTS,
        )
        pipeline.run(documents=documents, show_progress=False)
        docstore.persist(str(persist_path))
        changed = max(0, len(docstore.docs) - before)
        return {"changed": changed, "unchanged": max(0, len(documents) - changed)}

    def _condense(self, question: str, history: list[dict[str, str]]) -> str:
        if len(question.split()) > 5 or not history:
            return question
        previous = "\n".join(f"{item['role']}: {item['content']}" for item in history[-4:])
        return f"{previous}\nuser: {question}"

    def _retrieve(self, question: str, history: list[dict[str, str]]) -> tuple[str, list[Any]]:
        """Retrieve exactly once, including the optional reranking stage."""
        retrieval_question = self._condense(question, history)
        with rag_span(
            "RAG retrieval",
            {
                "openinference.span.kind": "RETRIEVER",
                "input.value": retrieval_question,
                "rag.similarity_top_k": self.similarity_top_k,
                "rag.reranker_enabled": self.reranker_enabled,
            },
        ) as span:
            nodes = self._index.as_retriever(similarity_top_k=self.similarity_top_k).retrieve(retrieval_question)
            if self.reranker_enabled and nodes:
                from app.services.reranker import CrossEncoderReranker

                nodes = CrossEncoderReranker(self.reranker_model).rerank(retrieval_question, nodes, top_n=5)
            if span is not None:
                span.set_attribute("rag.retrieved_document_count", len(nodes))
        return retrieval_question, nodes

    def _synthesize(self, question: str, nodes: list[Any]) -> str:
        from llama_index.core import PromptTemplate, get_response_synthesizer

        with rag_span(
            "RAG synthesis",
            {
                "openinference.span.kind": "CHAIN",
                "input.value": question,
                "rag.context_count": len(nodes),
            },
        ):
            synthesizer = get_response_synthesizer(text_qa_template=PromptTemplate(RAG_PROMPT))
            return str(synthesizer.synthesize(question, nodes))

    @staticmethod
    def _node_text(node: Any) -> str:
        get_content = getattr(node, "get_content", None)
        return str(get_content() if callable(get_content) else node.text)

    def _validate(self) -> None:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"RAG corpus directory is missing: {self.data_dir}")
        if self.chunk_size < 1 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("RAG_CHUNK_SIZE и RAG_CHUNK_OVERLAP заданы некорректно.")

    def _ensure_collection(self) -> None:
        assert self._client is not None
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(self.collection, vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE))

    @staticmethod
    def _source(node: Any, number: int) -> dict[str, Any]:
        metadata = node.metadata or {}
        return {"id": str(getattr(node, "node_id", number)), "file_name": metadata.get("file_name") or metadata.get("source"),
                "page": metadata.get("page_label") or metadata.get("page"), "score": round(float(node.score or 0.0), 3),
                "snippet": RAGService._node_text(node)[:300]}
