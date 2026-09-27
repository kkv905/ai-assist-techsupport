"""Стратегии разбиения документов для экспериментов RAG.

Импорты LlamaIndex находятся внутри функций: это сохраняет быстрыми unit-тесты
и не загружает embedding-модель при старте HTTP-приложения.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any


def russian_sentence_tokenizer(text: str) -> list[str]:
    """Разделяет русский текст на предложения, сохраняя знаки пунктуации."""
    return [part.strip() for part in re.split(r"(?<=[.!?…])\s+", text) if part.strip()]


def fixed_size(
    documents: Sequence[Any], chunk_size: int = 512, chunk_overlap: int = 64
) -> list[Any]:
    """Baseline: токеновый splitter без привязки к границам предложений."""
    from llama_index.core.node_parser import TokenTextSplitter

    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.get_nodes_from_documents(list(documents))


def recursive(
    documents: Sequence[Any], chunk_size: int = 512, chunk_overlap: int = 64
) -> list[Any]:
    """Paragraph/sentence-aware аналог RecursiveCharacterTextSplitter."""
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        chunking_tokenizer_fn=russian_sentence_tokenizer,
    )
    return splitter.get_nodes_from_documents(list(documents))


def semantic(
    documents: Sequence[Any],
    embed_model: Any,
    *,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
) -> list[Any]:
    """Создаёт семантические чанки по изменению embedding-сходства."""
    from llama_index.core.node_parser import SemanticSplitterNodeParser

    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
    )
    return splitter.get_nodes_from_documents(list(documents))


STRATEGIES: dict[str, Callable[..., list[Any]]] = {
    "fixed": fixed_size,
    "recursive": recursive,
    "semantic": semantic,
}
