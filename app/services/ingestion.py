"""Инкрементальная загрузка корпоративной базы знаний из нескольких форматов."""

from __future__ import annotations

import re
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}
logger = logging.getLogger(__name__)


def document_metadata(path: Path, data_dir: Path) -> dict[str, str]:
    """Возвращает стабильные полезные метаданные без передачи технических путей в эмбеддинги."""

    stat = path.stat()
    try:
        category = path.relative_to(data_dir).parts[0] if path.parent != data_dir else "general"
    except ValueError:
        category = "uploads"
    version_match = re.search(r"(?:^|[-_ ])v?(\d+(?:\.\d+){0,2})(?:[-_ .]|$)", path.stem)
    metadata = {
        "source": path.name,
        "file_name": path.name,
        "category": category,
        "last_modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
    }
    if version_match:
        metadata["version"] = version_match.group(1)
    if path.suffix.lower() == ".docx":
        author = _docx_author(path)
        if author:
            metadata["author"] = author
    return metadata


def load_documents(paths: list[Path], data_dir: Path) -> list[Any]:
    """Читает поддерживаемые файлы средствами LlamaIndex и дополняет каждый документ."""

    from llama_index.core import SimpleDirectoryReader

    readers = _readers()
    documents: list[Any] = []
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            loaded = SimpleDirectoryReader(
                input_files=[str(path)], file_extractor=readers, filename_as_id=True
            ).load_data()
        except Exception as error:
            failed = path.with_suffix(path.suffix + ".failed")
            path.rename(failed)
            logger.warning("rag_ingestion_file_failed", extra={"path": str(path), "failed": str(failed)}, exc_info=error)
            continue
        metadata = document_metadata(path, data_dir)
        for document in loaded:
            document.metadata.update(metadata)
            # Пути к файлам, отметки времени и детали загрузки создают шум для поиска.
            document.excluded_embed_metadata_keys = [
                "file_path", "file_name", "last_modified", "creation_date", "modified_date",
                "last_accessed_date", "file_size", "source", "author", "version",
            ]
            document.excluded_llm_metadata_keys = ["file_path", "last_modified", "file_size"]
        documents.extend(loaded)
    return documents


def _readers() -> dict[str, Any]:
    """Создаёт явные обработчики для PDF, DOCX, HTML и Markdown."""

    from llama_index.readers.file import DocxReader, MarkdownReader, PyMuPDFReader

    readers: dict[str, Any] = {
        ".pdf": PyMuPDFReader(),
        ".docx": DocxReader(),
        ".md": MarkdownReader(),
    }
    try:
        from llama_index.readers.file import HTMLTagReader

        readers[".html"] = HTMLTagReader()
        readers[".htm"] = HTMLTagReader()
    except ImportError:
        # Старые наборы reader-компонентов позволяют SimpleDirectoryReader выбрать встроенный HTML-обработчик.
        pass
    return readers


def _docx_author(path: Path) -> str | None:
    try:
        from docx import Document

        return Document(path).core_properties.author or None
    except Exception:
        return None
