"""Index the knowledge base incrementally: ``uv run python scripts/ingest.py data``."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.services.rag import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally ingest RAG documents")
    parser.add_argument("directory", nargs="?", default=None)
    args = parser.parse_args()
    settings = get_settings()
    service = RAGService.from_settings(settings)
    if args.directory:
        service.data_dir = Path(args.directory)
    try:
        result = service.ingest()
        print(f"{result['changed']} changed, {result['unchanged']} unchanged")
    finally:
        service.close()


if __name__ == "__main__":
    main()
