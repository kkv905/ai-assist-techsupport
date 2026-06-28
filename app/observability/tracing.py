from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse

_TRACING_INITIALIZED = False


def _resolve_phoenix_working_dir() -> str:
    """Подбирает каталог Phoenix с гарантированной возможностью записи."""

    candidates = [
        os.environ.get("PHOENIX_WORKING_DIR"),
        os.environ.get("PHOENIX_WORKINGDIR"),
        str(Path(tempfile.gettempdir()) / "phoenix"),
    ]

    for candidate in candidates:
        if not candidate:
            continue

        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return str(path)
        except OSError:
            continue

    return str(Path(tempfile.gettempdir()))


def _normalize_collector_endpoint(raw_endpoint: str) -> str:
    """Приводит endpoint Phoenix к формату, ожидаемому HTTP OTLP exporter-ом."""

    parsed = urlparse(raw_endpoint)
    if parsed.path.endswith("/v1/traces"):
        return raw_endpoint

    normalized_path = "/v1/traces" if not parsed.path else parsed.path.rstrip("/") + "/v1/traces"
    return urlunparse(parsed._replace(path=normalized_path))


def setup_tracing(project_name: str = "ai-assistant-techsupport") -> None:
    """Подключает Phoenix и автоинструментацию OpenAI до создания клиента SDK."""

    global _TRACING_INITIALIZED

    if _TRACING_INITIALIZED:
        return

    os.environ["PHOENIX_WORKING_DIR"] = _resolve_phoenix_working_dir()

    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from phoenix.otel import register
    except ImportError:
        logging.getLogger(__name__).warning(
            "Зависимости для трассировки не установлены, Phoenix-инструментация пропущена."
        )
        return
    except Exception:
        logging.getLogger(__name__).warning(
            "Не удалось инициализировать Phoenix-трассировку.",
            exc_info=True,
        )
        return

    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    normalized_endpoint = _normalize_collector_endpoint(endpoint)
    tracer_provider = register(
        project_name=project_name,
        endpoint=normalized_endpoint,
        protocol="http/protobuf",
    )
    instrumentor = OpenAIInstrumentor()
    is_instrumented = getattr(instrumentor, "is_instrumented_by_opentelemetry", False)
    if callable(is_instrumented):
        is_instrumented = is_instrumented()
    if not is_instrumented:
        instrumentor.instrument(tracer_provider=tracer_provider)
    _TRACING_INITIALIZED = True
