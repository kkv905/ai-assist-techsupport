from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import quote, urlsplit, urlunsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from redis.asyncio import Redis

from app.core.config import get_settings
from app.core.exceptions import LLMAuthError, LLMError, LLMRateLimitError, LLMTimeoutError
from app.routers.chat import router as chat_router
from app.routers.health import router as health_router
from app.routers.models import router as models_router

logger = logging.getLogger("llm-service")


def build_redis_url(settings) -> str:
    """Собирает строку подключения к Redis с учетом пароля из настроек."""

    password = settings.redis_password.get_secret_value() if settings.redis_password else None
    if not password:
        return settings.redis_url

    parsed = urlsplit(settings.redis_url)
    auth_password = quote(password, safe="")
    hostname = parsed.hostname or "localhost"
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f":{auth_password}@{hostname}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Инициализирует общие клиенты приложения и корректно закрывает их на shutdown."""

    settings = get_settings()
    app.state.openai = AsyncOpenAI(
        api_key=settings.llm.openai_api_key.get_secret_value(),
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )
    app.state.cache = Redis.from_url(build_redis_url(settings), decode_responses=True)

    try:
        yield
    finally:
        await app.state.openai.close()
        await app.state.cache.aclose()


def create_app() -> FastAPI:
    """Создает и настраивает экземпляр FastAPI-приложения."""

    settings = get_settings()
    app = FastAPI(
        title="AI Assist Techsupport",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        """Назначает идентификатор запроса, логирует время обработки и добавляет заголовок ответа."""

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        started_at = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.info(
            "Обработан HTTP-запрос.",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        """Преобразует доменные ошибки LLM в единый HTTP-формат."""

        if isinstance(exc, LLMRateLimitError):
            status_code = 429
            code = "llm_rate_limit"
        elif isinstance(exc, LLMTimeoutError):
            status_code = 504
            code = "llm_timeout"
        elif isinstance(exc, LLMAuthError):
            status_code = 502
            code = "llm_auth"
        else:
            status_code = 502
            code = "llm_error"

        return JSONResponse(
            status_code=status_code,
            content={"error": {"code": code, "message": str(exc)}},
            headers={"X-Request-ID": getattr(request.state, "request_id", "")},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Возвращает ошибки валидации в едином формате для клиентов API."""

        errors = [
            {
                "field": ".".join(str(part) for part in error["loc"] if part != "body"),
                "message": error["msg"],
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": "Ошибка валидации.", "details": errors}},
            headers={"X-Request-ID": getattr(request.state, "request_id", "")},
        )

    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(chat_router)
    return app


app = create_app()
