import asyncio

from fastapi import APIRouter, Request, Response, status

router = APIRouter(tags=["health"])

ERROR_RESPONSES = {
    200: {"description": "Сервис доступен."},
    422: {"description": "Ошибка валидации входных данных."},
    429: {"description": "Превышен лимит запросов к провайдеру."},
    502: {"description": "Ошибка авторизации или недоступность провайдера."},
    504: {"description": "Провайдер не успел ответить вовремя."},
}


@router.get("/health", summary="Проверить доступность HTTP-сервиса", responses=ERROR_RESPONSES)
async def healthcheck() -> dict[str, str]:
    """Возвращает статус сервиса без обращения к внешним зависимостям."""

    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
async def ready(request: Request, response: Response) -> dict:
    """Проверяет готовность сервиса и доступность Redis."""

    cache = getattr(request.app.state, "cache", None)
    redis_ok = False
    if cache is not None:
        try:
            await asyncio.wait_for(cache.ping(), timeout=1.5)
            redis_ok = True
        except (Exception, asyncio.TimeoutError):
            redis_ok = False

    if redis_ok:
        return {"status": "ok", "redis": "up"}

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "degraded", "redis": "down"}
