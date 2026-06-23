from fastapi import APIRouter

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
