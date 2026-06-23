from fastapi import APIRouter

from app.schemas.models import AVAILABLE_MODELS, ModelInfo

router = APIRouter(tags=["models"])

ERROR_RESPONSES = {
    200: {"description": "Список доступных моделей."},
    422: {"description": "Ошибка валидации входных данных."},
    429: {"description": "Превышен лимит запросов к провайдеру."},
    502: {"description": "Ошибка авторизации или недоступность провайдера."},
    504: {"description": "Провайдер не успел ответить вовремя."},
}


@router.get(
    "/models",
    response_model=list[ModelInfo],
    summary="Получить список доступных моделей",
    responses=ERROR_RESPONSES,
)
async def list_models() -> list[ModelInfo]:
    """Возвращает статический список моделей и ориентировочных цен."""

    return AVAILABLE_MODELS
