from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.deps.providers import LLMServiceDep
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

ERROR_RESPONSES = {
    200: {"description": "Успешный ответ сервиса."},
    422: {"description": "Ошибка валидации входных данных."},
    429: {"description": "Превышен лимит запросов к провайдеру."},
    502: {"description": "Ошибка авторизации или недоступность провайдера."},
    504: {"description": "Провайдер не успел ответить вовремя."},
}


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Получить полный ответ модели",
    responses=ERROR_RESPONSES,
)
async def complete_chat(request: ChatRequest, llm_service: LLMServiceDep) -> ChatResponse:
    """Возвращает единый JSON-ответ с текстом модели."""

    return await llm_service.complete(request)


@router.post(
    "/chat/stream",
    summary="Получить потоковый ответ модели",
    responses=ERROR_RESPONSES,
)
async def stream_chat(request: ChatRequest, llm_service: LLMServiceDep) -> StreamingResponse:
    """Проксирует ответ модели как SSE-поток."""

    async def event_stream() -> AsyncIterator[str]:
        """Сериализует события стрима в формат SSE вручную."""

        async for chunk in llm_service.stream(request):
            if chunk.content is not None:
                yield f"data: {chunk.content}\n\n"
                continue

            if chunk.usage is not None:
                usage_payload = json.dumps({"usage": chunk.usage.model_dump()}, ensure_ascii=False)
                yield f"data: {usage_payload}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
