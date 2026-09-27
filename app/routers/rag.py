from fastapi import APIRouter, HTTPException, Request

from app.schemas.rag import RAGQueryRequest, RAGQueryResponse

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGQueryResponse, summary="Ответить по RAG-корпусу")
async def query_rag(payload: RAGQueryRequest, request: Request) -> dict:
    """Использует индекс, подготовленный один раз в FastAPI lifespan."""

    service = getattr(request.app.state, "rag_service", None)
    if service is None or not service.ready:
        raise HTTPException(status_code=503, detail="RAG index is not available")
    return service.answer(payload.question)
