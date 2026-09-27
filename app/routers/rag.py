from pathlib import Path
from uuid import uuid4

import anyio
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile, status

from app.schemas.rag import RAGQueryRequest, RAGQueryResponse

router = APIRouter(prefix="/rag", tags=["rag"])
documents_router = APIRouter(prefix="/documents", tags=["rag"])


@router.post("/query", response_model=RAGQueryResponse, summary="Ответить по RAG-корпусу")
async def query_rag(payload: RAGQueryRequest, request: Request) -> dict:
    """Использует индекс, подготовленный один раз в FastAPI lifespan."""

    service = getattr(request.app.state, "rag_service", None)
    if service is None or not service.ready:
        raise HTTPException(status_code=503, detail="RAG index is not available")
    return service.answer(payload.question)


@documents_router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> dict[str, str]:
    """Store a supported document and index it without blocking the HTTP request."""

    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="RAG index is not available")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}:
        raise HTTPException(status_code=415, detail="Поддерживаются PDF, DOCX, HTML, Markdown и TXT.")
    uploads = service.data_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    target = uploads / f"{uuid4().hex}_{Path(file.filename or 'document').name}"
    target.write_bytes(await file.read())

    async def index_upload() -> None:
        await anyio.to_thread.run_sync(service.ingest, [target])

    background_tasks.add_task(index_upload)
    return {"status": "accepted", "file_name": target.name}
