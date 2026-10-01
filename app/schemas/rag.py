from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    """Вопрос к демонстрационному RAG-корпусу."""

    question: str = Field(min_length=1, max_length=4000)


class RAGSource(BaseModel):
    id: str
    file_name: str | None = None
    page: str | int | None = None
    score: float
    snippet: str


class RAGQueryResponse(BaseModel):
    answer: str
    top_score: float
    sources: list[RAGSource]
    confident: bool
    is_fallback: bool = Field(description="Ответ является каноническим fallback RAG.")
