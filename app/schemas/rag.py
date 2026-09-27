from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    """Вопрос к демонстрационному RAG-корпусу."""

    question: str = Field(min_length=1, max_length=4000)


class RAGSource(BaseModel):
    text: str
    source: str | None = None
    score: float


class RAGQueryResponse(BaseModel):
    answer: str
    top_score: float
    sources: list[RAGSource]
