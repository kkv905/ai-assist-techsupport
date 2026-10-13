"""Сборка метрик RAGAS 0.4, вынесенная из графа production-импортов."""

from __future__ import annotations

import asyncio
from typing import Any


def build_judge(model: str, api_key: str, base_url: str):
    """Создаёт судейскую модель DeepSeek через её OpenAI-совместимый асинхронный API."""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    # DeepSeek и OpenAI могут отклонить серию одновременных структурированных вызовов.
    # Ограниченные повторы делают отдельную метрику устойчивой без бесконечного ожидания.
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=45.0, max_retries=2)
    return llm_factory(model, provider="openai", client=client)


def build_metrics(judge: Any, embedding_model: str, embedding_api_key: str) -> dict[str, Any]:
    """Собирает четыре обязательные метрики коллекций и дискретную оценку цитирования."""
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    embeddings = OpenAIEmbeddings(
        client=AsyncOpenAI(api_key=embedding_api_key, timeout=45.0, max_retries=2),
        model=embedding_model,
    )
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
        "has_citation": make_has_citation(judge),
    }


def make_has_citation(llm: Any):
    """Возвращает бинарную русскоязычную метрику LLM-судьи, совместимую с RAGAS 0.4."""
    from pydantic import BaseModel
    from ragas.metrics import discrete_metric

    class CitationVerdict(BaseModel):
        value: str

    @discrete_metric(name="has_citation", allowed_values=["yes", "no"])
    async def has_citation(response: str) -> str:
        prompt = (
            "Содержит ли ответ ссылку на источник: маркер вида '[1]'/'[doc_id]', "
            "имя файла, или фразу 'согласно …'? Ответь строго yes либо no.\n\n"
            f"Ответ: {response}"
        )
        verdict = await llm.agenerate(prompt, response_model=CitationVerdict)
        return verdict.value.lower()

    return has_citation


async def eval_row(row: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Оценивает одну строку, сохраняя ошибки для аудита вместо остановки запуска."""
    result = dict(row)
    for name, metric in metrics.items():
        # Контрольные точки сохраняют успешные ячейки. При продолжении повторно
        # считаются только отсутствующие метрики, а не весь завершённый пример.
        if result.get(name) not in (None, ""):
            continue
        try:
            if name == "faithfulness":
                score = await metric.ascore(
                    user_input=row["user_input"],
                    response=row["response"],
                    retrieved_contexts=row["retrieved_contexts"],
                )
            elif name == "answer_relevancy":
                score = await metric.ascore(
                    user_input=row["user_input"], response=row["response"]
                )
            elif name == "context_precision":
                score = await _context_precision_with_retry(metric, row)
            elif name == "context_recall":
                score = await metric.ascore(
                    user_input=row["user_input"],
                    retrieved_contexts=row["retrieved_contexts"],
                    reference=row.get("reference") or "",
                )
            else:
                score = await metric.ascore(response=row["response"])
            value = getattr(score, "value", score)
            if value is None:
                reason = getattr(score, "reason", None) or "Metric returned no value."
                raise ValueError(reason)
            result[name] = 1.0 if value == "yes" else 0.0 if value == "no" else float(value)
            result.pop(f"{name}_error", None)
        except Exception as exc:  # Временная ошибка судейской модели должна остаться видимой в CSV.
            result[name] = None
            result[f"{name}_error"] = str(exc)
    return result


async def _context_precision_with_retry(metric: Any, row: dict[str, Any]) -> Any:
    """Повторяет расчёт всей ячейки ContextPrecision, если теряется вердикт по одному контексту.

    RAGAS evaluates every retrieved context independently. One transient judge
    connection error otherwise discards the complete precision score.
    """
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            return await metric.ascore(
                user_input=row["user_input"],
                reference=row.get("reference") or "",
                retrieved_contexts=row["retrieved_contexts"],
            )
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                raise
            await asyncio.sleep(2**attempt)
    raise RuntimeError("ContextPrecision retry loop exhausted") from last_error
