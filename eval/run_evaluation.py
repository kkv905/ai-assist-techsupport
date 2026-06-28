from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI

from app.core.config import Settings
from app.llm.parsing import extract_json_object
from app.schemas.chat import ChatRequest
from app.services.llm import LLMService


@dataclass(slots=True)
class GoldenItem:
    """Описывает один эталонный кейс evaluation-прогона."""

    id: str
    question: str
    expected_answer: str
    expected_keywords: list[str]
    category: str
    difficulty: str
    source: str
    must_not_contain: list[str]


class NullRedisCache:
    """Подменяет Redis в evaluation-прогоне, чтобы не требовать внешний сервис."""

    async def get(self, key: str) -> None:
        """Всегда сообщает, что кеш-попадания нет."""

        return None

    async def setex(self, key: str, ttl: int, value: str) -> None:
        """Игнорирует сохранение, потому что evaluation должен быть детерминированным."""

        return None


def parse_args() -> argparse.Namespace:
    """Читает аргументы командной строки для evaluation-прогона."""

    parser = argparse.ArgumentParser(description="Ручной evaluation-прогон golden dataset.")
    parser.add_argument("--golden", required=True, help="Путь к golden dataset JSON.")
    parser.add_argument("--judge", required=True, help="Модель судьи для G-Eval проверки.")
    parser.add_argument("--out", required=True, help="Путь для сохранения JSON-результата.")
    return parser.parse_args()


def load_golden_dataset(path: Path) -> tuple[int, list[GoldenItem]]:
    """Загружает golden dataset и валидирует базовую структуру."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    version = int(payload["version"])
    items = [
        GoldenItem(
            id=item["id"],
            question=item["question"],
            expected_answer=item["expected_answer"],
            expected_keywords=list(item.get("expected_keywords", [])),
            category=item["category"],
            difficulty=item["difficulty"],
            source=item["source"],
            must_not_contain=list(item.get("must_not_contain", [])),
        )
        for item in payload["items"]
    ]
    return version, items


async def generate_answer(service: LLMService, item: GoldenItem) -> str:
    """Запрашивает ответ приложения на один эталонный вопрос."""

    response = await service.complete(
        ChatRequest(
            messages=[{"role": "user", "content": item.question}],
            temperature=0,
            max_tokens=700,
        )
    )
    return response.content


async def judge_answer(
    client: AsyncOpenAI,
    judge_model: str,
    item: GoldenItem,
    answer: str,
) -> dict[str, Any]:
    """Оценивает ответ через judge-модель в стиле G-Eval."""

    judge_prompt = build_judge_prompt(item=item, answer=answer)
    response = await client.chat.completions.create(
        model=judge_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты оцениваешь ответы AI-ассистента техподдержки. "
                    "Сначала кратко рассуждай, затем выставь оценки по шкале от 1 до 5. "
                    "Верни только JSON-объект с полями в таком порядке: "
                    "reasoning, scores, explanation. "
                    "Поле scores должно содержать relevance, correctness, completeness."
                ),
            },
            {"role": "user", "content": judge_prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = extract_json_object(content)
    scores = parsed.get("scores", {})
    return {
        "scores": {
            "relevance": float(scores["relevance"]),
            "correctness": float(scores["correctness"]),
            "completeness": float(scores["completeness"]),
        },
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "explanation": str(parsed.get("explanation", "")).strip(),
    }


def build_judge_prompt(item: GoldenItem, answer: str) -> str:
    """Формирует детальный промпт для judge-модели."""

    return (
        "Оцени ответ модели на вопрос из техподдержки.\n\n"
        f"ID кейса: {item.id}\n"
        f"Категория: {item.category}\n"
        f"Сложность: {item.difficulty}\n"
        f"Вопрос: {item.question}\n\n"
        f"Ожидаемый ответ: {item.expected_answer}\n"
        f"Обязательные ключевые слова: {', '.join(item.expected_keywords)}\n"
        f"Запрещенные слова: {', '.join(item.must_not_contain) if item.must_not_contain else 'нет'}\n\n"
        f"Фактический ответ модели: {answer}\n\n"
        "Критерии:\n"
        "1. relevance: насколько ответ относится к вопросу и задаче.\n"
        "2. correctness: насколько ответ фактически корректен и безопасен.\n"
        "3. completeness: насколько ответ покрывает нужные шаги и детали.\n\n"
        "Верни только JSON-объект."
    )


def build_settings() -> Settings:
    """Загружает настройки приложения из окружения для ручного evaluation."""

    return Settings()


async def run_evaluation(golden_path: Path, judge_model: str, output_path: Path) -> dict[str, Any]:
    """Запускает полный evaluation-прогон по всем кейсам golden dataset."""

    settings = build_settings()
    version, golden_items = load_golden_dataset(golden_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    openai_client = AsyncOpenAI(
        api_key=settings.llm.openai_api_key.get_secret_value(),
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )
    service = LLMService(openai=openai_client, cache=NullRedisCache(), settings=settings)

    try:
        items_payload: list[dict[str, Any]] = []
        for item in golden_items:
            answer = await generate_answer(service=service, item=item)
            judgment = await judge_answer(
                client=openai_client,
                judge_model=judge_model,
                item=item,
                answer=answer,
            )
            items_payload.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "answer": answer,
                    "scores": judgment["scores"],
                    "reasoning": judgment["reasoning"],
                    "explanation": judgment["explanation"],
                }
            )

        run_payload = build_run_payload(
            items=items_payload,
            model_under_test=settings.llm.default_model,
            judge_model=judge_model,
            golden_version=version,
        )
        output_path.write_text(
            json.dumps(run_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return run_payload
    finally:
        await openai_client.close()


def build_run_payload(
    *,
    items: list[dict[str, Any]],
    model_under_test: str,
    judge_model: str,
    golden_version: int,
) -> dict[str, Any]:
    """Собирает финальный JSON-артефакт прогона с агрегатами."""

    relevance_scores = [float(item["scores"]["relevance"]) for item in items]
    correctness_scores = [float(item["scores"]["correctness"]) for item in items]
    completeness_scores = [float(item["scores"]["completeness"]) for item in items]
    timestamp = datetime.now(UTC).isoformat()
    return {
        "run_id": str(uuid4()),
        "timestamp": timestamp,
        "model_under_test": model_under_test,
        "judge_model": judge_model,
        "golden_version": golden_version,
        "items": items,
        "aggregates": {
            "relevance_avg": round(sum(relevance_scores) / len(relevance_scores), 4),
            "correctness_avg": round(sum(correctness_scores) / len(correctness_scores), 4),
            "completeness_avg": round(sum(completeness_scores) / len(completeness_scores), 4),
            "min_correctness": round(min(correctness_scores), 4),
        },
    }


def main() -> None:
    """Запускает evaluation из CLI и печатает краткий итог."""

    args = parse_args()
    result = asyncio.run(
        run_evaluation(
            golden_path=Path(args.golden),
            judge_model=args.judge,
            output_path=Path(args.out),
        )
    )
    aggregates = result["aggregates"]
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "model_under_test": result["model_under_test"],
                "judge_model": result["judge_model"],
                "aggregates": aggregates,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
