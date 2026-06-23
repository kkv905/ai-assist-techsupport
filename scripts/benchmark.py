import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.client import build_client, load_settings
from app.core.config import get_settings
from app.services.llm_client import AsyncLLMClient, InMemoryLLMCache

RESULTS_PATH = PROJECT_ROOT / "scripts" / "benchmark_results.md"
PROMPTS_COUNT = 20
INVALID_MODEL = "gpt-invalid-benchmark-model"


def build_prompts() -> list[str]:
    """Формирует набор однотипных промптов средней длины для бенчмарка."""
    return [
        (
            f"Объясни одним абзацем концепцию №{index}: "
            "что такое идемпотентность в API, где она полезна, и приведи короткий пример."
        )
        for index in range(1, PROMPTS_COUNT + 1)
    ]


def complete_sync(client: OpenAI, model: str, prompt: str) -> str:
    """Выполняет один синхронный запрос к chat completions."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def run_sync_benchmark(client: OpenAI, model: str, prompts: list[str]) -> tuple[list[str], float]:
    """Последовательно прогоняет все промпты через синхронный клиент."""
    started_at = time.perf_counter()
    results = [complete_sync(client=client, model=model, prompt=prompt) for prompt in prompts]
    duration = time.perf_counter() - started_at
    return results, duration


async def run_async_benchmark(
    client: AsyncLLMClient,
    prompts: list[str],
    concurrency: int,
) -> tuple[list[str | Exception], float]:
    """Прогоняет батч запросов через асинхронный клиент с заданной конкурентностью."""
    started_at = time.perf_counter()
    results = await client.batch_chat(prompts, concurrency=concurrency)
    duration = time.perf_counter() - started_at
    return results, duration


async def compare_batch_modes(client: AsyncLLMClient, prompts: list[str]) -> dict[str, Any]:
    """Сравнивает мягкий и строгий режим батча на искусственной ошибке."""
    model_overrides = {2: INVALID_MODEL}
    tolerant_results = await client.batch_chat(
        prompts,
        concurrency=5,
        model_overrides=model_overrides,
    )

    tolerant_summary = {
        "successful": sum(isinstance(item, str) for item in tolerant_results),
        "failed": sum(isinstance(item, Exception) for item in tolerant_results),
    }

    try:
        await client.batch_chat_strict(
            prompts,
            concurrency=5,
            model_overrides=model_overrides,
        )
    except* Exception as error_group:
        strict_summary = {
            "status": "ошибка",
            "exceptions": len(error_group.exceptions),
            "exception_types": sorted({type(item).__name__ for item in error_group.exceptions}),
        }
    else:
        strict_summary = {
            "status": "успех",
            "exceptions": 0,
            "exception_types": [],
        }

    return {
        "tolerant": tolerant_summary,
        "strict": strict_summary,
    }


def format_seconds(duration: float) -> str:
    """Форматирует длительность в секунды с тремя знаками после запятой."""
    return f"{duration:.3f} с"


def write_results(
    *,
    model: str,
    sync_duration: float,
    async_durations: dict[int, float],
    comparison: dict[str, Any],
) -> None:
    """Сохраняет результаты бенчмарка в Markdown-файл."""
    content = "\n".join(
        [
            "# Результаты benchmark M3B3",
            "",
            f"- Дата запуска: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Модель: `{model}`",
            f"- Количество промптов: {PROMPTS_COUNT}",
            "",
            "## Время выполнения",
            "",
            "| Режим | Общее время |",
            "| --- | --- |",
            f"| sync sequential | {format_seconds(sync_duration)} |",
            *[
                f"| async batch, concurrency={concurrency} | {format_seconds(duration)} |"
                for concurrency, duration in sorted(async_durations.items())
            ],
            "",
            "## Сравнение поведения batch-режимов",
            "",
            "- `batch_chat`: сохраняет частичные успехи и возвращает исключение на месте ошибочного запроса.",
            (
                f"  Успешных ответов: {comparison['tolerant']['successful']}, "
                f"ошибок: {comparison['tolerant']['failed']}."
            ),
            (
                f"- `batch_chat_strict`: статус `{comparison['strict']['status']}`, "
                f"количество исключений в `ExceptionGroup`: {comparison['strict']['exceptions']}."
            ),
            (
                f"  Типы исключений: {', '.join(comparison['strict']['exception_types']) or 'нет'}."
            ),
        ]
    )
    RESULTS_PATH.write_text(content, encoding="utf-8")


def write_failure_results(reason: str) -> None:
    """Сохраняет причину, по которой реальный benchmark не удалось выполнить."""
    content = "\n".join(
        [
            "# Результаты benchmark M3B3",
            "",
            "Реальный запуск benchmark не выполнен.",
            "",
            f"- Дата попытки запуска: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Причина: {reason}",
            "",
            "Ожидаемое поведение режимов:",
            "",
            "- `batch_chat` возвращает частичные успехи и объекты исключений на ошибочных позициях.",
            "- `batch_chat_strict` завершает `TaskGroup` с `ExceptionGroup` и не оставляет частичных результатов как успешный итог сценария.",
        ]
    )
    RESULTS_PATH.write_text(content, encoding="utf-8")


async def main() -> None:
    """Запускает полный сценарий бенчмарка для sync и async режимов."""
    try:
        legacy_settings = load_settings()
        settings = get_settings()
        prompts = build_prompts()

        sync_client = build_client(legacy_settings["api_key"])
        async_client = AsyncLLMClient(
            api_key=settings.llm.openai_api_key.get_secret_value(),
            model=settings.llm.default_model,
            timeout=settings.llm.request_timeout,
            max_retries=settings.llm.max_retries,
            cache=InMemoryLLMCache(ttl_seconds=settings.cache_ttl_seconds),
            fallback_models=["gpt-4.1-mini"],
        )

        print("Запуск последовательного sync-бенчмарка...")
        _, sync_duration = run_sync_benchmark(sync_client, legacy_settings["model"], prompts)
        print(f"sync sequential: {format_seconds(sync_duration)}")

        async_durations: dict[int, float] = {}
        for concurrency in (1, 5, 10):
            print(f"Запуск async batch с concurrency={concurrency}...")
            _, duration = await run_async_benchmark(async_client, prompts, concurrency)
            async_durations[concurrency] = duration
            print(f"async batch, concurrency={concurrency}: {format_seconds(duration)}")

        print("Сравнение batch_chat и batch_chat_strict на искусственной ошибке...")
        comparison = await compare_batch_modes(async_client, prompts)
        print(
            "Сравнение завершено: "
            f"мягкий режим сохранил {comparison['tolerant']['successful']} ответов, "
            f"строгий режим вернул статус {comparison['strict']['status']}."
        )

        write_results(
            model=settings.llm.default_model,
            sync_duration=sync_duration,
            async_durations=async_durations,
            comparison=comparison,
        )
        print(f"Результаты сохранены в {RESULTS_PATH}")
    except Exception as error:
        write_failure_results(str(error))
        print(f"Бенчмарк не выполнен: {error}")
    finally:
        async_client_instance = locals().get("async_client")
        if async_client_instance is not None:
            await async_client_instance.aclose()

        sync_client_instance = locals().get("sync_client")
        if sync_client_instance is not None:
            sync_client_instance.close()


if __name__ == "__main__":
    asyncio.run(main())
