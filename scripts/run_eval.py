"""Запускает вручную проверенный эталонный набор на настроенном RAG-сервисе.

Usage: uv sync --extra eval --extra tracing; uv run python scripts/run_eval.py --label baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.eval.metrics import build_judge, build_metrics, eval_row  # noqa: E402
from app.services.rag import RAGService  # noqa: E402

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "has_citation",
)


def load_checkpoint(path: Path) -> dict[str, dict]:
    """Читает готовые строки из дописываемой контрольной точки, игнорируя неполные строки."""
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            rows[row["user_input"]] = row
        except (json.JSONDecodeError, KeyError):
            # Процесс может быть прерван при записи последней строки. Ранее
            # завершённые строки остаются корректными, а вопрос будет повторён.
            continue
    return rows


async def run(
    dataset: list[dict],
    service: RAGService,
    metrics: dict,
    checkpoint_path: Path,
    concurrency: int,
) -> list[dict]:
    # API судейской модели ограничивает всплески запросов; малый лимит быстрее и
    # надёжнее, чем 31 одновременный запрос структурированного вывода.
    semaphore = asyncio.Semaphore(concurrency)

    completed = load_checkpoint(checkpoint_path)
    if completed:
        print(f"Контрольная точка восстановлена: {len(completed)}/{len(dataset)} строк.")

    async def one(item: dict, previous: dict | None = None) -> dict:
        async with semaphore:
            if previous is None:
                started = perf_counter()
                rag_output = await asyncio.to_thread(service.evaluate_inputs, item["user_input"])
                row = {
                    **item,
                    **rag_output,
                    "response": rag_output["answer"],
                    "latency_ms": round((perf_counter() - started) * 1000, 2),
                }
            else:
                row = previous
            return await eval_row(row, metrics)

    pending = [
        (item, completed.get(item["user_input"]))
        for item in dataset
        if any(completed.get(item["user_input"], {}).get(metric) in (None, "") for metric in METRIC_NAMES)
    ]
    if pending:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
            tasks = [asyncio.create_task(one(item, previous)) for item, previous in pending]
            for task in asyncio.as_completed(tasks):
                row = await task
                checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
                checkpoint.flush()
                completed[row["user_input"]] = row
                print(f"Контрольная точка сохранена: {len(completed)}/{len(dataset)} строк.")
    return [completed[item["user_input"]] for item in dataset]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=ROOT / "tests/eval/golden_dataset.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tests/eval/results")
    parser.add_argument("--label", required=True, help="For example baseline, chunk_1024 or top_k_5")
    parser.add_argument("--limit", type=int, help="Evaluate only first N rows (diagnostics only)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Concurrent rows; use 1–2 for stable DeepSeek/OpenAI eval (default: 2)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Append-only JSONL state; defaults to tests/eval/results/.<label>.checkpoint.jsonl",
    )
    args = parser.parse_args()
    dataset = json.loads(args.golden.read_text(encoding="utf-8"))
    if args.limit is not None:
        dataset = dataset[: args.limit]
    if not dataset:
        raise ValueError("Эталонный набор данных пуст.")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.output_dir / f".{args.label}.checkpoint.jsonl"
    settings = get_settings()
    service = RAGService.from_settings(settings)
    # Когда все строки набора уже есть в контрольной точке, при продолжении
    # судейская модель вызывается лишь для отсутствующих ячеек. Не загружаем
    # BGE-M3 и не обращаемся к Qdrant повторно.
    checkpoint_rows = load_checkpoint(checkpoint_path)
    needs_retrieval = any(item["user_input"] not in checkpoint_rows for item in dataset)
    if needs_retrieval:
        service.build()
    else:
        print("Контрольная точка содержит все ответы RAG; инициализация поиска и индекса пропущена.")
    try:
        judge_key = settings.rag_eval_judge_api_key
        if judge_key is None:
            raise ValueError("Перед оценкой задайте DEEPSEEK_API_KEY (или RAG_EVAL_JUDGE_API_KEY) в .env.")
        metrics = build_metrics(
            build_judge(
                settings.rag_eval_judge_model,
                judge_key.get_secret_value(),
                settings.rag_eval_judge_base_url,
            ),
            settings.rag_eval_embedding_model,
            settings.llm.openai_api_key.get_secret_value(),
        )
        rows = asyncio.run(run(dataset, service, metrics, checkpoint_path, args.concurrency))
    finally:
        service.close()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    csv_path = args.output_dir / f"{stamp}_{args.label}.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    aggregates = {
        "label": args.label,
        "rows": len(frame),
        "judge_model": settings.rag_eval_judge_model,
        "embedding_model": settings.rag_eval_embedding_model,
        "metrics": frame[["faithfulness", "answer_relevancy", "context_precision", "context_recall", "has_citation", "latency_ms"]].mean(numeric_only=True).to_dict(),
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(aggregates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранены файлы: {csv_path} и {json_path}")


if __name__ == "__main__":
    main()
