"""Run the manually reviewed golden set against the configured RAG service.

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
    """Read completed rows from an append-only checkpoint without trusting partial lines."""
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            rows[row["user_input"]] = row
        except (json.JSONDecodeError, KeyError):
            # A process may be interrupted while writing its final line. Earlier
            # completed rows remain valid and the question will be retried.
            continue
    return rows


async def run(
    dataset: list[dict],
    service: RAGService,
    metrics: dict,
    checkpoint_path: Path,
    concurrency: int,
) -> list[dict]:
    # Judge APIs rate-limit burst traffic; a small bound is faster and more reliable
    # than 31 simultaneous structured-output requests.
    semaphore = asyncio.Semaphore(concurrency)

    completed = load_checkpoint(checkpoint_path)
    if completed:
        print(f"Checkpoint restored: {len(completed)}/{len(dataset)} rows.")

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
                print(f"Checkpoint saved: {len(completed)}/{len(dataset)} rows.")
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
        raise ValueError("Golden dataset is empty")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.output_dir / f".{args.label}.checkpoint.jsonl"
    settings = get_settings()
    service = RAGService.from_settings(settings)
    # When all dataset rows already exist in checkpoint, a resume only calls
    # judges for missing cells. Avoid loading BGE-M3 and touching Qdrant again.
    checkpoint_rows = load_checkpoint(checkpoint_path)
    needs_retrieval = any(item["user_input"] not in checkpoint_rows for item in dataset)
    if needs_retrieval:
        service.build()
    else:
        print("Checkpoint contains all RAG outputs; skipping retrieval/index initialization.")
    try:
        judge_key = settings.rag_eval_judge_api_key
        if judge_key is None:
            raise ValueError("Set DEEPSEEK_API_KEY (or RAG_EVAL_JUDGE_API_KEY) in .env before eval.")
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
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
