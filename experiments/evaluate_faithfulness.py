"""Evaluate saved M6B5 answers with the configured LLM-as-a-judge."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.common import RESULTS_PATH, search_knowledge_base

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "quality_results.json"


def build_rows(results_path: Path = RESULTS_PATH) -> list[dict[str, Any]]:
    """Attach the exact shared-tool observation supplied to the LLM judge."""
    results = json.loads(results_path.read_text(encoding="utf-8"))
    return [
        {
            "implementation": result["implementation"],
            "user_input": result["question"],
            "response": result["answer"],
            "retrieved_contexts": [search_knowledge_base.invoke({"query": result["question"]})],
        }
        for result in results
    ]


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a binary faithfulness verdict, retaining failures for audit."""
    from app.core.config import get_settings
    from openai import OpenAI

    settings = get_settings()
    if settings.rag_eval_judge_api_key is None:
        raise ValueError("Set DEEPSEEK_API_KEY or RAG_EVAL_JUDGE_API_KEY in .env before evaluation.")
    prompt = (
        "You are a strict support-answer faithfulness judge. Score 1 only when every "
        "factual claim in ANSWER is supported by CONTEXT; score 0 otherwise. A refusal "
        "that says the context has no reliable data is supported when CONTEXT says that. "
        "Ignore citation-marker formatting. Reply with exactly one character: 0 or 1.\n\n"
        f"CONTEXT:\n{row['retrieved_contexts'][0]}\n\nANSWER:\n{row['response']}"
    )
    client = OpenAI(api_key=settings.rag_eval_judge_api_key.get_secret_value(), base_url=settings.rag_eval_judge_base_url, timeout=45.0, max_retries=1)
    try:
        response = client.chat.completions.create(
            model=settings.rag_eval_judge_model,
            messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=3,
        )
        verdict = response.choices[0].message.content or ""
        match = re.search(r"[01]", verdict)
        if match is None:
            raise ValueError(f"Unexpected judge verdict: {verdict!r}")
        return {**row, "faithfulness": float(match.group()), "judge_verdict": verdict.strip()}
    except Exception as error:
        return {**row, "faithfulness": None, "faithfulness_error": str(error)}
    finally:
        client.close()


def evaluate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use bounded parallelism, without retrieval, embeddings, or corpus changes."""
    # DeepSeek drops concurrent short structured verdicts intermittently; a
    # single bounded worker makes this small five-by-two comparison reliable.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return list(executor.map(_score_row, rows))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    values: dict[str, list[float]] = {"single": [], "multi": []}
    for row in rows:
        score = row.get("faithfulness")
        if isinstance(score, (int, float)):
            values[row["implementation"]].append(float(score))
    return {name: round(sum(scores) / len(scores), 4) if scores else None for name, scores in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = evaluate(build_rows())
    payload = {"metric": "LLM-as-a-judge faithfulness (DeepSeek)", "judge_model": "configured RAG_EVAL_JUDGE_MODEL", "rows": rows, "averages": aggregate(rows)}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["averages"], ensure_ascii=False))


if __name__ == "__main__":
    main()
