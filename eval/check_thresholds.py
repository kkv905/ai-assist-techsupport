from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
THRESHOLDS_PATH = BASE_DIR / "thresholds.yaml"


def load_thresholds(path: Path) -> dict[str, float]:
    """Загружает пороги оценки из YAML-файла."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    aggregates = payload.get("aggregates", {})
    return {
        "correctness_avg": float(aggregates["correctness_avg"]),
        "min_correctness": float(aggregates["min_correctness"]),
    }


def find_latest_run(runs_dir: Path) -> Path:
    """Находит последний JSON-артефакт evaluation-прогона."""

    run_files = sorted(runs_dir.glob("*.json"))
    if not run_files:
        raise FileNotFoundError("В каталоге eval/runs не найдено ни одного evaluation-прогона.")
    return run_files[-1]


def collect_failures(run_payload: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    """Сверяет агрегаты прогона с порогами и возвращает список нарушений."""

    failures: list[str] = []
    aggregates = run_payload.get("aggregates", {})
    for metric_name, threshold in thresholds.items():
        actual_value = float(aggregates.get(metric_name, 0))
        if actual_value < threshold:
            failures.append(
                f"Порог не выполнен: {metric_name}={actual_value:.4f}, требуется не ниже {threshold:.4f}."
            )
    return failures


def main() -> None:
    """Проверяет последний evaluation-прогон и завершает процесс с корректным кодом."""

    latest_run_path = find_latest_run(RUNS_DIR)
    thresholds = load_thresholds(THRESHOLDS_PATH)
    run_payload = json.loads(latest_run_path.read_text(encoding="utf-8"))
    failures = collect_failures(run_payload=run_payload, thresholds=thresholds)

    if failures:
        print(f"Проверка порогов не пройдена для файла: {latest_run_path}")
        for failure in failures:
            print(failure)
        sys.exit(1)

    print(f"Проверка порогов пройдена: {latest_run_path}")


if __name__ == "__main__":
    main()
