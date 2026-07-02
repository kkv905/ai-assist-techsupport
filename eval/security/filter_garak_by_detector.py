# скрипт читает garak report.jsonl, фильтрует записи:
# entry_type == "attempt"
# status == 2
# detector_results[<detector_name>][0] >= threshold
# Скрипт сам создаст файл рядом с отчётом, примерно такой:
# filtered_<detector_name>_gte_<threshold>.jsonl

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def make_safe_filename_part(value: str) -> str:
    """Преобразует имя detector в безопасную часть имени файла."""
    value = value.strip()
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE)
    value = value.strip("._")

    if not value:
        return "detector"

    return value


def get_detector_score(item: dict[str, Any], detector_name: str) -> float | None:
    """Возвращает первый score detector или None, если score не найден."""
    detector_results = item.get("detector_results")

    if not isinstance(detector_results, dict):
        return None

    detector_values = detector_results.get(detector_name)

    if not isinstance(detector_values, list):
        return None

    if not detector_values:
        return None

    first_score = detector_values[0]

    if not isinstance(first_score, int | float):
        return None

    return float(first_score)


def filter_garak_report(
    report_path: Path,
    detector_name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Фильтрует JSONL-отчёт garak по entry_type, status и score detector."""
    matched_items: list[dict[str, Any]] = []

    with report_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                print(
                    f"Строка {line_number}: ошибка разбора JSON: {error}",
                    file=sys.stderr,
                )
                continue

            if item.get("entry_type") != "attempt":
                continue

            if item.get("status") != 2:
                continue

            score = get_detector_score(item, detector_name)

            if score is None:
                continue

            if score >= threshold:
                matched_items.append(item)

    return matched_items


def save_jsonl(output_path: Path, items: list[dict[str, Any]]) -> None:
    """Сохраняет найденные записи в JSONL-файл."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False))
            file.write("\n")


def build_default_output_path(
    report_path: Path,
    detector_name: str,
    threshold: float,
) -> Path:
    """Создаёт путь выходного файла с именем detector."""
    safe_detector_name = make_safe_filename_part(detector_name)
    threshold_part = str(threshold).replace(".", "_")

    return report_path.parent / f"filtered_{safe_detector_name}_gte_{threshold_part}.jsonl"


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Фильтрация JSONL-отчёта garak по detector_results.",
    )

    parser.add_argument(
        "report_path",
        type=Path,
        help="Путь к report.jsonl файлу garak.",
    )

    parser.add_argument(
        "detector_name",
        help="Имя detector, например: mitigation.MitigationBypass",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Минимальный score detector. По умолчанию: 0.5",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Путь выходного JSONL-файла. Если не указан, имя будет создано по detector.",
    )

    return parser.parse_args()


def main() -> None:
    """Точка входа."""
    args = parse_args()

    report_path: Path = args.report_path
    detector_name: str = args.detector_name
    threshold: float = args.threshold

    if not report_path.exists():
        print(f"Файл не найден: {report_path}", file=sys.stderr)
        sys.exit(1)

    if not report_path.is_file():
        print(f"Указанный путь не является файлом: {report_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output

    if output_path is None:
        output_path = build_default_output_path(
            report_path=report_path,
            detector_name=detector_name,
            threshold=threshold,
        )

    matched_items = filter_garak_report(
        report_path=report_path,
        detector_name=detector_name,
        threshold=threshold,
    )

    save_jsonl(output_path=output_path, items=matched_items)

    print(f"Файл отчёта: {report_path}")
    print(f"Detector: {detector_name}")
    print(f"Порог: {threshold}")
    print(f"Найдено записей: {len(matched_items)}")
    print(f"Результат сохранён: {output_path}")


if __name__ == "__main__":
    main()