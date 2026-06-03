import json
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"

def load_knowledge_base() -> list[dict[str, Any]]:
    """ Читаем внешний JSON-файл базы знаний """
    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(f"Файл базы знаний не найден: {KNOWLEDGE_BASE_PATH}")

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Файл базы знаний должен содержать JSON-массив")

    return data

def normalize_text(value: str) -> str:
    """ Приводим текст к единому виду """
    value = value.lower()
    value = value.replace("ё", "е")
    value = re.sub(r"[^a-zа-я0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def calculate_score(record: dict[str, Any], query: str, ips: str) -> int:
    """
    Считает грубый рейтинг записи:
    +10 если совпало подразделение;
    +3 если подразделение пустое и запись общая;
    +1 за каждое совпавшее слово;
    +5 за точное совпадение симптома;
    """
    normalized_query = normalize_text(query)
    query_words = set(normalized_query.split())

    searchable_parts = [
        str(record.get("title", "")),
        str(record.get("problem", "")),
        str(record.get("solution", "")),
        " ".join(record.get("symptoms", [])),
        " ".join(record.get("diagnostics", [])),
    ]

    searchable_text = normalize_text(" ".join(searchable_parts))
    searchable_words = set(searchable_text.split())

    score = 0

    # Совпадение по подразделениям дает сильный вес
    record_ips = str(record.get("ips", ""))
    if ips and record_ips == ips:
        score += 10

    # Если подразделение не указано, общая запись тоже получает небольшой вес.
    if not ips and not record_ips:
        score += 3

    # Совпадение слов из запроса
    score += len(query_words.intersection(searchable_words))

    # Совпадение фраз из symptoms
    for symptom in record.get("symptoms", []):
        normalized_symptom = normalize_text(str(symptom))
        if normalized_symptom and normalized_symptom in normalized_query:
            score += 5

    return score

def search_knowledge_base(query: str, ips: str) -> dict[str, Any]:
    if not isinstance(query, str) or len(query.strip()) < 3:
        raise ValueError("query должен быть строкой не менее 3 символов")

    if not isinstance(ips, str):
        raise ValueError("ips должен быть строкой")

    query = query.strip()
    ips = ips.strip()

    knowledge_base = load_knowledge_base()

    scored_records = []

    for record in knowledge_base:
        score = calculate_score(record, query, ips)

        if score > 0:
            scored_records.append({
                "score": score,
                "record": record,
            })

    scored_records.sort(key=lambda item: item["score"], reverse=True)

    top_records = scored_records[:3]
    if not top_records:
        return {
            "success": True,
            "query": query,
            "ips": ips,
            "found": False,
            "results": [],
            "message": "В базе знаний не найдено подходящее решение.",
        }

    results = []

    for item in top_records:
        record = item["record"]

        results.append({
            "id": record.get("id"),
            "ips": record.get("ips"),
            "title": record.get("title"),
            "problem": record.get("problem"),
            "solution": record.get("solution"),
            "diagnostics": record.get("diagnostics"),
            "score": item["score"],
        })

    return {
        "success": True,
        "query": query,
        "ips": ips,
        "found": True,
        "results": results,
    }

TOOL_HANDLERS = {
    "search_knowledge_base": search_knowledge_base,
}