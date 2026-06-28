from __future__ import annotations

import hashlib
import re

PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "PHONE_RU": re.compile(
        r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
    ),
    "CARD": re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
    "INN": re.compile(r"\b(?:\d{10}|\d{12})\b"),
    "PASSPORT": re.compile(r"\b\d{2}\s?\d{2}\s?\d{6}\b"),
}


def redact_pii(text: str) -> str:
    """Маскирует известные шаблоны персональных данных в тексте."""

    redacted_text = text
    for name, pattern in PII_PATTERNS.items():
        redacted_text = pattern.sub(f"[{name}]", redacted_text)
    return redacted_text


def prompt_hash(text: str) -> str:
    """Возвращает короткий SHA-256-хеш для безопасной корреляции промптов."""

    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
