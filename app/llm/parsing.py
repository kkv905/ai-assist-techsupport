from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(slots=True)
class ParsedToolCall:
    """Хранит безопасно распарсенный tool call."""

    id: str
    name: str
    arguments: dict[str, Any]


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Извлекает JSON-объект из текста или markdown-fence."""

    normalized_text = _strip_markdown_fence(raw_text.strip())
    try:
        payload = json.loads(normalized_text)
    except json.JSONDecodeError as error:
        raise ValueError("Не удалось распарсить JSON-ответ модели.") from error

    if not isinstance(payload, dict):
        raise ValueError("Ожидался JSON-объект верхнего уровня.")
    return payload


def parse_tool_calls(tool_calls: Sequence[Any] | None) -> list[ParsedToolCall]:
    """Преобразует tool calls SDK в удобную доменную структуру."""

    parsed_calls: list[ParsedToolCall] = []
    for tool_call in tool_calls or []:
        function = getattr(tool_call, "function", None)
        if function is None:
            raise ValueError("Tool call не содержит описания функции.")

        parsed_calls.append(
            ParsedToolCall(
                id=str(getattr(tool_call, "id", "")),
                name=str(getattr(function, "name", "")),
                arguments=extract_json_object(str(getattr(function, "arguments", "{}"))),
            )
        )
    return parsed_calls


def _strip_markdown_fence(raw_text: str) -> str:
    """Удаляет markdown-обертку вокруг JSON, если модель вернула fenced-блок."""

    if not raw_text.startswith("```"):
        return raw_text

    lines = raw_text.splitlines()
    if not lines:
        return raw_text

    first_line = lines[0].strip().lower()
    if first_line not in {"```", "```json"}:
        return raw_text

    content_lines = lines[1:]
    if content_lines and content_lines[-1].strip() == "```":
        content_lines = content_lines[:-1]
    return "\n".join(content_lines).strip()
