from types import SimpleNamespace

import pytest

from app.llm.parsing import extract_json_object, parse_tool_calls


def test_extract_json_object_supports_markdown_fence() -> None:
    """Проверяет разбор JSON, завернутого в markdown-блок."""

    payload = extract_json_object('```json\n{"answer": "ok"}\n```')

    assert payload == {"answer": "ok"}


def test_extract_json_object_supports_plain_json() -> None:
    """Проверяет разбор обычного JSON без markdown-обертки."""

    payload = extract_json_object('{"answer": "ok", "score": 5}')

    assert payload["score"] == 5


def test_extract_json_object_raises_value_error_for_malformed_json() -> None:
    """Проверяет доменную ошибку при битом JSON-ответе."""

    with pytest.raises(ValueError, match="Не удалось распарсить JSON-ответ модели"):
        extract_json_object('{"answer": }')


def test_parse_tool_calls_reads_arguments_from_sdk_payload() -> None:
    """Проверяет разбор tool_calls из структуры OpenAI SDK."""

    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="search_knowledge_base",
            arguments='{"query": "Ошибка входа", "ips": "АИС Правоохрана"}',
        ),
    )

    parsed = parse_tool_calls([tool_call])

    assert parsed[0].id == "call_1"
    assert parsed[0].name == "search_knowledge_base"
    assert parsed[0].arguments == {"query": "Ошибка входа", "ips": "АИС Правоохрана"}
