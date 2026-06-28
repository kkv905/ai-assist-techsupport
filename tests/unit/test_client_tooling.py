from types import SimpleNamespace

import pytest

from app.client import build_initial_messages, execute_tool_call


def test_build_initial_messages_uses_imported_prompt_loader(mocker) -> None:
    """Проверяет сборку стартовых сообщений с патчем по месту импорта."""

    mocker.patch("app.client.load_prompt", return_value="Системный промпт")

    messages = build_initial_messages("Текст {пользователя}")

    assert messages == [
        {"role": "system", "content": "Системный промпт"},
        {"role": "user", "content": "Текст {{пользователя}}"},
    ]


def test_execute_tool_call_raises_for_unknown_tool() -> None:
    """Проверяет защиту от вызова неизвестного обработчика."""

    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="unknown_tool", arguments='{"value": 1}'),
    )

    with pytest.raises(ValueError, match="Неизвестный tool"):
        execute_tool_call(tool_call)
