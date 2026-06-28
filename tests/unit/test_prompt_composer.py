from app.prompts.composer import build_prompt_messages, escape_curly_braces
from app.schemas.chat import ChatMessage


def test_escape_curly_braces_protects_user_text() -> None:
    """Проверяет экранирование фигурных скобок в пользовательском тексте."""

    assert escape_curly_braces("Ошибка {code} в {service}") == "Ошибка {{code}} в {{service}}"


def test_build_prompt_messages_keeps_role_order_and_escapes_history() -> None:
    """Проверяет порядок ролей и безопасную подстановку истории."""

    history = [ChatMessage(role="assistant", content="Проверьте поле {login}")]

    messages = build_prompt_messages(
        system_prompt="Отвечай кратко.",
        user_message="У меня {error}",
        history=history,
    )

    assert [message["role"] for message in messages] == ["system", "assistant", "user"]
    assert messages[1]["content"] == "Проверьте поле {{login}}"
    assert messages[2]["content"] == "У меня {{error}}"
