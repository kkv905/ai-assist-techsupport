from __future__ import annotations

from app.schemas.chat import ChatMessage


def escape_curly_braces(text: str) -> str:
    """Экранирует фигурные скобки в пользовательском тексте для безопасной подстановки."""

    return text.replace("{", "{{").replace("}", "}}")


def build_prompt_messages(
    *,
    system_prompt: str,
    user_message: str,
    history: list[ChatMessage] | None = None,
) -> list[dict[str, str]]:
    """Собирает сообщения в порядке system -> history -> user."""

    messages: list[dict[str, str]] = []
    normalized_system_prompt = system_prompt.strip()
    if normalized_system_prompt:
        messages.append({"role": "system", "content": normalized_system_prompt})

    for message in history or []:
        messages.append(
            {
                "role": message.role,
                "content": escape_curly_braces(message.content),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": escape_curly_braces(user_message),
        }
    )
    return messages
