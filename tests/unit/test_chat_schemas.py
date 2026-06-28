import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatMessage, ChatRequest, MAX_MESSAGE_LENGTH


def test_chat_message_rejects_empty_content() -> None:
    """Проверяет запрет пустого текста сообщения."""

    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")


def test_chat_message_rejects_too_long_content() -> None:
    """Проверяет ограничение длины пользовательского сообщения."""

    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="x" * (MAX_MESSAGE_LENGTH + 1))


def test_chat_request_repr_masks_pii() -> None:
    """Проверяет маскирование PII в repr моделей Pydantic."""

    request = ChatRequest(
        messages=[
            {
                "role": "user",
                "content": "Мой email ivan@mail.ru и карта 4111 1111 1111 1111",
            }
        ]
    )

    representation = repr(request)

    assert "ivan@mail.ru" not in representation
    assert "4111 1111 1111 1111" not in representation
    assert "[EMAIL]" in representation
    assert "[CARD]" in representation
