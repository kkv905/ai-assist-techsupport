"""Inline-клавиатуры Telegram-бота."""

from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

TOPIC_TITLES: dict[str, str] = {
    "billing": "Биллинг",
    "setup": "Установка",
    "api_errors": "Ошибки API",
    "integrations": "Интеграции",
}


def topics_kb() -> InlineKeyboardMarkup:
    """Собирает клавиатуру выбора темы для сценария `/ask`."""

    builder = InlineKeyboardBuilder()
    for slug, title in TOPIC_TITLES.items():
        builder.button(text=title, callback_data=f"topic:{slug}")

    builder.button(text="Отмена", callback_data="topic:cancel")
    builder.adjust(2, 2, 1)
    return builder.as_markup()


def get_topic_title(slug: str) -> str:
    """Возвращает человекочитаемое название темы по ее идентификатору."""

    return TOPIC_TITLES.get(slug, slug)
