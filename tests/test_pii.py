from app.observability.pii import prompt_hash, redact_pii


def test_redact_pii_masks_sensitive_fragments() -> None:
    """Проверяет, что персональные данные заменяются плейсхолдерами."""

    source = "Мой email ivan@mail.ru, тел +7 (999) 123-45-67, карта 4111 1111 1111 1111"
    preview = redact_pii(source)[:120]

    assert "ivan@mail.ru" not in preview
    assert "+7 (999) 123-45-67" not in preview
    assert "4111 1111 1111 1111" not in preview
    assert "[EMAIL]" in preview
    assert "[PHONE_RU]" in preview
    assert "[CARD]" in preview


def test_prompt_hash_is_stable_and_safe() -> None:
    """Проверяет стабильность короткого хеша для корреляции промптов."""

    source = "Тестовый промпт"

    assert prompt_hash(source) == prompt_hash(source)
    assert prompt_hash(source).startswith("sha256:")
    assert len(prompt_hash(source)) == 23
