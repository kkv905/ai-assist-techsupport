"""Регрессия безопасного выбора PostgreSQL-базы для тестов."""

from __future__ import annotations

import pytest

from postgres_test_database import (
    UnsafeTestDatabaseUrlError,
    ensure_safe_test_database_url,
    get_configured_test_database_url,
)


def test_missing_test_database_url_does_not_fall_back_to_application_database() -> None:
    environment = {"DATABASE_URL": "postgresql+asyncpg://chat:secret@localhost:5432/chat"}

    assert get_configured_test_database_url(environment) is None


def test_chat_test_database_url_is_accepted() -> None:
    database_url = "postgresql+asyncpg://chat:secret@localhost:5432/chat_test"

    assert get_configured_test_database_url({"TEST_DATABASE_URL": database_url}) == database_url


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql+asyncpg://chat:secret@localhost:5432/chat",
        "postgresql+asyncpg://chat:secret@localhost:5432/support",
    ],
)
def test_non_test_database_url_is_rejected_before_destructive_operations(database_url: str) -> None:
    with pytest.raises(UnsafeTestDatabaseUrlError):
        ensure_safe_test_database_url(database_url)


def test_unsafe_url_error_does_not_expose_password() -> None:
    password = "very-secret-password"
    database_url = f"postgresql+asyncpg://chat:{password}@localhost:5432/chat"

    with pytest.raises(UnsafeTestDatabaseUrlError) as error:
        ensure_safe_test_database_url(database_url)

    assert password not in str(error.value)
