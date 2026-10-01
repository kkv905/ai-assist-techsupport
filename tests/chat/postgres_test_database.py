"""Проверка выделенной PostgreSQL-базы для контрактных тестов."""

from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


class UnsafeTestDatabaseUrlError(ValueError):
    """TEST_DATABASE_URL не соответствует безопасной тестовой базе."""


def get_configured_test_database_url(environ: Mapping[str, str]) -> str | None:
    """Возвращает только явно настроенный URL тестовой БД.

    DATABASE_URL намеренно не используется: он принадлежит приложению и может
    указывать на development-базу Docker Compose.
    """

    database_url = environ.get("TEST_DATABASE_URL")
    if not database_url:
        return None

    ensure_safe_test_database_url(database_url)
    return database_url


def ensure_safe_test_database_url(database_url: str) -> None:
    """Проверяет URL до любой изменяющей схему или данные операции."""

    try:
        url = make_url(database_url)
    except ArgumentError:
        raise UnsafeTestDatabaseUrlError(_safe_error_message()) from None

    database_name = url.database.casefold() if url.database else ""
    is_postgres = url.drivername.startswith("postgresql")
    is_test_database = "_test" in database_name or database_name.endswith("test")
    if not is_postgres or not is_test_database:
        raise UnsafeTestDatabaseUrlError(_safe_error_message())


def _safe_error_message() -> str:
    return (
        "TEST_DATABASE_URL должен указывать на PostgreSQL-базу с именем, "
        "содержащим '_test' или оканчивающимся на 'test'."
    )
