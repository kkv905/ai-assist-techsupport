"""Модели результатов модерации."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModerationResult(BaseModel):
    """Описывает итог проверки контента политиками модерации."""

    allowed: bool
    categories: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    blocked_by: str = "none"
