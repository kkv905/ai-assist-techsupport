"""Сервис модерации сообщений."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog
import yaml

from app.moderation.domain import ModerationResult

logger = structlog.get_logger("app.moderation")


class ModerationService:
    """Проверяет входной и выходной контент локальными и внешними правилами."""

    def __init__(
        self,
        *,
        keywords_path: Path,
        openai_client: Any | None = None,
        openai_enabled: bool = False,
        category_thresholds: dict[str, float] | None = None,
    ) -> None:
        """Загружает keyword-правила и сохраняет параметры внешней модерации."""

        self._rules = self._load_rules(keywords_path)
        self._openai_client = openai_client
        self._openai_enabled = openai_enabled
        self._category_thresholds = category_thresholds or {}

    async def check_input(self, content: str) -> ModerationResult:
        """Проверяет пользовательский ввод до вызова LLM."""

        keyword_result = self._check_keyword_layer(content, direction="input")
        if not keyword_result.allowed:
            return keyword_result
        return await self._check_openai_layer(content)

    async def check_output(self, content: str) -> ModerationResult:
        """Проверяет ответ модели перед отправкой пользователю."""

        keyword_result = self._check_keyword_layer(content, direction="output")
        if not keyword_result.allowed:
            return keyword_result
        return await self._check_openai_layer(content)

    def _check_keyword_layer(self, content: str, *, direction: str) -> ModerationResult:
        """Прогоняет текст через локальные regex-правила из YAML."""

        categories: list[str] = []
        reasons: list[str] = []
        blocked_by = "none"
        for rule in self._rules.get(direction, []):
            for pattern in rule["patterns"]:
                if re.search(pattern, content, flags=re.IGNORECASE):
                    categories.append(rule["category"])
                    reasons.append(f"Сработало правило: {pattern}")
                    blocked_by = rule.get("blocked_by", "keyword")
                    break

        if categories:
            return ModerationResult(allowed=False, categories=categories, reasons=reasons, blocked_by=blocked_by)
        return ModerationResult(allowed=True)

    async def _check_openai_layer(self, content: str) -> ModerationResult:
        """Проверяет текст через OpenAI Moderation API, если слой включен."""

        if not self._openai_enabled or self._openai_client is None or not content.strip():
            return ModerationResult(allowed=True)

        response = await self._openai_client.moderations.create(
            model="omni-moderation-latest",
            input=content,
        )
        result = response.results[0]
        category_scores = self._to_mapping(getattr(result, "category_scores", {}))
        categories = [
            name
            for name, score in category_scores.items()
            if score >= self._category_thresholds.get(name, 0.5)
        ]
        flagged = bool(getattr(result, "flagged", False) or categories)
        if not flagged:
            return ModerationResult(allowed=True)

        if not categories:
            categories = [name for name, value in self._to_mapping(getattr(result, "categories", {})).items() if value]
        reasons = [f"Категория {name} превысила порог" for name in categories] or ["OpenAI Moderation API пометил ответ"]
        return ModerationResult(allowed=False, categories=categories, reasons=reasons, blocked_by="openai")

    def _load_rules(self, keywords_path: Path) -> dict[str, list[dict[str, Any]]]:
        """Читает YAML-файл с локальными правилами модерации."""

        payload = yaml.safe_load(keywords_path.read_text(encoding="utf-8")) or {}
        return {
            "input": payload.get("input", []),
            "output": payload.get("output", []),
        }

    def _to_mapping(self, value: Any) -> dict[str, Any]:
        """Нормализует объекты SDK в обычный словарь для дальнейшей обработки."""

        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "__dict__"):
            return {key: item for key, item in vars(value).items() if not key.startswith("_")}
        return {}
