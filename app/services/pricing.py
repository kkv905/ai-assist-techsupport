from __future__ import annotations

from dataclasses import dataclass

from app.schemas.chat import Usage


@dataclass(frozen=True, slots=True)
class PricingPlan:
    """Описывает стоимость токенов для конкретной модели."""

    input_per_1k_tokens: float
    output_per_1k_tokens: float


MODEL_PRICING: dict[str, PricingPlan] = {
    "gpt-4o-mini": PricingPlan(input_per_1k_tokens=0.00015, output_per_1k_tokens=0.0006),
    "gpt-5-mini": PricingPlan(input_per_1k_tokens=0.00025, output_per_1k_tokens=0.002),
    "gpt-5.2": PricingPlan(input_per_1k_tokens=0.002, output_per_1k_tokens=0.008),
}


def calculate_request_cost(model: str, usage: Usage) -> float:
    """Вычисляет примерную стоимость запроса по расходу токенов."""

    pricing_plan = MODEL_PRICING.get(model)
    if pricing_plan is None:
        raise ValueError(f"Неизвестна стоимость модели: {model}")

    prompt_tokens = usage.prompt_tokens or 0
    completion_tokens = usage.completion_tokens or 0
    cost = (
        prompt_tokens / 1000 * pricing_plan.input_per_1k_tokens
        + completion_tokens / 1000 * pricing_plan.output_per_1k_tokens
    )
    return round(cost, 8)
