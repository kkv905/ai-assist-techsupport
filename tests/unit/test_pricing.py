from app.schemas.chat import Usage
from app.services.pricing import calculate_request_cost


def test_calculate_request_cost_uses_usage_fields() -> None:
    """Проверяет расчет стоимости по входным и выходным токенам."""

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    cost = calculate_request_cost("gpt-4o-mini", usage)

    assert cost == 0.00045
