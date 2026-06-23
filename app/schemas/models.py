from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Описывает модель OpenAI и ориентировочную стоимость."""

    id: str = Field(description="Идентификатор модели.")
    title: str = Field(description="Понятное название модели.")
    input_price_per_1m: float = Field(description="Цена за 1 млн входных токенов в USD.")
    output_price_per_1m: float = Field(description="Цена за 1 млн выходных токенов в USD.")


AVAILABLE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gpt-4o-mini",
        title="GPT-4o mini",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
    ),
    ModelInfo(
        id="gpt-4.1-mini",
        title="GPT-4.1 mini",
        input_price_per_1m=0.40,
        output_price_per_1m=1.60,
    ),
    ModelInfo(
        id="gpt-4.1",
        title="GPT-4.1",
        input_price_per_1m=2.00,
        output_price_per_1m=8.00,
    ),
]
