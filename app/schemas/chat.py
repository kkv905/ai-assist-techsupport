from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatMessage(BaseModel):
    """Описывает одно сообщение в истории диалога."""

    role: Literal["system", "user", "assistant"] = Field(description="Роль автора сообщения.")
    content: str = Field(min_length=1, description="Текст сообщения.")


class Usage(BaseModel):
    """Содержит информацию о расходе токенов."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ChatRequest(BaseModel):
    """Описывает входные параметры запроса к модели."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "messages": [{"role": "user", "content": "Привет! Кратко представься."}],
                    "temperature": 0.2,
                    "max_tokens": 300,
                },
                {
                    "messages": [
                        {"role": "system", "content": "Отвечай кратко и по делу."},
                        {"role": "user", "content": "Объясни, что такое DI в FastAPI."},
                    ],
                    "model": "gpt-4o-mini",
                    "temperature": 0.4,
                    "max_tokens": 500,
                    "user_id": "student-42",
                    "session_id": "session-7",
                },
            ]
        }
    )

    messages: list[ChatMessage] = Field(min_length=1, description="История диалога.")
    model: str | None = Field(default=None, description="Имя модели OpenAI.")
    temperature: float = Field(default=0.0, ge=0, le=2, description="Температура генерации.")
    max_tokens: int = Field(default=512, ge=1, le=16000, description="Лимит токенов ответа.")
    user_id: str | None = Field(default=None, description="Идентификатор пользователя.")
    session_id: str | None = Field(default=None, description="Идентификатор сессии.")


class ChatResponse(BaseModel):
    """Описывает итоговый ответ сервиса на запрос к модели."""

    content: str = Field(description="Текст ответа модели.")
    model: str = Field(description="Модель, которая обработала запрос.")
    usage: Usage = Field(description="Информация о расходе токенов.")
    finish_reason: str | None = Field(default=None, description="Причина завершения генерации.")
    cached: bool = Field(default=False, description="Признак ответа из кеша.")

    @classmethod
    def from_openai(cls, response: Any) -> ChatResponse:
        """Преобразует raw-ответ OpenAI SDK в доменную схему."""

        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)
        usage_data = getattr(response, "usage", None)
        usage = Usage(
            prompt_tokens=getattr(usage_data, "prompt_tokens", None),
            completion_tokens=getattr(usage_data, "completion_tokens", None),
            total_tokens=getattr(usage_data, "total_tokens", None),
        )
        return cls(
            content=getattr(message, "content", "") or "",
            model=getattr(response, "model", "") or "",
            usage=usage,
            finish_reason=getattr(choice, "finish_reason", None),
            cached=False,
        )


class ChatDelta(BaseModel):
    """Описывает один кадр streaming-ответа."""

    content: str | None = Field(default=None, description="Часть текстового ответа.")
    usage: Usage | None = Field(default=None, description="Финальная статистика токенов.")

    @model_validator(mode="after")
    def validate_payload(self) -> ChatDelta:
        """Проверяет, что кадр содержит либо текст, либо usage."""

        if self.content is None and self.usage is None:
            raise ValueError("Кадр стрима должен содержать текст или usage.")
        return self
