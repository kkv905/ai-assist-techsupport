import json
import os
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI

from app.llm.parsing import parse_tool_calls
from app.logging_config import setup_logger, write_log_event
from app.prompts.composer import build_prompt_messages
from app.prompts.loader import load_prompt
from app.tools.handlers import TOOL_HANDLERS
from app.tools.schemas import TOOLS


def load_settings() -> dict[str, str]:
    """Загружает настройки приложения из переменных окружения."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if not api_key:
        raise RuntimeError("Не задана переменная окружения OPENAI_API_KEY")

    return {"api_key": api_key, "model": model}


def build_client(api_key: str) -> OpenAI:
    """Создает клиента OpenAI SDK."""
    return OpenAI(api_key=api_key)


def build_initial_messages(user_text: str) -> list[dict[str, Any]]:
    """Формирует стартовую историю сообщений для модели."""
    system_prompt = load_prompt("system_v1.j2")
    return build_prompt_messages(system_prompt=system_prompt, user_message=user_text)


def execute_tool_call(tool_call: Any) -> dict[str, Any]:
    """Выполняет один tool_call и возвращает результат функции."""

    parsed_tool_call = parse_tool_calls([tool_call])[0]
    if parsed_tool_call.name not in TOOL_HANDLERS:
        raise ValueError(f"Неизвестный tool: {parsed_tool_call.name}")

    handler = TOOL_HANDLERS[parsed_tool_call.name]
    result = handler(**parsed_tool_call.arguments)

    return {
        "tool_call_id": parsed_tool_call.id,
        "tool_name": parsed_tool_call.name,
        "arguments": parsed_tool_call.arguments,
        "result": result,
    }


def append_tool_results(
    messages: list[dict[str, Any]],
    assistant_message: Any,
    executed_tools: list[dict[str, Any]],
) -> None:
    """Добавляет в историю сообщение ассистента с tool_calls и результаты tools."""
    assistant_payload: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [],
    }

    if assistant_message.content:
        assistant_payload["content"] = assistant_message.content

    for tool_call in assistant_message.tool_calls or []:
        assistant_payload["tool_calls"].append(
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )

    messages.append(assistant_payload)

    for executed_tool in executed_tools:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": executed_tool["tool_call_id"],
                "name": executed_tool["tool_name"],
                "content": json.dumps(
                    executed_tool["result"],
                    ensure_ascii=False,
                ),
            }
        )


def run_chat(user_text: str) -> dict[str, Any]:
    """Запускает полный цикл общения с моделью и обработки tool_calls."""
    settings = load_settings()
    client = build_client(settings["api_key"])
    model = settings["model"]

    logger = setup_logger()
    run_id = str(uuid4())

    write_log_event(
        logger=logger,
        event="user_input",
        run_id=run_id,
        user_text=user_text,
        model=model,
    )

    messages = build_initial_messages(user_text)

    first_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    first_choice = first_response.choices[0]
    assistant_message = first_choice.message
    tool_calls = assistant_message.tool_calls or []

    if not tool_calls:
        final_text = assistant_message.content or ""
        usage_total_tokens = (
            first_response.usage.total_tokens
            if first_response.usage
            else None
        )

        write_log_event(
            logger=logger,
            event="final_answer",
            run_id=run_id,
            used_tool=False,
            final_answer=final_text,
            usage_total_tokens=usage_total_tokens,
        )

        return {
            "used_tool": False,
            "tool_calls": [],
            "final_answer": final_text,
            "usage_total_tokens": usage_total_tokens,
        }

    executed_tools = []

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        raw_arguments = tool_call.function.arguments

        try:
            parsed_arguments = parse_tool_calls([tool_call])[0].arguments
        except ValueError:
            parsed_arguments = {
                "raw_arguments": raw_arguments,
                "parse_error": "Некорректный JSON в аргументах tool",
            }

        write_log_event(
            logger=logger,
            event="tool_selected",
            run_id=run_id,
            tool_name=tool_name,
            arguments=parsed_arguments,
        )

        executed_tool = execute_tool_call(tool_call)
        executed_tools.append(executed_tool)

        write_log_event(
            logger=logger,
            event="tool_result",
            run_id=run_id,
            tool_name=executed_tool["tool_name"],
            arguments=executed_tool["arguments"],
            result=executed_tool["result"],
        )

    append_tool_results(
        messages=messages,
        assistant_message=assistant_message,
        executed_tools=executed_tools,
    )

    second_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    second_choice = second_response.choices[0]
    final_text = second_choice.message.content or ""

    first_tokens = first_response.usage.total_tokens if first_response.usage else 0
    second_tokens = second_response.usage.total_tokens if second_response.usage else 0
    usage_total_tokens = first_tokens + second_tokens

    write_log_event(
        logger=logger,
        event="final_answer",
        run_id=run_id,
        used_tool=True,
        final_answer=final_text,
        usage_total_tokens=usage_total_tokens,
    )

    return {
        "used_tool": True,
        "tool_calls": [
            {
                "name": item["tool_name"],
                "arguments": item["arguments"],
                "result": item["result"],
            }
            for item in executed_tools
        ],
        "final_answer": final_text,
        "usage_total_tokens": usage_total_tokens,
    }


def main() -> None:
    """Запускает ручную проверку полного цикла function calling."""
    user_text = input("Введите запрос пользователя: ").strip()

    if not user_text:
        print("Запрос пользователя пустой. Завершение работы.")
        return

    result = run_chat(user_text)
    print("\n=== Итог выполнения ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
