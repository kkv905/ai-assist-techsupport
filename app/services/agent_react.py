"""Bounded native-tool-calling ReAct agent with lightweight self-reflection."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from typing import Any

import structlog
from dotenv import load_dotenv
from openai import OpenAI

from app.services.agent_naive import (
    get_current_time,
    search_knowledge_base,
    send_telegram_message,
)

log = structlog.get_logger(__name__)
MODEL_MAIN = "gpt-5.4-mini"
MODEL_PREMIUM = "gpt-5.4"
MODEL_CRITIC = "gpt-5.4-mini"

# Strict schemas keep the model from inventing undeclared arguments.  Each
# description says what the tool does, when it is appropriate, its inputs and
# what it returns (including its safety boundary).
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Находит один наиболее релевантный фрагмент базы знаний техподдержки. "
                "Используй, когда ответ должен быть подтверждён документацией или прежним "
                "обращением. Принимает query — точный поисковый запрос. Возвращает текст "
                "фрагмента либо сообщение, что надёжных данных нет; не генерирует ответ сам."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Поисковый запрос."}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Возвращает текущее локальное время в ISO 8601. Используй только когда нужен "
                "фактический текущий момент, а не справочная информация. Принимает timezone — "
                "имя IANA часового пояса. Возвращает ISO-время или ошибку невалидного пояса."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "description": "IANA timezone."}},
                "required": ["timezone"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_telegram_message",
            "description": (
                "Имитирует отправку текста в Telegram и не выполняет сетевой запрос. Используй "
                "только по явной просьбе пользователя отправить сообщение. Принимает chat_id и "
                "text. Возвращает подтверждение имитации; реальное сообщение никогда не отправляет."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string", "description": "Идентификатор чата."},
                    "text": {"type": "string", "description": "Текст сообщения."},
                },
                "required": ["chat_id", "text"],
                "additionalProperties": False,
            },
        },
    },
]

DISPATCH: dict[str, Callable[..., Any]] = {
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time,
    "send_telegram_message": send_telegram_message,
}

SYSTEM_PROMPT = (
    "Ты ReAct-ассистент технической поддержки. Перед действием кратко, одним "
    "предложением объясняй, что и зачем проверяешь. За шаг вызывай ровно один "
    "инструмент. Опирайся только на observation; не выдумывай данные при пустом "
    "результате. Когда данных достаточно, дай финальный ответ без инструментов. "
    "Если доступные инструменты не способны решить задачу, честно сообщи об этом."
)
CRITIC_PROMPT = (
    "Ты краткий критик плана ReAct-агента. Проверь, достаточно ли observation для "
    "цели пользователя и не содержит ли план выдумок или лишних действий. Ответь "
    "строго `OK` либо `REVISE: <короткая причина>`."
)


def _empty_usage() -> dict[str, int]:
    return {"prompt": 0, "completion": 0, "total": 0}


def _add_usage(total: dict[str, int], response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    reported_total = getattr(usage, "total_tokens", None)
    total["prompt"] += prompt
    total["completion"] += completion
    total["total"] += int(reported_total if reported_total is not None else prompt + completion)
    return {"prompt": prompt, "completion": completion, "total": int(reported_total or prompt + completion)}


def _result(answer: str, usage: dict[str, int], steps: int, trace: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"answer": answer, "usage": usage, "steps": steps, "trace": trace, **extra}


def run_react_with_reflection(
    question: str,
    tools: list[dict[str, Any]] | None = None,
    tool_dispatch: dict[str, Callable[..., Any]] | None = None,
    max_iterations: int = 10,
    timeout_per_iteration_sec: float = 10.0,
    max_revisions: int = 2,
    model_main: str = MODEL_MAIN,
    model_critic: str = MODEL_CRITIC,
    client: Any | None = None,
) -> dict[str, Any]:
    """Answer *question* with bounded tool use and at most two critic revisions."""
    if not 8 <= max_iterations <= 20:
        raise ValueError("max_iterations must be between 8 and 20")
    if not 5 <= timeout_per_iteration_sec <= 15:
        raise ValueError("timeout_per_iteration_sec must be between 5 and 15")
    if not 0 <= max_revisions <= 2:
        raise ValueError("max_revisions must be between 0 and 2")

    active_tools = tools if tools is not None else TOOLS
    dispatch = tool_dispatch if tool_dispatch is not None else DISPATCH
    usage_total = _empty_usage()
    trace: list[dict[str, Any]] = []
    messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    revisions_used = 0
    premium_next_step = False

    try:
        if client is None:
            load_dotenv()
            api_key = os.environ.get("LLM__OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return _result("", usage_total, 0, trace, error="Не задан ключ OpenAI API.")
            client = OpenAI(api_key=api_key)

        for step in range(1, max_iterations + 1):
            started = time.monotonic()
            selected_model = MODEL_PREMIUM if premium_next_step else model_main
            premium_next_step = False
            response = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                tools=active_tools,
                tool_choice="auto",
                timeout=timeout_per_iteration_sec,
            )
            main_usage = _add_usage(usage_total, response)
            if time.monotonic() - started > timeout_per_iteration_sec:
                return _result("Timeout", usage_total, step, trace)

            message = response.choices[0].message
            messages.append(message)
            calls = list(message.tool_calls or [])
            if not calls:
                latency = time.monotonic() - started
                record = {
                    "step": step, "tool_name": None, "tool_args": None, "observation": None,
                    "latency_sec": round(latency, 4), "usage": main_usage, "critic_usage": None,
                    "critic_verdict": None, "model": selected_model,
                }
                trace.append(record)
                log.info("react.step", **record)
                return _result(message.content or "", usage_total, step, trace, revisions_used=revisions_used)

            call = calls[0]
            name, raw_args = call.function.name, call.function.arguments
            try:
                args = json.loads(raw_args)
                if not isinstance(args, dict):
                    raise ValueError("аргументы должны быть JSON object")
                observation = str(dispatch[name](**args)) if name in dispatch else f"Неизвестный tool: {name}"
            except Exception as error:
                args = raw_args
                observation = f"Ошибка tool {name}: {error}"

            # A model that proposes multiple calls does not get to perform them:
            # one action is executed and all remaining calls receive a rejection.
            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
            for extra_call in calls[1:]:
                messages.append({
                    "role": "tool",
                    "tool_call_id": extra_call.id,
                    "content": "Отклонено: за одну итерацию разрешён ровно один инструмент.",
                })

            elapsed = time.monotonic() - started
            if elapsed > timeout_per_iteration_sec:
                return _result("Timeout", usage_total, step, trace)

            critic_usage: dict[str, int] | None = None
            verdict = "OK"
            try:
                critic_response = client.chat.completions.create(
                    model=model_critic,
                    messages=[
                        {"role": "system", "content": CRITIC_PROMPT},
                        {"role": "user", "content": f"Цель: {question}\nObservation: {observation}"},
                    ],
                    timeout=max(0.01, timeout_per_iteration_sec - elapsed),
                )
                critic_usage = _add_usage(usage_total, critic_response)
                verdict = (critic_response.choices[0].message.content or "OK").strip()
            except Exception as error:
                verdict = f"CRITIC_ERROR: {error}"

            latency = time.monotonic() - started
            record = {
                "step": step, "tool_name": name, "tool_args": args, "observation": observation[:500],
                "latency_sec": round(latency, 4), "usage": main_usage, "critic_usage": critic_usage,
                "critic_verdict": verdict, "model": selected_model,
            }
            trace.append(record)
            log.info("react.step", **record)
            if latency > timeout_per_iteration_sec:
                return _result("Timeout", usage_total, step, trace)

            if verdict.upper().startswith("REVISE:") and revisions_used < max_revisions:
                revisions_used += 1
                premium_next_step = True
                messages.append({"role": "system", "content": f"Критика предыдущего плана: {verdict}"})

        return _result("Превышен лимит итераций", usage_total, max_iterations, trace, revisions_used=revisions_used)
    except Exception as error:
        return _result("", usage_total, len(trace), trace, error=str(error), revisions_used=revisions_used)
    finally:
        # This runs for final answers, explicit limit exits, and API failures.
        log.info("react.usage", usage=usage_total, revisions_used=revisions_used)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    result = run_react_with_reflection(
        args.question, max_iterations=args.max_iterations, timeout_per_iteration_sec=args.timeout
    )
    print(result.get("answer") or result.get("error"))
    print(json.dumps({key: result[key] for key in ("usage", "steps", "revisions_used") if key in result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
