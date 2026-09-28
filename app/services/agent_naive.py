"""Minimal Chat Completions agent loop used to demonstrate tool calling."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)
MODEL = "gpt-5.4-mini"
TOOLS = [
    {"type": "function", "function": {"name": "search_knowledge_base", "description": "Ищет самый релевантный фрагмент в корпоративной базе знаний технической поддержки. Вызывай его, когда ответ должен опираться на документацию или известные обращения.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "Возвращает текущее время в указанном часовом поясе в формате ISO 8601. Вызывай его, когда для ответа или планирования нужно реальное локальное время.", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Europe/Moscow"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "send_telegram_message", "description": "Имитирует отправку текста в Telegram-чат и не выполняет сетевой запрос. Вызывай его только когда пользователь явно попросил отправить сообщение и известны chat_id и текст.", "parameters": {"type": "object", "properties": {"chat_id": {"type": "string"}, "text": {"type": "string"}}, "required": ["chat_id", "text"], "additionalProperties": False}}},
]


def search_knowledge_base(query: str) -> str:
    """Return the first retrieved RAG fragment, without generating a new answer."""
    from app.core.config import get_settings
    from app.services.rag import NOT_FOUND_ANSWER, RAGService

    service = RAGService.from_settings(get_settings())
    try:
        service.build()
        _, nodes = service._retrieve(query, [])
        if not nodes or float(nodes[0].score or 0.0) < service.score_threshold:
            return NOT_FOUND_ANSWER
        return service._node_text(nodes[0])
    finally:
        service.close()


def get_current_time(timezone: str = "Europe/Moscow") -> str:
    return datetime.now(ZoneInfo(timezone)).isoformat()


def send_telegram_message(chat_id: str, text: str) -> str:
    print(f"[TELEGRAM → {chat_id}] {text}")
    return f"Сообщение отправлено в {chat_id}"


DISPATCH = {"search_knowledge_base": search_knowledge_base, "get_current_time": get_current_time, "send_telegram_message": send_telegram_message}


def run_agent(task: str, max_steps: int = 6) -> dict[str, Any]:
    """Run the bounded LLM/tool loop and return its answer and audit trace."""
    messages: list[Any] = [{"role": "user", "content": task}]
    trace: list[dict[str, Any]] = []
    try:
        load_dotenv()
        api_key = os.environ.get("LLM__OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"answer": "", "steps": 0, "trace": trace, "error": "Не задан ключ OpenAI API."}
        client = OpenAI(api_key=api_key)
        for step in range(1, max_steps + 1):
            started = time.perf_counter()
            response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            message = response.choices[0].message
            messages.append(message)
            usage = response.usage
            calls = message.tool_calls or []
            if not calls:
                trace.append({"step": step, "tool_name": None, "tool_args": None, "tool_result": None, "llm_input_tokens": getattr(usage, "prompt_tokens", None), "llm_output_tokens": getattr(usage, "completion_tokens", None), "duration_ms": duration_ms})
                logger.info("agent.step=%s tool=none duration_ms=%s", step, duration_ms)
                return {"answer": message.content or "", "steps": step, "trace": trace}
            for call in calls:
                name, raw_args = call.function.name, call.function.arguments
                try:
                    args = json.loads(raw_args)
                    result = DISPATCH[name](**args) if name in DISPATCH else f"Неизвестный tool: {name}"
                except Exception as error:
                    args, result = raw_args, f"Ошибка tool {name}: {error}"
                result = str(result)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                trace.append({"step": step, "tool_name": name, "tool_args": args, "tool_result": result[:200], "llm_input_tokens": getattr(usage, "prompt_tokens", None), "llm_output_tokens": getattr(usage, "completion_tokens", None), "duration_ms": duration_ms})
                logger.info("agent.step=%s tool=%s duration_ms=%s", step, name, duration_ms)
        return {"answer": "", "steps": max_steps, "trace": trace, "error": "Достигнут лимит шагов агента."}
    except Exception as error:
        return {"answer": "", "steps": len({item["step"] for item in trace}), "trace": trace, "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()
    result = run_agent(args.task, args.max_steps)
    print(result.get("answer") or result.get("error"))
    if args.trace:
        print(json.dumps(result["trace"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
