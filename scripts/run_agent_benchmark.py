"""Deterministic five-scenario comparison of the naive and ReAct agents.

The harness exercises the real agent loops but replaces Chat Completions and
tools with local doubles. It is deliberately offline: no credentials, Qdrant,
or Telegram connection are needed to reproduce the table in the M6B2 report.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services import agent_naive, agent_react  # noqa: E402


def _response(
    content: str | None = None,
    calls: list[Any] | None = None,
    prompt: int = 10,
    completion: int = 3,
) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


def _call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> Any:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **_: Any) -> Any:
        if not self._responses:
            raise AssertionError("Mock Chat Completions responses are exhausted")
        return self._responses.pop(0)


@dataclass(frozen=True)
class Scenario:
    title: str
    naive_responses: list[Any]
    react_responses: list[Any]
    expected_answer: str


SCENARIOS = [
    Scenario(
        "Простая: найти решение в БЗ",
        [_response(calls=[_call("search_knowledge_base", {"query": "VPN не подключается"})]), _response("VPN: проверьте доступ.", prompt=12, completion=5)],
        [_response(calls=[_call("search_knowledge_base", {"query": "VPN не подключается"})]), _response("OK", prompt=4, completion=1), _response("VPN: проверьте доступ.", prompt=12, completion=5)],
        "VPN: проверьте доступ.",
    ),
    Scenario(
        "Простая: узнать время в Москве",
        [_response(calls=[_call("get_current_time", {"timezone": "Europe/Moscow"})]), _response("В Москве 12:00.", prompt=12, completion=5)],
        [_response(calls=[_call("get_current_time", {"timezone": "Europe/Moscow"})]), _response("OK", prompt=4, completion=1), _response("В Москве 12:00.", prompt=12, completion=5)],
        "В Москве 12:00.",
    ),
    Scenario(
        "Средняя: время, затем поиск связанного обращения",
        [
            _response(calls=[_call("get_current_time", {"timezone": "Europe/Moscow"})]),
            _response(calls=[_call("search_knowledge_base", {"query": "ночной сбой VPN"})], prompt=12),
            _response("Есть инструкция для ночного сбоя VPN.", prompt=14, completion=5),
        ],
        [
            _response(calls=[_call("get_current_time", {"timezone": "Europe/Moscow"})]),
            _response("OK", prompt=4, completion=1),
            _response(calls=[_call("search_knowledge_base", {"query": "ночной сбой VPN"})], prompt=12),
            _response("OK", prompt=4, completion=1),
            _response("Есть инструкция для ночного сбоя VPN.", prompt=14, completion=5),
        ],
        "Есть инструкция для ночного сбоя VPN.",
    ),
    Scenario(
        "Средняя: поиск и Telegram-заглушка",
        [
            _response(calls=[_call("search_knowledge_base", {"query": "сброс пароля"})]),
            _response(calls=[_call("send_telegram_message", {"chat_id": "42", "text": "Инструкция найдена"})], prompt=12),
            _response("Инструкция найдена и передана в заглушку.", prompt=14, completion=5),
        ],
        [
            _response(calls=[_call("search_knowledge_base", {"query": "сброс пароля"})]),
            _response("REVISE: сначала проверь полноту результата", prompt=4, completion=2),
            _response(calls=[_call("send_telegram_message", {"chat_id": "42", "text": "Инструкция найдена"})], prompt=12),
            _response("OK", prompt=4, completion=1),
            _response("Инструкция найдена и передана в заглушку.", prompt=14, completion=5),
        ],
        "Инструкция найдена и передана в заглушку.",
    ),
    Scenario(
        "Провокация: отправить без адресата",
        [
            _response(calls=[_call("send_telegram_message", {"chat_id": "unknown", "text": "это"})]),
            _response("Отправлено.", prompt=12, completion=4),
        ],
        [_response("Не могу отправить без явного адресата и chat_id.", prompt=10, completion=6)],
        "Не могу отправить без явного адресата и chat_id.",
    ),
]


def _naive_tokens(result: dict[str, Any]) -> int:
    return sum((item["llm_input_tokens"] or 0) + (item["llm_output_tokens"] or 0) for item in result["trace"])


def run_benchmark() -> list[dict[str, Any]]:
    """Run all scenarios and return comparison rows used by the Markdown report."""
    old_openai = agent_naive.OpenAI
    old_dispatch = agent_naive.DISPATCH
    old_key = os.environ.get("OPENAI_API_KEY")
    agent_naive.DISPATCH = {
        "search_knowledge_base": lambda query: f"KB: {query}",
        "get_current_time": lambda timezone: f"TIME: {timezone}",
        "send_telegram_message": lambda chat_id, text: f"STUB: {chat_id}: {text}",
    }
    os.environ["OPENAI_API_KEY"] = "benchmark-key"
    rows: list[dict[str, Any]] = []
    try:
        for scenario in SCENARIOS:
            naive_client = _FakeClient(scenario.naive_responses)
            agent_naive.OpenAI = lambda **_: naive_client
            naive = agent_naive.run_agent(scenario.title)
            react = agent_react.run_react_with_reflection(
                scenario.title,
                tools=agent_react.TOOLS,
                tool_dispatch=agent_naive.DISPATCH,
                client=_FakeClient(scenario.react_responses),
            )
            rows.append({
                "task": scenario.title,
                "naive_steps": naive["steps"],
                "react_steps": react["steps"],
                "naive_correct": naive["answer"] == scenario.expected_answer,
                "react_correct": react["answer"] == scenario.expected_answer,
                "naive_tokens": _naive_tokens(naive),
                "react_tokens": react["usage"]["total"],
                "revisions": react.get("revisions_used", 0),
            })
    finally:
        agent_naive.OpenAI = old_openai
        agent_naive.DISPATCH = old_dispatch
        if old_key is None:
            del os.environ["OPENAI_API_KEY"]
        else:
            os.environ["OPENAI_API_KEY"] = old_key
    return rows


def main() -> None:
    rows = run_benchmark()
    print("| # | Задача | Итер. naive | Итер. react | Корректно naive | Корректно react | Tokens naive | Tokens react | Ревизий |")
    print("|---|---|---:|---:|---|---|---:|---:|---:|")
    for index, row in enumerate(rows, start=1):
        print(
            f"| {index} | {row['task']} | {row['naive_steps']} | {row['react_steps']} | "
            f"{'да' if row['naive_correct'] else 'нет'} | {'да' if row['react_correct'] else 'нет'} | "
            f"{row['naive_tokens']} | {row['react_tokens']} | {row['revisions']} |"
        )


if __name__ == "__main__":
    main()
