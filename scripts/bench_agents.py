"""Offline, reproducible M6B3 benchmark: naive loop vs two LangGraph forms.

It uses scripted LangChain messages and local tools, so its measurements are
orchestration overhead rather than network latency.  Run it again with real
credentials only when an end-to-end provider benchmark is desired.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "agent-graph-report.md"
sys.path.insert(0, str(ROOT))

from app.services import agent_graph, agent_naive  # noqa: E402
from scripts.run_agent_benchmark import SCENARIOS, _FakeClient  # noqa: E402

REPEATS = 3


@tool
def lookup(query: str) -> str:
    """Return a deterministic support-knowledge-base result for a query."""
    return f"KB: {query}"


@tool
def clock(timezone: str) -> str:
    """Return a deterministic time value for a timezone."""
    return f"TIME: {timezone}"


@tool
def notify(chat_id: str, text: str) -> str:
    """Simulate notification delivery to an explicit chat identifier."""
    return f"STUB: {chat_id}: {text}"


OFFLINE_TOOLS = [lookup, clock, notify]


class ScriptedChatModel(BaseChatModel):
    """Small local ChatModel that exercises the normal LangGraph interfaces."""

    responses: list[AIMessage] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "m6b3-scripted"

    def bind_tools(self, _tools: Any, **_kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self, _messages: list[Any], _stop: Any = None, _run_manager: Any = None, **_kwargs: Any
    ) -> ChatResult:
        if not self.responses:
            raise AssertionError("Scripted model responses exhausted")
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


TASKS = [
    ("Простая: найти решение в БЗ", [("lookup", {"query": "VPN не подключается"})]),
    ("Простая: узнать время в Москве", [("clock", {"timezone": "Europe/Moscow"})]),
    ("Средняя: время, затем поиск связанного обращения", [("clock", {"timezone": "Europe/Moscow"}), ("lookup", {"query": "ночной сбой VPN"})]),
    ("Средняя: поиск и Telegram-заглушка", [("lookup", {"query": "сброс пароля"}), ("notify", {"chat_id": "42", "text": "Инструкция найдена"})]),
    ("Провокация: отправить без адресата", []),
]


def _message(content: str, calls: list[dict[str, Any]] | None = None) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=calls or [],
        usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
    )


def _script(calls: list[tuple[str, dict[str, Any]]]) -> list[AIMessage]:
    if not calls:
        return [_message("Не могу отправить сообщение без явного chat_id и адресата.")]
    responses = []
    for index, (name, args) in enumerate(calls, start=1):
        call = {"name": name, "args": args, "id": f"offline-call-{index}", "type": "tool_call"}
        responses.append(_message("", [call]))
    return [*responses, _message("Проверка выполнена; используйте полученный результат.")]


async def _custom(task: str, calls: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    old_model, old_tools = agent_graph.MODEL, agent_graph.TOOLS
    try:
        agent_graph.MODEL = ScriptedChatModel(responses=_script(calls))
        agent_graph.TOOLS = OFFLINE_TOOLS
        return await agent_graph.custom_graph.ainvoke(
            {"messages": [HumanMessage(content=task)], "iteration_count": 0, "tool_results": []},
            config={"configurable": {"thread_id": f"bench-{task[:8]}"}},
        )
    finally:
        agent_graph.MODEL, agent_graph.TOOLS = old_model, old_tools


async def _prebuilt(task: str, calls: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    graph = agent_graph.build_prebuilt_graph(ScriptedChatModel(responses=_script(calls)), OFFLINE_TOOLS)
    return await graph.ainvoke({"messages": [HumanMessage(content=task)]})


def _usage(result: dict[str, Any]) -> tuple[int, int]:
    messages = result["messages"]
    values = [getattr(message, "usage_metadata", None) or {} for message in messages]
    return sum(item.get("input_tokens", 0) for item in values), sum(item.get("output_tokens", 0) for item in values)


def _measure(coro: Any) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    result = asyncio.run(coro)
    return (time.perf_counter() - started) * 1000, result


def _naive(task: str) -> dict[str, Any]:
    """Run the original B6.2 loop against the matching offline transcript."""
    scenario = next(item for item in SCENARIOS if item.title == task)
    old_openai, old_dispatch = agent_naive.OpenAI, agent_naive.DISPATCH
    old_key = os.environ.get("OPENAI_API_KEY")
    try:
        agent_naive.OpenAI = lambda **_kwargs: _FakeClient(scenario.naive_responses)
        agent_naive.DISPATCH = {"search_knowledge_base": lambda query: f"KB: {query}", "get_current_time": lambda timezone: f"TIME: {timezone}", "send_telegram_message": lambda chat_id, text: f"STUB: {chat_id}: {text}"}
        os.environ["OPENAI_API_KEY"] = "offline-benchmark"
        return agent_naive.run_agent(task)
    finally:
        agent_naive.OpenAI, agent_naive.DISPATCH = old_openai, old_dispatch
        if old_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_key


def run() -> list[dict[str, Any]]:
    rows = []
    for task, calls in TASKS:
        for variant, invoke in (("custom", _custom), ("prebuilt", _prebuilt)):
            timings, last = [], None
            for _ in range(REPEATS):
                elapsed, last = _measure(invoke(task, calls))
                timings.append(elapsed)
            prompt, completion = _usage(last)
            rows.append({"task": task, "variant": variant, "latency_ms": statistics.mean(timings), "prompt": prompt, "completion": completion, "steps": last.get("iteration_count", len(last["messages"]) - 1)})
        timings, last = [], None
        for _ in range(REPEATS):
            started = time.perf_counter()
            last = _naive(task)
            timings.append((time.perf_counter() - started) * 1000)
        prompt = sum(item["llm_input_tokens"] or 0 for item in last["trace"])
        completion = sum(item["llm_output_tokens"] or 0 for item in last["trace"])
        rows.append({"task": task, "variant": "naive B6.2", "latency_ms": statistics.mean(timings), "prompt": prompt, "completion": completion, "steps": last["steps"]})
    return rows


def render_table(rows: list[dict[str, Any]]) -> str:
    """Render the benchmark section embedded between stable report markers."""
    lines = [
        "| Задача | Реализация | latency_ms (mean 3) | prompt_tokens | completion_tokens | total_steps |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['task']} | {row['variant']} | {row['latency_ms']:.2f} | "
        f"{row['prompt']} | {row['completion']} | {row['steps']} |"
        for row in rows
    )
    return "\n".join(lines)


def write_report_table(rows: list[dict[str, Any]]) -> None:
    """Replace only the generated table, retaining the explanatory report text."""
    report = REPORT_PATH.read_text(encoding="utf-8")
    start = "<!-- BENCHMARK_TABLE_START -->"
    end = "<!-- BENCHMARK_TABLE_END -->"
    if start not in report or end not in report:
        raise RuntimeError("Benchmark table markers are missing from docs/agent-graph-report.md")
    before, remainder = report.split(start, maxsplit=1)
    _, after = remainder.split(end, maxsplit=1)
    REPORT_PATH.write_text(f"{before}{start}\n{render_table(rows)}\n{end}{after}", encoding="utf-8")


def main() -> None:
    rows = run()
    write_report_table(rows)
    print(render_table(rows))


if __name__ == "__main__":
    main()
