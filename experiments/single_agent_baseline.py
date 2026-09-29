"""Run the five-question single-agent M6B5 baseline."""

from __future__ import annotations

import time
from typing import Any

from langchain.agents import create_agent

from experiments.common import DeterministicSupportModel, QUESTIONS, UsageCounter, search_knowledge_base, write_results

SYSTEM_PROMPT = """Ты единый агент техподдержки. Найди факты через search_knowledge_base,
затем дай краткое решение только по результату инструмента и процитируй источники [1], [2]."""


def run_question(question: str) -> dict[str, Any]:
    counter = UsageCounter()
    app = create_agent(
        model=DeterministicSupportModel(role="single", counter=counter),
        tools=[search_knowledge_base], system_prompt=SYSTEM_PROMPT, name="single_support_agent",
    )
    started = time.perf_counter()
    final: dict[str, Any] = {}
    for update in app.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="updates"):
        print(update)
        final = update
    answer = next((str(value["messages"][-1].content) for value in final.values() if value.get("messages")), "")
    return {"implementation": "single", "question": question, "answer": answer, "total_tokens": counter.input_tokens + counter.output_tokens, "llm_calls": counter.calls, "latency_ms": round((time.perf_counter() - started) * 1000, 2), "handoff_count": 0}


def main() -> None:
    rows = [run_question(question) for question in QUESTIONS]
    write_results(rows)
    print("Saved", len(rows), "single-agent measurements.")


if __name__ == "__main__":
    main()
