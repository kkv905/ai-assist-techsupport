"""Run a manual LangGraph supervisor/researcher/writer M6B5 experiment."""

from __future__ import annotations

import operator
import time
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from experiments.common import DeterministicSupportModel, QUESTIONS, UsageCounter, search_knowledge_base, write_results

RESEARCHER_PROMPT = "Собери факты через поиск и верни только маркированный список с источниками; финальный ответ не пиши."
WRITER_PROMPT = "По списку фактов собери связный ответ техподдержки с цитированием [1], [2]."


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    stage: Literal["start", "researched", "written"]
    handoff_count: Annotated[int, operator.add]


def build_app(counters: list[UsageCounter]) -> Any:
    researcher_counter, writer_counter = counters
    researcher = create_agent(DeterministicSupportModel(role="researcher", counter=researcher_counter), [search_knowledge_base], system_prompt=RESEARCHER_PROMPT, name="researcher")
    writer = create_agent(DeterministicSupportModel(role="writer", counter=writer_counter), [], system_prompt=WRITER_PROMPT, name="writer")

    def supervisor(state: State) -> Command:
        if state["stage"] == "start":
            return Command(goto="researcher", update={"handoff_count": 1})
        if state["stage"] == "researched":
            return Command(goto="writer", update={"handoff_count": 1})
        return Command(goto=END)

    def research(state: State) -> Command:
        result = researcher.invoke({"messages": state["messages"]})
        return Command(goto="supervisor", update={"messages": result["messages"][len(state["messages"]):], "stage": "researched"})

    def write(state: State) -> Command:
        result = writer.invoke({"messages": state["messages"]})
        return Command(goto="supervisor", update={"messages": result["messages"][len(state["messages"]):], "stage": "written"})

    graph = StateGraph(State)
    # ``destinations`` documents dynamic Command routes in draw_mermaid().
    graph.add_node("supervisor", supervisor, destinations=("researcher", "writer", END))
    graph.add_node("researcher", research, destinations=("supervisor",))
    graph.add_node("writer", write, destinations=("supervisor",))
    graph.add_edge(START, "supervisor")
    return graph.compile(checkpointer=InMemorySaver())


def run_question(question: str) -> dict[str, Any]:
    counters = [UsageCounter(), UsageCounter()]
    app = build_app(counters)
    started = time.perf_counter()
    config = {"configurable": {"thread_id": "exp-langgraph"}}
    for update in app.stream({"messages": [{"role": "user", "content": question}], "stage": "start", "handoff_count": 0}, config, stream_mode="updates"):
        print(update)
    state = app.get_state(config).values
    answer = str(state["messages"][-1].content)
    return {"implementation": "multi", "question": question, "answer": answer, "total_tokens": sum(item.input_tokens + item.output_tokens for item in counters), "llm_calls": sum(item.calls for item in counters), "latency_ms": round((time.perf_counter() - started) * 1000, 2), "handoff_count": state["handoff_count"]}


def main() -> None:
    app = build_app([UsageCounter(), UsageCounter()])
    (Path(__file__).resolve().parents[1] / "docs" / "architecture-multi-agent.md").write_text(app.get_graph().draw_mermaid() + "\n", encoding="utf-8")
    rows = [run_question(question) for question in QUESTIONS]
    write_results(rows)
    print("Saved", len(rows), "multi-agent measurements.")


if __name__ == "__main__":
    main()
