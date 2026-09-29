from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.services import agent_graph


class FakeBoundModel:
    def __init__(self, responses):
        self.responses = list(responses)

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_custom_graph_runs_tool_then_final_answer(monkeypatch) -> None:
    monkeypatch.setattr(agent_graph, "MODEL", FakeBoundModel([
        AIMessage(content="", tool_calls=[{"name": "multiply", "args": {"a": 2, "b": 3}, "id": "call-1"}]),
        AIMessage(content="Ответ: 6"),
    ]))
    multiply = SimpleNamespace(name="multiply", ainvoke=lambda args: _async_result(args["a"] * args["b"]))
    monkeypatch.setattr(agent_graph, "TOOLS", [multiply])
    result = await agent_graph.custom_graph.ainvoke({"messages": [HumanMessage(content="2*3")], "iteration_count": 0, "tool_results": []})
    assert result["messages"][-1].content == "Ответ: 6"
    assert result["iteration_count"] == 2
    assert result["tool_results"] == [{"name": "multiply", "args": {"a": 2, "b": 3}, "result": "6"}]


async def _async_result(value):
    return value


@pytest.mark.asyncio
async def test_unknown_tool_becomes_observation() -> None:
    result = await agent_graph.execute_tool({
        "messages": [AIMessage(content="", tool_calls=[{"name": "missing", "args": {}, "id": "call-2"}])],
        "iteration_count": 1,
        "tool_results": [],
    })
    assert "неизвестный" in result["messages"][0].content


def test_router_has_explicit_iteration_stop() -> None:
    state = {"messages": [AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "c"}])], "iteration_count": 6, "tool_results": []}
    assert agent_graph.route_after_model(state) == "force_finish"


@pytest.mark.asyncio
async def test_broken_tool_cannot_bypass_iteration_guard(monkeypatch) -> None:
    """A useless/unknown tool call ends explicitly instead of looping forever."""
    monkeypatch.setattr(
        agent_graph,
        "MODEL",
        FakeBoundModel([
            AIMessage(content="", tool_calls=[{"name": "broken", "args": {}, "id": f"call-{index}"}])
            for index in range(agent_graph.MAX_ITERATIONS)
        ]),
    )
    monkeypatch.setattr(agent_graph, "TOOLS", [])
    result = await agent_graph.custom_graph.ainvoke({
        "messages": [HumanMessage(content="Выполни бесполезную операцию")],
        "iteration_count": 0,
        "tool_results": [],
    })
    assert "лимит итераций" in result["messages"][-1].content
    assert result["iteration_count"] == agent_graph.MAX_ITERATIONS
