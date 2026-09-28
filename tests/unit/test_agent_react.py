import json
import time
from types import SimpleNamespace

import pytest

from app.services import agent_react


def response(content=None, calls=None, prompt=10, completion=3):
    message = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion),
    )


def tool_call(name, arguments, call_id="call_1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def client_with(responses):
    queue = list(responses)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: queue.pop(0)))
    )


def test_reflection_revises_once_and_counts_all_tokens() -> None:
    client = client_with([
        response(calls=[tool_call("time", {"timezone": "UTC"})], prompt=10, completion=2),
        response(content="REVISE: уточни вывод", prompt=4, completion=1),
        response(content="Время получено.", prompt=8, completion=2),
    ])

    result = agent_react.run_react_with_reflection(
        "Который час?", tools=[], tool_dispatch={"time": lambda timezone: "12:00"}, client=client
    )

    assert result["answer"] == "Время получено."
    assert result["revisions_used"] == 1
    assert result["usage"] == {"prompt": 22, "completion": 5, "total": 27}
    assert result["trace"][0]["critic_verdict"].startswith("REVISE:")
    assert result["trace"][1]["model"] == agent_react.MODEL_PREMIUM


def test_reflections_are_limited_across_iterations() -> None:
    client = client_with([
        response(calls=[tool_call("x", {})]), response(content="REVISE: one"),
        response(calls=[tool_call("x", {})]), response(content="REVISE: two"),
        response(calls=[tool_call("x", {})]), response(content="REVISE: three"),
        response(content="Готово"),
    ])
    result = agent_react.run_react_with_reflection(
        "test", tools=[], tool_dispatch={"x": lambda: "ok"}, client=client
    )

    assert result["answer"] == "Готово"
    assert result["revisions_used"] == 2
    assert [item["model"] for item in result["trace"][:3]] == [
        agent_react.MODEL_MAIN, agent_react.MODEL_PREMIUM, agent_react.MODEL_PREMIUM,
    ]


def test_max_iterations_returns_explicit_answer() -> None:
    responses = []
    for _ in range(8):
        responses.extend([response(calls=[tool_call("x", {})]), response(content="OK")])
    result = agent_react.run_react_with_reflection(
        "test", tools=[], tool_dispatch={"x": lambda: "ok"}, client=client_with(responses), max_iterations=8
    )
    assert result["answer"] == "Превышен лимит итераций"
    assert result["steps"] == 8


def test_timeout_returns_explicit_answer() -> None:
    client = client_with([response(calls=[tool_call("slow", {})])])
    result = agent_react.run_react_with_reflection(
        "test", tools=[], tool_dispatch={"slow": lambda: (time.sleep(5.05), "late")[1]},
        client=client, timeout_per_iteration_sec=5,
    )
    assert result["answer"] == "Timeout"


def test_provocative_task_can_finish_without_tools() -> None:
    result = agent_react.run_react_with_reflection(
        "Отправь это кому-нибудь", tools=[], tool_dispatch={}, client=client_with([response(content="Не могу отправить без чата и явного адресата.")])
    )
    assert result["answer"].startswith("Не могу")
    assert result["trace"][0]["tool_name"] is None


@pytest.mark.parametrize(("iterations", "timeout"), [(7, 10), (21, 10), (8, 4), (8, 16)])
def test_limit_ranges_are_enforced(iterations, timeout) -> None:
    with pytest.raises(ValueError):
        agent_react.run_react_with_reflection("test", max_iterations=iterations, timeout_per_iteration_sec=timeout)
