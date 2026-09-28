import json
from types import SimpleNamespace

from app.services import agent_naive


def response(content=None, calls=None, prompt=10, completion=3):
    message = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion))


def tool_call(name, arguments, call_id="call_1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def test_run_agent_executes_tool_and_returns_trace(monkeypatch) -> None:
    calls = [response(calls=[tool_call("get_current_time", {"timezone": "UTC"})]), response(content="Сейчас известно время.")]
    def fake_create(**kwargs):
        return calls.pop(0)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(agent_naive, "OpenAI", lambda **kwargs: fake_client)

    result = agent_naive.run_agent("Который час?")

    assert result["answer"] == "Сейчас известно время."
    assert result["steps"] == 2
    assert result["trace"][0]["tool_name"] == "get_current_time"
    assert result["trace"][0]["llm_input_tokens"] == 10
    assert result["trace"][1]["tool_result"] is None


def test_unknown_tool_is_returned_to_model(monkeypatch) -> None:
    calls = [response(calls=[tool_call("get_user_balance", {"user_id": "7"})]), response(content="Такого инструмента нет.")]
    received = []
    def create(**kwargs):
        received.append(list(kwargs["messages"]))
        return calls.pop(0)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(agent_naive, "OpenAI", lambda **kwargs: fake_client)

    result = agent_naive.run_agent("Проверь баланс")

    assert result["answer"] == "Такого инструмента нет."
    assert result["trace"][0]["tool_result"] == "Неизвестный tool: get_user_balance"
    assert received[1][-1]["content"] == "Неизвестный tool: get_user_balance"
