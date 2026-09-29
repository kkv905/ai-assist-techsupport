from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.services import agent_persistent


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "user_role": "write-with-approve"}}


def _input() -> dict[str, str]:
    return {
        "request_id": "REQ-42",
        "recipient": "customer@example.test",
        "subject": "Решение найдено",
        "body": "Перезапустите клиент и повторите вход.",
    }


@pytest.mark.asyncio
async def test_graph_interrupts_with_approval_node_visible() -> None:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        await checkpointer.setup()
        graph = agent_persistent.build_agent(checkpointer)
        config = _config("interrupt-test")

        result = await graph.ainvoke(_input(), config)
        snapshot = await graph.aget_state(config)

    assert "__interrupt__" in result
    assert snapshot.next == ("confirm_and_execute_email",)
    assert snapshot.tasks[0].interrupts[0].value["type"] == "approve_send_email"


@pytest.mark.asyncio
async def test_approved_resume_sends_email(monkeypatch: pytest.MonkeyPatch) -> None:
    send_mock = AsyncMock()
    monkeypatch.setattr(agent_persistent, "send_email", send_mock)
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        await checkpointer.setup()
        graph = agent_persistent.build_agent(checkpointer)
        config = _config("approved-test")

        await graph.ainvoke(_input(), config)
        result = await graph.ainvoke(Command(resume=True), config)

    assert result["sent"] is True
    send_mock.assert_awaited_once_with(result["draft"])


@pytest.mark.asyncio
async def test_rejected_resume_does_not_send_email(monkeypatch: pytest.MonkeyPatch) -> None:
    send_mock = AsyncMock()
    monkeypatch.setattr(agent_persistent, "send_email", send_mock)
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        await checkpointer.setup()
        graph = agent_persistent.build_agent(checkpointer)
        config = _config("rejected-test")

        await graph.ainvoke(_input(), config)
        result = await graph.ainvoke(Command(resume=False), config)

    assert result["sent"] is False
    send_mock.assert_not_called()


@pytest.mark.asyncio
async def test_stream_endpoint_emits_interrupt_and_resumes(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.main import app

    send_mock = AsyncMock()
    monkeypatch.setattr(agent_persistent, "send_email", send_mock)
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        await checkpointer.setup()
        monkeypatch.setattr(
            app.state, "agent_graph", agent_persistent.build_agent(checkpointer), raising=False
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.post("/agent/stream", json={"thread_id": "sse-test", "input": _input()})
            resumed = await client.post(
                "/agent/stream",
                json={"thread_id": "sse-test", "input": _input(), "resume": True},
            )

    assert first.status_code == 200
    assert 'event: interrupt\ndata: {"__interrupt__"' in first.text
    assert resumed.status_code == 200
    assert 'event: done\ndata: {"next": [], "sent": true}' in resumed.text
    send_mock.assert_awaited_once()
