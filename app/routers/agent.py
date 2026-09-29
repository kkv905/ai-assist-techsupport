"""SSE interface for the durable approval-gated LangGraph workflow."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass
import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langgraph.types import Command

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentInput(BaseModel):
    request_id: str = "support-request"
    recipient: str = "support@example.invalid"
    subject: str = "Ответ технической поддержки"
    body: str = "Подготовлен ответ по вашему обращению."


class AgentStreamRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    input: AgentInput
    user_role: Literal["read-only", "write-with-approve", "full"] = "write-with-approve"
    resume: bool | None = None


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _sse(event: str, payload: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=_json_default, ensure_ascii=False)}\n\n"


@router.post("/stream", response_class=StreamingResponse)
async def stream_agent(request: Request, payload: AgentStreamRequest) -> StreamingResponse:
    """Stream node updates and model-message events in standard SSE framing."""

    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="Persistent agent is not initialised")

    config = {
        "configurable": {
            "thread_id": payload.thread_id,
            "user_role": payload.user_role,
        }
    }

    async def events() -> AsyncIterator[str]:
        graph_input: Any = (
            Command(resume=payload.resume) if payload.resume is not None else payload.input.model_dump()
        )
        async for stream_type, event_payload in graph.astream(
            graph_input,
            config,
            stream_mode=["updates", "messages"],
        ):
            yield _sse(stream_type, event_payload)

        snapshot = await graph.aget_state(config)
        interrupts = [
            interrupt_payload
            for task in snapshot.tasks
            for interrupt_payload in getattr(task, "interrupts", ())
        ]
        if interrupts:
            yield _sse("interrupt", {"__interrupt__": interrupts})
        yield _sse("done", {"next": list(snapshot.next), "sent": snapshot.values.get("sent", False)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
