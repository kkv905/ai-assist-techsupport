"""Offline demonstration of LangGraph interrupts, history and separate branches."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.agent_persistent import build_agent  # noqa: E402


INPUT = {
    "request_id": "DEMO-1",
    "recipient": "customer@example.test",
    "subject": "Решение по обращению DEMO-1",
    "body": "Выполните повторный вход после очистки кэша.",
}


def config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id, "user_role": "write-with-approve"}}


async def run_branch(graph: Any, thread_id: str, decision: bool) -> dict[str, Any]:
    thread_config = config(thread_id)
    paused = await graph.ainvoke(INPUT, thread_config)
    print(f"{thread_id}: __interrupt__ = {paused['__interrupt__'][0].value}")
    return await graph.ainvoke(Command(resume=decision), thread_config)


async def main() -> None:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        await checkpointer.setup()
        graph = build_agent(checkpointer)
        demo_config = config("demo-history")

        paused = await graph.ainvoke(INPUT, demo_config)
        print("interrupt payload:", paused["__interrupt__"][0].value)

        history = [state async for state in graph.aget_state_history(demo_config)]
        print("\ncheckpoint_id                         next")
        for state in history[:5]:
            print(f"{state.config['configurable']['checkpoint_id']}  {state.next}")

        before_send = next(state for state in history if state.next == ("confirm_and_execute_email",))
        old_config = {
            "configurable": {
                "thread_id": "demo-history",
                "checkpoint_id": before_send.config["configurable"]["checkpoint_id"],
            }
        }
        old_state = await graph.aget_state(old_config)
        print("\nstate before send:", {"draft": old_state.values["draft"], "sent": old_state.values["sent"], "next": old_state.next})

        rejected = await run_branch(graph, "demo-rejected", False)
        approved = await run_branch(graph, "demo-approved", True)
        print("\nseparate lineages (resume is deterministic per lineage):")
        print("rejected sent =", rejected["sent"])
        print("approved sent =", approved["sent"])


if __name__ == "__main__":
    asyncio.run(main())
