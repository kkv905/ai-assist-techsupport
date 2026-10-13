"""Устойчивый LangGraph-процесс для уведомлений поддержки с подтверждением действий.

Unlike :mod:`app.services.agent_graph`, this graph deliberately has a small,
deterministic state machine.  It makes the checkpoint/resume boundary explicit:
the draft is safe to repeat, while the email is sent only after ``interrupt``
returns a confirmed decision.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class PersistentAgentState(TypedDict, total=False):
    """Состояние письма поддержки, требующего подтверждения."""

    request_id: str
    recipient: str
    subject: str
    body: str
    draft: dict[str, str]
    sent: bool


async def send_email(payload: dict[str, str]) -> None:
    """Точка интеграции для будущего почтового провайдера.

    The training project intentionally has no SMTP credentials yet.  Keeping
    this async boundary makes the real side effect mockable and prevents an
    accidental outbound delivery in local demos.
    """


async def prepare_email(state: PersistentAgentState) -> dict[str, Any]:
    """Создаёт детерминированные данные письма; этот узел идемпотентен."""

    return {
        "draft": {
            "request_id": state.get("request_id", "support-request"),
            "recipient": state.get("recipient", "support@example.invalid"),
            "subject": state.get("subject", "Ответ технической поддержки"),
            "body": state.get("body", "Подготовлен ответ по вашему обращению."),
        },
        "sent": False,
    }


async def confirm_and_execute_email(
    state: PersistentAgentState, config: RunnableConfig
) -> dict[str, bool]:
    """Приостанавливает работу для подтверждения, затем выполняет неидемпотентную операцию.

    LangGraph restarts this node on resume.  Consequently there must be no
    external call before ``interrupt``: only the resumed branch can send.
    """

    role = config.get("configurable", {}).get("user_role", "write-with-approve")
    if role == "read-only":
        return {"sent": False}

    decision = True if role == "full" else interrupt(
        {
            "type": "approve_send_email",
            "preview": state["draft"],
        }
    )
    if decision:
        await send_email(state["draft"])
        return {"sent": True}
    return {"sent": False}


def build_agent(checkpointer: Any) -> Any:
    """Компилирует постоянный граф с переданным хранилищем LangGraph."""

    builder = StateGraph(PersistentAgentState)
    builder.add_node("prepare_email", prepare_email)
    builder.add_node("confirm_and_execute_email", confirm_and_execute_email)
    builder.add_edge(START, "prepare_email")
    builder.add_edge("prepare_email", "confirm_and_execute_email")
    builder.add_edge("confirm_and_execute_email", END)
    return builder.compile(checkpointer=checkpointer)


def _psycopg_uri(database_url: str) -> str:
    """Преобразует URI SQLAlchemy asyncpg приложения для psycopg v3."""

    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


@asynccontextmanager
async def agent_lifespan(settings: Any) -> AsyncIterator[tuple[Any, Any]]:
    """Открывает одно хранилище на время жизни приложения и однократно создаёт его схему."""

    stack = AsyncExitStack()
    try:
        backend: Literal["memory", "sqlite", "postgres"] = settings.agent_checkpointer
        if backend == "memory":
            checkpointer: Any = InMemorySaver()
        elif backend == "sqlite":
            sqlite_path = settings.agent_sqlite_path
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            checkpointer = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(sqlite_path))
            )
        else:
            checkpointer = await stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(_psycopg_uri(settings.database_url))
            )

        setup = getattr(checkpointer, "setup", None)
        if setup is not None:
            await setup()
        yield build_agent(checkpointer), checkpointer
    finally:
        await stack.aclose()
