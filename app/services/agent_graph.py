"""Реализации ReAct-агента поддержки M6 на LangGraph.

Модуль намеренно предоставляет и вручную собранный граф, и готового агента
LangChain. Пока ни один граф не использует хранилище контрольных точек, но при
вызове уже можно передавать ``configurable.thread_id``.
"""

from __future__ import annotations

import operator
import os
from typing import Annotated, Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.services.agent_naive import (
    get_current_time as _get_current_time,
    search_knowledge_base as _search_knowledge_base,
    send_telegram_message as _send_telegram_message,
)

MAX_ITERATIONS = 6
SYSTEM_PROMPT = (
    "Ты ассистент технической поддержки. Используй инструмент только когда он "
    "действительно нужен, а затем дай краткий, проверяемый ответ."
)


@tool
def search_knowledge_base(query: str) -> str:
    """Ищи подтверждённое решение или похожее обращение в базе знаний техподдержки."""
    return _search_knowledge_base(query)


@tool
def get_current_time(timezone: str = "Europe/Moscow") -> str:
    """Возвращай фактическое текущее время для IANA часового пояса."""
    return _get_current_time(timezone)


@tool
def send_telegram_message(chat_id: str, text: str) -> str:
    """Имитируй отправку Telegram-сообщения только по явной просьбе и с chat_id."""
    return _send_telegram_message(chat_id, text)


TOOLS: list[BaseTool] = [search_knowledge_base, get_current_time, send_telegram_message]
# Клиент и ключ API намеренно не входят в состояние графа.
# Безопасная заглушка позволяет исследовать граф и запускать тесты без учётных
# данных. Для реального вызова всё равно нужен OPENAI_API_KEY (или LLM__OPENAI_API_KEY).
MODEL = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0,
    api_key=os.environ.get("LLM__OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or "not-configured",
)


class AgentState(TypedDict):
    """Сериализуемое состояние графа; редьюсеры сохраняют диалог и аудит инструментов."""

    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    tool_results: Annotated[list[dict[str, Any]], operator.add]


async def call_model(state: AgentState) -> dict[str, Any]:
    """Запрашивает у связанной модели следующее действие или итоговый ответ."""
    response = await MODEL.bind_tools(TOOLS).ainvoke(state["messages"])
    return {"messages": [response], "iteration_count": state["iteration_count"] + 1}


async def execute_tool(state: AgentState) -> dict[str, Any]:
    """Выполняет запрошенные инструменты и возвращает наблюдения как ToolMessage."""
    last_message = state["messages"][-1]
    by_name = {tool_.name: tool_ for tool_ in TOOLS}
    messages: list[ToolMessage] = []
    results: list[dict[str, Any]] = []
    for tool_call in getattr(last_message, "tool_calls", []):
        name = tool_call["name"]
        args = tool_call.get("args", {})
        if name not in by_name:
            content = f"Ошибка: неизвестный инструмент '{name}'."
        else:
            try:
                content = str(await by_name[name].ainvoke(args))
            except Exception as error:  # Ошибки инструментов передаются модели как наблюдения.
                content = f"Ошибка инструмента {name}: {error}"
        messages.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))
        results.append({"name": name, "args": args, "result": content})
    return {"messages": messages, "tool_results": results}


async def force_finish(state: AgentState) -> dict[str, Any]:
    """Создаёт явное итоговое сообщение, если ограничитель остановил работу инструментов."""
    last_message = state["messages"][-1]
    if state["iteration_count"] >= MAX_ITERATIONS and getattr(last_message, "tool_calls", None):
        return {"messages": [AIMessage(content="Достигнут лимит итераций; не удалось получить итоговый ответ.")]}
    return {}


def route_after_model(state: AgentState) -> Literal["execute_tool", "force_finish"]:
    """Чистый детерминированный переход завершения для пользовательского графа."""
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "force_finish"
    last_message = state["messages"][-1]
    return "execute_tool" if getattr(last_message, "tool_calls", None) else "force_finish"


def build_custom_graph() -> Any:
    """Собирает явный StateGraph для случаев, когда нужна настройка оркестрации."""
    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("force_finish", force_finish)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {"execute_tool": "execute_tool", "force_finish": "force_finish"},
    )
    builder.add_edge("execute_tool", "call_model")
    builder.add_edge("force_finish", END)
    return builder.compile()


def build_prebuilt_graph(model: Any = MODEL, tools: list[BaseTool] = TOOLS) -> Any:
    """Собирает управляемый ReAct-граф LangChain из тех же инструментов."""
    return create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)


custom_graph = build_custom_graph()
prebuilt_graph = build_prebuilt_graph()
