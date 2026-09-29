"""Shared corpus, deterministic LLM and measurement helpers for M6B5.

The experiment deliberately uses a local mock retriever and deterministic chat
model.  This keeps both implementations comparable and executable without an
API key; replace only ``search_knowledge_base`` with ``RAGService.answer`` for
an online experiment, keeping the exact same tool in both scripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import ConfigDict

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "experiments" / "results.json"
QUESTIONS = [
    "Не могу войти в АИС Правоохрана: access denied.",
    "Почему не загружается файл в АИС ЦРСВЭД?",
    "В АИС Постконтроль страница выдаёт 500, что собрать для второй линии?",
    "В АИС ЦРСВЭД файл в неверной кодировке: что проверить и куда эскалировать?",
    "Как сбросить пароль от корпоративной почты?",
]


def _documents() -> list[dict[str, Any]]:
    return json.loads((ROOT / "app" / "data" / "knowledge_base.json").read_text(encoding="utf-8"))


def _score(query: str, document: dict[str, Any]) -> int:
    query_words = set(query.lower().replace("?", "").replace(".", "").split())
    fields = [document["title"], document["problem"], document["solution"], *document["symptoms"]]
    return len(query_words & set(" ".join(fields).lower().replace(",", "").split()))


def retrieve(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Return 2--3 deterministic fake RAG documents with an explicit score."""
    ranked = sorted(_documents(), key=lambda item: _score(query, item), reverse=True)
    return [item for item in ranked[:limit] if _score(query, item) > 0]


@tool
def search_knowledge_base(query: str) -> str:
    """Find support facts in the shared local mock of the M5 knowledge base."""
    found = retrieve(query)
    if not found:
        return "Надёжных сведений в базе нет. Не придумывай решение; предложи эскалацию."
    return "\n".join(
        f"[{number}] {item['title']} ({item['id']}, score={_score(query, item)}): {item['solution']}"
        for number, item in enumerate(found, start=1)
    )


def _answer(observation: str) -> str:
    if observation.startswith("Надёжных"):
        return "По базе не нашёл подтверждённого решения для корпоративной почты; передам обращение на уточнение."
    first = observation.splitlines()[0]
    solution = first.split(": ", maxsplit=1)[-1]
    citations = " [1]"
    if "\n[2]" in observation:
        citations += " [2]"
    return f"Рекомендуемые действия: {solution}{citations}"


@dataclass
class UsageCounter:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> dict[str, int]:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1
        return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}


class DeterministicSupportModel(BaseChatModel):
    """Minimal tool-capable offline chat model used only by the experiment."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    role: Literal["single", "researcher", "writer"]
    counter: UsageCounter = field(default_factory=UsageCounter)

    @property
    def _llm_type(self) -> str:
        return "m6b5-deterministic-support-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "DeterministicSupportModel":
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        del stop, run_manager, kwargs
        last_tool = next((message for message in reversed(messages) if isinstance(message, ToolMessage)), None)
        if self.role in {"single", "researcher"} and last_tool is None:
            question = str(messages[-1].content)
            usage = self.counter.add(28, 8)
            message = AIMessage(
                content="",
                tool_calls=[{"name": "search_knowledge_base", "args": {"query": question}, "id": "search-1"}],
                usage_metadata=usage,
            )
        else:
            observation = str(last_tool.content) if last_tool else next(
                (str(message.content) for message in reversed(messages) if isinstance(message, AIMessage) and message.content), ""
            )
            content = observation if self.role == "researcher" else _answer(observation)
            usage = self.counter.add(36, 32 if self.role != "researcher" else 24)
            message = AIMessage(content=content, usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=message)])


def write_results(rows: list[dict[str, Any]]) -> None:
    current: list[dict[str, Any]] = []
    if RESULTS_PATH.exists():
        current = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    implementations = {row["implementation"] for row in rows}
    current = [row for row in current if row.get("implementation") not in implementations]
    current.extend(rows)
    RESULTS_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
