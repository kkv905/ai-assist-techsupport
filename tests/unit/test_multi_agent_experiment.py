import json
from pathlib import Path

from experiments.common import QUESTIONS, search_knowledge_base
from experiments.evaluate_faithfulness import aggregate, build_rows
from experiments.multi_agent_langgraph import run_question
from experiments.single_agent_baseline import run_question as run_single_question


def test_shared_tool_returns_sources_and_refuses_unknown_question() -> None:
    assert "[1]" in search_knowledge_base.invoke({"query": QUESTIONS[0]})
    assert "Надёжных сведений" in search_knowledge_base.invoke({"query": QUESTIONS[-1]})


def test_both_agents_use_comparable_measurement_contract() -> None:
    single = run_single_question(QUESTIONS[0])
    multi = run_question(QUESTIONS[0])
    keys = {"implementation", "question", "answer", "total_tokens", "llm_calls", "latency_ms", "handoff_count"}
    assert set(single) == set(multi) == keys
    assert single["handoff_count"] == 0
    assert multi["handoff_count"] == 2
    assert multi["total_tokens"] > single["total_tokens"]
    assert "[1]" in multi["answer"]


def test_saved_results_have_ten_raw_measurements() -> None:
    rows = json.loads(Path("experiments/results.json").read_text(encoding="utf-8"))
    assert len(rows) == 10
    assert {(row["implementation"], row["question"]) for row in rows} == {
        (implementation, question) for implementation in ("single", "multi") for question in QUESTIONS
    }


def test_faithfulness_input_reuses_shared_tool_context() -> None:
    rows = build_rows()
    assert len(rows) == 10
    assert all(row["retrieved_contexts"] for row in rows)
    assert aggregate([
        {"implementation": "single", "faithfulness": 1.0},
        {"implementation": "multi", "faithfulness": 0.5},
    ]) == {"single": 1.0, "multi": 0.5}
