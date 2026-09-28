from scripts.run_agent_benchmark import run_benchmark


def test_agent_benchmark_runs_five_scenarios_for_both_agents() -> None:
    rows = run_benchmark()

    assert len(rows) == 5
    assert all(row["naive_tokens"] > 0 and row["react_tokens"] > 0 for row in rows)
    assert rows[-1]["naive_correct"] is False
    assert rows[-1]["react_correct"] is True
    assert rows[3]["revisions"] == 1
