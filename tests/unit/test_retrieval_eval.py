from app.services.retrieval_eval import evaluate_retrieval, hit_rate_at_k, mrr_at_k, recall_at_k


def test_retrieval_metrics_respect_rank_cutoffs_and_paths() -> None:
    retrieved = ["noise.txt", "data/02-vpn.txt", "01-password-reset.md"]

    assert hit_rate_at_k(retrieved, ["01-password-reset.md"], k=2) == 0.0
    assert mrr_at_k(retrieved, ["02-vpn.txt"], k=10) == 0.5
    assert recall_at_k(retrieved, ["02-vpn.txt", "01-password-reset.md"], k=10) == 1.0


def test_evaluate_retrieval_averages_all_required_metrics() -> None:
    data = [
        {"question": "vpn", "relevant_doc_ids": ["02-vpn.txt"]},
        {"question": "missing", "relevant_doc_ids": ["03-email.md"]},
    ]

    metrics = evaluate_retrieval(data, lambda question: ["02-vpn.txt"] if question == "vpn" else [])

    assert metrics == {"hit_rate_at_5": 0.5, "mrr_at_10": 0.5, "recall_at_10": 0.5}
