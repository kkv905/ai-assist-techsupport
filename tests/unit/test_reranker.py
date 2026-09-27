from app.services.reranker import CrossEncoderReranker


def test_reranker_returns_original_candidates_in_cross_encoder_order() -> None:
    reranker = CrossEncoderReranker()

    class Model:
        def predict(self, pairs):
            assert pairs == [("vpn", "first"), ("vpn", "second"), ("vpn", "third")]
            return [0.1, 0.9, 0.4]

    reranker._model = Model()
    candidates = ["first", "second", "third"]

    assert reranker.rerank("vpn", candidates, top_n=2) == ["second", "third"]
