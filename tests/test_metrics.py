from rag_eval.metrics import evaluate


def test_metrics_at_multiple_cutoffs():
    qrels = {
        "q1": {"d1": 1, "d2": 1},
        "q2": {"d3": 2},
    }
    run = {
        "q1": {"d1": 2.0, "d9": 1.0, "d2": 0.5},
        "q2": {"d9": 2.0, "d3": 1.0},
    }

    metrics = evaluate(qrels, run, (1, 3))

    assert metrics["Precision@1"] == 0.5
    assert metrics["Recall@1"] == 0.25
    assert metrics["MRR@1"] == 0.5
    assert metrics["NDCG@1"] == 0.5
    assert metrics["Precision@3"] == 0.5
    assert metrics["Recall@3"] == 1.0
    assert metrics["MRR@3"] == 0.75
    assert metrics["NDCG@3"] == 0.77533


def test_missing_run_query_scores_zero():
    metrics = evaluate({"q1": {"d1": 1}}, {}, (1,))

    assert metrics == {
        "Precision@1": 0.0,
        "Recall@1": 0.0,
        "MRR@1": 0.0,
        "NDCG@1": 0.0,
    }
