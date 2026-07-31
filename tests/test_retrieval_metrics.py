from rag_eval.retrieval.metrics import (
    aggregate_evidence_availability,
    score_ranked_evidence_ids,
)


def test_score_ranked_evidence_ids_uses_best_complete_reference():
    score = score_ranked_evidence_ids(
        ("p2", "noise", "p1"),
        ({"p1", "p2"}, {"p3"}),
    )

    assert score["evidence_hit"] is True
    assert score["best_reference_evidence_recall"] == 1.0
    assert score["best_reference_evidence_precision"] == 2 / 3
    assert score["best_reference_evidence_mrr"] == 1.0
    assert score["complete_reference_evidence_available"] is True


def test_aggregate_evidence_availability_excludes_cases_without_gold_evidence():
    complete = score_ranked_evidence_ids(("p1",), ({"p1"},))
    unavailable = score_ranked_evidence_ids(("p2",), ())

    result = aggregate_evidence_availability((complete, unavailable))

    assert result["case_count"] == 2
    assert result["evaluable_answerable_case_count"] == 1
    assert result["complete_reference_set_rate"] == 1.0
