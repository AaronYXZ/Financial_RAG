from rag_eval.evaluation.attribution import (
    aggregate_failure_attribution,
    classify_failure_attribution,
)


def _valid_case(**overrides):
    case = {
        "status": "valid",
        "answerability": "answerable",
        "abstention_outcome": "answered_answerable",
        "complete_reference_evidence_available": True,
        "best_reference_evidence_precision": 1.0,
        "answer_normalized_exact_match": 1.0,
        "citation_valid": True,
        "citation_f1": 1.0,
    }
    case.update(overrides)
    return case


def test_retrieval_miss_precedes_generation_failure():
    case = _valid_case(
        complete_reference_evidence_available=False,
        answer_normalized_exact_match=0.0,
    )

    assert (
        classify_failure_attribution(case, track="retrieved-context")
        == "retrieval_miss"
    )


def test_aggregate_attribution_keeps_retrieval_noise_secondary():
    case = _valid_case(best_reference_evidence_precision=0.5)
    case["failure_attribution"] = classify_failure_attribution(
        case,
        track="retrieved-context",
    )

    result = aggregate_failure_attribution((case,), track="retrieved-context")

    assert result["primary_outcome_counts"] == {"correct_answer": 1}
    assert result["retrieval_noise_secondary_count"] == 1
