"""Deterministic attribution across retrieval and generation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def classify_failure_attribution(
    per_case: Mapping[str, Any],
    *,
    track: str,
) -> str:
    """Assign one deterministic primary outcome without semantic judging."""

    status = str(per_case["status"])
    if status in {"request_error", "timeout", "missing_prediction"}:
        return "request_failure"
    if status == "invalid_citation":
        return "citation_failure"
    if status != "valid":
        return "format_failure"

    answerability = per_case["answerability"]
    outcome = per_case["abstention_outcome"]
    if answerability == "ambiguous":
        return "ambiguous_answerability"
    if answerability == "unanswerable":
        return (
            "correct_abstention"
            if outcome == "correct_abstention"
            else "false_answer"
        )
    if (
        track == "retrieved-context"
        and per_case["complete_reference_evidence_available"] is None
    ):
        return "evidence_unavailable_for_attribution"
    if (
        track == "retrieved-context"
        and per_case["complete_reference_evidence_available"] is False
    ):
        return "retrieval_miss"
    if outcome == "false_abstention":
        return "false_abstention"
    if float(per_case["answer_normalized_exact_match"]) < 1.0:
        return "answer_failure_despite_sufficient_evidence"
    if (
        per_case["citation_valid"] is not True
        or per_case["citation_f1"] is None
        or float(per_case["citation_f1"]) < 1.0
    ):
        return "citation_failure"
    return "correct_answer"


def aggregate_failure_attribution(
    per_case: Sequence[Mapping[str, Any]],
    *,
    track: str,
) -> dict[str, Any]:
    """Aggregate primary outcomes and the secondary retrieval-noise flag."""

    counts = Counter(str(item["failure_attribution"]) for item in per_case)
    retrieval_noise_count = sum(
        track == "retrieved-context"
        and item["complete_reference_evidence_available"] is True
        and item["best_reference_evidence_precision"] is not None
        and float(item["best_reference_evidence_precision"]) < 1.0
        for item in per_case
    )
    denominator = len(per_case) or 1
    return {
        "case_count": len(per_case),
        "primary_outcome_counts": dict(sorted(counts.items())),
        "primary_outcome_rates": {
            label: count / denominator for label, count in sorted(counts.items())
        },
        "retrieval_noise_secondary_count": retrieval_noise_count,
        "retrieval_noise_secondary_rate": retrieval_noise_count / denominator,
        "correctness_rule": "normalized_exact_match_equals_1",
        "sufficient_evidence_rule": "any_complete_reference_evidence_set_in_context",
    }
