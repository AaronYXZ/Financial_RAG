"""Deterministic evidence-availability metrics for ranked passages."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def score_ranked_evidence_ids(
    context_ids: Sequence[str],
    reference_sets: Sequence[set[str]],
) -> dict[str, Any]:
    """Score ranked passage IDs against the best complete human evidence set."""

    context_set = set(context_ids)
    reference_scores = []
    for reference_ids in reference_sets:
        relevant_ranks = [
            rank
            for rank, passage_id in enumerate(context_ids, start=1)
            if passage_id in reference_ids
        ]
        overlap = len(reference_ids & context_set)
        dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
        ideal_count = min(len(reference_ids), len(context_ids))
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_count + 1)
        )
        reference_scores.append(
            {
                "hit": bool(relevant_ranks),
                "recall": overlap / len(reference_ids),
                "precision": overlap / len(context_set) if context_set else 0.0,
                "mrr": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
                "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
                "complete": overlap == len(reference_ids),
            }
        )
    winning = (
        max(
            reference_scores,
            key=lambda item: (
                item["complete"],
                item["recall"],
                item["ndcg"],
                item["precision"],
            ),
        )
        if reference_scores
        else None
    )
    return {
        "context_passage_count": len(context_ids),
        "reference_evidence_set_count": len(reference_sets),
        "evidence_hit": winning["hit"] if winning else None,
        "best_reference_evidence_recall": winning["recall"] if winning else None,
        "best_reference_evidence_precision": (
            winning["precision"] if winning else None
        ),
        "best_reference_evidence_mrr": winning["mrr"] if winning else None,
        "best_reference_evidence_ndcg": winning["ndcg"] if winning else None,
        "complete_reference_evidence_available": (
            winning["complete"] if winning else None
        ),
    }


def aggregate_evidence_availability(
    per_case: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate passage-level evidence scores over evaluable cases."""

    evaluable = [
        item
        for item in per_case
        if item["complete_reference_evidence_available"] is not None
    ]

    def mean(field: str) -> float | None:
        return (
            sum(float(item[field]) for item in evaluable) / len(evaluable)
            if evaluable
            else None
        )

    return {
        "case_count": len(per_case),
        "evaluable_answerable_case_count": len(evaluable),
        "hit_rate": (
            sum(item["evidence_hit"] is True for item in evaluable)
            / len(evaluable)
            if evaluable
            else None
        ),
        "best_reference_recall": mean("best_reference_evidence_recall"),
        "best_reference_precision": mean("best_reference_evidence_precision"),
        "best_reference_mrr": mean("best_reference_evidence_mrr"),
        "best_reference_ndcg": mean("best_reference_evidence_ndcg"),
        "complete_reference_set_rate": (
            sum(
                item["complete_reference_evidence_available"] is True
                for item in evaluable
            )
            / len(evaluable)
            if evaluable
            else None
        ),
    }
