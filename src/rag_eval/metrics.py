"""Retrieval metrics with BEIR-compatible names."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Mapping[str, float]]


def evaluate(
    qrels: Qrels,
    run: Run,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Compute macro-averaged Precision, Recall, MRR, and nDCG at each cutoff."""
    if not qrels:
        raise ValueError("qrels must contain at least one query")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")

    totals = {
        f"{metric}@{k}": 0.0
        for k in k_values
        for metric in ("Precision", "Recall", "MRR", "NDCG")
    }
    for query_id, judgments in qrels.items():
        ranked = sorted(
            run.get(query_id, {}).items(),
            key=lambda item: (-item[1], item[0]),
        )
        positive_count = sum(relevance > 0 for relevance in judgments.values())
        for k in k_values:
            top_ids = [doc_id for doc_id, _ in ranked[:k]]
            relevant_retrieved = sum(judgments.get(doc_id, 0) > 0 for doc_id in top_ids)
            totals[f"Precision@{k}"] += relevant_retrieved / k
            totals[f"Recall@{k}"] += relevant_retrieved / positive_count if positive_count else 0.0

            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(top_ids, start=1):
                if judgments.get(doc_id, 0) > 0:
                    reciprocal_rank = 1 / rank
                    break
            totals[f"MRR@{k}"] += reciprocal_rank

            dcg = _dcg([judgments.get(doc_id, 0) for doc_id in top_ids])
            ideal = sorted(judgments.values(), reverse=True)[:k]
            ideal_dcg = _dcg(ideal)
            totals[f"NDCG@{k}"] += dcg / ideal_dcg if ideal_dcg else 0.0

    query_count = len(qrels)
    return {name: round(value / query_count, 5) for name, value in totals.items()}


def _dcg(relevances: Sequence[int]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )
