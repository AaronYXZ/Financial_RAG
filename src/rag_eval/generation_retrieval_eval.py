"""Evaluate frozen QASPER retrieval contexts against oracle evidence before generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .generation_metrics import (
    aggregate_evidence_availability,
    load_generation_cases,
    score_ranked_evidence_ids,
)
from .generation_retrieval import (
    file_sha256,
    frozen_contexts_by_case,
    load_frozen_context_manifest,
)


SELECTION_METRICS = (
    "complete_reference_set_rate",
    "best_reference_recall",
    "hit_rate",
    "best_reference_ndcg",
    "best_reference_mrr",
    "best_reference_precision",
)


def _score_context(
    context_ids: Sequence[str],
    *,
    reference_sets: Sequence[set[str]],
) -> dict[str, Any]:
    return score_ranked_evidence_ids(tuple(context_ids), reference_sets)


def compare_retrieval_to_oracle(
    *,
    cases_file: Path,
    context_manifest_files: Sequence[Path],
) -> dict[str, Any]:
    """Rank retrieval configurations using evidence availability before generation."""

    if not context_manifest_files:
        raise ValueError("At least one frozen context manifest is required")

    cases = load_generation_cases(cases_file)
    cases_sha256 = file_sha256(cases_file)
    evaluations: list[dict[str, Any]] = []
    expected_case_ids: list[str] | None = None

    for path in context_manifest_files:
        manifest = load_frozen_context_manifest(path)
        if manifest["cases_sha256"] != cases_sha256:
            raise ValueError(f"Cases checksum mismatch for {path}")
        case_ids = list(manifest["eligible_case_ids"])
        if expected_case_ids is None:
            expected_case_ids = case_ids
        elif case_ids != expected_case_ids:
            raise ValueError(
                "Retrieval manifests must contain identical ordered eligible case IDs"
            )

        retrieved_contexts = frozen_contexts_by_case(manifest)
        retrieved_rows: list[dict[str, Any]] = []
        oracle_rows: list[dict[str, Any]] = []
        for case_id in case_ids:
            case = cases.get(case_id)
            if case is None:
                raise ValueError(f"Case {case_id!r} is missing from {cases_file}")
            reference_sets = [
                set(reference.evidence_ids)
                for reference in case.references
                if not reference.unanswerable and reference.evidence_ids
            ]
            retrieved_rows.append(
                _score_context(
                    retrieved_contexts[case_id],
                    reference_sets=reference_sets,
                )
            )
            oracle_rows.append(
                _score_context(
                    case.oracle_passage_ids,
                    reference_sets=reference_sets,
                )
            )

        retrieved = aggregate_evidence_availability(retrieved_rows)
        oracle = aggregate_evidence_availability(oracle_rows)
        gap = {
            metric: (
                float(retrieved[metric]) - float(oracle[metric])
                if retrieved[metric] is not None and oracle[metric] is not None
                else None
            )
            for metric in SELECTION_METRICS
        }
        evaluations.append(
            {
                "context_manifest": str(path),
                "context_manifest_sha256": file_sha256(path),
                "retriever": manifest["retriever"],
                "retrieval_scope": manifest["retrieval_scope"],
                "top_k": manifest["retriever"]["parameters"]["top_k"],
                "retrieved": retrieved,
                "oracle": oracle,
                "retrieved_minus_oracle": gap,
            }
        )

    ranked = sorted(
        evaluations,
        key=lambda item: tuple(
            -(float(item["retrieved"][metric]) if item["retrieved"][metric] is not None else -1.0)
            for metric in SELECTION_METRICS
        ),
    )
    return {
        "schema_version": 1,
        "comparison": "retrieval_evidence_availability_vs_oracle",
        "cases_file": str(cases_file),
        "cases_sha256": cases_sha256,
        "eligible_case_count": len(expected_case_ids or []),
        "selection_policy": {
            "method": "descending_lexicographic",
            "metrics": list(SELECTION_METRICS),
            "generation_metrics_used": False,
        },
        "selected_context_manifest": ranked[0]["context_manifest"],
        "ranked_configurations": ranked,
    }
