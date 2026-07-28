"""Matched response-validity comparisons for generation runs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .generation_metrics import (
    EVALUATION_STATUSES,
    classify_prediction_status,
    load_eligibility_manifest,
    load_prediction_rows,
)


def _run_statuses(
    predictions_file: Path,
    eligibility: Mapping[str, Any],
    *,
    track: str,
    model_id: str,
) -> tuple[dict[str, int], set[str | None]]:
    rows = load_prediction_rows(predictions_file, track=track, model_id=model_id)
    counts: Counter[str] = Counter()
    prompt_versions: set[str | None] = set()
    for case_id in eligibility["eligible_case_ids"]:
        candidates = rows.get(case_id, [])
        prediction = candidates[-1] if candidates else None
        counts[classify_prediction_status(prediction)] += 1
        if prediction is not None:
            prompt_versions.add(prediction.get("prompt_version"))
    return (
        {status: counts[status] for status in EVALUATION_STATUSES},
        prompt_versions,
    )


def compare_response_validity(
    *,
    baseline_predictions_file: Path,
    baseline_eligibility_file: Path,
    candidate_predictions_file: Path,
    candidate_eligibility_file: Path,
    track: str,
    model_id: str,
) -> dict[str, Any]:
    """Compare response statuses only when both runs use identical eligible cases."""

    baseline_eligibility = load_eligibility_manifest(baseline_eligibility_file)
    candidate_eligibility = load_eligibility_manifest(candidate_eligibility_file)
    baseline_ids = baseline_eligibility["eligible_case_ids"]
    candidate_ids = candidate_eligibility["eligible_case_ids"]
    if baseline_ids != candidate_ids:
        raise ValueError("Comparison runs must have identical ordered eligible case IDs")
    if baseline_eligibility["track"] != track or candidate_eligibility["track"] != track:
        raise ValueError("Comparison eligibility track does not match requested track")
    if (
        baseline_eligibility["model_id"] != model_id
        or candidate_eligibility["model_id"] != model_id
    ):
        raise ValueError("Comparison eligibility model does not match requested model")

    baseline_counts, baseline_prompts = _run_statuses(
        baseline_predictions_file,
        baseline_eligibility,
        track=track,
        model_id=model_id,
    )
    candidate_counts, candidate_prompts = _run_statuses(
        candidate_predictions_file,
        candidate_eligibility,
        track=track,
        model_id=model_id,
    )
    count = len(baseline_ids)

    def summarize(
        predictions_file: Path,
        eligibility_file: Path,
        counts: Mapping[str, int],
        prompt_versions: set[str | None],
    ) -> dict[str, Any]:
        denominator = count or 1
        return {
            "predictions_file": str(predictions_file),
            "eligibility_file": str(eligibility_file),
            "prompt_versions": sorted(
                ("unversioned" if item is None else item) for item in prompt_versions
            ),
            "status_counts": dict(counts),
            "status_rates": {
                status: counts[status] / denominator for status in EVALUATION_STATUSES
            },
            "valid_response_rate": counts["valid"] / denominator,
            "invalid_schema_rate": counts["invalid_schema"] / denominator,
        }

    baseline = summarize(
        baseline_predictions_file,
        baseline_eligibility_file,
        baseline_counts,
        baseline_prompts,
    )
    candidate = summarize(
        candidate_predictions_file,
        candidate_eligibility_file,
        candidate_counts,
        candidate_prompts,
    )
    return {
        "schema_version": 1,
        "comparison": "matched_response_validity",
        "track": track,
        "model_id": model_id,
        "eligible_case_count": count,
        "eligible_case_ids_match": True,
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_baseline": {
            "valid_response_rate": (
                candidate["valid_response_rate"] - baseline["valid_response_rate"]
            ),
            "invalid_schema_rate": (
                candidate["invalid_schema_rate"] - baseline["invalid_schema_rate"]
            ),
        },
    }


def write_comparison(path: Path, comparison: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
