"""Matched response-validity comparisons for generation runs."""

from __future__ import annotations

import json
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .generation_metrics import (
    BOOTSTRAP_CONFIDENCE_LEVEL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EVALUATION_STATUSES,
    bootstrap_point_estimates,
    classify_prediction_status,
    load_eligibility_manifest,
    load_prediction_rows,
)


LOWER_IS_BETTER = {
    "abstention.false_abstention_rate",
    "abstention.false_answer_rate",
    "confidence.area_under_risk_coverage_curve",
    "confidence.expected_calibration_error",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def intersect_eligibility_manifests(
    paths: Sequence[Path],
) -> dict[str, Any]:
    """Create an ordered common-case manifest for matched system comparisons."""

    if len(paths) < 2:
        raise ValueError("At least two eligibility manifests are required")
    manifests = [load_eligibility_manifest(path) for path in paths]
    tracks = {manifest["track"] for manifest in manifests}
    if len(tracks) != 1:
        raise ValueError("Eligibility manifests must use the same track")
    prompt_versions = {manifest.get("prompt_version") for manifest in manifests}
    if len(prompt_versions) != 1:
        raise ValueError("Eligibility manifests must use the same prompt version")

    common_ids = set(manifests[0]["eligible_case_ids"])
    for manifest in manifests[1:]:
        common_ids &= set(manifest["eligible_case_ids"])
    ordered_ids = [
        case_id
        for case_id in manifests[0]["eligible_case_ids"]
        if case_id in common_ids
    ]
    return {
        "schema_version": 3,
        "track": manifests[0]["track"],
        "model_id": "matched-common",
        "prompt_version": manifests[0].get("prompt_version"),
        "eligible_case_count": len(ordered_ids),
        "eligible_case_ids": ordered_ids,
        "source_manifests": [
            {
                "path": str(path),
                "sha256": _file_sha256(path),
                "model_id": manifest["model_id"],
                "eligible_case_count": manifest["eligible_case_count"],
            }
            for path, manifest in zip(paths, manifests, strict=True)
        ],
        "intersection_policy": "ordered_intersection_preserving_first_manifest_order",
    }


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


def _load_per_case_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"Missing case_id at {path}:{line_number}")
            if case_id in seen:
                raise ValueError(f"Duplicate case ID {case_id!r} in {path}")
            seen.add(case_id)
            rows.append(row)
    return rows


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile without values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compare_evaluated_runs(
    *,
    baseline_per_case_file: Path,
    candidate_per_case_file: Path,
    track: str,
    baseline_label: str,
    candidate_label: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Compare matched evaluated runs with paired paper-clustered bootstrap."""

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    baseline = _load_per_case_metrics(baseline_per_case_file)
    candidate = _load_per_case_metrics(candidate_per_case_file)
    baseline_ids = [row["case_id"] for row in baseline]
    candidate_ids = [row["case_id"] for row in candidate]
    if baseline_ids != candidate_ids:
        raise ValueError("Comparison runs must have identical ordered case IDs")
    if any(row.get("track") != track for row in [*baseline, *candidate]):
        raise ValueError("Per-case metric track does not match requested track")
    for baseline_row, candidate_row in zip(baseline, candidate, strict=True):
        if baseline_row.get("paper_id") != candidate_row.get("paper_id"):
            raise ValueError(
                f"Paper mismatch for case {baseline_row['case_id']!r}"
            )

    baseline_points = bootstrap_point_estimates(baseline, track=track)
    candidate_points = bootstrap_point_estimates(candidate, track=track)
    metrics = sorted(set(baseline_points) & set(candidate_points))
    point_differences = {
        metric: candidate_points[metric] - baseline_points[metric]
        for metric in metrics
    }

    baseline_by_paper: dict[str, list[dict[str, Any]]] = {}
    candidate_by_paper: dict[str, list[dict[str, Any]]] = {}
    for baseline_row, candidate_row in zip(baseline, candidate, strict=True):
        paper_id = str(baseline_row["paper_id"])
        baseline_by_paper.setdefault(paper_id, []).append(baseline_row)
        candidate_by_paper.setdefault(paper_id, []).append(candidate_row)

    paper_ids = sorted(baseline_by_paper)
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    generator = random.Random(seed)
    for _ in range(resamples):
        baseline_sample: list[dict[str, Any]] = []
        candidate_sample: list[dict[str, Any]] = []
        for paper_id in generator.choices(paper_ids, k=len(paper_ids)):
            baseline_sample.extend(baseline_by_paper[paper_id])
            candidate_sample.extend(candidate_by_paper[paper_id])
        baseline_replicate = bootstrap_point_estimates(
            baseline_sample,
            track=track,
        )
        candidate_replicate = bootstrap_point_estimates(
            candidate_sample,
            track=track,
        )
        for metric in metrics:
            if metric in baseline_replicate and metric in candidate_replicate:
                samples[metric].append(
                    candidate_replicate[metric] - baseline_replicate[metric]
                )

    alpha = (1.0 - confidence_level) / 2.0
    differences: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        values = samples[metric]
        lower_is_better = metric in LOWER_IS_BETTER
        wins = sum(
            value < 0.0 if lower_is_better else value > 0.0
            for value in values
        )
        ties = sum(value == 0.0 for value in values)
        differences[metric] = {
            "baseline": baseline_points[metric],
            "candidate": candidate_points[metric],
            "candidate_minus_baseline": point_differences[metric],
            "lower_is_better": lower_is_better,
            "confidence_interval_lower": (
                _percentile(values, alpha) if values else None
            ),
            "confidence_interval_upper": (
                _percentile(values, 1.0 - alpha) if values else None
            ),
            "candidate_win_probability": (
                (wins + 0.5 * ties) / len(values) if values else None
            ),
            "valid_resample_count": len(values),
        }

    return {
        "schema_version": 1,
        "comparison": "paired_paper_clustered_metric_difference",
        "track": track,
        "baseline": {
            "label": baseline_label,
            "per_case_metrics_file": str(baseline_per_case_file),
        },
        "candidate": {
            "label": candidate_label,
            "per_case_metrics_file": str(candidate_per_case_file),
        },
        "matched_case_count": len(baseline),
        "paper_count": len(paper_ids),
        "case_ids_match": True,
        "bootstrap": {
            "cluster_unit": "paper_id",
            "resamples": resamples,
            "seed": seed,
            "confidence_level": confidence_level,
        },
        "metric_differences": differences,
    }
