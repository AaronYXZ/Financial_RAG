import json
from pathlib import Path

import pytest

from rag_eval.generation.comparison import (
    compare_evaluated_runs,
    compare_response_validity,
    intersect_eligibility_manifests,
)


MODEL = "test-model"
TRACK = "complete-paper"


def _write_json(path: Path, value):
    path.write_text(json.dumps(value))


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _eligibility(case_ids):
    return {
        "schema_version": 2,
        "track": TRACK,
        "model_id": MODEL,
        "eligible_case_count": len(case_ids),
        "eligible_case_ids": case_ids,
    }


def test_compare_response_validity_reports_matched_rate_delta(tmp_path: Path):
    baseline_eligibility = tmp_path / "baseline.eligibility.json"
    candidate_eligibility = tmp_path / "candidate.eligibility.json"
    _write_json(baseline_eligibility, _eligibility(["q1", "q2"]))
    _write_json(candidate_eligibility, _eligibility(["q1", "q2"]))
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    common = {"track": TRACK, "model_id": MODEL}
    _write_jsonl(
        baseline,
        [
            {"case_id": "q1", **common, "parsed_response": {}, "error": None},
            {
                "case_id": "q2",
                **common,
                "parsed_response": None,
                "error": "confidence must be numeric",
                "error_stage": "response_validation",
            },
        ],
    )
    _write_jsonl(
        candidate,
        [
            {"case_id": "q1", **common, "parsed_response": {}, "error": None},
            {"case_id": "q2", **common, "parsed_response": {}, "error": None},
        ],
    )

    result = compare_response_validity(
        baseline_predictions_file=baseline,
        baseline_eligibility_file=baseline_eligibility,
        candidate_predictions_file=candidate,
        candidate_eligibility_file=candidate_eligibility,
        track=TRACK,
        model_id=MODEL,
    )

    assert result["baseline"]["valid_response_rate"] == 0.5
    assert result["candidate"]["valid_response_rate"] == 1.0
    assert result["delta_candidate_minus_baseline"]["valid_response_rate"] == 0.5
    assert result["delta_candidate_minus_baseline"]["invalid_schema_rate"] == -0.5


def test_compare_response_validity_requires_identical_ordered_cases(tmp_path: Path):
    baseline_eligibility = tmp_path / "baseline.eligibility.json"
    candidate_eligibility = tmp_path / "candidate.eligibility.json"
    _write_json(baseline_eligibility, _eligibility(["q1", "q2"]))
    _write_json(candidate_eligibility, _eligibility(["q2", "q1"]))
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    baseline.write_text("")
    candidate.write_text("")

    with pytest.raises(ValueError, match="identical ordered"):
        compare_response_validity(
            baseline_predictions_file=baseline,
            baseline_eligibility_file=baseline_eligibility,
            candidate_predictions_file=candidate,
            candidate_eligibility_file=candidate_eligibility,
            track=TRACK,
            model_id=MODEL,
        )


def test_intersect_eligibility_preserves_first_manifest_order(
    tmp_path: Path,
):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_payload = _eligibility(["q1", "q2", "q3"])
    first_payload["prompt_version"] = "v3"
    second_payload = {
        **_eligibility(["q3", "q1"]),
        "model_id": "other-model",
        "prompt_version": "v3",
    }
    _write_json(first, first_payload)
    _write_json(second, second_payload)

    result = intersect_eligibility_manifests([first, second])

    assert result["model_id"] == "matched-common"
    assert result["eligible_case_ids"] == ["q1", "q3"]
    assert result["eligible_case_count"] == 2
    assert len(result["source_manifests"]) == 2


def _per_case(case_id: str, paper_id: str, token_f1: float):
    return {
        "case_id": case_id,
        "paper_id": paper_id,
        "track": "oracle-evidence",
        "status": "valid",
        "winning_answer_type": "abstractive",
        "answer_token_f1": token_f1,
        "answer_normalized_exact_match": token_f1,
        "response_mode": "answer",
        "citation_evaluable": True,
        "citation_precision": token_f1,
        "citation_recall": token_f1,
        "citation_f1": token_f1,
        "citation_valid": True,
        "has_citation": True,
        "missing_reference_evidence": False,
        "complete_reference_evidence_available": True,
        "evidence_hit": True,
        "best_reference_evidence_recall": 1.0,
        "best_reference_evidence_precision": 1.0,
        "best_reference_evidence_mrr": 1.0,
        "best_reference_evidence_ndcg": 1.0,
    }


def test_compare_evaluated_runs_bootstraps_paired_paper_differences(
    tmp_path: Path,
):
    baseline = tmp_path / "baseline-per-case.jsonl"
    candidate = tmp_path / "candidate-per-case.jsonl"
    _write_jsonl(
        baseline,
        [_per_case("q1", "p1", 0.0), _per_case("q2", "p2", 0.0)],
    )
    _write_jsonl(
        candidate,
        [_per_case("q1", "p1", 1.0), _per_case("q2", "p2", 1.0)],
    )

    comparison = compare_evaluated_runs(
        baseline_per_case_file=baseline,
        candidate_per_case_file=candidate,
        track="oracle-evidence",
        baseline_label="baseline",
        candidate_label="candidate",
        resamples=200,
        seed=7,
    )

    answer = comparison["metric_differences"]["answer.token_f1"]
    assert comparison["matched_case_count"] == 2
    assert comparison["paper_count"] == 2
    assert answer["candidate_minus_baseline"] == 1.0
    assert answer["confidence_interval_lower"] == 1.0
    assert answer["confidence_interval_upper"] == 1.0
    assert answer["candidate_win_probability"] == 1.0


def test_compare_evaluated_runs_requires_identical_ordered_cases(
    tmp_path: Path,
):
    baseline = tmp_path / "baseline-per-case.jsonl"
    candidate = tmp_path / "candidate-per-case.jsonl"
    _write_jsonl(
        baseline,
        [_per_case("q1", "p1", 0.0), _per_case("q2", "p2", 0.0)],
    )
    _write_jsonl(
        candidate,
        [_per_case("q2", "p2", 1.0), _per_case("q1", "p1", 1.0)],
    )

    with pytest.raises(ValueError, match="identical ordered"):
        compare_evaluated_runs(
            baseline_per_case_file=baseline,
            candidate_per_case_file=candidate,
            track="oracle-evidence",
            baseline_label="baseline",
            candidate_label="candidate",
            resamples=10,
        )
