import json
from pathlib import Path

import pytest

from rag_eval.generation_comparison import compare_response_validity


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
