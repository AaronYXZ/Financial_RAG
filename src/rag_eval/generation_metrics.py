"""Stage 0-6 evaluation for fixed-context QASPER generation runs."""

from __future__ import annotations

import json
import math
import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .generation_data import GenerationCase, ReferenceAnswer, load_generation_cases
from .retrieval.metrics import (
    aggregate_evidence_availability,
    score_ranked_evidence_ids,
)


EVALUATION_STATUSES = (
    "valid",
    "invalid_json",
    "invalid_schema",
    "invalid_citation",
    "request_error",
    "timeout",
    "missing_prediction",
)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
ANSWERABILITY_TRACKS = ("complete-paper", "retrieved-context")


@dataclass(frozen=True)
class EvaluationRecord:
    case: GenerationCase
    track: str
    model_id: str
    status: str
    prediction: Mapping[str, Any] | None
    duplicate_prediction_count: int

    def to_dict(self) -> dict[str, Any]:
        prediction = self.prediction or {}
        return {
            "case_id": self.case.case_id,
            "paper_id": self.case.paper_id,
            "split": self.case.split,
            "track": self.track,
            "model_id": self.model_id,
            "prompt_version": prediction.get("prompt_version"),
            "answerability": self.case.answerability,
            "status": self.status,
            "prediction_present": self.prediction is not None,
            "duplicate_prediction_count": self.duplicate_prediction_count,
            "parsed_response": prediction.get("parsed_response"),
            "error": prediction.get("error"),
            "error_stage": prediction.get("error_stage"),
            "error_type": prediction.get("error_type"),
            "attempts": prediction.get("attempts"),
            "counted_input_tokens": prediction.get("counted_input_tokens"),
            "server_input_tokens": prediction.get("server_input_tokens"),
            "server_output_tokens": prediction.get("server_output_tokens"),
            "latency_seconds": prediction.get("latency_seconds"),
        }


def load_eligibility_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "track",
        "model_id",
        "eligible_case_count",
        "eligible_case_ids",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Eligibility manifest is missing keys: {missing}")
    case_ids = payload["eligible_case_ids"]
    if not isinstance(case_ids, list) or not all(isinstance(item, str) for item in case_ids):
        raise ValueError("eligible_case_ids must be a list of strings")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Eligibility manifest contains duplicate case IDs")
    if payload["eligible_case_count"] != len(case_ids):
        raise ValueError("eligible_case_count does not match eligible_case_ids")
    return payload


def load_prediction_rows(
    path: Path,
    *,
    track: str,
    model_id: str,
    prompt_version: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("track") != track or row.get("model_id") != model_id:
                continue
            if prompt_version is not None and row.get("prompt_version") != prompt_version:
                continue
            grouped[row["case_id"]].append(row)
    return dict(grouped)


def classify_prediction_status(prediction: Mapping[str, Any] | None) -> str:
    if prediction is None:
        return "missing_prediction"
    if prediction.get("error") is None and isinstance(prediction.get("parsed_response"), Mapping):
        return "valid"

    message = str(prediction.get("error") or "").lower()
    stage = prediction.get("error_stage")
    if stage == "generation":
        return "timeout" if "timed out" in message or "timeout" in message else "request_error"
    if "generation request failed" in message:
        return "timeout" if "timed out" in message or "timeout" in message else "request_error"
    if "not a valid json object" in message:
        return "invalid_json"
    if "unknown citation ids" in message:
        return "invalid_citation"
    if stage == "response_validation" or prediction.get("parsed_response") is None:
        return "invalid_schema"
    return "invalid_schema"


def build_evaluation_records(
    cases: Mapping[str, GenerationCase],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    eligibility: Mapping[str, Any],
    *,
    track: str,
    model_id: str,
) -> list[EvaluationRecord]:
    if eligibility["track"] != track:
        raise ValueError("Eligibility track does not match requested track")
    if eligibility["model_id"] != model_id:
        raise ValueError("Eligibility model_id does not match requested model_id")

    records: list[EvaluationRecord] = []
    for case_id in eligibility["eligible_case_ids"]:
        if case_id not in cases:
            raise ValueError(f"Eligible case {case_id!r} is missing from the case file")
        rows = list(predictions.get(case_id, ()))
        prediction = rows[-1] if rows else None
        records.append(
            EvaluationRecord(
                case=cases[case_id],
                track=track,
                model_id=model_id,
                status=classify_prediction_status(prediction),
                prediction=prediction,
                duplicate_prediction_count=max(0, len(rows) - 1),
            )
        )
    return records


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def numeric_summary(values: Iterable[int | float | None]) -> dict[str, int | float | None]:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    if not numbers:
        return {
            "count": 0,
            "sum": None,
            "mean": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": len(numbers),
        "sum": sum(numbers),
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "p50": _percentile(numbers, 0.50),
        "p95": _percentile(numbers, 0.95),
        "p99": _percentile(numbers, 0.99),
    }


def reliability_and_efficiency(records: Sequence[EvaluationRecord]) -> dict[str, Any]:
    eligible_count = len(records)
    status_counts = Counter(record.status for record in records)
    attempted = [record for record in records if record.prediction is not None]
    predictions = [record.prediction or {} for record in attempted]
    attempts = [int(row.get("attempts") or 1) for row in predictions]
    total_tokens = [
        row["server_input_tokens"] + row["server_output_tokens"]
        for row in predictions
        if isinstance(row.get("server_input_tokens"), (int, float))
        and isinstance(row.get("server_output_tokens"), (int, float))
    ]
    denominator = eligible_count or 1
    attempted_denominator = len(attempted) or 1
    providers = {
        str(row.get("provider") or "local")
        for row in predictions
    }
    openai_run = "openai" in providers
    return {
        "eligible_case_count": eligible_count,
        "attempted_case_count": len(attempted),
        "status_counts": {status: status_counts.get(status, 0) for status in EVALUATION_STATUSES},
        "status_rates": {
            status: status_counts.get(status, 0) / denominator for status in EVALUATION_STATUSES
        },
        "valid_response_rate": status_counts.get("valid", 0) / denominator,
        "retry_case_count": sum(value > 1 for value in attempts),
        "retry_rate": sum(value > 1 for value in attempts) / attempted_denominator,
        "attempts": numeric_summary(attempts),
        "latency_seconds": numeric_summary(row.get("latency_seconds") for row in predictions),
        "server_input_tokens": numeric_summary(
            row.get("server_input_tokens") for row in predictions
        ),
        "server_output_tokens": numeric_summary(
            row.get("server_output_tokens") for row in predictions
        ),
        "total_tokens": numeric_summary(total_tokens),
        "estimated_cost_usd": {
            "pricing_basis": (
                "unavailable_without_frozen_provider_price_table"
                if openai_run
                else "local_inference"
            ),
            "observed_total": None if openai_run else 0.0,
            "per_attempted_case": None if openai_run else 0.0,
            "per_1000_attempted_cases": None if openai_run else 0.0,
        },
    }


def normalize_answer(text: str) -> str:
    """Apply the official QASPER/SQuAD answer normalization."""

    lowered = text.lower()
    without_punctuation = "".join(char for char in lowered if char not in set(string.punctuation))
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def token_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def normalized_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def reference_answer(reference: ReferenceAnswer) -> tuple[str, str]:
    if reference.unanswerable:
        return "Unanswerable", "none"
    if reference.answer_type == "extractive":
        return ", ".join(reference.extractive_spans), "extractive"
    if reference.answer_type == "free_form":
        return reference.text, "abstractive"
    if reference.answer_type == "yes_no":
        return reference.text, "boolean"
    raise ValueError(f"Unsupported reference answer type: {reference.answer_type}")


def score_answer_record(record: EvaluationRecord) -> dict[str, Any]:
    response = (record.prediction or {}).get("parsed_response")
    if record.status == "valid" and isinstance(response, Mapping):
        candidate = "Unanswerable" if response["abstain"] else str(response["answer"])
    else:
        candidate = ""

    reference_scores = []
    for reference in record.case.references:
        answer, answer_type = reference_answer(reference)
        reference_scores.append(
            {
                "annotation_id": reference.annotation_id,
                "answer_type": answer_type,
                "token_f1": token_f1_score(candidate, answer),
                "normalized_exact_match": normalized_exact_match(candidate, answer),
            }
        )
    if not reference_scores:
        raise ValueError(f"Case {record.case.case_id} has no reference answers")
    winning = max(reference_scores, key=lambda item: item["token_f1"])
    citation_score = score_citation_record(record)
    abstention_score = score_abstention_record(record)
    confidence_score = score_confidence_record(
        record,
        answer_token_f1=float(winning["token_f1"]),
    )
    evidence_score = score_evidence_availability(record)
    result = {
        **citation_score,
        **abstention_score,
        **confidence_score,
        **evidence_score,
        "case_id": record.case.case_id,
        "paper_id": record.case.paper_id,
        "track": record.track,
        "model_id": record.model_id,
        "status": record.status,
        "answerability": record.case.answerability,
        "candidate_answer": candidate,
        "answer_token_f1": winning["token_f1"],
        "answer_normalized_exact_match": max(
            item["normalized_exact_match"] for item in reference_scores
        ),
        "winning_annotation_id": winning["annotation_id"],
        "winning_answer_type": winning["answer_type"],
        "reference_scores": reference_scores,
    }
    result["failure_attribution"] = classify_failure_attribution(
        result,
        track=record.track,
    )
    return result


def score_evidence_availability(record: EvaluationRecord) -> dict[str, Any]:
    """Measure deterministic gold-evidence availability in the supplied context."""

    prediction = record.prediction or {}
    raw_context_ids = prediction.get("context_passage_ids")
    if isinstance(raw_context_ids, list):
        context_ids = tuple(str(item) for item in raw_context_ids)
    elif record.track == "oracle-evidence":
        context_ids = record.case.oracle_passage_ids
    elif record.track == "complete-paper":
        context_ids = tuple(
            passage.passage_id for passage in record.case.paper_passages
        )
    else:
        context_ids = ()

    reference_sets = [
        set(reference.evidence_ids)
        for reference in record.case.references
        if not reference.unanswerable and reference.evidence_ids
    ]
    return score_ranked_evidence_ids(context_ids, reference_sets)


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


def aggregate_answer_quality(per_case: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not per_case:
        return {
            "case_count": 0,
            "token_f1": 0.0,
            "normalized_exact_match": 0.0,
            "by_answer_type": {},
        }

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in per_case:
        grouped[str(item["winning_answer_type"])].append(item)

    def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
        return {
            "case_count": len(items),
            "token_f1": sum(float(item["answer_token_f1"]) for item in items) / len(items),
            "normalized_exact_match": sum(
                float(item["answer_normalized_exact_match"]) for item in items
            )
            / len(items),
        }

    summary = summarize(per_case)
    summary["by_answer_type"] = {
        answer_type: summarize(items) for answer_type, items in sorted(grouped.items())
    }
    return summary


def evaluate_prediction_files(
    *,
    cases_file: Path,
    predictions_file: Path,
    eligibility_file: Path,
    output_dir: Path,
    track: str,
    model_id: str,
) -> dict[str, Any]:
    cases = load_generation_cases(cases_file)
    eligibility = load_eligibility_manifest(eligibility_file)
    predictions = load_prediction_rows(
        predictions_file,
        track=track,
        model_id=model_id,
        prompt_version=eligibility.get("prompt_version"),
    )
    records = build_evaluation_records(
        cases,
        predictions,
        eligibility,
        track=track,
        model_id=model_id,
    )
    per_case = [score_answer_record(record) for record in records]
    summary = {
        "schema_version": 1,
        "track": track,
        "model_id": model_id,
        "prompt_version": eligibility.get("prompt_version"),
        "reliability_and_efficiency": reliability_and_efficiency(records),
        "answer_quality": aggregate_answer_quality(per_case),
        "citation_quality": aggregate_citation_quality(per_case),
        "evidence_availability": aggregate_evidence_availability(per_case),
        "abstention_quality": aggregate_abstention_quality(per_case, track=track),
        "confidence_and_calibration": aggregate_confidence_and_calibration(
            per_case,
            track=track,
        ),
        "failure_attribution": aggregate_failure_attribution(
            per_case,
            track=track,
        ),
        "paper_clustered_bootstrap": paper_clustered_bootstrap_intervals(
            per_case,
            track=track,
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "evaluation_records.jsonl", (record.to_dict() for record in records))
    _write_jsonl(output_dir / "per_case_metrics.jsonl", per_case)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def citation_precision_recall_f1(
    predicted_ids: Iterable[str],
    reference_ids: Iterable[str],
) -> dict[str, float]:
    predicted = set(predicted_ids)
    reference = set(reference_ids)
    if not reference:
        raise ValueError("Citation scoring requires at least one reference evidence ID")
    overlap = len(predicted & reference)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(reference)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
    }


def score_citation_record(record: EvaluationRecord) -> dict[str, Any]:
    result: dict[str, Any] = {
        "response_mode": "invalid",
        "citation_evaluable": False,
        "citation_valid": False if record.status == "invalid_citation" else None,
        "has_citation": None,
        "citation_precision": None,
        "citation_recall": None,
        "citation_f1": None,
        "winning_citation_annotation_id": None,
        "reference_evidence_set_count": 0,
        "missing_reference_evidence": False,
        "citation_reference_scores": [],
    }
    response = (record.prediction or {}).get("parsed_response")
    if record.status != "valid" or not isinstance(response, Mapping):
        return result
    if response["abstain"]:
        result.update(
            {
                "response_mode": "abstain",
                "citation_valid": None,
                "has_citation": False,
            }
        )
        return result

    predicted_ids = tuple(dict.fromkeys(response["citations"]))
    context_ids = set((record.prediction or {}).get("context_passage_ids") or ())
    result.update(
        {
            "response_mode": "answer",
            "citation_valid": all(item in context_ids for item in predicted_ids),
            "has_citation": bool(predicted_ids),
        }
    )

    reference_scores: list[dict[str, Any]] = []
    for reference in record.case.references:
        if not reference.evidence_ids:
            continue
        reference_scores.append(
            {
                "annotation_id": reference.annotation_id,
                "reference_evidence_ids": list(reference.evidence_ids),
                **citation_precision_recall_f1(
                    predicted_ids,
                    reference.evidence_ids,
                ),
            }
        )

    result["reference_evidence_set_count"] = len(reference_scores)
    result["citation_reference_scores"] = reference_scores
    if not reference_scores:
        result["missing_reference_evidence"] = True
        return result

    winning = max(
        reference_scores,
        key=lambda item: (
            item["citation_f1"],
            item["citation_recall"],
            item["citation_precision"],
        ),
    )
    result.update(
        {
            "citation_evaluable": True,
            "citation_precision": winning["citation_precision"],
            "citation_recall": winning["citation_recall"],
            "citation_f1": winning["citation_f1"],
            "winning_citation_annotation_id": winning["annotation_id"],
        }
    )
    return result


def aggregate_citation_quality(
    per_case: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    answered = [item for item in per_case if item["response_mode"] == "answer"]
    abstained = [item for item in per_case if item["response_mode"] == "abstain"]
    scorable = [item for item in answered if item["citation_evaluable"]]
    invalid_citation_count = sum(
        item["status"] == "invalid_citation" for item in per_case
    )
    validity_evaluable_count = len(answered) + invalid_citation_count
    valid_citation_count = sum(item["citation_valid"] is True for item in answered)
    answered_with_citation_count = sum(item["has_citation"] is True for item in answered)

    def mean(field: str) -> float | None:
        if not scorable:
            return None
        return sum(float(item[field]) for item in scorable) / len(scorable)

    return {
        "case_count": len(per_case),
        "answered_case_count": len(answered),
        "abstained_case_count": len(abstained),
        "invalid_or_missing_response_count": len(per_case) - len(answered) - len(abstained),
        "scorable_answered_case_count": len(scorable),
        "missing_reference_evidence_case_count": sum(
            item["missing_reference_evidence"] for item in answered
        ),
        "invalid_citation_case_count": invalid_citation_count,
        "citation_validity_evaluable_count": validity_evaluable_count,
        "citation_validity_rate": (
            valid_citation_count / validity_evaluable_count
            if validity_evaluable_count
            else None
        ),
        "answered_with_citation_count": answered_with_citation_count,
        "answered_with_citation_rate": (
            answered_with_citation_count / len(answered) if answered else None
        ),
        "citation_precision": mean("citation_precision"),
        "citation_recall": mean("citation_recall"),
        "citation_f1": mean("citation_f1"),
    }


def score_abstention_record(record: EvaluationRecord) -> dict[str, Any]:
    expected_abstain = {
        "answerable": False,
        "unanswerable": True,
        "ambiguous": None,
    }.get(record.case.answerability)
    if record.track not in ANSWERABILITY_TRACKS:
        return {
            "abstention_primary_eligible": False,
            "expected_abstain": expected_abstain,
            "predicted_abstain": None,
            "abstention_outcome": "not_applicable",
        }

    response = (record.prediction or {}).get("parsed_response")
    predicted_abstain = (
        bool(response["abstain"])
        if record.status == "valid" and isinstance(response, Mapping)
        else None
    )
    if expected_abstain is None:
        ambiguous_outcome = {
            True: "ambiguous_abstain",
            False: "ambiguous_answer",
            None: "ambiguous_no_decision",
        }
        return {
            "abstention_primary_eligible": False,
            "expected_abstain": None,
            "predicted_abstain": predicted_abstain,
            "abstention_outcome": ambiguous_outcome[predicted_abstain],
        }

    if predicted_abstain is None:
        outcome = "no_decision"
    elif expected_abstain and predicted_abstain:
        outcome = "correct_abstention"
    elif expected_abstain:
        outcome = "false_answer"
    elif predicted_abstain:
        outcome = "false_abstention"
    else:
        outcome = "correct_answer"
    return {
        "abstention_primary_eligible": True,
        "expected_abstain": expected_abstain,
        "predicted_abstain": predicted_abstain,
        "abstention_outcome": outcome,
    }


def aggregate_abstention_quality(
    per_case: Sequence[Mapping[str, Any]],
    *,
    track: str,
) -> dict[str, Any]:
    if track not in ANSWERABILITY_TRACKS:
        return {
            "applicable": False,
            "reason": "Abstention metrics require a non-oracle context track",
            "case_count": len(per_case),
        }

    primary = [item for item in per_case if item["abstention_primary_eligible"]]
    ambiguous = [
        item
        for item in per_case
        if item["answerability"] == "ambiguous"
    ]
    answerable = [item for item in primary if item["expected_abstain"] is False]
    unanswerable = [item for item in primary if item["expected_abstain"] is True]

    outcomes = Counter(item["abstention_outcome"] for item in primary)
    true_positive = outcomes["correct_abstention"]
    false_positive = outcomes["false_abstention"]
    true_negative = outcomes["correct_answer"]
    explicit_false_answer = outcomes["false_answer"]
    no_decision_unanswerable = sum(
        item["abstention_outcome"] == "no_decision" for item in unanswerable
    )
    no_decision_answerable = sum(
        item["abstention_outcome"] == "no_decision" for item in answerable
    )
    false_negative = explicit_false_answer + no_decision_unanswerable

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    recall = true_positive / len(unanswerable) if unanswerable else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0 if precision is not None and recall is not None else None
    )
    ambiguous_outcomes = Counter(
        item["abstention_outcome"] for item in ambiguous
    )

    return {
        "applicable": True,
        "case_count": len(per_case),
        "primary_case_count": len(primary),
        "ambiguous_case_count": len(ambiguous),
        "answerable_case_count": len(answerable),
        "unanswerable_case_count": len(unanswerable),
        "valid_decision_count": sum(
            item["predicted_abstain"] is not None for item in primary
        ),
        "no_decision_count": outcomes["no_decision"],
        "no_decision_rate": (
            outcomes["no_decision"] / len(primary) if primary else None
        ),
        "answerability_accuracy": (
            (true_positive + true_negative) / len(primary) if primary else None
        ),
        "abstention_precision": precision,
        "abstention_recall": recall,
        "abstention_f1": f1,
        "false_answer_count": explicit_false_answer,
        "false_answer_rate": (
            explicit_false_answer / len(unanswerable) if unanswerable else None
        ),
        "false_abstention_count": false_positive,
        "false_abstention_rate": (
            false_positive / len(answerable) if answerable else None
        ),
        "confusion_matrix": {
            "true_positive_correct_abstention": true_positive,
            "false_positive_false_abstention": false_positive,
            "true_negative_correct_answer": true_negative,
            "false_negative_answer_or_no_decision": false_negative,
            "explicit_false_answer": explicit_false_answer,
            "no_decision_unanswerable": no_decision_unanswerable,
            "no_decision_answerable": no_decision_answerable,
        },
        "ambiguous_outcomes": {
            "answer": ambiguous_outcomes["ambiguous_answer"],
            "abstain": ambiguous_outcomes["ambiguous_abstain"],
            "no_decision": ambiguous_outcomes["ambiguous_no_decision"],
        },
    }


def score_confidence_record(
    record: EvaluationRecord,
    *,
    answer_token_f1: float,
) -> dict[str, Any]:
    """Create the Track B quality target paired with declared confidence."""

    result: dict[str, Any] = {
        "confidence_primary_eligible": False,
        "confidence_evaluable": False,
        "declared_confidence": None,
        "calibration_quality_score": None,
    }
    if record.track not in ANSWERABILITY_TRACKS or record.case.answerability == "ambiguous":
        return result

    result["confidence_primary_eligible"] = True
    response = (record.prediction or {}).get("parsed_response")
    if record.status != "valid" or not isinstance(response, Mapping):
        return result

    confidence = response.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return result

    predicted_abstain = bool(response["abstain"])
    if record.case.answerability == "unanswerable":
        quality = 1.0 if predicted_abstain else 0.0
    else:
        quality = 0.0 if predicted_abstain else answer_token_f1

    result.update(
        {
            "confidence_evaluable": True,
            "declared_confidence": float(confidence),
            "calibration_quality_score": float(quality),
        }
    )
    return result


def _calibration_bins(
    evaluable: Sequence[Mapping[str, Any]],
    *,
    bin_count: int,
) -> tuple[list[dict[str, Any]], float | None]:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")

    grouped: list[list[Mapping[str, Any]]] = [[] for _ in range(bin_count)]
    for item in evaluable:
        confidence = float(item["declared_confidence"])
        index = min(int(confidence * bin_count), bin_count - 1)
        grouped[index].append(item)

    bins: list[dict[str, Any]] = []
    weighted_gap = 0.0
    for index, items in enumerate(grouped):
        mean_confidence = (
            sum(float(item["declared_confidence"]) for item in items) / len(items)
            if items
            else None
        )
        mean_quality = (
            sum(float(item["calibration_quality_score"]) for item in items) / len(items)
            if items
            else None
        )
        absolute_gap = (
            abs(mean_confidence - mean_quality)
            if mean_confidence is not None and mean_quality is not None
            else None
        )
        if absolute_gap is not None:
            weighted_gap += absolute_gap * len(items)
        bins.append(
            {
                "bin_index": index,
                "lower_bound_inclusive": index / bin_count,
                "upper_bound_inclusive": (index + 1) / bin_count
                if index == bin_count - 1
                else None,
                "upper_bound_exclusive": (index + 1) / bin_count
                if index < bin_count - 1
                else None,
                "count": len(items),
                "mean_confidence": mean_confidence,
                "mean_quality": mean_quality,
                "absolute_gap": absolute_gap,
            }
        )

    ece = weighted_gap / len(evaluable) if evaluable else None
    return bins, ece


def _risk_coverage_curve(
    evaluable: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], float | None]:
    if not evaluable:
        return [], None

    by_confidence: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for item in evaluable:
        by_confidence[float(item["declared_confidence"])].append(item)

    selected_count = 0
    selected_quality = 0.0
    previous_coverage = 0.0
    aurc = 0.0
    curve: list[dict[str, Any]] = []
    denominator = len(evaluable)
    for threshold in sorted(by_confidence, reverse=True):
        group = by_confidence[threshold]
        selected_count += len(group)
        selected_quality += sum(
            float(item["calibration_quality_score"]) for item in group
        )
        coverage = selected_count / denominator
        mean_quality = selected_quality / selected_count
        risk = 1.0 - mean_quality
        aurc += risk * (coverage - previous_coverage)
        previous_coverage = coverage
        curve.append(
            {
                "confidence_threshold": threshold,
                "selected_case_count": selected_count,
                "coverage": coverage,
                "mean_quality": mean_quality,
                "risk": risk,
            }
        )
    return curve, aurc


def aggregate_confidence_and_calibration(
    per_case: Sequence[Mapping[str, Any]],
    *,
    track: str,
    bin_count: int = 10,
) -> dict[str, Any]:
    """Aggregate Track B ECE and confidence-threshold risk-coverage."""

    if track not in ANSWERABILITY_TRACKS:
        return {
            "applicable": False,
            "reason": "Confidence calibration requires a non-oracle context track",
            "case_count": len(per_case),
        }

    primary = [item for item in per_case if item["confidence_primary_eligible"]]
    evaluable = [item for item in primary if item["confidence_evaluable"]]
    bins, ece = _calibration_bins(evaluable, bin_count=bin_count)
    curve, aurc = _risk_coverage_curve(evaluable)
    primary_count = len(primary)
    evaluable_count = len(evaluable)
    return {
        "applicable": True,
        "case_count": len(per_case),
        "primary_case_count": primary_count,
        "ambiguous_case_count": sum(
            item["answerability"] == "ambiguous" for item in per_case
        ),
        "confidence_evaluable_count": evaluable_count,
        "confidence_unavailable_count": primary_count - evaluable_count,
        "confidence_availability_rate": (
            evaluable_count / primary_count if primary_count else None
        ),
        "quality_target": "abstention_correctness_or_answer_token_f1",
        "binning_policy": f"{bin_count}_equal_width_validation_frozen",
        "bin_count": bin_count,
        "expected_calibration_error": ece,
        "calibration_bins": bins,
        "area_under_risk_coverage_curve": aurc,
        "risk_coverage_curve": curve,
    }


def bootstrap_point_estimates(
    per_case: Sequence[Mapping[str, Any]],
    *,
    track: str,
) -> dict[str, float]:
    answer = aggregate_answer_quality(per_case)
    citation = aggregate_citation_quality(per_case)
    evidence_evaluable = [
        item
        for item in per_case
        if item.get("complete_reference_evidence_available") is not None
    ]
    estimates: dict[str, float | None] = {
        "reliability.valid_response_rate": (
            sum(item.get("status") == "valid" for item in per_case) / len(per_case)
            if per_case
            else 0.0
        ),
        "answer.token_f1": answer["token_f1"],
        "answer.normalized_exact_match": answer["normalized_exact_match"],
        "citation.precision": citation["citation_precision"],
        "citation.recall": citation["citation_recall"],
        "citation.f1": citation["citation_f1"],
        "citation.validity_rate": citation["citation_validity_rate"],
        "citation.answered_with_citation_rate": citation[
            "answered_with_citation_rate"
        ],
        "evidence.hit_rate": (
            sum(item.get("evidence_hit") is True for item in evidence_evaluable)
            / len(evidence_evaluable)
            if evidence_evaluable
            else None
        ),
        "evidence.best_reference_recall": (
            sum(
                float(item["best_reference_evidence_recall"])
                for item in evidence_evaluable
            )
            / len(evidence_evaluable)
            if evidence_evaluable
            else None
        ),
        "evidence.best_reference_precision": (
            sum(
                float(item["best_reference_evidence_precision"])
                for item in evidence_evaluable
            )
            / len(evidence_evaluable)
            if evidence_evaluable
            else None
        ),
        "evidence.best_reference_mrr": (
            sum(
                float(item["best_reference_evidence_mrr"])
                for item in evidence_evaluable
            )
            / len(evidence_evaluable)
            if evidence_evaluable
            else None
        ),
        "evidence.best_reference_ndcg": (
            sum(
                float(item["best_reference_evidence_ndcg"])
                for item in evidence_evaluable
            )
            / len(evidence_evaluable)
            if evidence_evaluable
            else None
        ),
        "evidence.complete_reference_set_rate": (
            sum(
                item.get("complete_reference_evidence_available") is True
                for item in evidence_evaluable
            )
            / len(evidence_evaluable)
            if evidence_evaluable
            else None
        ),
    }

    if track in ANSWERABILITY_TRACKS:
        abstention = aggregate_abstention_quality(per_case, track=track)
        calibration = aggregate_confidence_and_calibration(per_case, track=track)
        estimates.update(
            {
                "abstention.answerability_accuracy": abstention[
                    "answerability_accuracy"
                ],
                "abstention.precision": abstention["abstention_precision"],
                "abstention.recall": abstention["abstention_recall"],
                "abstention.f1": abstention["abstention_f1"],
                "abstention.false_answer_rate": abstention["false_answer_rate"],
                "abstention.false_abstention_rate": abstention[
                    "false_abstention_rate"
                ],
                "confidence.expected_calibration_error": calibration[
                    "expected_calibration_error"
                ],
                "confidence.area_under_risk_coverage_curve": calibration[
                    "area_under_risk_coverage_curve"
                ],
            }
        )

    return {
        metric: float(value)
        for metric, value in estimates.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def paper_clustered_bootstrap_intervals(
    per_case: Sequence[Mapping[str, Any]],
    *,
    track: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Calculate percentile intervals by resampling whole QASPER papers."""

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    by_paper: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in per_case:
        paper_id = item.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            raise ValueError("Every per-case metric row must have a paper_id")
        by_paper[paper_id].append(item)

    paper_ids = sorted(by_paper)
    point_estimates = bootstrap_point_estimates(per_case, track=track)
    samples: dict[str, list[float]] = {
        metric: [] for metric in point_estimates
    }
    if paper_ids:
        random_generator = random.Random(seed)
        for _ in range(resamples):
            sampled_rows: list[Mapping[str, Any]] = []
            for paper_id in random_generator.choices(
                paper_ids,
                k=len(paper_ids),
            ):
                sampled_rows.extend(by_paper[paper_id])
            replicate = bootstrap_point_estimates(sampled_rows, track=track)
            for metric in samples:
                value = replicate.get(metric)
                if value is not None:
                    samples[metric].append(value)

    alpha = (1.0 - confidence_level) / 2.0
    intervals: dict[str, dict[str, Any]] = {}
    for metric, point_estimate in point_estimates.items():
        values = samples[metric]
        intervals[metric] = {
            "point_estimate": point_estimate,
            "lower": _percentile(values, alpha) if values else None,
            "upper": _percentile(values, 1.0 - alpha) if values else None,
            "valid_resample_count": len(values),
        }

    return {
        "method": "percentile_cluster_bootstrap",
        "cluster_unit": "paper_id",
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "paper_count": len(paper_ids),
        "case_count": len(per_case),
        "intervals": intervals,
    }
