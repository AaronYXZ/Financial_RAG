"""Stage 0-2 evaluation for fixed-context QASPER generation runs."""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .generation_data import GenerationCase, ReferenceAnswer, generation_case_from_dict


EVALUATION_STATUSES = (
    "valid",
    "invalid_json",
    "invalid_schema",
    "invalid_citation",
    "request_error",
    "timeout",
    "missing_prediction",
)


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


def load_generation_cases(path: Path) -> dict[str, GenerationCase]:
    cases: dict[str, GenerationCase] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            case = generation_case_from_dict(json.loads(line))
            if case.case_id in cases:
                raise ValueError(f"Duplicate case ID {case.case_id!r} at line {line_number}")
            cases[case.case_id] = case
    return cases


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
            "pricing_basis": "local_inference",
            "observed_total": 0.0,
            "per_attempted_case": 0.0,
            "per_1000_attempted_cases": 0.0,
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
    return {
        **citation_score,
        **abstention_score,
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
        "abstention_quality": aggregate_abstention_quality(per_case, track=track),
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
    if record.track != "complete-paper":
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
    if track != "complete-paper":
        return {
            "applicable": False,
            "reason": "Abstention metrics require the complete-paper track",
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
