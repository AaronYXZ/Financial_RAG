import json
from dataclasses import replace
from pathlib import Path

import pytest

from rag_eval.generation_data import GenerationCase, PaperPassage, ReferenceAnswer
from rag_eval.generation_metrics import (
    EvaluationRecord,
    aggregate_abstention_quality,
    aggregate_answer_quality,
    aggregate_citation_quality,
    build_evaluation_records,
    citation_precision_recall_f1,
    evaluate_prediction_files,
    normalize_answer,
    normalized_exact_match,
    reference_answer,
    reliability_and_efficiency,
    score_answer_record,
    token_f1_score,
)


PASSAGE = PaperPassage(
    "paper::paragraph::0001",
    "paper",
    "paragraph",
    "Results",
    "Relevant evidence",
    0,
)


def make_reference(
    annotation_id: str,
    *,
    answer_type: str = "free_form",
    text: str = "The relevant result",
    unanswerable: bool = False,
    extractive_spans: tuple[str, ...] = (),
) -> ReferenceAnswer:
    return ReferenceAnswer(
        annotation_id=annotation_id,
        answer_type=answer_type,
        text=text,
        unanswerable=unanswerable,
        extractive_spans=extractive_spans,
        evidence_ids=(PASSAGE.passage_id,),
        evidence_texts=(PASSAGE.text,),
        highlighted_evidence=(),
        unresolved_evidence=(),
    )


def make_case(
    case_id: str,
    references: tuple[ReferenceAnswer, ...],
    *,
    answerability: str = "answerable",
) -> GenerationCase:
    return GenerationCase(
        case_id=case_id,
        split="validation",
        paper_id="paper",
        title="Paper",
        question="What is reported?",
        answerability=answerability,
        paper_passages=(PASSAGE,),
        oracle_passage_ids=(PASSAGE.passage_id,),
        references=references,
    )


def prediction(case_id: str, **overrides):
    row = {
        "case_id": case_id,
        "track": "oracle-evidence",
        "model_id": "test-model",
        "context_passage_ids": [PASSAGE.passage_id],
        "parsed_response": {
            "answer": "relevant result",
            "abstain": False,
            "citations": [PASSAGE.passage_id],
            "confidence": 0.9,
        },
        "error": None,
        "attempts": 1,
        "latency_seconds": 0.5,
        "server_input_tokens": 100,
        "server_output_tokens": 20,
    }
    row.update(overrides)
    return row


def test_qasper_normalization_token_f1_and_exact_match():
    assert normalize_answer("The, Relevant Result!") == "relevant result"
    assert token_f1_score("relevant result", "The relevant result") == 1.0
    assert token_f1_score("relevant", "relevant result") == pytest.approx(2 / 3)
    assert normalized_exact_match("A result.", "result") == 1.0


def test_extractive_reference_uses_official_comma_join():
    reference = make_reference(
        "a1",
        answer_type="extractive",
        text="ignored",
        extractive_spans=("first span", "second span"),
    )

    assert reference_answer(reference) == ("first span, second span", "extractive")


def test_join_uses_frozen_denominator_last_retry_and_missing_prediction():
    cases = {
        "q1": make_case("q1", (make_reference("a1"),)),
        "q2": make_case("q2", (make_reference("a2"),)),
        "q3": make_case("q3", (make_reference("a3"),)),
    }
    predictions = {
        "q1": [prediction("q1")],
        "q2": [
            prediction("q2"),
            prediction(
                "q2",
                parsed_response=None,
                error="Generation request failed: temporary",
                error_stage="generation",
                attempts=2,
            ),
        ],
    }
    eligibility = {
        "track": "oracle-evidence",
        "model_id": "test-model",
        "eligible_case_ids": ["q1", "q2", "q3"],
    }

    records = build_evaluation_records(
        cases,
        predictions,
        eligibility,
        track="oracle-evidence",
        model_id="test-model",
    )

    assert [record.status for record in records] == [
        "valid",
        "request_error",
        "missing_prediction",
    ]
    assert records[1].duplicate_prediction_count == 1

    reliability = reliability_and_efficiency(records)
    assert reliability["eligible_case_count"] == 3
    assert reliability["valid_response_rate"] == pytest.approx(1 / 3)
    assert reliability["retry_rate"] == 0.5
    assert reliability["latency_seconds"]["p50"] == 0.5
    assert reliability["total_tokens"]["sum"] == 240.0


def test_answer_scoring_maximizes_over_references_and_scores_failures_zero():
    case = make_case(
        "q1",
        (
            make_reference("a1", text="different answer"),
            make_reference("a2", text="The relevant result"),
        ),
    )
    valid = EvaluationRecord(
        case=case,
        track="oracle-evidence",
        model_id="test-model",
        status="valid",
        prediction=prediction("q1"),
        duplicate_prediction_count=0,
    )
    missing = EvaluationRecord(
        case=case,
        track="oracle-evidence",
        model_id="test-model",
        status="missing_prediction",
        prediction=None,
        duplicate_prediction_count=0,
    )

    valid_score = score_answer_record(valid)
    missing_score = score_answer_record(missing)
    summary = aggregate_answer_quality([valid_score, missing_score])

    assert valid_score["answer_token_f1"] == 1.0
    assert valid_score["winning_annotation_id"] == "a2"
    assert missing_score["answer_token_f1"] == 0.0
    assert summary["token_f1"] == 0.5


def test_abstention_maps_to_official_unanswerable_label():
    case = make_case(
        "q1",
        (make_reference("a1", text="", unanswerable=True),),
        answerability="unanswerable",
    )
    row = prediction(
        "q1",
        parsed_response={
            "answer": "",
            "abstain": True,
            "citations": [],
            "confidence": 0.8,
        },
    )
    record = EvaluationRecord(
        case=case,
        track="complete-paper",
        model_id="test-model",
        status="valid",
        prediction=row,
        duplicate_prediction_count=0,
    )

    score = score_answer_record(record)

    assert score["candidate_answer"] == "Unanswerable"
    assert score["answer_token_f1"] == 1.0
    assert score["answer_normalized_exact_match"] == 1.0


def test_evaluator_writes_stage_zero_to_four_artifacts(tmp_path: Path):
    case = make_case("q1", (make_reference("a1"),))
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(json.dumps(case.to_dict()) + "\n", encoding="utf-8")
    predictions_file = tmp_path / "predictions.jsonl"
    predictions_file.write_text(json.dumps(prediction("q1")) + "\n", encoding="utf-8")
    eligibility_file = tmp_path / "predictions.eligibility.json"
    eligibility_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "track": "oracle-evidence",
                "model_id": "test-model",
                "eligible_case_count": 1,
                "eligible_case_ids": ["q1"],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "metrics"

    summary = evaluate_prediction_files(
        cases_file=cases_file,
        predictions_file=predictions_file,
        eligibility_file=eligibility_file,
        output_dir=output_dir,
        track="oracle-evidence",
        model_id="test-model",
    )

    assert summary["answer_quality"]["token_f1"] == 1.0
    assert summary["citation_quality"]["citation_f1"] == 1.0
    assert summary["abstention_quality"]["applicable"] is False
    assert (output_dir / "evaluation_records.jsonl").is_file()
    assert (output_dir / "per_case_metrics.jsonl").is_file()
    assert json.loads((output_dir / "summary.json").read_text()) == summary


def test_citation_scoring_selects_best_human_evidence_set():
    second_passage_id = "paper::paragraph::0002"
    case = make_case(
        "q1",
        (
            replace(
                make_reference("a1"),
                evidence_ids=(PASSAGE.passage_id, second_passage_id),
            ),
            replace(make_reference("a2"), evidence_ids=(PASSAGE.passage_id,)),
        ),
    )
    record = EvaluationRecord(
        case=case,
        track="oracle-evidence",
        model_id="test-model",
        status="valid",
        prediction=prediction(
            "q1",
            context_passage_ids=[PASSAGE.passage_id, second_passage_id],
        ),
        duplicate_prediction_count=0,
    )

    score = score_answer_record(record)

    assert score["citation_evaluable"] is True
    assert score["winning_citation_annotation_id"] == "a2"
    assert score["citation_precision"] == 1.0
    assert score["citation_recall"] == 1.0
    assert score["citation_f1"] == 1.0
    assert score["citation_reference_scores"][0]["citation_f1"] == pytest.approx(2 / 3)


def test_citation_aggregate_reports_abstention_invalid_and_missing_evidence():
    answered_case = make_case("answered", (make_reference("a1"),))
    abstained_case = make_case(
        "abstained",
        (make_reference("a2", text="", unanswerable=True),),
        answerability="unanswerable",
    )
    missing_evidence_case = make_case(
        "missing-evidence",
        (replace(make_reference("a3"), evidence_ids=()),),
    )
    scores = [
        score_answer_record(
            EvaluationRecord(
                answered_case,
                "complete-paper",
                "test-model",
                "valid",
                prediction("answered"),
                0,
            )
        ),
        score_answer_record(
            EvaluationRecord(
                abstained_case,
                "complete-paper",
                "test-model",
                "valid",
                prediction(
                    "abstained",
                    parsed_response={
                        "answer": "",
                        "abstain": True,
                        "citations": [],
                        "confidence": 0.8,
                    },
                ),
                0,
            )
        ),
        score_answer_record(
            EvaluationRecord(
                missing_evidence_case,
                "complete-paper",
                "test-model",
                "valid",
                prediction("missing-evidence"),
                0,
            )
        ),
        score_answer_record(
            EvaluationRecord(
                answered_case,
                "complete-paper",
                "test-model",
                "invalid_citation",
                prediction(
                    "invalid",
                    parsed_response=None,
                    error="Unknown citation IDs",
                    error_stage="response_validation",
                ),
                0,
            )
        ),
    ]

    summary = aggregate_citation_quality(scores)

    assert summary["case_count"] == 4
    assert summary["answered_case_count"] == 2
    assert summary["abstained_case_count"] == 1
    assert summary["invalid_or_missing_response_count"] == 1
    assert summary["scorable_answered_case_count"] == 1
    assert summary["missing_reference_evidence_case_count"] == 1
    assert summary["invalid_citation_case_count"] == 1
    assert summary["citation_validity_evaluable_count"] == 3
    assert summary["citation_validity_rate"] == pytest.approx(2 / 3)
    assert summary["answered_with_citation_rate"] == 1.0
    assert summary["citation_f1"] == 1.0


def test_citation_set_metrics_handle_no_overlap_and_empty_reference():
    score = citation_precision_recall_f1(["predicted"], ["reference-1", "reference-2"])

    assert score == {
        "citation_precision": 0.0,
        "citation_recall": 0.0,
        "citation_f1": 0.0,
    }
    with pytest.raises(ValueError, match="reference evidence ID"):
        citation_precision_recall_f1(["predicted"], [])


def test_track_b_abstention_metrics_include_no_decisions_and_exclude_ambiguous():
    def scored(
        case_id: str,
        answerability: str,
        predicted_abstain: bool | None,
    ):
        case = make_case(
            case_id,
            (
                make_reference(
                    f"{case_id}-annotation",
                    text="" if answerability == "unanswerable" else "answer",
                    unanswerable=answerability == "unanswerable",
                ),
            ),
            answerability=answerability,
        )
        if predicted_abstain is None:
            row = prediction(
                case_id,
                parsed_response=None,
                error="Generation response keys are invalid",
                error_stage="response_validation",
            )
            status = "invalid_schema"
        else:
            row = prediction(
                case_id,
                parsed_response={
                    "answer": "" if predicted_abstain else "answer",
                    "abstain": predicted_abstain,
                    "citations": [] if predicted_abstain else [PASSAGE.passage_id],
                    "confidence": 0.8,
                },
            )
            status = "valid"
        return score_answer_record(
            EvaluationRecord(
                case,
                "complete-paper",
                "test-model",
                status,
                row,
                0,
            )
        )

    per_case = [
        scored("correct-answer", "answerable", False),
        scored("false-abstention", "answerable", True),
        scored("correct-abstention", "unanswerable", True),
        scored("false-answer", "unanswerable", False),
        scored("no-decision", "unanswerable", None),
        scored("ambiguous", "ambiguous", False),
    ]

    summary = aggregate_abstention_quality(per_case, track="complete-paper")

    assert summary["applicable"] is True
    assert summary["primary_case_count"] == 5
    assert summary["ambiguous_case_count"] == 1
    assert summary["answerable_case_count"] == 2
    assert summary["unanswerable_case_count"] == 3
    assert summary["answerability_accuracy"] == pytest.approx(2 / 5)
    assert summary["abstention_precision"] == 0.5
    assert summary["abstention_recall"] == pytest.approx(1 / 3)
    assert summary["abstention_f1"] == pytest.approx(0.4)
    assert summary["false_answer_rate"] == pytest.approx(1 / 3)
    assert summary["false_abstention_rate"] == 0.5
    assert summary["no_decision_rate"] == pytest.approx(1 / 5)
    assert summary["confusion_matrix"] == {
        "true_positive_correct_abstention": 1,
        "false_positive_false_abstention": 1,
        "true_negative_correct_answer": 1,
        "false_negative_answer_or_no_decision": 2,
        "explicit_false_answer": 1,
        "no_decision_unanswerable": 1,
        "no_decision_answerable": 0,
    }
    assert summary["ambiguous_outcomes"] == {
        "answer": 1,
        "abstain": 0,
        "no_decision": 0,
    }


def test_abstention_metrics_are_not_applicable_to_oracle_evidence_track():
    case = make_case("q1", (make_reference("a1"),))
    per_case = [
        score_answer_record(
            EvaluationRecord(
                case,
                "oracle-evidence",
                "test-model",
                "valid",
                prediction("q1"),
                0,
            )
        )
    ]

    summary = aggregate_abstention_quality(per_case, track="oracle-evidence")

    assert summary == {
        "applicable": False,
        "reason": "Abstention metrics require the complete-paper track",
        "case_count": 1,
    }
