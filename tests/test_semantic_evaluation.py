import json
from pathlib import Path

import pytest

from rag_eval.generation.adapter import AdapterResult
from rag_eval.generation.data import GenerationCase, PaperPassage, ReferenceAnswer
from rag_eval.semantic_evaluation import (
    aggregate_semantic_judgments,
    load_blinded_semantic_inputs,
    prepare_blinded_semantic_inputs,
    run_semantic_judgments,
    semantic_manifest_path,
)


PASSAGE = PaperPassage(
    passage_id="paper::paragraph::0001",
    paper_id="paper",
    kind="paragraph",
    section_name="Results",
    text="The method improved accuracy by five points.",
    order=0,
)
REFERENCE = ReferenceAnswer(
    annotation_id="a1",
    answer_type="extractive",
    text="Accuracy improved by five points.",
    unanswerable=False,
    extractive_spans=("improved accuracy by five points",),
    evidence_ids=(PASSAGE.passage_id,),
    evidence_texts=(PASSAGE.text,),
    highlighted_evidence=(),
    unresolved_evidence=(),
)
CASE = GenerationCase(
    case_id="q1",
    split="validation",
    paper_id="paper",
    title="Paper",
    question="What improved?",
    answerability="answerable",
    paper_passages=(PASSAGE,),
    oracle_passage_ids=(PASSAGE.passage_id,),
    references=(REFERENCE,),
)

JUDGMENT = {
    "claims": [
        {
            "claim_id": "claim_1",
            "claim_text": "Accuracy improved by five points.",
            "cited_context_ids": ["C1"],
            "support_label": "supported",
            "support_context_ids": ["C1"],
            "citation_entailment": "entailed",
            "rationale": "C1 directly supports the claim.",
        }
    ],
    "semantic_correctness": {"score": 4, "rationale": "Fully correct."},
    "completeness": {"score": 4, "rationale": "Complete."},
}


class FakeJudge:
    provider = "fake"
    model_id = "independent-judge"
    fallback_model_ids = ()

    def count_tokens(self, system_prompt, user_prompt):
        return 123

    def generate(self, system_prompt, user_prompt):
        assert "generator-model" not in user_prompt
        assert PASSAGE.passage_id not in user_prompt
        return AdapterResult(
            text=json.dumps(JUDGMENT),
            latency_seconds=0.1,
            input_tokens=120,
            output_tokens=30,
        )


def _write_sources(tmp_path: Path) -> tuple[Path, Path]:
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(json.dumps(CASE.to_dict()) + "\n", encoding="utf-8")
    predictions_file = tmp_path / "predictions.jsonl"
    prediction = {
        "case_id": CASE.case_id,
        "paper_id": CASE.paper_id,
        "track": "retrieved-context",
        "model_id": "generator-model",
        "provider": "secret-provider",
        "latency_seconds": 9.2,
        "prompt_version": "qasper-generation-v3",
        "context_passage_ids": [PASSAGE.passage_id],
        "parsed_response": {
            "answer": "Accuracy improved by five points.",
            "abstain": False,
            "citations": [PASSAGE.passage_id],
            "confidence": 0.99,
        },
        "error": None,
    }
    predictions_file.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    return cases_file, predictions_file


def test_prepare_creates_blinded_inputs_and_separate_provenance(tmp_path: Path):
    cases_file, predictions_file = _write_sources(tmp_path)
    output_file = tmp_path / "blinded.jsonl"

    manifest = prepare_blinded_semantic_inputs(
        cases_file=cases_file,
        predictions_file=predictions_file,
        output_file=output_file,
        track="retrieved-context",
        generator_model_id="generator-model",
    )
    row = json.loads(output_file.read_text())
    serialized_input = json.dumps(row["judge_input"])

    assert row["judge_input"]["context"][0]["context_id"] == "C1"
    assert row["judge_input"]["candidate"]["citations"] == ["C1"]
    assert row["citation_map"] == {"C1": PASSAGE.passage_id}
    assert "generator-model" not in serialized_input
    assert "secret-provider" not in serialized_input
    assert "confidence" not in serialized_input
    assert "latency" not in serialized_input
    assert manifest["generator_model_id"] == "generator-model"
    assert manifest["prepared_record_count"] == 1


def test_loading_blinded_inputs_detects_tampering(tmp_path: Path):
    cases_file, predictions_file = _write_sources(tmp_path)
    output_file = tmp_path / "blinded.jsonl"
    prepare_blinded_semantic_inputs(
        cases_file=cases_file,
        predictions_file=predictions_file,
        output_file=output_file,
        track="retrieved-context",
        generator_model_id="generator-model",
    )
    output_file.write_text(output_file.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        load_blinded_semantic_inputs(output_file)


def test_runner_uses_only_blinded_input_and_resumes(tmp_path: Path):
    cases_file, predictions_file = _write_sources(tmp_path)
    inputs_file = tmp_path / "blinded.jsonl"
    output_file = tmp_path / "judgments.jsonl"
    prepare_blinded_semantic_inputs(
        cases_file=cases_file,
        predictions_file=predictions_file,
        output_file=inputs_file,
        track="retrieved-context",
        generator_model_id="generator-model",
    )

    first = run_semantic_judgments(
        inputs_file=inputs_file,
        adapter=FakeJudge(),
        output_file=output_file,
    )
    second = run_semantic_judgments(
        inputs_file=inputs_file,
        adapter=FakeJudge(),
        output_file=output_file,
    )
    row = json.loads(output_file.read_text())
    manifest = json.loads(semantic_manifest_path(output_file).read_text())

    assert first["completed"] == 1
    assert second["skipped"] == 1
    assert row["parsed_judgment"]["claims"][0]["support_label"] == "supported"
    assert row["judge_prompt_hash"] == manifest["judge_prompt_hash"]
    assert manifest["judge_model_id"] == "independent-judge"


def test_runner_rejects_generator_as_judge(tmp_path: Path):
    cases_file, predictions_file = _write_sources(tmp_path)
    inputs_file = tmp_path / "blinded.jsonl"
    prepare_blinded_semantic_inputs(
        cases_file=cases_file,
        predictions_file=predictions_file,
        output_file=inputs_file,
        track="retrieved-context",
        generator_model_id="generator-model",
    )
    judge = FakeJudge()
    judge.model_id = "generator-model"

    with pytest.raises(ValueError, match="cannot be its own"):
        run_semantic_judgments(
            inputs_file=inputs_file,
            adapter=judge,
            output_file=tmp_path / "judgments.jsonl",
        )


def test_aggregate_reports_claim_faithfulness_citations_and_rubrics(tmp_path: Path):
    judgments_file = tmp_path / "judgments.jsonl"
    judgments_file.write_text(
        json.dumps(
            {
                "record_id": "r1",
                "case_id": "q1",
                "paper_id": "paper",
                "parsed_judgment": JUDGMENT,
                "error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = aggregate_semantic_judgments(
        judgments_file=judgments_file,
        output_dir=tmp_path / "metrics",
    )

    assert summary["claim_support"]["supported_claim_rate"] == 1.0
    assert summary["claim_support"]["fully_faithful_rate"] == 1.0
    assert summary["citation_entailment"]["entailed_claim_rate"] == 1.0
    assert summary["citation_entailment"]["citation_complete_case_rate"] == 1.0
    assert summary["rubric_scores"]["semantic_correctness_mean"] == 4.0
    assert summary["rubric_scores"]["completeness_mean"] == 4.0
