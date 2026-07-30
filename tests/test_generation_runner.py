import json
from dataclasses import replace
from pathlib import Path

import pytest

from rag_eval.generation_adapter import AdapterResult
from rag_eval.generation_data import GenerationCase, PaperPassage, ReferenceAnswer
from rag_eval.generation_prompt import PROMPT_VERSION
from rag_eval.generation_runner import eligibility_manifest_path, run_generation_cases


class FakeAdapter:
    model_id = "fake-model"

    def count_tokens(self, system_prompt, user_prompt):
        return 100

    def generate(self, system_prompt, user_prompt):
        return AdapterResult(
            text=(
                '{"answer":"Evidence","abstain":false,'
                '"citations":["paper::paragraph::0001"],"confidence":0.9}'
            ),
            latency_seconds=0.1,
            input_tokens=100,
            output_tokens=20,
        )


class InvalidAdapter(FakeAdapter):
    def generate(self, system_prompt, user_prompt):
        return AdapterResult(
            text='{"answer":"Evidence"}',
            latency_seconds=0.1,
            input_tokens=100,
            output_tokens=5,
        )


class FakeOpenAIAdapter(FakeAdapter):
    provider = "openai"
    model_id = "gpt-test"


class FakeFallbackAdapter(FakeAdapter):
    provider = "openrouter"
    model_id = "openai/gpt-5.6-luna-pro"
    fallback_model_ids = (
        "qwen/qwen3.7-plus",
        "deepseek/deepseek-v4-flash",
    )

    def generate(self, system_prompt, user_prompt):
        result = super().generate(system_prompt, user_prompt)
        return replace(result, resolved_model_id="qwen/qwen3.7-plus")


PASSAGE = PaperPassage(
    "paper::paragraph::0001", "paper", "paragraph", "Results", "Evidence", 0
)
REFERENCE = ReferenceAnswer(
    annotation_id="a1",
    answer_type="extractive",
    text="Evidence",
    unanswerable=False,
    extractive_spans=("Evidence",),
    evidence_ids=(PASSAGE.passage_id,),
    evidence_texts=("Evidence",),
    highlighted_evidence=(),
    unresolved_evidence=(),
)
CASE = GenerationCase(
    case_id="q1",
    split="validation",
    paper_id="paper",
    title="Paper",
    question="What is reported?",
    answerability="answerable",
    paper_passages=(PASSAGE,),
    oracle_passage_ids=(PASSAGE.passage_id,),
    references=(REFERENCE,),
)


def test_runner_persists_valid_prediction_manifest_and_resumes(tmp_path: Path):
    output = tmp_path / "predictions.jsonl"

    first = run_generation_cases(
        [CASE], adapter=FakeAdapter(), track="oracle-evidence", output_file=output
    )
    second = run_generation_cases(
        [CASE], adapter=FakeAdapter(), track="oracle-evidence", output_file=output
    )

    assert first["completed"] == 1
    assert second["skipped"] == 1
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["parsed_response"]["answer"] == "Evidence"
    assert rows[0]["prompt_version"] == PROMPT_VERSION
    manifest = json.loads(eligibility_manifest_path(output).read_text())
    assert manifest["eligible_case_count"] == 1
    assert manifest["eligible_case_ids"] == ["q1"]
    assert manifest["model_id"] == "fake-model"
    assert manifest["schema_version"] == 2
    assert manifest["prompt_version"] == PROMPT_VERSION


def test_resume_does_not_retry_a_persisted_invalid_response(tmp_path: Path):
    output = tmp_path / "invalid.jsonl"

    first = run_generation_cases(
        [CASE],
        adapter=InvalidAdapter(),
        track="oracle-evidence",
        output_file=output,
    )
    second = run_generation_cases(
        [CASE],
        adapter=InvalidAdapter(),
        track="oracle-evidence",
        output_file=output,
    )

    rows = output.read_text().splitlines()
    assert first["errors"] == 1
    assert second["skipped"] == 1
    assert len(rows) == 1


def test_runner_excludes_case_over_shared_context_limit(tmp_path: Path):
    output = tmp_path / "predictions.jsonl"
    counts = run_generation_cases(
        [CASE],
        adapter=FakeAdapter(),
        track="oracle-evidence",
        output_file=output,
        max_context_tokens=500,
        max_output_tokens=512,
    )

    assert counts["ineligible"] == 1
    assert counts["selected"] == 0
    assert counts["completed"] == 0
    manifest = json.loads(eligibility_manifest_path(output).read_text())
    assert manifest["eligible_case_ids"] == []
    assert manifest["excluded_counts"]["context_limit"] == 1


def test_resume_rejects_a_changed_eligibility_denominator(tmp_path: Path):
    output = tmp_path / "predictions.jsonl"
    run_generation_cases(
        [CASE],
        adapter=FakeAdapter(),
        track="oracle-evidence",
        output_file=output,
    )

    with pytest.raises(ValueError, match="eligibility manifest"):
        run_generation_cases(
            [CASE],
            adapter=FakeAdapter(),
            track="oracle-evidence",
            output_file=output,
            max_cases=None,
        )


def test_runner_uses_frozen_retrieved_context_and_records_manifest_hash(tmp_path: Path):
    output = tmp_path / "retrieved.jsonl"
    counts = run_generation_cases(
        [CASE],
        adapter=FakeAdapter(),
        track="retrieved-context",
        output_file=output,
        max_cases=None,
        retrieved_contexts={CASE.case_id: (PASSAGE.passage_id,)},
        context_manifest_sha256="abc123",
    )

    assert counts["completed"] == 1
    row = json.loads(output.read_text())
    assert row["context_passage_ids"] == [PASSAGE.passage_id]
    assert row["context_manifest_sha256"] == "abc123"
    manifest = json.loads(eligibility_manifest_path(output).read_text())
    assert manifest["track"] == "retrieved-context"
    assert manifest["context_manifest_sha256"] == "abc123"


def test_runner_resolves_retrieved_passages_from_the_full_case_corpus(tmp_path: Path):
    other_passage = PaperPassage(
        "other-paper::paragraph::0001",
        "other-paper",
        "paragraph",
        "Results",
        "Cross-paper evidence.",
        0,
    )
    other_case = GenerationCase(
        case_id="q2",
        split="validation",
        paper_id="other-paper",
        title="Other paper",
        question="What else is reported?",
        answerability="answerable",
        paper_passages=(other_passage,),
        oracle_passage_ids=(other_passage.passage_id,),
        references=(REFERENCE,),
    )

    class CrossPaperAdapter(FakeAdapter):
        def generate(self, system_prompt, user_prompt):
            return AdapterResult(
                text=(
                    '{"answer":"Cross-paper evidence.","abstain":false,'
                    '"citations":["other-paper::paragraph::0001"],"confidence":0.9}'
                ),
                latency_seconds=0.1,
                input_tokens=100,
                output_tokens=20,
            )

    output = tmp_path / "cross-paper-retrieved.jsonl"
    counts = run_generation_cases(
        [CASE, other_case],
        adapter=CrossPaperAdapter(),
        track="retrieved-context",
        output_file=output,
        max_cases=None,
        retrieved_contexts={CASE.case_id: (other_passage.passage_id,)},
        context_manifest_sha256="abc123",
    )

    assert counts["completed"] == 1
    row = json.loads(output.read_text())
    assert row["context_passage_ids"] == [other_passage.passage_id]


def test_runner_requires_frozen_context_identity(tmp_path: Path):
    with pytest.raises(ValueError, match="requires frozen contexts"):
        run_generation_cases(
            [CASE],
            adapter=FakeAdapter(),
            track="retrieved-context",
            output_file=tmp_path / "retrieved.jsonl",
        )


def test_runner_records_openai_provider_without_changing_run_identity(tmp_path: Path):
    output = tmp_path / "openai.jsonl"
    run_generation_cases(
        [CASE],
        adapter=FakeOpenAIAdapter(),
        track="oracle-evidence",
        output_file=output,
    )

    row = json.loads(output.read_text())
    manifest = json.loads(eligibility_manifest_path(output).read_text())
    assert row["provider"] == "openai"
    assert row["model_id"] == "gpt-test"
    assert manifest["provider"] == "openai"
    assert manifest["model_id"] == "gpt-test"


def test_runner_records_any_named_provider_in_eligibility_manifest(tmp_path: Path):
    adapter = FakeOpenAIAdapter()
    adapter.provider = "openrouter"
    output = tmp_path / "openrouter.jsonl"

    run_generation_cases(
        [CASE],
        adapter=adapter,
        track="oracle-evidence",
        output_file=output,
    )

    row = json.loads(output.read_text())
    manifest = json.loads(eligibility_manifest_path(output).read_text())
    assert row["provider"] == "openrouter"
    assert manifest["provider"] == "openrouter"


def test_runner_records_fallback_chain_and_resolved_model(tmp_path: Path):
    output = tmp_path / "fallback.jsonl"
    run_generation_cases(
        [CASE],
        adapter=FakeFallbackAdapter(),
        track="oracle-evidence",
        output_file=output,
    )

    row = json.loads(output.read_text())
    manifest = json.loads(eligibility_manifest_path(output).read_text())
    expected_fallbacks = [
        "qwen/qwen3.7-plus",
        "deepseek/deepseek-v4-flash",
    ]
    assert row["model_id"] == "openai/gpt-5.6-luna-pro"
    assert row["fallback_model_ids"] == expected_fallbacks
    assert row["resolved_model_id"] == "qwen/qwen3.7-plus"
    assert manifest["fallback_model_ids"] == expected_fallbacks


def test_oracle_runner_excludes_answerable_case_without_resolved_evidence(
    tmp_path: Path,
):
    unresolved = replace(CASE, case_id="unresolved", oracle_passage_ids=())
    output = tmp_path / "oracle.jsonl"

    counts = run_generation_cases(
        [unresolved],
        adapter=FakeAdapter(),
        track="oracle-evidence",
        output_file=output,
        max_cases=None,
    )

    manifest = json.loads(eligibility_manifest_path(output).read_text())
    assert counts["selected"] == 0
    assert manifest["excluded_counts"]["missing_oracle_evidence"] == 1
