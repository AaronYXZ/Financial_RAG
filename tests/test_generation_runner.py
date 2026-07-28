import json
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
