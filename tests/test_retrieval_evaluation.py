import json
from pathlib import Path

import pytest

from rag_eval.generation_data import GenerationCase, PaperPassage, ReferenceAnswer
from rag_eval.retrieval.evaluation import compare_retrieval_to_oracle


def _write_json(path: Path, value):
    path.write_text(json.dumps(value))


def _case() -> GenerationCase:
    passages = (
        PaperPassage("p::1", "p", "paragraph", "S", "one", 0),
        PaperPassage("p::2", "p", "paragraph", "S", "two", 1),
        PaperPassage("p::3", "p", "paragraph", "S", "three", 2),
    )
    reference = ReferenceAnswer(
        annotation_id="a1",
        answer_type="extractive",
        text="one two",
        unanswerable=False,
        extractive_spans=("one two",),
        evidence_ids=("p::1", "p::2"),
        evidence_texts=("one", "two"),
        highlighted_evidence=(),
        unresolved_evidence=(),
    )
    return GenerationCase(
        case_id="q1",
        split="validation",
        paper_id="p",
        title="Paper",
        question="What?",
        answerability="answerable",
        paper_passages=passages,
        oracle_passage_ids=("p::1", "p::2"),
        references=(reference,),
    )


def _manifest(cases_file: Path, passage_ids):
    import hashlib

    checksum = hashlib.sha256(cases_file.read_bytes()).hexdigest()
    return {
        "schema_version": 2,
        "track": "retrieved-context",
        "retrieval_scope": "paper",
        "retriever": {
            "name": "dense",
            "implementation": "test",
            "parameters": {"scope": "paper", "top_k": 2},
        },
        "cases_file": str(cases_file),
        "cases_sha256": checksum,
        "source_eligibility_file": "eligible.json",
        "source_eligibility_sha256": "abc",
        "eligible_case_count": 1,
        "eligible_case_ids": ["q1"],
        "contexts": [{"case_id": "q1", "passage_ids": passage_ids, "scores": [1, 0]}],
    }


def test_compare_retrieval_selects_configuration_closest_to_oracle(tmp_path: Path):
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(json.dumps(_case().to_dict()) + "\n")
    weak = tmp_path / "weak.json"
    strong = tmp_path / "strong.json"
    _write_json(weak, _manifest(cases_file, ["p::1", "p::3"]))
    _write_json(strong, _manifest(cases_file, ["p::1", "p::2"]))

    result = compare_retrieval_to_oracle(
        cases_file=cases_file,
        context_manifest_files=[weak, strong],
    )

    assert result["selected_context_manifest"] == str(strong)
    assert result["selection_policy"]["generation_metrics_used"] is False
    assert result["ranked_configurations"][0]["retrieved"][
        "complete_reference_set_rate"
    ] == 1.0
    assert result["ranked_configurations"][1]["retrieved"][
        "complete_reference_set_rate"
    ] == 0.0


def test_compare_retrieval_requires_identical_ordered_cases(tmp_path: Path):
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(json.dumps(_case().to_dict()) + "\n")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_json(first, _manifest(cases_file, ["p::1", "p::2"]))
    payload = _manifest(cases_file, ["p::1", "p::2"])
    payload["eligible_case_ids"] = []
    payload["eligible_case_count"] = 0
    payload["contexts"] = []
    _write_json(second, payload)

    with pytest.raises(ValueError, match="identical ordered"):
        compare_retrieval_to_oracle(
            cases_file=cases_file,
            context_manifest_files=[first, second],
        )
