import json
from pathlib import Path

import pytest

from rag_eval.generation_data import GenerationCase, PaperPassage, ReferenceAnswer
from rag_eval.generation_retrieval import (
    freeze_bm25_contexts,
    frozen_contexts_by_case,
    load_frozen_context_manifest,
    write_frozen_context_manifest,
)


PASSAGES = (
    PaperPassage("p0", "paper", "paragraph", "Background", "irrelevant words", 0),
    PaperPassage("p1", "paper", "paragraph", "Results", "unique evidence term", 1),
)
REFERENCE = ReferenceAnswer(
    annotation_id="a1",
    answer_type="extractive",
    text="unique evidence",
    unanswerable=False,
    extractive_spans=("unique evidence",),
    evidence_ids=("p1",),
    evidence_texts=("unique evidence term",),
    highlighted_evidence=(),
    unresolved_evidence=(),
)
CASE = GenerationCase(
    case_id="q1",
    split="validation",
    paper_id="paper",
    title="Paper",
    question="What unique evidence is reported?",
    answerability="answerable",
    paper_passages=PASSAGES,
    oracle_passage_ids=("p1",),
    references=(REFERENCE,),
)


def test_freeze_bm25_contexts_is_ranked_and_case_scoped():
    contexts = freeze_bm25_contexts([CASE], eligible_case_ids=["q1"], top_k=1)

    assert contexts[0]["case_id"] == "q1"
    assert contexts[0]["passage_ids"] == ["p1"]
    assert len(contexts[0]["scores"]) == 1


def test_frozen_context_manifest_records_inputs_and_round_trips(tmp_path: Path):
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text("{}\n")
    eligibility_file = tmp_path / "eligible.json"
    eligibility_file.write_text(json.dumps({"eligible_case_ids": ["q1"]}))
    output = tmp_path / "frozen.json"
    contexts = [{"case_id": "q1", "passage_ids": ["p1"], "scores": [1.0]}]

    payload = write_frozen_context_manifest(
        output,
        cases_file=cases_file,
        eligibility_file=eligibility_file,
        eligible_case_ids=["q1"],
        contexts=contexts,
        top_k=1,
    )
    loaded = load_frozen_context_manifest(output)

    assert loaded == payload
    assert loaded["retriever"]["parameters"]["top_k"] == 1
    assert frozen_contexts_by_case(loaded) == {"q1": ("p1",)}


def test_frozen_context_manifest_rejects_mismatched_cases(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "track": "retrieved-context",
                "eligible_case_count": 1,
                "eligible_case_ids": ["q1"],
                "contexts": [],
            }
        )
    )

    with pytest.raises(ValueError, match="do not match"):
        load_frozen_context_manifest(path)
