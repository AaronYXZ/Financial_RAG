import json
from pathlib import Path

import rag_eval.end_to_end.workflow as workflow
from rag_eval.generation.adapter import AdapterResult
from rag_eval.generation.data import GenerationCase, PaperPassage, ReferenceAnswer
from rag_eval.retrieval.context import (
    retriever_manifest,
    write_frozen_context_manifest,
)


class FakeAdapter:
    model_id = "fake-generator"

    def count_tokens(self, system_prompt: str, user_prompt: str) -> int:
        return 10

    def generate(self, system_prompt: str, user_prompt: str) -> AdapterResult:
        return AdapterResult(
            text=json.dumps(
                {
                    "answer": "Supported result",
                    "abstain": False,
                    "citations": ["p::paragraph::0001"],
                    "confidence": 0.9,
                }
            ),
            latency_seconds=0.01,
            input_tokens=10,
            output_tokens=12,
        )


def test_retrieve_then_generate_freezes_before_generation(monkeypatch, tmp_path: Path):
    calls = []
    manifest = tmp_path / "contexts.json"

    def fake_freeze(**kwargs):
        calls.append(("retrieve", kwargs))
        return {"eligible_case_count": 2, "retriever": {"name": "bm25"}}

    def fake_generate(**kwargs):
        calls.append(("generate", kwargs))
        return {
            "selected": 2,
            "completed": 2,
            "skipped": 0,
            "ineligible": 0,
            "errors": 0,
        }

    monkeypatch.setattr(workflow, "freeze_context_manifest", fake_freeze)
    monkeypatch.setattr(workflow, "run_retrieved_context_generation", fake_generate)

    retrieval, generation = workflow.run_retrieve_then_generate(
        cases_file=tmp_path / "cases.jsonl",
        eligibility_file=tmp_path / "eligibility.json",
        context_manifest_file=manifest,
        adapter=object(),
        output_file=tmp_path / "predictions.jsonl",
        top_k=5,
        retriever="bm25",
        retrieval_scope="paper",
        dense_model="dense-model",
        dense_batch_size=16,
        hybrid_rrf_k=60,
        hybrid_candidate_k=None,
        max_context_tokens=32_768,
        max_output_tokens=1024,
        max_cases=None,
        resume=True,
    )

    assert retrieval["eligible_case_count"] == 2
    assert generation["completed"] == 2
    assert [name for name, _ in calls] == ["retrieve", "generate"]
    assert calls[0][1]["output_file"] == manifest
    assert calls[1][1]["context_manifest_file"] == manifest


def test_retrieved_context_workflow_generates_from_validated_manifest(
    tmp_path: Path,
):
    passage = PaperPassage(
        "p::paragraph::0001",
        "p",
        "paragraph",
        "Results",
        "Supported result",
        0,
    )
    reference = ReferenceAnswer(
        annotation_id="a1",
        answer_type="extractive",
        text="Supported result",
        unanswerable=False,
        extractive_spans=("Supported result",),
        evidence_ids=(passage.passage_id,),
        evidence_texts=(passage.text,),
        highlighted_evidence=(),
        unresolved_evidence=(),
    )
    case = GenerationCase(
        case_id="q1",
        split="validation",
        paper_id="p",
        title="Paper",
        question="What result?",
        answerability="answerable",
        paper_passages=(passage,),
        oracle_passage_ids=(passage.passage_id,),
        references=(reference,),
    )
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(json.dumps(case.to_dict()) + "\n")
    eligibility_file = tmp_path / "eligibility.json"
    eligibility_file.write_text("{}\n")
    manifest_file = tmp_path / "contexts.json"
    write_frozen_context_manifest(
        manifest_file,
        cases_file=cases_file,
        eligibility_file=eligibility_file,
        eligible_case_ids=[case.case_id],
        contexts=[
            {
                "case_id": case.case_id,
                "passage_ids": [passage.passage_id],
                "scores": [1.0],
            }
        ],
        top_k=1,
        retrieval_scope="paper",
        retriever=retriever_manifest(
            method="bm25",
            scope="paper",
            top_k=1,
        ),
    )
    predictions_file = tmp_path / "predictions.jsonl"

    counts = workflow.run_retrieved_context_generation(
        cases_file=cases_file,
        context_manifest_file=manifest_file,
        adapter=FakeAdapter(),
        output_file=predictions_file,
        max_context_tokens=1024,
        max_output_tokens=64,
        max_cases=None,
        resume=False,
    )

    prediction = json.loads(predictions_file.read_text())
    assert counts["completed"] == 1
    assert counts["errors"] == 0
    assert prediction["context_passage_ids"] == [passage.passage_id]
    assert prediction["context_manifest_sha256"]
