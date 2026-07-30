import json
from pathlib import Path

import pytest

import rag_eval.generation_retrieval as generation_retrieval
from rag_eval.benchmark_retrievers import SearchResult
from rag_eval.generation_data import GenerationCase, PaperPassage, ReferenceAnswer
from rag_eval.generation_retrieval import (
    freeze_bm25_contexts,
    freeze_retrieved_contexts,
    frozen_contexts_by_case,
    load_frozen_context_manifest,
    retriever_manifest,
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
OTHER_PASSAGE = PaperPassage(
    "other-p0",
    "other-paper",
    "paragraph",
    "Results",
    "unique evidence term unique evidence term",
    0,
)
OTHER_CASE = GenerationCase(
    case_id="q2",
    split="validation",
    paper_id="other-paper",
    title="Other paper",
    question="What else is reported?",
    answerability="answerable",
    paper_passages=(OTHER_PASSAGE,),
    oracle_passage_ids=(OTHER_PASSAGE.passage_id,),
    references=(REFERENCE,),
)


def test_freeze_bm25_contexts_is_ranked_and_case_scoped():
    contexts = freeze_bm25_contexts([CASE], eligible_case_ids=["q1"], top_k=1)

    assert contexts[0]["case_id"] == "q1"
    assert contexts[0]["passage_ids"] == ["p1"]
    assert len(contexts[0]["scores"]) == 1


def test_bm25_scope_controls_whether_cross_paper_passages_are_candidates():
    paper_context = freeze_retrieved_contexts(
        [CASE, OTHER_CASE],
        eligible_case_ids=[CASE.case_id],
        top_k=1,
        method="bm25",
        scope="paper",
    )
    corpus_context = freeze_retrieved_contexts(
        [CASE, OTHER_CASE],
        eligible_case_ids=[CASE.case_id],
        top_k=1,
        method="bm25",
        scope="corpus",
    )

    assert paper_context[0]["passage_ids"] == ["p1"]
    assert corpus_context[0]["passage_ids"] == [OTHER_PASSAGE.passage_id]


def test_dense_and_hybrid_use_the_selected_model(monkeypatch):
    calls = []

    class FakeDenseRetriever:
        def __init__(
            self,
            corpus,
            model_name,
            batch_size=32,
            model=None,
        ):
            calls.append((set(corpus), model_name, batch_size, model))
            self.corpus = corpus
            self.model = model if model is not None else object()

        def search(self, queries, top_k):
            selected = list(self.corpus)[:top_k]
            return SearchResult(
                run={
                    query_id: {
                        passage_id: float(len(selected) - index)
                        for index, passage_id in enumerate(selected)
                    }
                    for query_id in queries
                },
                query_latencies_ms=tuple(0.0 for _ in queries),
            )

    monkeypatch.setattr(
        generation_retrieval,
        "PreparedDenseRetriever",
        FakeDenseRetriever,
    )
    dense_contexts = freeze_retrieved_contexts(
        [CASE, OTHER_CASE],
        eligible_case_ids=[CASE.case_id, OTHER_CASE.case_id],
        top_k=1,
        method="dense",
        scope="paper",
        dense_model="dense-model-2",
        dense_batch_size=8,
    )
    hybrid_contexts = freeze_retrieved_contexts(
        [CASE],
        eligible_case_ids=[CASE.case_id],
        top_k=1,
        method="hybrid",
        scope="paper",
        dense_model="dense-model-1",
        hybrid_candidate_k=2,
    )

    assert [item["case_id"] for item in dense_contexts] == ["q1", "q2"]
    assert hybrid_contexts[0]["passage_ids"]
    assert calls[0][1:3] == ("dense-model-2", 8)
    assert calls[0][3] is None
    assert calls[1][3] is not None
    assert calls[-1][1] == "dense-model-1"


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
        retrieval_scope="paper",
        retriever=retriever_manifest(
            method="dense",
            scope="paper",
            top_k=1,
            dense_model="dense-model-1",
        ),
    )
    loaded = load_frozen_context_manifest(output)

    assert loaded == payload
    assert loaded["schema_version"] == 2
    assert loaded["retrieval_scope"] == "paper"
    assert loaded["retriever"]["name"] == "dense"
    assert loaded["retriever"]["parameters"]["dense_model"] == "dense-model-1"
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


def test_frozen_context_manifest_keeps_schema_v1_read_compatibility(tmp_path: Path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "track": "retrieved-context",
                "eligible_case_count": 1,
                "eligible_case_ids": ["q1"],
                "contexts": [
                    {
                        "case_id": "q1",
                        "passage_ids": ["p1"],
                        "scores": [1.0],
                    }
                ],
            }
        )
    )

    loaded = load_frozen_context_manifest(path)

    assert loaded["schema_version"] == 1
    assert frozen_contexts_by_case(loaded) == {"q1": ("p1",)}
