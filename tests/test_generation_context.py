import pytest

from rag_eval.generation_context import build_fixed_context
from rag_eval.generation_data import GenerationCase, PaperPassage, ReferenceAnswer


PASSAGES = (
    PaperPassage("p0", "paper", "title", "Title", "Title", 0),
    PaperPassage("p1", "paper", "paragraph", "Results", "Evidence", 1),
)
REFERENCE = ReferenceAnswer(
    annotation_id="a1",
    answer_type="extractive",
    text="Evidence",
    unanswerable=False,
    extractive_spans=("Evidence",),
    evidence_ids=("p1",),
    evidence_texts=("Evidence",),
    highlighted_evidence=(),
    unresolved_evidence=(),
)


def _case(answerability="answerable", oracle_ids=("p1",)):
    return GenerationCase(
        case_id="q1",
        split="validation",
        paper_id="paper",
        title="Title",
        question="Question?",
        answerability=answerability,
        paper_passages=PASSAGES,
        oracle_passage_ids=oracle_ids,
        references=(REFERENCE,),
    )


def test_oracle_track_returns_only_evidence_in_document_order():
    assert build_fixed_context(_case(), "oracle-evidence") == (PASSAGES[1],)


def test_complete_paper_track_returns_every_passage():
    assert build_fixed_context(_case(), "complete-paper") == PASSAGES


def test_oracle_track_rejects_unanswerable_or_missing_evidence():
    with pytest.raises(ValueError):
        build_fixed_context(_case(answerability="unanswerable"), "oracle-evidence")
    with pytest.raises(ValueError):
        build_fixed_context(_case(oracle_ids=()), "oracle-evidence")
