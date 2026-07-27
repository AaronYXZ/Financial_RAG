from rag_eval.generation_data import (
    generation_case_from_dict,
    normalize_qasper_row,
    select_passages,
)


def _row():
    evidence = "The model improves accuracy by five points."
    return {
        "id": "paper-1",
        "title": "A Test Paper",
        "abstract": "A short abstract.",
        "full_text": {
            "section_name": ["Results"],
            "paragraphs": [[evidence, "A distractor paragraph."]],
        },
        "qas": {
            "question": ["How much does accuracy improve?", "Is recall reported?"],
            "question_id": ["q1", "q2"],
            "answers": [
                {
                    "answer": [
                        {
                            "unanswerable": False,
                            "extractive_spans": ["five points"],
                            "yes_no": None,
                            "free_form_answer": "",
                            "evidence": [evidence],
                            "highlighted_evidence": ["five points"],
                        }
                    ],
                    "annotation_id": ["a1"],
                    "worker_id": ["w1"],
                },
                {
                    "answer": [
                        {
                            "unanswerable": True,
                            "extractive_spans": [],
                            "yes_no": None,
                            "free_form_answer": "",
                            "evidence": [],
                            "highlighted_evidence": [],
                        },
                        {
                            "unanswerable": False,
                            "extractive_spans": [],
                            "yes_no": False,
                            "free_form_answer": "",
                            "evidence": [evidence],
                            "highlighted_evidence": [],
                        },
                    ],
                    "annotation_id": ["a2", "a3"],
                    "worker_id": ["w2", "w3"],
                },
            ],
        },
        "figures_and_tables": {"caption": [], "file": []},
    }


def test_normalizes_column_oriented_qasper_row():
    answerable, ambiguous = normalize_qasper_row(_row(), "validation")

    assert answerable.case_id == "q1"
    assert answerable.answerability == "answerable"
    assert generation_case_from_dict(answerable.to_dict()) == answerable

    assert answerable.references[0].text == "five points"
    assert answerable.references[0].unresolved_evidence == ()
    assert len(answerable.oracle_passage_ids) == 1
    selected = select_passages(answerable, answerable.oracle_passage_ids)
    assert selected[0].text == "The model improves accuracy by five points."

    assert ambiguous.answerability == "ambiguous"
    assert ambiguous.references[1].answer_type == "yes_no"
    assert ambiguous.references[1].text == "No"


def test_unmatched_evidence_is_reported_not_silently_discarded():
    row = _row()
    row["qas"]["answers"][0]["answer"][0]["evidence"] = ["Missing evidence"]

    case = normalize_qasper_row(row, "validation")[0]

    assert case.oracle_passage_ids == ()
    assert case.references[0].unresolved_evidence == ("Missing evidence",)
