import json

import pytest

from rag_eval.data import BeirDataset, load_dataset, sample_dataset


def test_sample_retains_all_positive_documents():
    dataset = BeirDataset(
        corpus={f"d{index}": {"title": "", "text": str(index)} for index in range(10)},
        queries={"q1": "one", "q2": "two"},
        qrels={"q1": {"d8": 1}, "q2": {"d9": 2}},
    )

    sampled = sample_dataset(dataset, max_documents=4, max_queries=2, seed=7)

    assert len(sampled.corpus) == 4
    assert {"d8", "d9"}.issubset(sampled.corpus)
    assert sampled == sample_dataset(dataset, max_documents=4, max_queries=2, seed=7)


def test_sample_rejects_limit_below_positive_count():
    dataset = BeirDataset(
        corpus={"d1": {}, "d2": {}},
        queries={"q1": "query"},
        qrels={"q1": {"d1": 1, "d2": 1}},
    )

    with pytest.raises(ValueError, match="too small"):
        sample_dataset(dataset, max_documents=1)


def test_load_beir_files(tmp_path):
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": "T", "text": "Body"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "q1", "text": "Question"}),
                json.dumps({"_id": "unused", "text": "No qrel"}),
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
        encoding="utf-8",
    )

    dataset = load_dataset(tmp_path)

    assert dataset.corpus["d1"]["title"] == "T"
    assert dataset.queries == {"q1": "Question"}
    assert dataset.qrels == {"q1": {"d1": 1}}
