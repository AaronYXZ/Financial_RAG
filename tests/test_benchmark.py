from pathlib import Path

from rag_eval.benchmark import BenchmarkConfig, run_benchmark
from rag_eval.data import BeirDataset


def test_whole_document_bm25_control_runs_without_chunking(
    tmp_path: Path,
):
    dataset = BeirDataset(
        corpus={
            "d1": {"title": "Cats", "text": "Cats purr."},
            "d2": {"title": "Rockets", "text": "Rockets reach orbit."},
        },
        queries={"q1": "rocket orbit"},
        qrels={"q1": {"d2": 1}},
    )
    config = BenchmarkConfig(
        chunk_strategies=(),
        k_values=(1,),
        repetitions=1,
        include_whole_document_bm25=True,
    )

    summary = run_benchmark(dataset, config, tmp_path)

    assert len(summary["runs"]) == 1
    run = summary["runs"][0]
    assert run["run_id"] == "scidocs-whole-document-bm25"
    assert run["metrics"]["Precision@1"] == 1.0
