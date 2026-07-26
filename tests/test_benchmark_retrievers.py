from rag_eval.benchmark_retrievers import (
    PreparedBM25Retriever,
    collapse_to_parents,
    reciprocal_rank_fusion,
)


def test_collapse_uses_best_chunk_and_returns_unique_parents():
    chunk_run = {
        "q1": {
            "d1::chunk::0000": 0.4,
            "d1::chunk::0001": 0.9,
            "d2::chunk::0000": 0.7,
        }
    }
    parents = {
        "d1::chunk::0000": "d1",
        "d1::chunk::0001": "d1",
        "d2::chunk::0000": "d2",
    }

    run, winners = collapse_to_parents(chunk_run, parents, top_k=2)

    assert list(run["q1"]) == ["d1", "d2"]
    assert run["q1"]["d1"] == 0.9
    assert winners["q1"]["d1"] == "d1::chunk::0001"


def test_collapse_breaks_chunk_score_ties_by_chunk_id():
    run, winners = collapse_to_parents(
        {"q": {"d::chunk::0001": 1.0, "d::chunk::0000": 1.0}},
        {"d::chunk::0001": "d", "d::chunk::0000": "d"},
        top_k=1,
    )

    assert run == {"q": {"d": 1.0}}
    assert winners == {"q": {"d": "d::chunk::0000"}}


def test_hybrid_fuses_parent_rankings():
    lexical = {"q": {"a": 2.0, "b": 1.0}}
    dense = {"q": {"b": 2.0, "c": 1.0}}

    run, latencies = reciprocal_rank_fusion(lexical, dense, top_k=2)

    assert next(iter(run["q"])) == "b"

    assert len(latencies) == 1

def test_prepared_bm25_ranks_matching_document_first():
    corpus = {
        "d1": {"title": "Cats", "text": "Cats purr and sleep."},
        "d2": {"title": "Rockets", "text": "A rocket reaches orbit."},
    }
    retriever = PreparedBM25Retriever(corpus)

    result = retriever.search({"q1": "rocket orbit"}, top_k=2)

    assert next(iter(result.run["q1"])) == "d2"
    assert len(result.query_latencies_ms) == 1
