from rag_eval.retrievers import BM25Retriever, HybridRetriever, tokenize


def test_bm25_ranks_matching_document_first():
    corpus = {
        "d1": {"title": "Cats", "text": "Cats purr and sleep."},
        "d2": {"title": "Rockets", "text": "A rocket reaches orbit."},
    }

    run = BM25Retriever().search(corpus, {"q1": "rocket orbit"}, top_k=2)

    assert next(iter(run["q1"])) == "d2"


def test_tokenizer_is_case_insensitive_and_unicode_aware():
    assert tokenize("Café CAFÉ retrieval") == ["café", "café", "retrieval"]


class FakeRetriever:
    def __init__(self, run):
        self.run = run

    def search(self, corpus, queries, top_k):
        return self.run


def test_hybrid_uses_reciprocal_rank_fusion():
    lexical = FakeRetriever({"q": {"a": 2.0, "b": 1.0}})
    dense = FakeRetriever({"q": {"b": 2.0, "c": 1.0}})
    retriever = HybridRetriever(lexical, dense)

    run = retriever.search({"a": {}, "b": {}, "c": {}}, {"q": "x"}, top_k=2)

    assert next(iter(run["q"])) == "b"
