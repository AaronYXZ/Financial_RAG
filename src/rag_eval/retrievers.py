"""Simple lexical, dense, and hybrid retrievers."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Protocol

Corpus = Mapping[str, Mapping[str, str]]
Queries = Mapping[str, str]
Run = dict[str, dict[str, float]]


class Retriever(Protocol):
    def search(self, corpus: Corpus, queries: Queries, top_k: int) -> Run: ...


class BM25Retriever:
    """Dependency-free Okapi BM25 baseline."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def search(self, corpus: Corpus, queries: Queries, top_k: int) -> Run:
        if not corpus:
            return {query_id: {} for query_id in queries}

        doc_ids = list(corpus)
        token_counts: list[Counter[str]] = []
        document_frequency: Counter[str] = Counter()
        lengths: list[int] = []

        for doc_id in doc_ids:
            document = corpus[doc_id]
            tokens = tokenize(f"{document.get('title', '')} {document.get('text', '')}")
            counts = Counter(tokens)
            token_counts.append(counts)
            document_frequency.update(counts)
            lengths.append(len(tokens))

        document_count = len(doc_ids)
        average_length = sum(lengths) / document_count or 1.0
        run: Run = {}
        for query_id, query in queries.items():
            scores: list[tuple[str, float]] = []
            for doc_id, counts, length in zip(doc_ids, token_counts, lengths, strict=True):
                score = 0.0
                for term in tokenize(query):
                    frequency = counts.get(term, 0)
                    if not frequency:
                        continue
                    doc_frequency = document_frequency[term]
                    inverse_document_frequency = math.log(
                        1 + (document_count - doc_frequency + 0.5) / (doc_frequency + 0.5)
                    )
                    denominator = frequency + self.k1 * (
                        1 - self.b + self.b * length / average_length
                    )
                    score += inverse_document_frequency * frequency * (self.k1 + 1) / denominator
                if score > 0:
                    scores.append((doc_id, score))
            scores.sort(key=lambda item: (-item[1], item[0]))
            run[query_id] = dict(scores[:top_k])
        return run


class DenseRetriever:
    """Exact cosine search using a sentence-transformers model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

    def search(self, corpus: Corpus, queries: Queries, top_k: int) -> Run:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Dense retrieval requires the optional dependencies. "
                "Install them with: pip install -e '.[dense]'"
            ) from error

        model = SentenceTransformer(self.model_name)
        doc_ids = list(corpus)
        doc_texts = [
            f"{corpus[doc_id].get('title', '')}\n{corpus[doc_id].get('text', '')}".strip()
            for doc_id in doc_ids
        ]
        query_ids = list(queries)
        query_texts = [queries[query_id] for query_id in query_ids]
        doc_embeddings = model.encode(
            doc_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        query_embeddings = model.encode(
            query_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarities = np.asarray(query_embeddings) @ np.asarray(doc_embeddings).T

        run: Run = {}
        limit = min(top_k, len(doc_ids))
        for row_index, query_id in enumerate(query_ids):
            indexes = np.argpartition(-similarities[row_index], limit - 1)[:limit]
            ranked = sorted(
                ((doc_ids[index], float(similarities[row_index, index])) for index in indexes),
                key=lambda item: (-item[1], item[0]),
            )
            run[query_id] = dict(ranked)
        return run


class HybridRetriever:
    """Fuse lexical and dense rankings with reciprocal rank fusion."""

    def __init__(self, lexical: Retriever, dense: Retriever, rrf_k: int = 60):
        self.lexical = lexical
        self.dense = dense
        self.rrf_k = rrf_k

    def search(self, corpus: Corpus, queries: Queries, top_k: int) -> Run:
        candidate_k = min(len(corpus), max(top_k * 4, 100))
        lexical_run = self.lexical.search(corpus, queries, candidate_k)
        dense_run = self.dense.search(corpus, queries, candidate_k)
        fused: Run = {}
        for query_id in queries:
            scores: defaultdict[str, float] = defaultdict(float)
            for run in (lexical_run, dense_run):
                ranked_ids = sorted(
                    run.get(query_id, {}),
                    key=lambda doc_id: (-run[query_id][doc_id], doc_id),
                )
                for rank, doc_id in enumerate(ranked_ids, start=1):
                    scores[doc_id] += 1 / (self.rrf_k + rank)
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            fused[query_id] = dict(ranked[:top_k])
        return fused


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
