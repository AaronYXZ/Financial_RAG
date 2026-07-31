"""Prepared lexical and dense retrieval engines."""

from __future__ import annotations

import math
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

Corpus = Mapping[str, Mapping[str, str]]
Queries = Mapping[str, str]
Run = dict[str, dict[str, float]]


def tokenize(text: str) -> list[str]:
    """Normalize text into Unicode word tokens."""

    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)


@dataclass(frozen=True)
class SearchResult:
    """Ranked results and measured latency for each query."""

    run: Run
    query_latencies_ms: tuple[float, ...]


class PreparedBM25Retriever:
    """Okapi BM25 with index construction separated from query execution."""

    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = list(corpus)
        self.postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        self.document_frequency: Counter[str] = Counter()
        self.lengths: list[int] = []
        for doc_index, doc_id in enumerate(self.doc_ids):
            document = corpus[doc_id]
            tokens = tokenize(f"{document.get('title', '')} {document.get('text', '')}")
            counts = Counter(tokens)
            for term, frequency in counts.items():
                self.postings[term].append((doc_index, frequency))
            self.document_frequency.update(counts)
            self.lengths.append(len(tokens))
        self.average_length = (
            sum(self.lengths) / len(self.doc_ids) if self.doc_ids else 1.0
        ) or 1.0

    def search(self, queries: Queries, top_k: int) -> SearchResult:
        run: Run = {}
        latencies: list[float] = []
        document_count = len(self.doc_ids)
        for query_id, query in queries.items():
            started = time.perf_counter()
            scores_by_index: defaultdict[int, float] = defaultdict(float)
            for term in tokenize(query):
                doc_frequency = self.document_frequency.get(term, 0)
                if not doc_frequency:
                    continue
                inverse_document_frequency = math.log(
                    1
                    + (document_count - doc_frequency + 0.5)
                    / (doc_frequency + 0.5)
                )
                for doc_index, frequency in self.postings[term]:
                    denominator = frequency + self.k1 * (
                        1
                        - self.b
                        + self.b * self.lengths[doc_index] / self.average_length
                    )
                    scores_by_index[doc_index] += (
                        inverse_document_frequency
                        * frequency
                        * (self.k1 + 1)
                        / denominator
                    )
            scores = [
                (self.doc_ids[doc_index], score)
                for doc_index, score in scores_by_index.items()
            ]
            scores.sort(key=lambda item: (-item[1], item[0]))
            run[query_id] = dict(scores[:top_k])
            latencies.append((time.perf_counter() - started) * 1000)
        return SearchResult(run=run, query_latencies_ms=tuple(latencies))

    @property
    def index_size_bytes(self) -> int:
        return (
            sys.getsizeof(self.doc_ids)
            + sys.getsizeof(self.postings)
            + sys.getsizeof(self.document_frequency)
            + sys.getsizeof(self.lengths)
            + sum(sys.getsizeof(postings) for postings in self.postings.values())
        )


class PreparedDenseRetriever:
    """Exact cosine retrieval with cached normalized document embeddings."""

    def __init__(
        self,
        corpus: Corpus,
        model_name: str,
        batch_size: int = 32,
        model: Any | None = None,
    ):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Dense retrieval requires retrieval dependencies. "
                "Install them with: pip install -e '.[retrieval]'"
            ) from error

        self.np = np
        self.model_name = model_name
        self.model = model if model is not None else SentenceTransformer(model_name)
        self.doc_ids = list(corpus)
        document_texts = [
            f"{corpus[doc_id].get('title', '')}\n{corpus[doc_id].get('text', '')}".strip()
            for doc_id in self.doc_ids
        ]
        self.document_embeddings = self.model.encode(
            document_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, queries: Queries, top_k: int) -> SearchResult:
        run: Run = {}
        latencies: list[float] = []
        limit = min(top_k, len(self.doc_ids))
        document_embeddings = self.np.asarray(self.document_embeddings)
        for query_id, query in queries.items():
            started = time.perf_counter()
            if limit == 0:
                run[query_id] = {}
            else:
                query_embedding = self.model.encode(
                    [query],
                    batch_size=1,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]
                similarities = self.np.asarray(query_embedding) @ document_embeddings.T
                indexes = self.np.argpartition(-similarities, limit - 1)[:limit]
                ranked = sorted(
                    (
                        (self.doc_ids[index], float(similarities[index]))
                        for index in indexes
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
                run[query_id] = dict(ranked)
            latencies.append((time.perf_counter() - started) * 1000)
        return SearchResult(run=run, query_latencies_ms=tuple(latencies))

    @property
    def index_size_bytes(self) -> int:
        return int(self.np.asarray(self.document_embeddings).nbytes)
