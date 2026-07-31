"""Rank fusion for retrieval engines."""

from __future__ import annotations

import time
from collections import defaultdict

from .engines import Run


def reciprocal_rank_fusion(
    lexical_run: Run,
    dense_run: Run,
    top_k: int,
    rrf_k: int = 60,
) -> tuple[Run, tuple[float, ...]]:
    """Fuse two rankings and return per-query fusion latency."""

    fused: Run = {}
    latencies: list[float] = []
    query_ids = list(dict.fromkeys([*lexical_run, *dense_run]))
    for query_id in query_ids:
        started = time.perf_counter()
        scores: defaultdict[str, float] = defaultdict(float)
        for run in (lexical_run, dense_run):
            ranked_ids = sorted(
                run.get(query_id, {}),
                key=lambda doc_id: (-run[query_id][doc_id], doc_id),
            )
            for rank, doc_id in enumerate(ranked_ids, start=1):
                scores[doc_id] += 1 / (rrf_k + rank)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        fused[query_id] = dict(ranked[:top_k])
        latencies.append((time.perf_counter() - started) * 1000)
    return fused, tuple(latencies)
