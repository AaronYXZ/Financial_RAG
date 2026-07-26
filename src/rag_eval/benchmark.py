"""Run the controlled SciDocs chunking and retrieval experiment matrix."""

from __future__ import annotations

import csv
import json
import platform
import resource
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from rag_eval.benchmark_retrievers import (
    PreparedBM25Retriever,
    PreparedDenseRetriever,
    SearchResult,
    collapse_to_parents,
    reciprocal_rank_fusion,
)
from rag_eval.chunking import ChunkedCorpus, ChunkingConfig, chunk_corpus, write_chunk_manifest
from rag_eval.data import BeirDataset
from rag_eval.metrics import evaluate
from rag_eval.retrievers import Run


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str = "scidocs"
    split: str = "test"
    chunk_strategies: tuple[str, ...] = ("fixed", "recursive")
    chunk_size: int = 256
    chunk_overlap: int = 32
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    k_values: tuple[int, ...] = (1, 3, 5, 10, 100)
    candidate_multiplier: int = 4
    minimum_candidates: int = 100
    rrf_k: int = 60
    repetitions: int = 3
    seed: int = 42


def run_benchmark(
    dataset: BeirDataset,
    config: BenchmarkConfig,
    output_dir: Path,
) -> dict:
    """Run fixed and recursive chunking against BM25, dense, and hybrid search."""
    _validate_config(config)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    session_id = f"{timestamp}-{config.dataset}-phase2"
    session_dir = output_dir / config.dataset / session_id
    reports: list[dict] = []

    for strategy in config.chunk_strategies:
        chunk_config = ChunkingConfig(
            strategy=strategy,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            model_name=config.model_name,
        )
        chunk_started = time.perf_counter()
        chunked = chunk_corpus(dataset.corpus, chunk_config)
        chunk_seconds = time.perf_counter() - chunk_started
        _validate_chunked_corpus(chunked, dataset)
        chunk_path = session_dir / "chunks" / f"{chunked.manifest_hash}.jsonl"
        write_chunk_manifest(chunked, chunk_path)

        candidate_k = min(
            len(chunked.corpus),
            max(config.minimum_candidates, config.candidate_multiplier * max(config.k_values)),
        )
        parent_k = max(config.k_values)

        lexical_build_started = time.perf_counter()
        lexical = PreparedBM25Retriever(chunked.corpus)
        lexical_build_seconds = time.perf_counter() - lexical_build_started
        lexical_result = _search_repeated(
            lexical, dataset, candidate_k, config.repetitions
        )
        lexical_parent, lexical_winners = collapse_to_parents(
            lexical_result.run, chunked.parent_by_chunk, parent_k
        )

        dense_build_started = time.perf_counter()
        dense = PreparedDenseRetriever(
            chunked.corpus,
            model_name=config.model_name,
            batch_size=config.batch_size,
        )
        dense_build_seconds = time.perf_counter() - dense_build_started
        dense_result = _search_repeated(dense, dataset, candidate_k, config.repetitions)
        dense_parent, dense_winners = collapse_to_parents(
            dense_result.run, chunked.parent_by_chunk, parent_k
        )

        hybrid_parent, fusion_latencies = reciprocal_rank_fusion(
            lexical_parent,
            dense_parent,
            top_k=parent_k,
            rrf_k=config.rrf_k,
        )
        repeated_fusion_latencies = fusion_latencies * config.repetitions
        hybrid_latencies = tuple(
            lexical_ms + dense_ms + fusion_ms
            for lexical_ms, dense_ms, fusion_ms in zip(
                lexical_result.query_latencies_ms,
                dense_result.query_latencies_ms,
                repeated_fusion_latencies,
                strict=True,
            )
        )

        common = {
            "session_id": session_id,
            "chunker": strategy,
            "chunk_manifest_hash": chunked.manifest_hash,
            "chunk_stats": chunked.stats,
            "chunking_seconds": round(chunk_seconds, 4),
        }
        reports.append(
            _build_report(
                dataset,
                config,
                run_id=f"{config.dataset}-{strategy}-bm25",
                retriever="bm25",
                run=lexical_parent,
                winners=lexical_winners,
                latencies=lexical_result.query_latencies_ms,
                build_seconds=lexical_build_seconds,
                index_size_bytes=lexical.index_size_bytes,
                common=common,
                session_dir=session_dir,
            )
        )
        reports.append(
            _build_report(
                dataset,
                config,
                run_id=f"{config.dataset}-{strategy}-dense",
                retriever="dense",
                run=dense_parent,
                winners=dense_winners,
                latencies=dense_result.query_latencies_ms,
                build_seconds=dense_build_seconds,
                index_size_bytes=dense.index_size_bytes,
                common=common,
                session_dir=session_dir,
            )
        )
        reports.append(
            _build_report(
                dataset,
                config,
                run_id=f"{config.dataset}-{strategy}-hybrid",
                retriever="hybrid",
                run=hybrid_parent,
                winners=_merge_winners(lexical_winners, dense_winners),
                latencies=hybrid_latencies,
                build_seconds=lexical_build_seconds + dense_build_seconds,
                index_size_bytes=lexical.index_size_bytes + dense.index_size_bytes,
                common=common,
                session_dir=session_dir,
            )
        )

    summary = {
        "session_id": session_id,
        "created_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "dataset_stats": _dataset_stats(dataset),
        "runs": [
            {
                "run_id": report["run_id"],
                "metrics": report["metrics"],
                "timing": report["timing"],
            }
            for report in reports
        ],
    }
    _write_json(session_dir / "summary.json", summary)
    _write_summary_csv(session_dir / "phase2-summary.csv", reports)
    return summary


def _search_repeated(retriever, dataset: BeirDataset, top_k: int, repetitions: int) -> SearchResult:
    warmup_query_id = next(iter(dataset.queries))
    retriever.search({warmup_query_id: dataset.queries[warmup_query_id]}, top_k)
    last_run: Run = {}
    latencies: list[float] = []
    for _ in range(repetitions):
        result = retriever.search(dataset.queries, top_k)
        last_run = result.run
        latencies.extend(result.query_latencies_ms)
    return SearchResult(run=last_run, query_latencies_ms=tuple(latencies))


def _build_report(
    dataset: BeirDataset,
    config: BenchmarkConfig,
    *,
    run_id: str,
    retriever: str,
    run: Run,
    winners: dict[str, dict[str, str]],
    latencies: tuple[float, ...],
    build_seconds: float,
    index_size_bytes: int,
    common: dict,
    session_dir: Path,
) -> dict:
    report = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "config": {**asdict(config), "retriever": retriever},
        "dataset_stats": _dataset_stats(dataset),
        **common,
        "metrics": evaluate(dataset.qrels, run, config.k_values),
        "timing": {
            "index_build_seconds": round(build_seconds, 4),
            "query_latency_ms": _latency_summary(latencies),
        },
        "resources": {
            "index_size_bytes": index_size_bytes,
            "process_peak_rss_bytes": _peak_rss_bytes(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    _write_json(session_dir / "metrics" / f"{run_id}.json", report)
    _write_rankings(session_dir / "rankings" / f"{run_id}.jsonl", run, winners)
    return report


def _write_rankings(
    path: Path,
    run: Run,
    winners: dict[str, dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for query_id, scores in run.items():
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            for rank, (parent_doc_id, score) in enumerate(ranked, start=1):
                row = {
                    "query_id": query_id,
                    "parent_doc_id": parent_doc_id,
                    "winning_chunk_id": winners.get(query_id, {}).get(parent_doc_id, ""),
                    "rank": rank,
                    "score": score,
                }
                file.write(json.dumps(row, sort_keys=True) + "\n")


def _merge_winners(
    lexical: dict[str, dict[str, str]],
    dense: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for query_id in dict.fromkeys([*lexical, *dense]):
        parent_ids = dict.fromkeys(
            [*lexical.get(query_id, {}), *dense.get(query_id, {})]
        )
        merged[query_id] = {
            parent_id: "|".join(
                filter(
                    None,
                    (
                        lexical.get(query_id, {}).get(parent_id),
                        dense.get(query_id, {}).get(parent_id),
                    ),
                )
            )
            for parent_id in parent_ids
        }
    return merged


def _latency_summary(latencies: tuple[float, ...]) -> dict[str, float | int]:
    return {
        "samples": len(latencies),
        "mean": round(statistics.fmean(latencies), 4) if latencies else 0.0,
        "p50": _percentile(latencies, 50),
        "p95": _percentile(latencies, 95),
        "p99": _percentile(latencies, 99),
    }


def _percentile(values: tuple[float, ...], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 4)


def _dataset_stats(dataset: BeirDataset) -> dict[str, int]:
    return {
        "documents": len(dataset.corpus),
        "queries": len(dataset.queries),
        "positive_judgments": sum(
            relevance > 0
            for judgments in dataset.qrels.values()
            for relevance in judgments.values()
        ),
    }


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_summary_csv(path: Path, reports: list[dict]) -> None:
    rows = [
        {
            "run_id": report["run_id"],
            "chunker": report["chunker"],
            "retriever": report["config"]["retriever"],
            "chunks": report["chunk_stats"]["chunks"],
            "index_build_seconds": report["timing"]["index_build_seconds"],
            "query_p95_ms": report["timing"]["query_latency_ms"]["p95"],
            "index_size_bytes": report["resources"]["index_size_bytes"],
            **report["metrics"],
        }
        for report in reports
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_config(config: BenchmarkConfig) -> None:
    if not config.chunk_strategies:
        raise ValueError("at least one chunk strategy is required")
    if config.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not config.k_values or any(k <= 0 for k in config.k_values):
        raise ValueError("k_values must contain positive integers")


def _validate_chunked_corpus(chunked: ChunkedCorpus, dataset: BeirDataset) -> None:
    if len(chunked.parent_by_chunk) != len(chunked.corpus):
        raise ValueError("every chunk must have exactly one parent")
    if len(set(chunked.parent_by_chunk)) != len(chunked.parent_by_chunk):
        raise ValueError("duplicate chunk IDs detected")
    missing_parents = set(chunked.parent_by_chunk.values()) - set(dataset.corpus)
    if missing_parents:
        raise ValueError(f"chunks reference missing parents: {sorted(missing_parents)[:3]}")
