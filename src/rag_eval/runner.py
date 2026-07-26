"""Orchestrate retrieval experiments and persist reproducible reports."""

from __future__ import annotations

import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from rag_eval.data import BeirDataset
from rag_eval.metrics import evaluate
from rag_eval.retrievers import Retriever


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    split: str
    retriever: str
    model: str | None
    k_values: tuple[int, ...]
    max_documents: int | None
    max_queries: int | None
    seed: int


def run_experiment(
    dataset: BeirDataset,
    retriever: Retriever,
    config: ExperimentConfig,
    output_dir: Path,
) -> dict:
    started = time.perf_counter()
    run = retriever.search(dataset.corpus, dataset.queries, max(config.k_values))
    elapsed = time.perf_counter() - started
    metrics = evaluate(dataset.qrels, run, config.k_values)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"{timestamp}-{config.dataset}-{config.retriever}"

    report = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "dataset_stats": {
            "documents": len(dataset.corpus),
            "queries": len(dataset.queries),
            "positive_judgments": sum(
                relevance > 0
                for judgments in dataset.qrels.values()
                for relevance in judgments.values()
            ),
        },
        "metrics": metrics,
        "timing": {
            "total_seconds": round(elapsed, 4),
            "milliseconds_per_query": round(elapsed * 1000 / len(dataset.queries), 4),
        },
        "environment": {"python": platform.python_version()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{experiment_id}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _append_summary(output_dir / "experiments.csv", report)
    return report


def _append_summary(path: Path, report: dict) -> None:
    row = {
        "experiment_id": report["experiment_id"],
        "dataset": report["config"]["dataset"],
        "retriever": report["config"]["retriever"],
        "model": report["config"]["model"] or "",
        "documents": report["dataset_stats"]["documents"],
        "queries": report["dataset_stats"]["queries"],
        "total_seconds": report["timing"]["total_seconds"],
        **report["metrics"],
    }
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
