"""Command-line interface for retrieval evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_eval.data import download_dataset, load_dataset, sample_dataset
from rag_eval.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from rag_eval.runner import ExperimentConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Evaluate BM25, dense, or hybrid retrieval on a BEIR dataset.",
    )
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name")
    parser.add_argument("--split", default="test", help="qrels split")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--retriever",
        choices=("bm25", "dense", "hybrid"),
        default="bm25",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model for dense and hybrid retrieval",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k", type=_parse_k_values, default=(1, 3, 5, 10))
    parser.add_argument("--max-documents", type=_positive_int)
    parser.add_argument("--max-queries", type=_positive_int)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset_dir = download_dataset(args.dataset, args.data_dir)
    dataset = load_dataset(dataset_dir, split=args.split)
    dataset = sample_dataset(
        dataset,
        max_documents=args.max_documents,
        max_queries=args.max_queries,
        seed=args.seed,
    )

    dense = DenseRetriever(model_name=args.model, batch_size=args.batch_size)
    if args.retriever == "bm25":
        retriever = BM25Retriever()
        model = None
    elif args.retriever == "dense":
        retriever = dense
        model = args.model
    else:
        retriever = HybridRetriever(BM25Retriever(), dense)
        model = args.model

    config = ExperimentConfig(
        dataset=args.dataset,
        split=args.split,
        retriever=args.retriever,
        model=model,
        k_values=args.k,
        max_documents=args.max_documents,
        max_queries=args.max_queries,
        seed=args.seed,
    )
    report = run_experiment(dataset, retriever, config, args.output_dir)
    print(json.dumps(report, indent=2))


def _parse_k_values(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--k must be comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("--k values must be positive")
    return values


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


if __name__ == "__main__":
    main()
