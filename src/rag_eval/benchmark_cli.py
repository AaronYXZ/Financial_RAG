"""Command-line interface for the SciDocs retrieval benchmark matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_eval.benchmark import BenchmarkConfig, run_benchmark
from rag_eval.data import download_dataset, load_dataset, sample_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-benchmark",
        description=(
            "Benchmark fixed and recursive LangChain chunking with BM25, dense, "
            "and hybrid retrieval."
        ),
    )
    parser.add_argument("--dataset", default="scidocs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--chunker",
        choices=("both", "fixed", "recursive"),
        default="both",
    )
    parser.add_argument("--chunk-size", type=_positive_int, default=256)
    parser.add_argument("--chunk-overlap", type=_non_negative_int, default=32)
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--k", type=_parse_k_values, default=(1, 3, 5, 10, 100))
    parser.add_argument("--repetitions", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-documents", type=_positive_int)
    parser.add_argument("--max-queries", type=_positive_int)
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and validate the dataset without running experiments.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset_dir = download_dataset(args.dataset, args.data_dir)
    dataset = load_dataset(dataset_dir, split=args.split)
    if args.download_only:
        print(
            json.dumps(
                {
                    "dataset_dir": str(dataset_dir),
                    "documents": len(dataset.corpus),
                    "queries": len(dataset.queries),
                },
                indent=2,
            )
        )
        return

    dataset = sample_dataset(
        dataset,
        max_documents=args.max_documents,
        max_queries=args.max_queries,
        seed=args.seed,
    )
    strategies = (
        ("fixed", "recursive") if args.chunker == "both" else (args.chunker,)
    )
    config = BenchmarkConfig(
        dataset=args.dataset,
        split=args.split,
        chunk_strategies=strategies,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model,
        batch_size=args.batch_size,
        k_values=args.k,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    summary = run_benchmark(dataset, config, args.output_dir)


    print(json.dumps(summary, indent=2))


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


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


if __name__ == "__main__":
    main()
