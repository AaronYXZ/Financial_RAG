"""Stable parser composition for all QASPER benchmark commands."""

from __future__ import annotations

import argparse

from ..end_to_end.cli import _generate_end_to_end, _generate_retrieved
from ..generation.cli import (
    _compare_metrics,
    _compare_responses,
    _estimate_cost,
    _generate_oracle,
    _intersect_eligibility,
    _metrics,
    _prepare,
    _run,
)
from ..generation.data import QASPER_PARQUET_REVISION
from ..retrieval.cli import (
    _add_retrieval_arguments,
    _compare_retrieval,
    _freeze_context,
)
from .common import (
    LOCAL_MODEL_ID,
    OPENAI_MODEL_ID,
    OPENAI_MODEL_IDS,
    _add_generation_arguments,
    _add_provider_arguments,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-generation",
        description="Prepare QASPER cases for fixed-context generation evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Download and normalize a QASPER split")
    prepare.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    prepare.add_argument("--output-dir", default="data/generation/qasper-v1")
    prepare.add_argument("--cache-dir")
    prepare.add_argument("--revision", default=QASPER_PARQUET_REVISION)
    prepare.add_argument("--limit-papers", type=int)
    prepare.set_defaults(handler=_prepare)
    run = subparsers.add_parser("run", help="Run a fixed-context generation smoke test")
    run.add_argument("--cases-file", default="data/generation/qasper-v1/validation.cases.jsonl")
    tracks = ("oracle-evidence", "complete-paper", "retrieved-context")
    run.add_argument("--track", choices=tracks, default="oracle-evidence")
    _add_provider_arguments(run)
    run.add_argument(
        "--output-file",
        default="results/generation/qasper-v1/predictions/qwen3-4b-track-a-v2.jsonl",
    )
    run.add_argument(
        "--context-manifest",
        help="Frozen retrieval manifest. Required for the retrieved-context track.",
    )
    run.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    run.add_argument(
        "--model",
        default=LOCAL_MODEL_ID,
        help="Model ID exposed by the local OpenAI-compatible server.",
    )
    run.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    run.add_argument("--max-context-tokens", type=int, default=32_768)
    run.add_argument("--max-output-tokens", type=int, default=1024)
    run.add_argument("--max-cases", type=int, default=25)
    run.add_argument(
        "--all-cases",
        action="store_const",
        const=None,
        dest="max_cases",
        help="Run every eligible case instead of the 25-case smoke default.",
    )
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--timeout", type=float, default=300.0)
    run.add_argument("--retries", type=int, default=1)
    run.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Overwrite the prediction file instead of resuming it",
    )
    run.set_defaults(handler=_run, resume=True)
    oracle = subparsers.add_parser(
        "generate-oracle",
        help="Generate answers from oracle evidence only",
    )
    _add_generation_arguments(
        oracle,
        default_output_file=(
            "results/generation/qasper-v1/predictions/"
            "qwen3-4b-oracle-v2.jsonl"
        ),
        include_max_cases=True,
    )
    oracle.set_defaults(handler=_generate_oracle)
    retrieved = subparsers.add_parser(
        "generate-retrieved",
        help="Generate answers from a previously frozen retrieval manifest",
    )
    _add_generation_arguments(
        retrieved,
        default_output_file=(
            "results/generation/qasper-v1/predictions/"
            "qwen3-4b-retrieved-v3.jsonl"
        ),
        include_max_cases=False,
    )
    retrieved.add_argument(
        "--context-manifest",
        required=True,
        help="Previously frozen retrieval manifest.",
    )
    retrieved.add_argument(
        "--max-cases",
        type=int,
        help="Optional ordered pilot size. Defaults to the full frozen manifest.",
    )
    retrieved.set_defaults(handler=_generate_retrieved)
    end_to_end = subparsers.add_parser(
        "generate-end-to-end",
        help="Retrieve, freeze context, and generate in one reproducible command",
    )
    _add_generation_arguments(
        end_to_end,
        default_output_file=(
            "results/generation/qasper-v1/predictions/"
            "qwen3-4b-end-to-end-v3.jsonl"
        ),
        include_max_cases=False,
    )
    end_to_end.add_argument(
        "--eligibility-file",
        required=True,
        help="Frozen eligible-case set to retrieve and generate for.",
    )
    end_to_end.add_argument(
        "--context-manifest",
        default="data/generation/qasper-v1/retrieval/end-to-end-retrieval-v2.json",
        help="Path where the end-to-end command freezes its retrieval output.",
    )
    end_to_end.add_argument("--top-k", type=int, default=5)
    _add_retrieval_arguments(end_to_end)
    end_to_end.set_defaults(handler=_generate_end_to_end)
    freeze = subparsers.add_parser(
        "freeze-context",
        help="Freeze BM25, dense, or hybrid passages for fixed-context generation",
    )
    freeze.add_argument(
        "--cases-file",
        default="data/generation/qasper-v1/validation.cases.jsonl",
    )
    freeze.add_argument(
        "--eligibility-file",
        required=True,
        help="Source Track A or Track B eligibility manifest that fixes the questions.",
    )
    freeze.add_argument(
        "--output-file",
        default="data/generation/qasper-v1/retrieval/bm25-paper-top5.json",
    )
    freeze.add_argument("--top-k", type=int, default=5)
    _add_retrieval_arguments(freeze)
    freeze.set_defaults(handler=_freeze_context)
    metrics = subparsers.add_parser(
        "metrics",
        help="Evaluate generation predictions against the frozen eligible-case set",
    )
    metrics.add_argument(
        "--cases-file",
        default="data/generation/qasper-v1/validation.cases.jsonl",
    )
    metrics.add_argument(
        "--predictions-file",
        default="results/generation/qasper-v1/predictions/qwen3-4b-track-a-v2.jsonl",
    )
    metrics.add_argument(
        "--eligibility-file",
        help="Eligibility manifest. Defaults to the prediction sidecar.",
    )
    metrics.add_argument(
        "--output-dir",
        default="results/generation/qasper-v1/metrics/qwen3-4b-track-a-v2",
    )
    metrics.add_argument(
        "--track", choices=tracks, default="oracle-evidence"
    )
    metrics.add_argument(
        "--model", default="mlx-community/Qwen3-4B-Instruct-2507-4bit"
    )
    metrics.set_defaults(handler=_metrics)
    compare = subparsers.add_parser(
        "compare-responses",
        help="Compare response validity on identical eligible cases",
    )
    compare.add_argument("--baseline-predictions-file", required=True)
    compare.add_argument("--baseline-eligibility-file", required=True)
    compare.add_argument("--candidate-predictions-file", required=True)
    compare.add_argument("--candidate-eligibility-file", required=True)
    compare.add_argument("--output-file")
    compare.add_argument("--track", choices=tracks, default="complete-paper")
    compare.add_argument(
        "--model", default="mlx-community/Qwen3-4B-Instruct-2507-4bit"
    )
    compare.set_defaults(handler=_compare_responses)
    compare_metrics = subparsers.add_parser(
        "compare-metrics",
        help="Compare matched per-case metrics with paired paper bootstrap",
    )
    compare_metrics.add_argument("--baseline-per-case-file", required=True)
    compare_metrics.add_argument("--candidate-per-case-file", required=True)
    compare_metrics.add_argument("--baseline-label", required=True)
    compare_metrics.add_argument("--candidate-label", required=True)
    compare_metrics.add_argument("--output-file")
    compare_metrics.add_argument("--track", choices=tracks, required=True)
    compare_metrics.add_argument("--bootstrap-resamples", type=int, default=10_000)
    compare_metrics.add_argument("--bootstrap-seed", type=int, default=42)
    compare_metrics.set_defaults(handler=_compare_metrics)
    intersect = subparsers.add_parser(
        "intersect-eligibility",
        help="Freeze the ordered common eligible cases across matched systems",
    )
    intersect.add_argument(
        "--eligibility-file",
        action="append",
        required=True,
        help="Eligibility manifest. Pass once per system.",
    )
    intersect.add_argument("--output-file", required=True)
    intersect.set_defaults(handler=_intersect_eligibility)
    retrieval_compare = subparsers.add_parser(
        "compare-retrieval",
        help="Rank frozen retrieval contexts against the oracle evidence ceiling",
    )
    retrieval_compare.add_argument(
        "--cases-file",
        default="data/generation/qasper-v1/validation.cases.jsonl",
    )
    retrieval_compare.add_argument(
        "--context-manifest",
        action="append",
        required=True,
        help="Frozen context manifest. Pass once per retrieval configuration.",
    )
    retrieval_compare.add_argument("--output-file", required=True)
    retrieval_compare.set_defaults(handler=_compare_retrieval)
    cost = subparsers.add_parser(
        "estimate-cost",
        help="Estimate full OpenAI run cost from observed pilot token usage",
    )
    cost.add_argument("--predictions-file", required=True)
    cost.add_argument(
        "--output-usage-file",
        help=(
            "Optional comparable pilot used only for output-token distribution "
            "when the primary pilot has no successful outputs."
        ),
    )
    cost.add_argument("--model", choices=OPENAI_MODEL_IDS, default=OPENAI_MODEL_ID)
    cost.add_argument("--target-case-count", type=int)
    cost.add_argument("--max-output-tokens", type=int, default=1024)
    cost.add_argument("--retries", type=int, default=1)
    cost.add_argument("--budget-usd", type=float)
    cost.add_argument(
        "--budget-basis",
        choices=(
            "expected",
            "observed_p95_output",
            "ceiling",
            "ceiling_with_retries",
        ),
        default="ceiling_with_retries",
    )
    cost.add_argument("--output-file")
    cost.set_defaults(handler=_estimate_cost)
    return parser
