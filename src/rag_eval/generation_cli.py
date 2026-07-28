"""Command-line preparation and validation for Phase 3 QASPER data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from .generation_adapter import OpenAICompatibleAdapter
from .generation_comparison import compare_response_validity, write_comparison
from .generation_metrics import evaluate_prediction_files, load_eligibility_manifest
from .generation_runner import eligibility_manifest_path, run_generation_cases
from .generation_retrieval import (
    file_sha256,
    freeze_bm25_contexts,
    frozen_contexts_by_case,
    load_frozen_context_manifest,
    write_frozen_context_manifest,
)

from .generation_data import (
    QASPER_DATASET,
    QASPER_PARQUET_REVISION,
    QASPER_VERSION,
    generation_case_from_dict,
    load_qasper_cases,
    qasper_parquet_url,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare(args: argparse.Namespace) -> int:
    cases = load_qasper_cases(
        args.split,
        cache_dir=args.cache_dir,
        revision=args.revision,
        limit_papers=args.limit_papers,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / f"{args.split}.cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    labels = Counter(case.answerability for case in cases)
    unresolved = sum(
        len(reference.unresolved_evidence)
        for case in cases
        for reference in case.references
    )
    manifest = {
        "schema_version": 1,
        "dataset": QASPER_DATASET,
        "dataset_config": "qasper",
        "dataset_revision": args.revision,
        "dataset_version": QASPER_VERSION,
        "source_parquet_url": qasper_parquet_url(args.split, args.revision),
        "split": args.split,
        "limit_papers": args.limit_papers,
        "case_count": len(cases),
        "answerability_counts": dict(sorted(labels.items())),
        "unresolved_evidence_count": unresolved,
        "cases_file": cases_path.name,
        "cases_sha256": _sha256(cases_path),
    }
    manifest_path = output_dir / f"{args.split}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    cases_path = Path(args.cases_file)
    with cases_path.open(encoding="utf-8") as handle:
        cases = [generation_case_from_dict(json.loads(line)) for line in handle]
    adapter = OpenAICompatibleAdapter(
        base_url=args.base_url,
        model_id=args.model,
        tokenizer_id=args.tokenizer,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
        runtime_retries=args.retries,
    )
    retrieved_contexts = None
    context_manifest_sha256 = None
    if args.track == "retrieved-context":
        if not args.context_manifest:
            raise ValueError("--context-manifest is required for retrieved-context")
        context_manifest_path = Path(args.context_manifest)
        context_manifest = load_frozen_context_manifest(context_manifest_path)
        if context_manifest["cases_sha256"] != file_sha256(cases_path):
            raise ValueError("Frozen context manifest cases checksum does not match")
        retrieved_contexts = frozen_contexts_by_case(context_manifest)
        context_manifest_sha256 = file_sha256(context_manifest_path)

    counts = run_generation_cases(
        cases,
        adapter=adapter,
        track=args.track,
        output_file=Path(args.output_file),
        max_context_tokens=args.max_context_tokens,
        max_output_tokens=args.max_output_tokens,
        max_cases=None if args.track == "retrieved-context" else args.max_cases,
        resume=args.resume,
        retrieved_contexts=retrieved_contexts,
        context_manifest_sha256=context_manifest_sha256,
    )
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 1 if counts["errors"] else 0


def _freeze_context(args: argparse.Namespace) -> int:
    cases_path = Path(args.cases_file)
    with cases_path.open(encoding="utf-8") as handle:
        cases = [generation_case_from_dict(json.loads(line)) for line in handle]
    eligibility_file = Path(args.eligibility_file)
    eligibility = load_eligibility_manifest(eligibility_file)
    case_ids = eligibility["eligible_case_ids"]
    contexts = freeze_bm25_contexts(
        cases,
        eligible_case_ids=case_ids,
        top_k=args.top_k,
    )
    payload = write_frozen_context_manifest(
        Path(args.output_file),
        cases_file=cases_path,
        eligibility_file=eligibility_file,
        eligible_case_ids=case_ids,
        contexts=contexts,
        top_k=args.top_k,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _metrics(args: argparse.Namespace) -> int:
    predictions_file = Path(args.predictions_file)
    eligibility_file = (
        Path(args.eligibility_file)
        if args.eligibility_file
        else eligibility_manifest_path(predictions_file)
    )
    summary = evaluate_prediction_files(
        cases_file=Path(args.cases_file),
        predictions_file=predictions_file,
        eligibility_file=eligibility_file,
        output_dir=Path(args.output_dir),
        track=args.track,
        model_id=args.model,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _compare_responses(args: argparse.Namespace) -> int:
    comparison = compare_response_validity(
        baseline_predictions_file=Path(args.baseline_predictions_file),
        baseline_eligibility_file=Path(args.baseline_eligibility_file),
        candidate_predictions_file=Path(args.candidate_predictions_file),
        candidate_eligibility_file=Path(args.candidate_eligibility_file),
        track=args.track,
        model_id=args.model,
    )
    if args.output_file:
        write_comparison(Path(args.output_file), comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


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
        default="mlx-community/Qwen3-4B-Instruct-2507-4bit",
    )
    run.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    run.add_argument("--max-context-tokens", type=int, default=32_768)
    run.add_argument("--max-output-tokens", type=int, default=512)
    run.add_argument("--max-cases", type=int, default=25)
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
    freeze = subparsers.add_parser(
        "freeze-context",
        help="Freeze BM25 top-K passages for an existing eligible-case set",
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
        default="data/generation/qasper-v1/retrieval/qwen3-4b-bm25-top5.json",
    )
    freeze.add_argument("--top-k", type=int, default=5)
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
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
