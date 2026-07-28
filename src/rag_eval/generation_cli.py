"""Command-line preparation and validation for Phase 3 QASPER data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from .generation_adapter import OpenAICompatibleAdapter, OpenAIResponsesAdapter
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

LOCAL_MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
OPENAI_MODEL_ID = "gpt-5"
OPENAI_MODEL_IDS = ("gpt-5", "gpt-5.6-sol", "gpt-5.6-luna")
OPENAI_REASONING_EFFORTS = {
    "gpt-5": ("minimal", "low", "medium", "high"),
    "gpt-5.6-sol": ("none", "low", "medium", "high", "xhigh", "max"),
    "gpt-5.6-luna": ("none", "low", "medium", "high", "xhigh", "max"),
}


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


def _load_cases(cases_path: Path):
    with cases_path.open(encoding="utf-8") as handle:
        return [generation_case_from_dict(json.loads(line)) for line in handle]


def _execute_generation(
    args: argparse.Namespace,
    *,
    track: str,
    context_manifest: str | None = None,
) -> int:
    cases_path = Path(args.cases_file)
    cases = _load_cases(cases_path)
    if args.provider == "openai":
        allowed_efforts = OPENAI_REASONING_EFFORTS[args.openai_model]
        if args.openai_reasoning_effort not in allowed_efforts:
            raise ValueError(
                f"{args.openai_model} supports reasoning efforts {allowed_efforts}, "
                f"not {args.openai_reasoning_effort!r}"
            )
        adapter = OpenAIResponsesAdapter(
            model_id=args.openai_model,
            env_file=Path(args.env_file),
            api_key_env=args.openai_api_key_env,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.openai_reasoning_effort,
            timeout_seconds=args.timeout,
            runtime_retries=args.retries,
        )
    else:
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
    if track == "retrieved-context":
        if not context_manifest:
            raise ValueError("--context-manifest is required for retrieved-context")
        context_manifest_path = Path(context_manifest)
        context_manifest = load_frozen_context_manifest(context_manifest_path)
        if context_manifest["cases_sha256"] != file_sha256(cases_path):
            raise ValueError("Frozen context manifest cases checksum does not match")
        retrieved_contexts = frozen_contexts_by_case(context_manifest)
        context_manifest_sha256 = file_sha256(context_manifest_path)

    counts = run_generation_cases(
        cases,
        adapter=adapter,
        track=track,
        output_file=Path(args.output_file),
        max_context_tokens=args.max_context_tokens,
        max_output_tokens=args.max_output_tokens,
        max_cases=(
            None
            if track == "retrieved-context"
            else getattr(args, "max_cases", None)
        ),
        resume=args.resume,
        retrieved_contexts=retrieved_contexts,
        context_manifest_sha256=context_manifest_sha256,
    )
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 1 if counts["errors"] else 0


def _run(args: argparse.Namespace) -> int:
    return _execute_generation(
        args,
        track=args.track,
        context_manifest=args.context_manifest,
    )


def _generate_oracle(args: argparse.Namespace) -> int:
    return _execute_generation(args, track="oracle-evidence")


def _generate_retrieved(args: argparse.Namespace) -> int:
    return _execute_generation(
        args,
        track="retrieved-context",
        context_manifest=args.context_manifest,
    )


def _freeze_context_payload(
    *,
    cases_path: Path,
    eligibility_file: Path,
    output_file: Path,
    top_k: int,
) -> dict:
    cases = _load_cases(cases_path)
    eligibility = load_eligibility_manifest(eligibility_file)
    case_ids = eligibility["eligible_case_ids"]
    contexts = freeze_bm25_contexts(
        cases,
        eligible_case_ids=case_ids,
        top_k=top_k,
    )
    return write_frozen_context_manifest(
        output_file,
        cases_file=cases_path,
        eligibility_file=eligibility_file,
        eligible_case_ids=case_ids,
        contexts=contexts,
        top_k=top_k,
    )


def _freeze_context(args: argparse.Namespace) -> int:
    payload = _freeze_context_payload(
        cases_path=Path(args.cases_file),
        eligibility_file=Path(args.eligibility_file),
        output_file=Path(args.output_file),
        top_k=args.top_k,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _generate_end_to_end(args: argparse.Namespace) -> int:
    context_manifest = Path(args.context_manifest)
    payload = _freeze_context_payload(
        cases_path=Path(args.cases_file),
        eligibility_file=Path(args.eligibility_file),
        output_file=context_manifest,
        top_k=args.top_k,
    )
    print(
        json.dumps(
            {
                "retrieval": {
                    "context_manifest": str(context_manifest),
                    "eligible_case_count": payload["eligible_case_count"],
                    "retriever": payload["retriever"],
                }
            },
            indent=2,
            sort_keys=True,
        )
    )
    return _execute_generation(
        args,
        track="retrieved-context",
        context_manifest=str(context_manifest),
    )


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


def _add_generation_arguments(
    command: argparse.ArgumentParser,
    *,
    default_output_file: str,
    include_max_cases: bool,
) -> None:
    command.add_argument(
        "--cases-file",
        default="data/generation/qasper-v1/validation.cases.jsonl",
    )
    command.add_argument("--output-file", default=default_output_file)
    _add_provider_arguments(command)
    command.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    command.add_argument(
        "--model",
        default=LOCAL_MODEL_ID,
        help="Model ID exposed by the local OpenAI-compatible server.",
    )
    command.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    command.add_argument("--max-context-tokens", type=int, default=32_768)
    command.add_argument("--max-output-tokens", type=int, default=1024)
    if include_max_cases:
        command.add_argument("--max-cases", type=int, default=25)
    command.add_argument("--temperature", type=float, default=0.0)
    command.add_argument("--timeout", type=float, default=300.0)
    command.add_argument("--retries", type=int, default=1)
    command.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Overwrite the prediction file instead of resuming it",
    )
    command.set_defaults(resume=True)


def _add_provider_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--provider",
        choices=("local", "openai"),
        default="local",
    )
    command.add_argument(
        "--openai-model",
        choices=OPENAI_MODEL_IDS,
        default=OPENAI_MODEL_ID,
        help="OpenAI model used when --provider openai is selected.",
    )
    command.add_argument(
        "--env-file",
        default=".env",
        help="Environment file containing the OpenAI API key.",
    )
    command.add_argument(
        "--openai-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that contains the OpenAI API key.",
    )
    command.add_argument(
        "--openai-reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default="low",
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
            "qwen3-4b-retrieved-bm25-top5-v1.jsonl"
        ),
        include_max_cases=False,
    )
    retrieved.add_argument(
        "--context-manifest",
        required=True,
        help="Previously frozen retrieval manifest.",
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
            "qwen3-4b-end-to-end-bm25-top5-v1.jsonl"
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
        default="data/generation/qasper-v1/retrieval/end-to-end-bm25-top5-v1.json",
        help="Path where the end-to-end command freezes its retrieval output.",
    )
    end_to_end.add_argument("--top-k", type=int, default=5)
    end_to_end.set_defaults(handler=_generate_end_to_end)
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
