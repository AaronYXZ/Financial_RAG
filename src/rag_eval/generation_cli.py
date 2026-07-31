"""Stable CLI façade for QASPER benchmark workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from .end_to_end.workflow import (
    freeze_context_manifest,
    run_retrieved_context_generation,
    run_retrieve_then_generate,
)
from .generation.adapter import (
    OpenAICompatibleAdapter,
    OpenAIResponsesAdapter,
    OpenRouterChatAdapter,
)
from .generation.cost import estimate_openai_cost
from .generation.comparison import (
    compare_evaluated_runs,
    compare_response_validity,
    intersect_eligibility_manifests,
    write_comparison,
)
from .generation.metrics import evaluate_prediction_files
from .generation.runner import eligibility_manifest_path, run_generation_cases
from .retrieval.context import (
    DENSE_MODEL_1,
    DENSE_MODEL_2,
)
from .retrieval.evaluation import compare_retrieval_to_oracle

from .generation.data import (
    QASPER_DATASET,
    QASPER_PARQUET_REVISION,
    QASPER_VERSION,
    generation_case_from_dict,
    load_qasper_cases,
    qasper_parquet_sha256,
    qasper_parquet_url,
)

LOCAL_MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
OPENAI_MODEL_ID = "gpt-5"
OPENROUTER_MODEL_ID = "openai/gpt-5.6-luna-pro"
OPENROUTER_FALLBACK_MODEL_IDS = (
    "qwen/qwen3.7-plus",
    "deepseek/deepseek-v4-flash",
)
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
        "schema_version": 2,
        "dataset": QASPER_DATASET,
        "dataset_config": "qasper",
        "dataset_revision": args.revision,
        "dataset_version": QASPER_VERSION,
        "source_parquet_sha256": qasper_parquet_sha256(
            args.split,
            args.revision,
        ),
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


def _openrouter_fallback_models(args: argparse.Namespace) -> tuple[str, ...]:
    configured = args.openrouter_fallback_model
    if configured is None:
        return OPENROUTER_FALLBACK_MODEL_IDS
    return tuple(configured)


def _execute_generation(
    args: argparse.Namespace,
    *,
    track: str,
    context_manifest: str | None = None,
) -> int:
    cases_path = Path(args.cases_file)
    adapter = _build_adapter(args)
    if track == "retrieved-context":
        if not context_manifest:
            raise ValueError("--context-manifest is required for retrieved-context")
        counts = run_retrieved_context_generation(
            cases_file=cases_path,
            context_manifest_file=Path(context_manifest),
            adapter=adapter,
            output_file=Path(args.output_file),
            max_context_tokens=args.max_context_tokens,
            max_output_tokens=args.max_output_tokens,
            max_cases=getattr(args, "max_cases", None),
            resume=args.resume,
        )
    else:
        counts = run_generation_cases(
            _load_cases(cases_path),
            adapter=adapter,
            track=track,
            output_file=Path(args.output_file),
            max_context_tokens=args.max_context_tokens,
            max_output_tokens=args.max_output_tokens,
            max_cases=getattr(args, "max_cases", None),
            resume=args.resume,
        )
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 1 if counts["errors"] else 0


def _build_adapter(args: argparse.Namespace):
    if args.provider == "openai":
        allowed_efforts = OPENAI_REASONING_EFFORTS[args.openai_model]
        if args.openai_reasoning_effort not in allowed_efforts:
            raise ValueError(
                f"{args.openai_model} supports reasoning efforts {allowed_efforts}, "
                f"not {args.openai_reasoning_effort!r}"
            )
        return OpenAIResponsesAdapter(
            model_id=args.openai_model,
            env_file=Path(args.env_file),
            api_key_env=args.openai_api_key_env,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.openai_reasoning_effort,
            timeout_seconds=args.timeout,
            runtime_retries=args.retries,
        )
    elif args.provider == "openrouter":
        return OpenRouterChatAdapter(
            model_id=args.openrouter_model,
            env_file=Path(args.env_file),
            api_key_env=args.openrouter_api_key_env,
            base_url=args.openrouter_base_url,
            http_referer=args.openrouter_http_referer,
            app_title=args.openrouter_app_title,
            fallback_model_ids=_openrouter_fallback_models(args),
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            timeout_seconds=args.timeout,
            runtime_retries=args.retries,
        )
    else:
        return OpenAICompatibleAdapter(
            base_url=args.base_url,
            model_id=args.model,
            tokenizer_id=args.tokenizer,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            timeout_seconds=args.timeout,
            runtime_retries=args.retries,
        )


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


def _freeze_context(args: argparse.Namespace) -> int:
    payload = freeze_context_manifest(
        cases_file=Path(args.cases_file),
        eligibility_file=Path(args.eligibility_file),
        output_file=Path(args.output_file),
        top_k=args.top_k,
        retriever=args.retriever,
        retrieval_scope=args.retrieval_scope,
        dense_model=args.dense_model,
        dense_batch_size=args.dense_batch_size,
        hybrid_rrf_k=args.hybrid_rrf_k,
        hybrid_candidate_k=args.hybrid_candidate_k,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _generate_end_to_end(args: argparse.Namespace) -> int:
    context_manifest = Path(args.context_manifest)
    payload, counts = run_retrieve_then_generate(
        cases_file=Path(args.cases_file),
        eligibility_file=Path(args.eligibility_file),
        context_manifest_file=context_manifest,
        adapter=_build_adapter(args),
        output_file=Path(args.output_file),
        top_k=args.top_k,
        retriever=args.retriever,
        retrieval_scope=args.retrieval_scope,
        dense_model=args.dense_model,
        dense_batch_size=args.dense_batch_size,
        hybrid_rrf_k=args.hybrid_rrf_k,
        hybrid_candidate_k=args.hybrid_candidate_k,
        max_context_tokens=args.max_context_tokens,
        max_output_tokens=args.max_output_tokens,
        max_cases=getattr(args, "max_cases", None),
        resume=args.resume,
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
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 1 if counts["errors"] else 0


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


def _compare_metrics(args: argparse.Namespace) -> int:
    comparison = compare_evaluated_runs(
        baseline_per_case_file=Path(args.baseline_per_case_file),
        candidate_per_case_file=Path(args.candidate_per_case_file),
        track=args.track,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    if args.output_file:
        write_comparison(Path(args.output_file), comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


def _intersect_eligibility(args: argparse.Namespace) -> int:
    payload = intersect_eligibility_manifests(
        [Path(path) for path in args.eligibility_file]
    )
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _compare_retrieval(args: argparse.Namespace) -> int:
    payload = compare_retrieval_to_oracle(
        cases_file=Path(args.cases_file),
        context_manifest_files=[Path(path) for path in args.context_manifest],
    )
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _estimate_cost(args: argparse.Namespace) -> int:
    payload = estimate_openai_cost(
        predictions_file=Path(args.predictions_file),
        output_usage_file=(
            Path(args.output_usage_file) if args.output_usage_file else None
        ),
        model=args.model,
        target_case_count=args.target_case_count,
        max_output_tokens=args.max_output_tokens,
        retries=args.retries,
        budget_usd=args.budget_usd,
        budget_basis=args.budget_basis,
    )
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["budget_gate"]["approved"] is not False else 2


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
        command.add_argument(
            "--all-cases",
            action="store_const",
            const=None,
            dest="max_cases",
            help="Run every eligible case instead of the 25-case smoke default.",
        )
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
        choices=("local", "openai", "openrouter"),
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
        help="Environment file containing provider API keys.",
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
    command.add_argument(
        "--openrouter-model",
        default=OPENROUTER_MODEL_ID,
        help=(
            "OpenRouter model slug used when --provider openrouter is selected. "
            "Any model with structured-output support may be specified."
        ),
    )
    command.add_argument(
        "--openrouter-api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable that contains the OpenRouter API key.",
    )
    command.add_argument(
        "--openrouter-base-url",
        default="https://openrouter.ai/api/v1",
        help="OpenRouter-compatible API base URL.",
    )
    command.add_argument(
        "--openrouter-http-referer",
        help="Optional site URL sent in OpenRouter's HTTP-Referer attribution header.",
    )
    command.add_argument(
        "--openrouter-app-title",
        default="Project Local RAG",
        help="Optional app name sent in OpenRouter's X-OpenRouter-Title header.",
    )
    command.add_argument(
        "--openrouter-fallback-model",
        action="append",
        default=None,
        help=(
            "Fallback model slug, tried in order after the primary. Repeat to add "
            "models. Defaults to Qwen3.7 Plus, then DeepSeek V4 Flash."
        ),
    )
    command.add_argument(
        "--no-openrouter-fallbacks",
        action="store_const",
        const=[],
        dest="openrouter_fallback_model",
        help="Disable the default OpenRouter model fallback chain.",
    )


def _add_retrieval_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--retriever",
        choices=("bm25", "dense", "hybrid"),
        default="bm25",
        help="Retrieval method used to create the frozen context manifest.",
    )
    command.add_argument(
        "--retrieval-scope",
        choices=("paper", "corpus"),
        default="paper",
        help="Search only the associated paper or the complete cases-file corpus.",
    )
    command.add_argument(
        "--dense-model",
        default=DENSE_MODEL_1,
        help=(
            "Sentence-transformers model used by dense and hybrid retrieval. "
            f"Common alternatives: {DENSE_MODEL_1}, {DENSE_MODEL_2}."
        ),
    )
    command.add_argument("--dense-batch-size", type=int, default=32)
    command.add_argument("--hybrid-rrf-k", type=int, default=60)
    command.add_argument(
        "--hybrid-candidate-k",
        type=int,
        help="Candidates per component before RRF. Defaults to max(top_k * 4, 100).",
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

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
