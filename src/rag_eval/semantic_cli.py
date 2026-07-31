"""CLI for blinded semantic evaluation of QASPER generation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .generation.adapter import OpenAIResponsesAdapter, OpenRouterChatAdapter
from .semantic.evaluation import (
    aggregate_semantic_judgments,
    prepare_blinded_semantic_inputs,
    run_semantic_judgments,
)
from .semantic.contracts import SEMANTIC_RESPONSE_SCHEMA


DEFAULT_OPENROUTER_JUDGE = "anthropic/claude-sonnet-4.5"


def _prepare(args: argparse.Namespace) -> int:
    manifest = prepare_blinded_semantic_inputs(
        cases_file=Path(args.cases_file),
        predictions_file=Path(args.predictions_file),
        output_file=Path(args.output_file),
        track=args.track,
        generator_model_id=args.generator_model,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.provider == "openai":
        adapter = OpenAIResponsesAdapter(
            model_id=args.judge_model,
            env_file=Path(args.env_file),
            api_key_env=args.openai_api_key_env,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=args.timeout,
            runtime_retries=args.retries,
            response_schema=SEMANTIC_RESPONSE_SCHEMA,
            response_schema_name="semantic_judgment",
        )
    else:
        adapter = OpenRouterChatAdapter(
            model_id=args.judge_model,
            env_file=Path(args.env_file),
            api_key_env=args.openrouter_api_key_env,
            base_url=args.openrouter_base_url,
            fallback_model_ids=tuple(args.fallback_model),
            max_output_tokens=args.max_output_tokens,
            temperature=0.0,
            timeout_seconds=args.timeout,
            runtime_retries=args.retries,
            response_schema=SEMANTIC_RESPONSE_SCHEMA,
            response_schema_name="semantic_judgment",
        )
    counts = run_semantic_judgments(
        inputs_file=Path(args.inputs_file),
        manifest_file=Path(args.manifest_file) if args.manifest_file else None,
        adapter=adapter,
        output_file=Path(args.output_file),
        max_cases=args.max_cases,
        resume=args.resume,
    )
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 1 if counts["errors"] else 0


def _summarize(args: argparse.Namespace) -> int:
    summary = aggregate_semantic_judgments(
        judgments_file=Path(args.judgments_file),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-semantic-judge",
        description="Prepare, run, and aggregate blinded semantic judgments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Create model-blinded judge inputs from generation predictions",
    )
    prepare.add_argument("--cases-file", required=True)
    prepare.add_argument("--predictions-file", required=True)
    prepare.add_argument("--output-file", required=True)
    prepare.add_argument(
        "--track",
        required=True,
        choices=("oracle-evidence", "complete-paper", "retrieved-context"),
    )
    prepare.add_argument("--generator-model", required=True)
    prepare.set_defaults(handler=_prepare)

    run = subparsers.add_parser(
        "run",
        help="Call a configured LLM judge on prepared blinded inputs",
    )
    run.add_argument("--inputs-file", required=True)
    run.add_argument("--manifest-file")
    run.add_argument("--output-file", required=True)
    run.add_argument("--provider", choices=("openai", "openrouter"), default="openrouter")
    run.add_argument("--judge-model", default=DEFAULT_OPENROUTER_JUDGE)
    run.add_argument("--env-file", default=".env")
    run.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY")
    run.add_argument(
        "--openrouter-base-url",
        default="https://openrouter.ai/api/v1",
    )
    run.add_argument("--fallback-model", action="append", default=[])
    run.add_argument("--reasoning-effort", default="low")
    run.add_argument("--max-output-tokens", type=int, default=4096)
    run.add_argument("--max-cases", type=int)
    run.add_argument("--timeout", type=float, default=300.0)
    run.add_argument("--retries", type=int, default=1)
    run.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Overwrite the output instead of skipping existing record IDs",
    )
    run.set_defaults(handler=_run, resume=True)

    summarize = subparsers.add_parser(
        "summarize",
        help="Aggregate completed semantic judgments",
    )
    summarize.add_argument("--judgments-file", required=True)
    summarize.add_argument("--output-dir", required=True)
    summarize.set_defaults(handler=_summarize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
