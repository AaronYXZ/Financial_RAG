"""CLI handlers for data preparation and component generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..cli.common import OPENAI_MODEL_IDS, _build_adapter
from ..end_to_end.workflow import run_retrieved_context_generation
from .comparison import (
    compare_evaluated_runs,
    compare_response_validity,
    intersect_eligibility_manifests,
    write_comparison,
)
from .cost import estimate_openai_cost
from .data import (
    QASPER_DATASET,
    QASPER_VERSION,
    generation_case_from_dict,
    load_qasper_cases,
    qasper_parquet_sha256,
    qasper_parquet_url,
)
from .metrics import evaluate_prediction_files
from .runner import eligibility_manifest_path, run_generation_cases


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

def _run(args: argparse.Namespace) -> int:
    return _execute_generation(
        args,
        track=args.track,
        context_manifest=args.context_manifest,
    )

def _generate_oracle(args: argparse.Namespace) -> int:
    return _execute_generation(args, track="oracle-evidence")

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
