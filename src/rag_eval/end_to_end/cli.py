"""CLI handlers for retrieved-context and retrieve-then-generate runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..cli.common import _build_adapter
from ..generation.cli import _execute_generation
from .workflow import run_retrieve_then_generate


def _generate_retrieved(args: argparse.Namespace) -> int:
    return _execute_generation(
        args,
        track="retrieved-context",
        context_manifest=args.context_manifest,
    )

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
