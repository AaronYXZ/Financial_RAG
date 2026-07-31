"""CLI handlers and arguments for retrieval-component evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .context import DENSE_MODEL_1, DENSE_MODEL_2
from .evaluation import compare_retrieval_to_oracle
from .workflow import freeze_context_manifest


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
