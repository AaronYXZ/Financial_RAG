"""File-based orchestration for retrieval-component runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..generation.data import load_generation_cases
from ..generation.metrics import load_eligibility_manifest
from .context import (
    RetrievalMethod,
    RetrievalScope,
    freeze_retrieved_contexts,
    retriever_manifest,
    write_frozen_context_manifest,
)


def freeze_context_manifest(
    *,
    cases_file: Path,
    eligibility_file: Path,
    output_file: Path,
    top_k: int,
    retriever: RetrievalMethod,
    retrieval_scope: RetrievalScope,
    dense_model: str,
    dense_batch_size: int,
    hybrid_rrf_k: int,
    hybrid_candidate_k: int | None,
) -> dict[str, Any]:
    """Retrieve passages for a frozen eligible set and persist the manifest."""

    cases = list(load_generation_cases(cases_file).values())
    eligibility = load_eligibility_manifest(eligibility_file)
    case_ids = eligibility["eligible_case_ids"]
    contexts = freeze_retrieved_contexts(
        cases,
        eligible_case_ids=case_ids,
        top_k=top_k,
        method=retriever,
        scope=retrieval_scope,
        dense_model=dense_model,
        dense_batch_size=dense_batch_size,
        hybrid_rrf_k=hybrid_rrf_k,
        hybrid_candidate_k=hybrid_candidate_k,
    )
    metadata = retriever_manifest(
        method=retriever,
        scope=retrieval_scope,
        top_k=top_k,
        dense_model=dense_model,
        dense_batch_size=dense_batch_size,
        hybrid_rrf_k=hybrid_rrf_k,
        hybrid_candidate_k=hybrid_candidate_k,
    )
    return write_frozen_context_manifest(
        output_file,
        cases_file=cases_file,
        eligibility_file=eligibility_file,
        eligible_case_ids=case_ids,
        contexts=contexts,
        top_k=top_k,
        retrieval_scope=retrieval_scope,
        retriever=metadata,
    )
