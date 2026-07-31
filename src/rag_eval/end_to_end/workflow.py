"""Workflow composition for retrieved-context generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..generation.adapter import GenerationAdapter
from ..generation.data import load_generation_cases
from ..generation.metrics import load_eligibility_manifest
from ..generation.runner import run_generation_cases
from ..retrieval.context import (
    RetrievalMethod,
    RetrievalScope,
    file_sha256,
    freeze_retrieved_contexts,
    frozen_contexts_by_case,
    load_frozen_context_manifest,
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


def run_retrieved_context_generation(
    *,
    cases_file: Path,
    context_manifest_file: Path,
    adapter: GenerationAdapter,
    output_file: Path,
    max_context_tokens: int,
    max_output_tokens: int,
    max_cases: int | None,
    resume: bool,
) -> dict[str, int]:
    """Generate from a validated, frozen retrieval manifest."""

    cases = list(load_generation_cases(cases_file).values())
    manifest = load_frozen_context_manifest(context_manifest_file)
    if manifest["cases_sha256"] != file_sha256(cases_file):
        raise ValueError("Frozen context manifest cases checksum does not match")
    return run_generation_cases(
        cases,
        adapter=adapter,
        track="retrieved-context",
        output_file=output_file,
        max_context_tokens=max_context_tokens,
        max_output_tokens=max_output_tokens,
        max_cases=max_cases,
        resume=resume,
        retrieved_contexts=frozen_contexts_by_case(manifest),
        context_manifest_sha256=file_sha256(context_manifest_file),
    )


def run_retrieve_then_generate(
    *,
    cases_file: Path,
    eligibility_file: Path,
    context_manifest_file: Path,
    adapter: GenerationAdapter,
    output_file: Path,
    top_k: int,
    retriever: RetrievalMethod,
    retrieval_scope: RetrievalScope,
    dense_model: str,
    dense_batch_size: int,
    hybrid_rrf_k: int,
    hybrid_candidate_k: int | None,
    max_context_tokens: int,
    max_output_tokens: int,
    max_cases: int | None,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Freeze retrieval first, then generate from that exact manifest."""

    retrieval = freeze_context_manifest(
        cases_file=cases_file,
        eligibility_file=eligibility_file,
        output_file=context_manifest_file,
        top_k=top_k,
        retriever=retriever,
        retrieval_scope=retrieval_scope,
        dense_model=dense_model,
        dense_batch_size=dense_batch_size,
        hybrid_rrf_k=hybrid_rrf_k,
        hybrid_candidate_k=hybrid_candidate_k,
    )
    generation = run_retrieved_context_generation(
        cases_file=cases_file,
        context_manifest_file=context_manifest_file,
        adapter=adapter,
        output_file=output_file,
        max_context_tokens=max_context_tokens,
        max_output_tokens=max_output_tokens,
        max_cases=max_cases,
        resume=resume,
    )
    return retrieval, generation
