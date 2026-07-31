"""Workflow composition for retrieved-context generation."""

from __future__ import annotations

from pathlib import Path

from ..generation.adapter import GenerationAdapter
from ..generation.data import load_generation_cases
from ..generation.runner import run_generation_cases
from ..retrieval.context import (
    RetrievalMethod,
    RetrievalScope,
    file_sha256,
    frozen_contexts_by_case,
    load_frozen_context_manifest,
)
from ..retrieval.workflow import freeze_context_manifest


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
