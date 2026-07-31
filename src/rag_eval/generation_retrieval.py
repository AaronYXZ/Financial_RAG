"""Freeze retrieved QASPER passages before generation evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from .retrieval import (
    PreparedBM25Retriever,
    PreparedDenseRetriever,
    reciprocal_rank_fusion,
)
from .generation_data import GenerationCase, PaperPassage


FROZEN_CONTEXT_SCHEMA_VERSION = 2
SUPPORTED_FROZEN_CONTEXT_SCHEMA_VERSIONS = (1, 2)
RetrievalMethod = Literal["bm25", "dense", "hybrid"]
RetrievalScope = Literal["paper", "corpus"]

DENSE_MODEL_1 = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_MODEL_2 = "sentence-transformers/all-mpnet-base-v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_passages(cases: Iterable[GenerationCase]) -> dict[str, PaperPassage]:
    passages: dict[str, PaperPassage] = {}
    for case in cases:
        for passage in case.paper_passages:
            previous = passages.setdefault(passage.passage_id, passage)
            if previous != passage:
                raise ValueError(f"Conflicting passage payload for {passage.passage_id}")
    return passages


def freeze_bm25_contexts(
    cases: Iterable[GenerationCase],
    *,
    eligible_case_ids: Iterable[str],
    top_k: int,
    scope: RetrievalScope = "corpus",
) -> list[dict[str, Any]]:
    """Backward-compatible BM25 wrapper over the configurable retrieval path."""

    return freeze_retrieved_contexts(
        cases,
        eligible_case_ids=eligible_case_ids,
        top_k=top_k,
        method="bm25",
        scope=scope,
    )


def freeze_retrieved_contexts(
    cases: Iterable[GenerationCase],
    *,
    eligible_case_ids: Iterable[str],
    top_k: int,
    method: RetrievalMethod,
    scope: RetrievalScope,
    dense_model: str = DENSE_MODEL_1,
    dense_batch_size: int = 32,
    hybrid_rrf_k: int = 60,
    hybrid_candidate_k: int | None = None,
) -> list[dict[str, Any]]:
    """Freeze ranked contexts with an explicit retrieval method and corpus scope."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if method not in ("bm25", "dense", "hybrid"):
        raise ValueError(f"Unsupported retrieval method: {method}")
    if scope not in ("paper", "corpus"):
        raise ValueError(f"Unsupported retrieval scope: {scope}")
    if dense_batch_size <= 0:
        raise ValueError("dense_batch_size must be positive")
    if hybrid_rrf_k <= 0:
        raise ValueError("hybrid_rrf_k must be positive")
    if hybrid_candidate_k is not None and hybrid_candidate_k < top_k:
        raise ValueError("hybrid_candidate_k must be at least top_k")

    case_list = list(cases)
    cases_by_id = {case.case_id: case for case in case_list}
    case_ids = list(eligible_case_ids)
    missing = [case_id for case_id in case_ids if case_id not in cases_by_id]
    if missing:
        raise ValueError(f"Eligible cases are missing from the case file: {missing[:3]}")

    passages = unique_passages(case_list)
    query_groups: list[tuple[list[str], dict[str, PaperPassage]]] = []
    if scope == "corpus":
        query_groups.append((case_ids, passages))
    else:
        case_ids_by_paper: defaultdict[str, list[str]] = defaultdict(list)
        for case_id in case_ids:
            case_ids_by_paper[cases_by_id[case_id].paper_id].append(case_id)
        passages_by_paper: defaultdict[str, dict[str, PaperPassage]] = defaultdict(dict)
        for passage_id, passage in passages.items():
            passages_by_paper[passage.paper_id][passage_id] = passage
        query_groups.extend(
            (paper_case_ids, passages_by_paper[paper_id])
            for paper_id, paper_case_ids in case_ids_by_paper.items()
        )

    scores_by_case: dict[str, dict[str, float]] = {}
    shared_dense_model: Any | None = None
    for group_case_ids, group_passages in query_groups:
        corpus = {
            passage_id: {
                "title": passage.section_name,
                "text": passage.text,
            }
            for passage_id, passage in group_passages.items()
        }
        queries = {
            case_id: cases_by_id[case_id].question for case_id in group_case_ids
        }
        if method == "bm25":
            scores_by_case.update(
                PreparedBM25Retriever(corpus).search(queries, top_k=top_k).run
            )
            continue

        dense = PreparedDenseRetriever(
            corpus,
            model_name=dense_model,
            batch_size=dense_batch_size,
            model=shared_dense_model,
        )
        shared_dense_model = dense.model
        if method == "dense":
            scores_by_case.update(dense.search(queries, top_k=top_k).run)
            continue

        candidate_k = min(
            len(corpus),
            hybrid_candidate_k
            if hybrid_candidate_k is not None
            else max(top_k * 4, 100),
        )
        lexical_run = PreparedBM25Retriever(corpus).search(
            queries, top_k=candidate_k
        ).run
        dense_run = dense.search(queries, top_k=candidate_k).run
        fused_run, _ = reciprocal_rank_fusion(
            lexical_run,
            dense_run,
            top_k=top_k,
            rrf_k=hybrid_rrf_k,
        )
        scores_by_case.update(fused_run)

    contexts: list[dict[str, Any]] = []
    for case_id in case_ids:
        scores = scores_by_case.get(case_id, {})
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        contexts.append(
            {
                "case_id": case_id,
                "passage_ids": [passage_id for passage_id, _ in ranked],
                "scores": [score for _, score in ranked],
            }
        )
    return contexts


def write_frozen_context_manifest(
    path: Path,
    *,
    cases_file: Path,
    eligibility_file: Path,
    eligible_case_ids: list[str],
    contexts: list[dict[str, Any]],
    top_k: int,
    retrieval_scope: RetrievalScope = "corpus",
    retriever: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if retriever is None:
        retriever = retriever_manifest(
            method="bm25",
            scope=retrieval_scope,
            top_k=top_k,
        )
    payload = {
        "schema_version": FROZEN_CONTEXT_SCHEMA_VERSION,
        "track": "retrieved-context",
        "retrieval_scope": retrieval_scope,
        "retriever": dict(retriever),
        "cases_file": str(cases_file),
        "cases_sha256": file_sha256(cases_file),
        "source_eligibility_file": str(eligibility_file),
        "source_eligibility_sha256": file_sha256(eligibility_file),
        "eligible_case_count": len(eligible_case_ids),
        "eligible_case_ids": eligible_case_ids,
        "contexts": contexts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def retriever_manifest(
    *,
    method: RetrievalMethod,
    scope: RetrievalScope,
    top_k: int,
    dense_model: str = DENSE_MODEL_1,
    dense_batch_size: int = 32,
    hybrid_rrf_k: int = 60,
    hybrid_candidate_k: int | None = None,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "scope": scope,
        "top_k": top_k,
    }
    implementations = {
        "bm25": "rag_eval.retrieval.PreparedBM25Retriever",
        "dense": "rag_eval.retrieval.PreparedDenseRetriever",
        "hybrid": "rag_eval.retrieval.reciprocal_rank_fusion",
    }
    if method in ("bm25", "hybrid"):
        parameters.update({"k1": 1.5, "b": 0.75})
    if method in ("dense", "hybrid"):
        parameters.update(
            {
                "dense_model": dense_model,
                "dense_batch_size": dense_batch_size,
            }
        )
    if method == "hybrid":
        parameters.update(
            {
                "rrf_k": hybrid_rrf_k,
                "candidate_k": (
                    hybrid_candidate_k
                    if hybrid_candidate_k is not None
                    else max(top_k * 4, 100)
                ),
            }
        )
    return {
        "name": method,
        "implementation": implementations[method],
        "parameters": parameters,
    }


def load_frozen_context_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in SUPPORTED_FROZEN_CONTEXT_SCHEMA_VERSIONS:
        raise ValueError("Unsupported frozen context manifest schema version")
    if payload.get("track") != "retrieved-context":
        raise ValueError("Frozen context manifest track must be retrieved-context")
    if payload["schema_version"] >= 2:
        if payload.get("retrieval_scope") not in ("paper", "corpus"):
            raise ValueError("Frozen context manifest has an invalid retrieval_scope")
        retriever = payload.get("retriever")
        if not isinstance(retriever, Mapping) or retriever.get("name") not in (
            "bm25",
            "dense",
            "hybrid",
        ):
            raise ValueError("Frozen context manifest has invalid retriever metadata")
        parameters = retriever.get("parameters")
        if (
            not isinstance(parameters, Mapping)
            or parameters.get("scope") != payload["retrieval_scope"]
        ):
            raise ValueError(
                "Frozen context retriever scope does not match retrieval_scope"
            )
    eligible = payload.get("eligible_case_ids")
    contexts = payload.get("contexts")
    if not isinstance(eligible, list) or not all(isinstance(item, str) for item in eligible):
        raise ValueError("eligible_case_ids must be a list of strings")
    if payload.get("eligible_case_count") != len(eligible) or len(set(eligible)) != len(eligible):
        raise ValueError("Frozen context eligibility metadata is inconsistent")
    if not isinstance(contexts, list):
        raise ValueError("contexts must be a list")

    contexts_by_id: dict[str, tuple[str, ...]] = {}
    for item in contexts:
        if not isinstance(item, Mapping) or not isinstance(item.get("case_id"), str):
            raise ValueError("Every frozen context must contain a case_id")
        passage_ids = item.get("passage_ids")
        if not isinstance(passage_ids, list) or not all(
            isinstance(passage_id, str) for passage_id in passage_ids
        ):
            raise ValueError("Every frozen context must contain passage_ids")
        case_id = item["case_id"]
        if case_id in contexts_by_id:
            raise ValueError(f"Duplicate frozen context for case {case_id}")
        contexts_by_id[case_id] = tuple(passage_ids)
    if set(contexts_by_id) != set(eligible):
        raise ValueError("Frozen context cases do not match eligible_case_ids")
    return payload


def frozen_contexts_by_case(payload: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    return {
        str(item["case_id"]): tuple(item["passage_ids"])
        for item in payload["contexts"]
    }
