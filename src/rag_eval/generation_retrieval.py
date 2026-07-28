"""Freeze retrieved QASPER passages before generation evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .benchmark_retrievers import PreparedBM25Retriever
from .generation_data import GenerationCase, PaperPassage


FROZEN_CONTEXT_SCHEMA_VERSION = 1


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
) -> list[dict[str, Any]]:
    """Retrieve once over the normalized QASPER passage corpus and freeze rankings."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    cases_by_id = {case.case_id: case for case in cases}
    case_ids = list(eligible_case_ids)
    missing = [case_id for case_id in case_ids if case_id not in cases_by_id]
    if missing:
        raise ValueError(f"Eligible cases are missing from the case file: {missing[:3]}")

    passages = unique_passages(cases_by_id.values())
    corpus = {
        passage_id: {
            "title": passage.section_name,
            "text": passage.text,
        }
        for passage_id, passage in passages.items()
    }
    queries = {case_id: cases_by_id[case_id].question for case_id in case_ids}
    result = PreparedBM25Retriever(corpus).search(queries, top_k=top_k)

    contexts: list[dict[str, Any]] = []
    for case_id in case_ids:
        scores = result.run.get(case_id, {})
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
) -> dict[str, Any]:
    payload = {
        "schema_version": FROZEN_CONTEXT_SCHEMA_VERSION,
        "track": "retrieved-context",
        "retriever": {
            "name": "bm25",
            "implementation": "rag_eval.benchmark_retrievers.PreparedBM25Retriever",
            "parameters": {"k1": 1.5, "b": 0.75, "top_k": top_k},
        },
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


def load_frozen_context_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FROZEN_CONTEXT_SCHEMA_VERSION:
        raise ValueError("Unsupported frozen context manifest schema version")
    if payload.get("track") != "retrieved-context":
        raise ValueError("Frozen context manifest track must be retrieved-context")
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
