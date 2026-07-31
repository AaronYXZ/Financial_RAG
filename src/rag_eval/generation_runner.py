"""Sequential fixed-context generation runner with JSONL resume support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .generation_adapter import GenerationAdapter
from .generation_context import ContextTrack, build_fixed_context
from .generation_data import GenerationCase, PaperPassage
from .generation_prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    parse_generation_response,
    prompt_hash,
    render_user_prompt,
)
from .retrieval.context import unique_passages


@dataclass(frozen=True)
class EligibleCase:
    case: GenerationCase
    context_passage_ids: tuple[str, ...]
    user_prompt: str
    input_tokens: int


def _completed_keys(path: Path) -> set[tuple[str, str, str, str | None]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, str, str, str | None]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            completed.add(
                (
                    row["case_id"],
                    row["track"],
                    row["model_id"],
                    row.get("prompt_version"),
                )
            )
    return completed


def eligibility_manifest_path(output_file: Path) -> Path:
    return output_file.with_suffix(".eligibility.json")


def _select_eligible_cases(
    cases: Iterable[GenerationCase],
    *,
    adapter: GenerationAdapter,
    track: ContextTrack,
    max_context_tokens: int,
    max_output_tokens: int,
    max_cases: int | None,
    retrieved_contexts: Mapping[str, tuple[str, ...]] | None = None,
    passage_lookup: Mapping[str, PaperPassage] | None = None,
) -> tuple[list[EligibleCase], dict[str, int]]:
    eligible: list[EligibleCase] = []
    exclusions = {
        "track_filter": 0,
        "missing_oracle_evidence": 0,
        "context_limit": 0,
    }
    case_list = list(cases)
    if passage_lookup is None:
        passage_lookup = unique_passages(case_list)
    for case in case_list:
        if track == "oracle-evidence" and case.answerability != "answerable":
            exclusions["track_filter"] += 1
            continue
        if track == "oracle-evidence" and not case.oracle_passage_ids:
            exclusions["missing_oracle_evidence"] += 1
            continue
        if track == "complete-paper" and case.answerability == "ambiguous":
            exclusions["track_filter"] += 1
            continue
        if track == "retrieved-context" and (
            retrieved_contexts is None or case.case_id not in retrieved_contexts
        ):
            exclusions["track_filter"] += 1
            continue
        if max_cases is not None and len(eligible) >= max_cases:
            break

        passages = build_fixed_context(
            case,
            track,
            retrieved_passage_ids=(
                retrieved_contexts[case.case_id]
                if track == "retrieved-context" and retrieved_contexts is not None
                else None
            ),
            passage_lookup=passage_lookup,
        )
        user_prompt = render_user_prompt(case.question, passages)
        input_tokens = adapter.count_tokens(SYSTEM_PROMPT, user_prompt)
        if input_tokens + max_output_tokens > max_context_tokens:
            exclusions["context_limit"] += 1
            continue
        eligible.append(
            EligibleCase(
                case=case,
                context_passage_ids=tuple(passage.passage_id for passage in passages),
                user_prompt=user_prompt,
                input_tokens=input_tokens,
            )
        )
    return eligible, exclusions


def _write_eligibility_manifest(
    path: Path,
    *,
    eligible: list[EligibleCase],
    exclusions: dict[str, int],
    adapter: GenerationAdapter,
    track: ContextTrack,
    max_context_tokens: int,
    max_output_tokens: int,
    max_cases: int | None,
    overwrite: bool,
    context_manifest_sha256: str | None,
) -> None:
    payload = {
        "schema_version": 2,
        "prompt_version": PROMPT_VERSION,
        "track": track,
        "model_id": adapter.model_id,
        "max_context_tokens": max_context_tokens,
        "max_output_tokens": max_output_tokens,
        "max_cases": max_cases,
        "eligible_case_count": len(eligible),
        "eligible_case_ids": [item.case.case_id for item in eligible],
        "excluded_counts": exclusions,
    }
    provider = getattr(adapter, "provider", None)
    if provider is not None:
        payload["provider"] = provider
    fallback_model_ids = getattr(adapter, "fallback_model_ids", ())
    if fallback_model_ids:
        payload["fallback_model_ids"] = list(fallback_model_ids)
    if context_manifest_sha256 is not None:
        payload["context_manifest_sha256"] = context_manifest_sha256
    if path.exists() and not overwrite:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                "Existing eligibility manifest does not match this resumed run. "
                "Use a different output file or --no-resume."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_generation_cases(
    cases: Iterable[GenerationCase],
    *,
    adapter: GenerationAdapter,
    track: ContextTrack,
    output_file: Path,
    max_context_tokens: int = 32_768,
    max_output_tokens: int = 1024,
    max_cases: int | None = 25,
    resume: bool = True,
    eligibility_file: Path | None = None,
    retrieved_contexts: Mapping[str, tuple[str, ...]] | None = None,
    context_manifest_sha256: str | None = None,
) -> dict[str, int]:
    case_list = list(cases)
    passage_lookup = unique_passages(case_list)
    if track == "retrieved-context":
        if retrieved_contexts is None or not context_manifest_sha256:
            raise ValueError(
                "The retrieved-context track requires frozen contexts and their manifest hash"
            )
        cases_by_id = {case.case_id: case for case in case_list}
        missing = [case_id for case_id in retrieved_contexts if case_id not in cases_by_id]
        if missing:
            raise ValueError(f"Frozen context cases are missing from the case file: {missing[:3]}")
        case_list = [cases_by_id[case_id] for case_id in retrieved_contexts]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    eligible, exclusions = _select_eligible_cases(
        case_list,
        adapter=adapter,
        track=track,
        max_context_tokens=max_context_tokens,
        max_output_tokens=max_output_tokens,
        max_cases=max_cases,
        retrieved_contexts=retrieved_contexts,
        passage_lookup=passage_lookup,
    )
    _write_eligibility_manifest(
        eligibility_file or eligibility_manifest_path(output_file),
        eligible=eligible,
        exclusions=exclusions,
        adapter=adapter,
        track=track,
        max_context_tokens=max_context_tokens,
        max_output_tokens=max_output_tokens,
        max_cases=max_cases,
        overwrite=not resume,
        context_manifest_sha256=context_manifest_sha256,
    )

    completed = _completed_keys(output_file) if resume else set()
    counts = {
        "selected": len(eligible),
        "completed": 0,
        "skipped": 0,
        "ineligible": sum(exclusions.values()),
        "errors": 0,
    }

    with output_file.open("a" if resume else "w", encoding="utf-8") as handle:
        for item in eligible:
            case = item.case
            key = (case.case_id, track, adapter.model_id, PROMPT_VERSION)
            if key in completed:
                counts["skipped"] += 1
                continue

            row: dict[str, object] = {
                "case_id": case.case_id,
                "paper_id": case.paper_id,
                "split": case.split,
                "track": track,
                "model_id": adapter.model_id,
                "answerability": case.answerability,
                "context_passage_ids": list(item.context_passage_ids),
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": prompt_hash(SYSTEM_PROMPT, item.user_prompt),
                "counted_input_tokens": item.input_tokens,
            }
            provider = getattr(adapter, "provider", None)
            if provider is not None:
                row["provider"] = provider
            fallback_model_ids = getattr(adapter, "fallback_model_ids", ())
            if fallback_model_ids:
                row["fallback_model_ids"] = list(fallback_model_ids)
            if context_manifest_sha256 is not None:
                row["context_manifest_sha256"] = context_manifest_sha256
            error_stage = "generation"
            try:
                result = adapter.generate(SYSTEM_PROMPT, item.user_prompt)
                row.update(
                    {
                        "raw_response": result.text,
                        "latency_seconds": result.latency_seconds,
                        "server_input_tokens": result.input_tokens,
                        "server_output_tokens": result.output_tokens,
                        "attempts": result.attempts,
                    }
                )
                if result.resolved_model_id is not None:
                    row["resolved_model_id"] = result.resolved_model_id
                error_stage = "response_validation"
                parsed = parse_generation_response(
                    result.text,
                    allowed_citation_ids=item.context_passage_ids,
                )
                row.update(
                    {
                        "parsed_response": {
                            "answer": parsed.answer,
                            "abstain": parsed.abstain,
                            "citations": list(parsed.citations),
                            "confidence": parsed.confidence,
                        },
                        "error": None,
                        "error_stage": None,
                        "error_type": None,
                    }
                )
                counts["completed"] += 1
            except Exception as exc:
                row.update(
                    {
                        "parsed_response": None,
                        "error": str(exc),
                        "error_stage": error_stage,
                        "error_type": type(exc).__name__,
                    }
                )
                if "attempts" not in row and hasattr(exc, "attempts"):
                    row["attempts"] = getattr(exc, "attempts")
                counts["errors"] += 1
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    return counts
