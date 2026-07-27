"""Sequential fixed-context generation runner with JSONL resume support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .generation_adapter import GenerationAdapter
from .generation_context import ContextTrack, build_fixed_context
from .generation_data import GenerationCase
from .generation_prompt import (
    SYSTEM_PROMPT,
    parse_generation_response,
    prompt_hash,
    render_user_prompt,
)


@dataclass(frozen=True)
class EligibleCase:
    case: GenerationCase
    context_passage_ids: tuple[str, ...]
    user_prompt: str
    input_tokens: int


def _completed_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("error") is None:
                completed.add((row["case_id"], row["track"], row["model_id"]))
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
) -> tuple[list[EligibleCase], dict[str, int]]:
    eligible: list[EligibleCase] = []
    exclusions = {"track_filter": 0, "context_limit": 0}
    for case in cases:
        if track == "oracle-evidence" and case.answerability != "answerable":
            exclusions["track_filter"] += 1
            continue
        if track == "complete-paper" and case.answerability == "ambiguous":
            exclusions["track_filter"] += 1
            continue
        if max_cases is not None and len(eligible) >= max_cases:
            break

        passages = build_fixed_context(case, track)
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
) -> None:
    payload = {
        "schema_version": 1,
        "track": track,
        "model_id": adapter.model_id,
        "max_context_tokens": max_context_tokens,
        "max_output_tokens": max_output_tokens,
        "max_cases": max_cases,
        "eligible_case_count": len(eligible),
        "eligible_case_ids": [item.case.case_id for item in eligible],
        "excluded_counts": exclusions,
    }
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
    max_output_tokens: int = 512,
    max_cases: int | None = 25,
    resume: bool = True,
    eligibility_file: Path | None = None,
) -> dict[str, int]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    eligible, exclusions = _select_eligible_cases(
        cases,
        adapter=adapter,
        track=track,
        max_context_tokens=max_context_tokens,
        max_output_tokens=max_output_tokens,
        max_cases=max_cases,
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
    )

    completed = _completed_keys(output_file) if resume else set()
    counts = {
        "selected": len(eligible),
        "completed": 0,
        "skipped": 0,
        "ineligible": exclusions["context_limit"],
        "errors": 0,
    }

    with output_file.open("a" if resume else "w", encoding="utf-8") as handle:
        for item in eligible:
            case = item.case
            key = (case.case_id, track, adapter.model_id)
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
                "prompt_hash": prompt_hash(SYSTEM_PROMPT, item.user_prompt),
                "counted_input_tokens": item.input_tokens,
            }
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
