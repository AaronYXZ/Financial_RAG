"""Prepare, run, and aggregate blinded semantic judgments."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .generation_adapter import AdapterResult, GenerationRequestError
from .generation_data import GenerationCase, load_generation_cases
from .retrieval.context import file_sha256, unique_passages
from .semantic_judge import (
    BLINDING_VERSION,
    SEMANTIC_OUTPUT_SCHEMA_VERSION,
    SEMANTIC_PARSING_POLICY,
    SEMANTIC_PROMPT_VERSION,
    SEMANTIC_RESPONSE_SCHEMA,
    SEMANTIC_SYSTEM_PROMPT,
    anonymous_context_id,
    content_hash,
    parse_semantic_judgment,
    render_semantic_user_prompt,
    semantic_prompt_hash,
    validate_blinded_judge_input,
)


class SemanticJudgeAdapter(Protocol):
    model_id: str

    def count_tokens(self, system_prompt: str, user_prompt: str) -> int: ...

    def generate(self, system_prompt: str, user_prompt: str) -> AdapterResult: ...


def semantic_manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def _reference_texts(case: GenerationCase) -> list[str]:
    values: list[str] = []
    for reference in case.references:
        text = reference.text.strip()
        if reference.unanswerable and not text:
            text = "[UNANSWERABLE]"
        if text and text not in values:
            values.append(text)
    return values


def _latest_prediction_rows(
    path: Path,
    *,
    track: str,
    generator_model_id: str,
) -> tuple[list[str], dict[str, dict[str, Any]], int]:
    order: list[str] = []
    latest: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for row in _load_jsonl(path):
        if row.get("track") != track or row.get("model_id") != generator_model_id:
            continue
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            raise ValueError("Prediction row is missing a string case_id")
        if case_id in latest:
            duplicate_count += 1
        else:
            order.append(case_id)
        latest[case_id] = row
    return order, latest, duplicate_count


def prepare_blinded_semantic_inputs(
    *,
    cases_file: Path,
    predictions_file: Path,
    output_file: Path,
    track: str,
    generator_model_id: str,
) -> dict[str, Any]:
    cases = load_generation_cases(cases_file)
    passage_lookup = unique_passages(cases.values())
    order, predictions, duplicate_count = _latest_prediction_rows(
        predictions_file,
        track=track,
        generator_model_id=generator_model_id,
    )
    if not predictions:
        raise ValueError("No prediction rows match the requested track and generator model")

    rows: list[dict[str, Any]] = []
    excluded = Counter()
    prompt_versions: set[str] = set()
    context_manifest_hashes: set[str] = set()
    for case_id in order:
        prediction = predictions[case_id]
        if case_id not in cases:
            raise ValueError(f"Prediction case {case_id!r} is missing from the case file")
        if prediction.get("error") is not None or not isinstance(
            prediction.get("parsed_response"), Mapping
        ):
            excluded["invalid_prediction"] += 1
            continue
        case = cases[case_id]
        response = prediction["parsed_response"]
        context_ids = prediction.get("context_passage_ids")
        if not isinstance(context_ids, list) or not all(
            isinstance(item, str) for item in context_ids
        ):
            raise ValueError(f"Prediction {case_id!r} has invalid context_passage_ids")
        if len(context_ids) != len(set(context_ids)):
            raise ValueError(f"Prediction {case_id!r} has duplicate context passages")
        missing = [item for item in context_ids if item not in passage_lookup]
        if missing:
            raise ValueError(
                f"Prediction {case_id!r} references unknown context passages: {missing[:3]}"
            )
        citation_map = {
            anonymous_context_id(index): passage_id
            for index, passage_id in enumerate(context_ids, start=1)
        }
        reverse_map = {value: key for key, value in citation_map.items()}
        citations = response.get("citations")
        if not isinstance(citations, list) or not all(
            isinstance(item, str) for item in citations
        ):
            raise ValueError(f"Prediction {case_id!r} has invalid citations")
        unknown_citations = sorted(set(citations) - set(reverse_map))
        if unknown_citations:
            raise ValueError(
                f"Prediction {case_id!r} cites passages outside its supplied context"
            )
        answer = response.get("answer")
        abstain = response.get("abstain")
        if not isinstance(answer, str) or not isinstance(abstain, bool):
            raise ValueError(f"Prediction {case_id!r} has an invalid parsed response")

        judge_input = {
            "question": case.question,
            "context": [
                {
                    "context_id": anonymous_id,
                    "text": passage_lookup[passage_id].text,
                }
                for anonymous_id, passage_id in citation_map.items()
            ],
            "references": _reference_texts(case),
            "candidate": {
                "answer": answer,
                "abstain": abstain,
                "citations": [reverse_map[item] for item in citations],
            },
        }
        validate_blinded_judge_input(judge_input)
        record_id = content_hash(
            {
                "blinding_version": BLINDING_VERSION,
                "case_id": case_id,
                "judge_input": judge_input,
            }
        )
        rows.append(
            {
                "schema_version": 1,
                "record_id": record_id,
                "case_id": case_id,
                "paper_id": case.paper_id,
                "judge_input": judge_input,
                "citation_map": citation_map,
            }
        )
        if isinstance(prediction.get("prompt_version"), str):
            prompt_versions.add(prediction["prompt_version"])
        if isinstance(prediction.get("context_manifest_sha256"), str):
            context_manifest_hashes.add(prediction["context_manifest_sha256"])

    _write_jsonl(output_file, rows)
    manifest = {
        "schema_version": 1,
        "artifact_type": "blinded_semantic_inputs",
        "blinding_version": BLINDING_VERSION,
        "track": track,
        "generator_model_id": generator_model_id,
        "generation_prompt_versions": sorted(prompt_versions),
        "context_manifest_sha256s": sorted(context_manifest_hashes),
        "cases_file": str(cases_file),
        "cases_sha256": file_sha256(cases_file),
        "predictions_file": str(predictions_file),
        "predictions_sha256": file_sha256(predictions_file),
        "source_prediction_count": len(predictions),
        "prepared_record_count": len(rows),
        "excluded_counts": dict(sorted(excluded.items())),
        "duplicate_source_prediction_count": duplicate_count,
        "record_ids": [row["record_id"] for row in rows],
        "output_sha256": file_sha256(output_file),
    }
    manifest_path = semantic_manifest_path(output_file)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_blinded_semantic_inputs(
    path: Path,
    *,
    manifest_file: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = manifest_file or semantic_manifest_path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "blinded_semantic_inputs":
        raise ValueError("Input manifest has the wrong artifact_type")
    if manifest.get("output_sha256") != file_sha256(path):
        raise ValueError("Blinded semantic input checksum does not match its manifest")
    rows = _load_jsonl(path)
    if manifest.get("record_ids") != [row.get("record_id") for row in rows]:
        raise ValueError("Blinded semantic input record IDs do not match the manifest")
    if manifest.get("prepared_record_count") != len(rows):
        raise ValueError("Blinded semantic input count does not match the manifest")
    for row in rows:
        validate_blinded_judge_input(row["judge_input"])
    return rows, manifest


def _completed_record_ids(path: Path) -> set[str]:
    return {
        str(row["record_id"])
        for row in _load_jsonl(path)
        if isinstance(row.get("record_id"), str)
    } if path.exists() else set()


def run_semantic_judgments(
    *,
    inputs_file: Path,
    adapter: SemanticJudgeAdapter,
    output_file: Path,
    manifest_file: Path | None = None,
    max_cases: int | None = None,
    resume: bool = True,
) -> dict[str, int]:
    rows, input_manifest = load_blinded_semantic_inputs(
        inputs_file,
        manifest_file=manifest_file,
    )
    generator_model_id = input_manifest.get("generator_model_id")
    judge_models = {
        adapter.model_id,
        *getattr(adapter, "fallback_model_ids", ()),
    }
    if generator_model_id in judge_models:
        raise ValueError("A generator cannot be its own only semantic judge")

    selected = rows if max_cases is None else rows[:max_cases]
    completed_ids = _completed_record_ids(output_file) if resume else set()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "selected": len(selected),
        "completed": 0,
        "skipped": 0,
        "errors": 0,
    }
    with output_file.open("a" if resume else "w", encoding="utf-8") as handle:
        for input_row in selected:
            record_id = input_row["record_id"]
            if record_id in completed_ids:
                counts["skipped"] += 1
                continue
            judge_input = input_row["judge_input"]
            user_prompt = render_semantic_user_prompt(judge_input)
            row: dict[str, Any] = {
                "schema_version": 1,
                "record_id": record_id,
                "case_id": input_row["case_id"],
                "paper_id": input_row["paper_id"],
                "judge_model_id": adapter.model_id,
                "judge_provider": getattr(adapter, "provider", None),
                "judge_prompt_version": SEMANTIC_PROMPT_VERSION,
                "judge_prompt_hash": semantic_prompt_hash(),
                "judge_output_schema_version": SEMANTIC_OUTPUT_SCHEMA_VERSION,
                "judge_parsing_policy": SEMANTIC_PARSING_POLICY,
                "judge_input_hash": content_hash(judge_input),
                "counted_input_tokens": adapter.count_tokens(
                    SEMANTIC_SYSTEM_PROMPT,
                    user_prompt,
                ),
            }
            error_stage = "judge_request"
            try:
                result = adapter.generate(SEMANTIC_SYSTEM_PROMPT, user_prompt)
                row.update(
                    {
                        "raw_response": result.text,
                        "latency_seconds": result.latency_seconds,
                        "server_input_tokens": result.input_tokens,
                        "server_output_tokens": result.output_tokens,
                        "attempts": result.attempts,
                    }
                )
                resolved_model = result.resolved_model_id or adapter.model_id
                row["resolved_judge_model_id"] = resolved_model
                if resolved_model == generator_model_id:
                    raise ValueError("The resolved judge model matches the generator")
                error_stage = "judge_response_validation"
                context_ids = [
                    item["context_id"] for item in judge_input["context"]
                ]
                judgment = parse_semantic_judgment(
                    result.text,
                    allowed_context_ids=context_ids,
                    candidate_citation_ids=judge_input["candidate"]["citations"],
                    candidate_answer=judge_input["candidate"]["answer"],
                    candidate_abstain=judge_input["candidate"]["abstain"],
                )
                row.update(
                    {
                        "parsed_judgment": judgment.to_dict(),
                        "error": None,
                        "error_stage": None,
                        "error_type": None,
                    }
                )
                counts["completed"] += 1
            except Exception as exc:
                row.update(
                    {
                        "parsed_judgment": None,
                        "error": str(exc),
                        "error_stage": error_stage,
                        "error_type": type(exc).__name__,
                    }
                )
                if "attempts" not in row and isinstance(exc, GenerationRequestError):
                    row["attempts"] = exc.attempts
                counts["errors"] += 1
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    output_manifest = {
        "schema_version": 1,
        "artifact_type": "semantic_judgments",
        "inputs_file": str(inputs_file),
        "inputs_sha256": file_sha256(inputs_file),
        "input_manifest_sha256": file_sha256(
            manifest_file or semantic_manifest_path(inputs_file)
        ),
        "judge_provider": getattr(adapter, "provider", None),
        "judge_model_id": adapter.model_id,
        "fallback_model_ids": list(getattr(adapter, "fallback_model_ids", ())),
        "judge_prompt_version": SEMANTIC_PROMPT_VERSION,
        "judge_prompt_hash": semantic_prompt_hash(),
        "judge_output_schema_version": SEMANTIC_OUTPUT_SCHEMA_VERSION,
        "judge_output_schema_hash": content_hash(SEMANTIC_RESPONSE_SCHEMA),
        "judge_parsing_policy": SEMANTIC_PARSING_POLICY,
        "selected_record_count": len(selected),
    }
    semantic_manifest_path(output_file).write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return counts


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_semantic_judgments(
    *,
    judgments_file: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    rows = _load_jsonl(judgments_file)
    valid = [
        row
        for row in rows
        if row.get("error") is None and isinstance(row.get("parsed_judgment"), Mapping)
    ]
    claims = [
        claim
        for row in valid
        for claim in row["parsed_judgment"]["claims"]
    ]
    support_counts = Counter(claim["support_label"] for claim in claims)
    citation_counts = Counter(claim["citation_entailment"] for claim in claims)
    claim_count = len(claims)
    claim_cases = [row for row in valid if row["parsed_judgment"]["claims"]]

    per_case: list[dict[str, Any]] = []
    for row in rows:
        judgment = row.get("parsed_judgment")
        if row.get("error") is not None or not isinstance(judgment, Mapping):
            per_case.append(
                {
                    "record_id": row.get("record_id"),
                    "case_id": row.get("case_id"),
                    "paper_id": row.get("paper_id"),
                    "evaluator_status": "error",
                    "error": row.get("error"),
                }
            )
            continue
        case_claims = judgment["claims"]
        count = len(case_claims)
        supported = sum(item["support_label"] == "supported" for item in case_claims)
        entailed = sum(
            item["citation_entailment"] == "entailed" for item in case_claims
        )
        per_case.append(
            {
                "record_id": row["record_id"],
                "case_id": row["case_id"],
                "paper_id": row["paper_id"],
                "evaluator_status": "valid",
                "claim_count": count,
                "supported_claim_rate": supported / count if count else None,
                "fully_faithful": supported == count if count else None,
                "citation_entailed_claim_rate": entailed / count if count else None,
                "citation_complete": entailed == count if count else None,
                "semantic_correctness": judgment["semantic_correctness"]["score"],
                "completeness": judgment["completeness"]["score"],
            }
        )

    valid_case_metrics = [
        item for item in per_case if item["evaluator_status"] == "valid"
    ]
    summary = {
        "schema_version": 1,
        "judge_prompt_version": SEMANTIC_PROMPT_VERSION,
        "judge_prompt_hash": semantic_prompt_hash(),
        "case_count": len(rows),
        "evaluated_case_count": len(valid),
        "evaluator_failure_count": len(rows) - len(valid),
        "evaluator_failure_rate": (len(rows) - len(valid)) / len(rows) if rows else None,
        "claim_count": claim_count,
        "claim_support": {
            "counts": dict(sorted(support_counts.items())),
            "supported_claim_rate": (
                support_counts["supported"] / claim_count if claim_count else None
            ),
            "contradicted_claim_rate": (
                support_counts["contradicted"] / claim_count if claim_count else None
            ),
            "unsupported_claim_rate": (
                support_counts["not_in_context"] / claim_count if claim_count else None
            ),
            "case_macro_supported_claim_rate": _mean(
                [
                    float(item["supported_claim_rate"])
                    for item in valid_case_metrics
                    if item["supported_claim_rate"] is not None
                ]
            ),
            "fully_faithful_rate": _mean(
                [
                    float(item["fully_faithful"])
                    for item in valid_case_metrics
                    if item["fully_faithful"] is not None
                ]
            ),
            "faithfulness_evaluable_case_count": len(claim_cases),
        },
        "citation_entailment": {
            "counts": dict(sorted(citation_counts.items())),
            "entailed_claim_rate": (
                citation_counts["entailed"] / claim_count if claim_count else None
            ),
            "contradicted_claim_rate": (
                citation_counts["contradicted"] / claim_count if claim_count else None
            ),
            "not_entailed_claim_rate": (
                citation_counts["not_entailed"] / claim_count if claim_count else None
            ),
            "uncited_claim_rate": (
                citation_counts["not_cited"] / claim_count if claim_count else None
            ),
            "citation_complete_case_rate": _mean(
                [
                    float(item["citation_complete"])
                    for item in valid_case_metrics
                    if item["citation_complete"] is not None
                ]
            ),
        },
        "rubric_scores": {
            "semantic_correctness_mean": _mean(
                [float(item["semantic_correctness"]) for item in valid_case_metrics]
            ),
            "completeness_mean": _mean(
                [float(item["completeness"]) for item in valid_case_metrics]
            ),
            "semantic_correctness_distribution": dict(
                sorted(
                    Counter(
                        str(item["semantic_correctness"])
                        for item in valid_case_metrics
                    ).items()
                )
            ),
            "completeness_distribution": dict(
                sorted(
                    Counter(str(item["completeness"]) for item in valid_case_metrics).items()
                )
            ),
            "completeness_interpretation": "rubric_score_not_exact_required_fact_recall",
        },
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "per_case_semantic_metrics.jsonl", per_case)
        (output_dir / "semantic_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary
