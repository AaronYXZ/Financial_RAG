"""Frozen prompt and strict response contract for Phase 3 generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .generation_data import PaperPassage


SYSTEM_PROMPT = """Answer the question using only the supplied context.
If the context does not support an answer, abstain.
Every factual answer must cite one or more supplied passage IDs.
Return exactly one JSON object with keys answer, abstain, citations, and confidence.
Do not include markdown or additional text."""


@dataclass(frozen=True)
class GenerationResponse:
    answer: str
    abstain: bool
    citations: tuple[str, ...]
    confidence: float


def render_user_prompt(question: str, passages: Sequence[PaperPassage]) -> str:
    context = "\n\n".join(
        f"[{passage.passage_id}]\n{passage.text}" for passage in passages
    )
    return f"Context:\n{context}\n\nQuestion:\n{question}\n"


def prompt_hash(system_prompt: str, user_prompt: str) -> str:
    payload = json.dumps(
        {"system": system_prompt, "user": user_prompt},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_generation_response(
    raw: str | Mapping[str, Any],
    *,
    allowed_citation_ids: Iterable[str],
) -> GenerationResponse:
    try:
        payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("Generation response is not a valid JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("Generation response must be a JSON object")

    expected = {"answer", "abstain", "citations", "confidence"}
    if set(payload) != expected:
        raise ValueError(f"Generation response keys must be exactly {sorted(expected)}")
    if not isinstance(payload["answer"], str):
        raise ValueError("answer must be a string")
    if not isinstance(payload["abstain"], bool):
        raise ValueError("abstain must be a boolean")
    if not isinstance(payload["citations"], list) or not all(
        isinstance(item, str) for item in payload["citations"]
    ):
        raise ValueError("citations must be a list of strings")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    citations = tuple(payload["citations"])
    unknown = sorted(set(citations) - set(allowed_citation_ids))
    if unknown:
        raise ValueError(f"Unknown citation IDs: {unknown}")
    if payload["abstain"]:
        if payload["answer"].strip() or citations:
            raise ValueError("An abstention must have an empty answer and no citations")
    elif not payload["answer"].strip():
        raise ValueError("A non-abstaining response must contain an answer")
    elif not citations:
        raise ValueError("A non-abstaining response must contain at least one citation")

    return GenerationResponse(
        answer=payload["answer"].strip(),
        abstain=payload["abstain"],
        citations=tuple(dict.fromkeys(citations)),
        confidence=float(confidence),
    )
