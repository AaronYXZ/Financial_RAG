"""Frozen prompt and strict response contract for Phase 3 generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .generation_data import PaperPassage


PROMPT_VERSION = "qasper-generation-v3"

RESPONSE_CONTRACT = """Return exactly one JSON object with exactly these four keys:
{"answer":"","abstain":true,"citations":[],"confidence":0.0}

Field requirements:
- answer must be a JSON string of at most 120 words.
- abstain must be a JSON boolean: true or false, never a quoted string.
- citations must be a JSON array containing at most 5 passage ID strings copied
  exactly from the context labels, without the surrounding "[" and "]".
  Never construct or otherwise modify an ID.
- Example: for context label [paper::paragraph::0001], return
  "citations":["paper::paragraph::0001"].
- confidence must be a JSON number from 0.0 to 1.0, never a string such as
  "high", "medium", or "low".
- If abstain is true, answer must be "" and citations must be [].
- If abstain is false, answer must be non-empty and citations must contain at
  least one exact context passage ID.
- Do not include markdown, code fences, comments, explanations, or additional keys."""

SYSTEM_PROMPT = f"""Answer the question using only the supplied context.
If the context does not support an answer, abstain.
Every factual answer must cite one or more supplied passage IDs.

{RESPONSE_CONTRACT}"""


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
    return (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Response contract reminder:\n{RESPONSE_CONTRACT}\n"
    )


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
    if len(payload["answer"].split()) > 120:
        raise ValueError("answer must contain at most 120 words")
    if len(payload["citations"]) > 5:
        raise ValueError("citations must contain at most 5 passage IDs")
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
