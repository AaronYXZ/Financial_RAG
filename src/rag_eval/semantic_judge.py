"""Blinded prompts and strict response parsing for semantic evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SEMANTIC_PROMPT_VERSION = "qasper-semantic-judge-v2"
SEMANTIC_OUTPUT_SCHEMA_VERSION = 1
SEMANTIC_PARSING_POLICY = "strict-json-v1"
BLINDING_VERSION = "qasper-semantic-blinding-v1"

SUPPORT_LABELS = ("supported", "contradicted", "not_in_context")
CITATION_ENTAILMENT_LABELS = (
    "entailed",
    "contradicted",
    "not_entailed",
    "not_cited",
)

SEMANTIC_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": 30,
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "claim_text": {"type": "string"},
                    "cited_context_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "support_label": {
                        "type": "string",
                        "enum": list(SUPPORT_LABELS),
                    },
                    "support_context_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "citation_entailment": {
                        "type": "string",
                        "enum": list(CITATION_ENTAILMENT_LABELS),
                    },
                    "rationale": {"type": "string"},
                },
                "required": [
                    "claim_id",
                    "claim_text",
                    "cited_context_ids",
                    "support_label",
                    "support_context_ids",
                    "citation_entailment",
                    "rationale",
                ],
                "additionalProperties": False,
            },
        },
        "semantic_correctness": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 4},
                "rationale": {"type": "string"},
            },
            "required": ["score", "rationale"],
            "additionalProperties": False,
        },
        "completeness": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 4},
                "rationale": {"type": "string"},
            },
            "required": ["score", "rationale"],
            "additionalProperties": False,
        },
    },
    "required": ["claims", "semantic_correctness", "completeness"],
    "additionalProperties": False,
}

SEMANTIC_SYSTEM_PROMPT = """You are a blinded evaluator of a question-answering system.
You receive only a question, supplied context, reference answers, and one anonymous
candidate answer with anonymous citations. Do not infer the candidate's source.

First split a non-empty candidate answer into atomic material claims. A material
claim is an externally verifiable factual assertion needed for the answer. Exclude
pure discourse, hedging, and statements that only describe the answer format.
Use claim IDs claim_1, claim_2, and so on, in answer order. Return no claims for
an empty abstention.

Candidate citations are answer-level rather than pre-attached to individual
claims. For every claim, evaluate the full candidate.citations list:
- support_label is supported only when the supplied context entails the claim.
- support_label is contradicted when the context provides conflicting evidence.
- support_label is not_in_context when the context neither entails nor contradicts it.
- support_context_ids list every anonymous context item needed for that decision.
- If one or more candidate citations entail the claim, cited_context_ids list
  those citations and citation_entailment is entailed.
- If candidate citations exist but none entail the claim, cited_context_ids list
  the citations used for the verdict. Use contradicted when they conflict with
  the claim and not_entailed otherwise.
- Use not_cited with an empty cited_context_ids list only when candidate.citations
  is empty.
- rationale must be short and evidence-specific.

Score semantic_correctness from 0 to 4:
0 wholly incorrect or an unjustified answer, 1 mostly incorrect, 2 mixed,
3 mostly correct with a minor error, 4 fully correct relative to the references
and supplied context.

Score completeness from 0 to 4:
0 omits essentially all information needed to answer, 1 major omissions,
2 partial, 3 nearly complete with a minor omission, 4 complete for the question
given the references and supplied context. This is a rubric score, not exact
required-fact recall.

Judge only the supplied material. Return exactly the required JSON object."""


@dataclass(frozen=True)
class SemanticClaim:
    claim_id: str
    claim_text: str
    cited_context_ids: tuple[str, ...]
    support_label: str
    support_context_ids: tuple[str, ...]
    citation_entailment: str
    rationale: str


@dataclass(frozen=True)
class RubricScore:
    score: int
    rationale: str


@dataclass(frozen=True)
class SemanticJudgment:
    claims: tuple[SemanticClaim, ...]
    semantic_correctness: RubricScore
    completeness: RubricScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "claim_text": claim.claim_text,
                    "cited_context_ids": list(claim.cited_context_ids),
                    "support_label": claim.support_label,
                    "support_context_ids": list(claim.support_context_ids),
                    "citation_entailment": claim.citation_entailment,
                    "rationale": claim.rationale,
                }
                for claim in self.claims
            ],
            "semantic_correctness": {
                "score": self.semantic_correctness.score,
                "rationale": self.semantic_correctness.rationale,
            },
            "completeness": {
                "score": self.completeness.score,
                "rationale": self.completeness.rationale,
            },
        }


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def semantic_prompt_hash() -> str:
    return content_hash(
        {
            "prompt_version": SEMANTIC_PROMPT_VERSION,
            "system_prompt": SEMANTIC_SYSTEM_PROMPT,
            "output_schema": SEMANTIC_RESPONSE_SCHEMA,
            "parsing_policy": SEMANTIC_PARSING_POLICY,
        }
    )


def render_semantic_user_prompt(judge_input: Mapping[str, Any]) -> str:
    validate_blinded_judge_input(judge_input)
    return (
        "Evaluate this blinded record. Context IDs are anonymous and local to this "
        "record.\n\n"
        + json.dumps(judge_input, ensure_ascii=False, sort_keys=True, indent=2)
    )


def validate_blinded_judge_input(judge_input: Mapping[str, Any]) -> None:
    expected = {"question", "context", "references", "candidate"}
    if set(judge_input) != expected:
        raise ValueError(f"Judge input keys must be exactly {sorted(expected)}")
    if not isinstance(judge_input["question"], str) or not judge_input["question"].strip():
        raise ValueError("Judge question must be a non-empty string")
    context = judge_input["context"]
    if not isinstance(context, list):
        raise ValueError("Judge context must be a list")
    context_ids: list[str] = []
    for item in context:
        if not isinstance(item, Mapping) or set(item) != {"context_id", "text"}:
            raise ValueError("Each context item must contain context_id and text")
        if not isinstance(item["context_id"], str) or not isinstance(item["text"], str):
            raise ValueError("Context IDs and text must be strings")
        context_ids.append(item["context_id"])
    if len(context_ids) != len(set(context_ids)):
        raise ValueError("Judge context contains duplicate anonymous IDs")
    if not isinstance(judge_input["references"], list) or not all(
        isinstance(item, str) for item in judge_input["references"]
    ):
        raise ValueError("Judge references must be a list of strings")
    candidate = judge_input["candidate"]
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "answer",
        "abstain",
        "citations",
    }:
        raise ValueError("Judge candidate must contain answer, abstain, and citations")
    if not isinstance(candidate["answer"], str) or not isinstance(candidate["abstain"], bool):
        raise ValueError("Candidate answer and abstain fields have invalid types")
    if not isinstance(candidate["citations"], list) or not all(
        isinstance(item, str) for item in candidate["citations"]
    ):
        raise ValueError("Candidate citations must be a list of strings")
    unknown = sorted(set(candidate["citations"]) - set(context_ids))
    if unknown:
        raise ValueError(f"Candidate contains unknown anonymous citations: {unknown}")


def _strict_object(
    value: Any,
    *,
    expected_keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if set(value) != expected_keys:
        raise ValueError(f"{label} keys must be exactly {sorted(expected_keys)}")
    return value


def _string_list(value: Any, *, label: str, allowed: set[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must not contain duplicates")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown context IDs: {unknown}")
    return tuple(value)


def _rubric_score(value: Any, *, label: str) -> RubricScore:
    payload = _strict_object(
        value,
        expected_keys={"score", "rationale"},
        label=label,
    )
    score = payload["score"]
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
        raise ValueError(f"{label}.score must be an integer from 0 to 4")
    rationale = payload["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"{label}.rationale must be a non-empty string")
    return RubricScore(score=score, rationale=rationale.strip())


def parse_semantic_judgment(
    raw: str | Mapping[str, Any],
    *,
    allowed_context_ids: Sequence[str],
    candidate_citation_ids: Sequence[str],
    candidate_answer: str | None = None,
    candidate_abstain: bool | None = None,
) -> SemanticJudgment:
    try:
        decoded = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("Semantic judge response is not a valid JSON object") from exc
    payload = _strict_object(
        decoded,
        expected_keys={"claims", "semantic_correctness", "completeness"},
        label="Semantic judge response",
    )
    raw_claims = payload["claims"]
    if not isinstance(raw_claims, list):
        raise ValueError("claims must be a list")
    if len(raw_claims) > 30:
        raise ValueError("claims must contain at most 30 items")

    allowed = set(allowed_context_ids)
    candidate_citations = set(candidate_citation_ids)
    claims: list[SemanticClaim] = []
    claim_keys = {
        "claim_id",
        "claim_text",
        "cited_context_ids",
        "support_label",
        "support_context_ids",
        "citation_entailment",
        "rationale",
    }
    for index, value in enumerate(raw_claims, start=1):
        claim = _strict_object(
            value,
            expected_keys=claim_keys,
            label=f"claims[{index - 1}]",
        )
        expected_id = f"claim_{index}"
        if claim["claim_id"] != expected_id:
            raise ValueError(f"Claim IDs must be sequential. Expected {expected_id!r}")
        claim_text = claim["claim_text"]
        rationale = claim["rationale"]
        if not isinstance(claim_text, str) or not claim_text.strip():
            raise ValueError("claim_text must be a non-empty string")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("claim rationale must be a non-empty string")
        support_label = claim["support_label"]
        if support_label not in SUPPORT_LABELS:
            raise ValueError(f"Unknown support label: {support_label!r}")
        entailment = claim["citation_entailment"]
        if entailment not in CITATION_ENTAILMENT_LABELS:
            raise ValueError(f"Unknown citation entailment label: {entailment!r}")
        cited = _string_list(
            claim["cited_context_ids"],
            label="cited_context_ids",
            allowed=allowed,
        )
        if set(cited) - candidate_citations:
            raise ValueError("A claim cites context IDs absent from the candidate answer")
        support = _string_list(
            claim["support_context_ids"],
            label="support_context_ids",
            allowed=allowed,
        )
        if support_label in {"supported", "contradicted"} and not support:
            raise ValueError(
                "A supported or contradicted claim must identify decision context"
            )
        if candidate_citations and entailment == "not_cited":
            raise ValueError(
                "A claim cannot be not_cited when the candidate supplied citations"
            )
        if not candidate_citations and entailment != "not_cited":
            raise ValueError(
                "A claim cannot have citation entailment without candidate citations"
            )
        if entailment == "not_cited" and cited:
            raise ValueError("A not_cited claim cannot have cited_context_ids")
        if entailment != "not_cited" and not cited:
            raise ValueError("A cited claim must identify cited_context_ids")
        claims.append(
            SemanticClaim(
                claim_id=expected_id,
                claim_text=claim_text.strip(),
                cited_context_ids=cited,
                support_label=support_label,
                support_context_ids=support,
                citation_entailment=entailment,
                rationale=rationale.strip(),
            )
        )

    if candidate_abstain is True and claims:
        raise ValueError("An abstaining candidate must not produce extracted claims")
    if candidate_abstain is False and candidate_answer and not claims:
        raise ValueError("A non-empty candidate answer must produce at least one claim")

    return SemanticJudgment(
        claims=tuple(claims),
        semantic_correctness=_rubric_score(
            payload["semantic_correctness"],
            label="semantic_correctness",
        ),
        completeness=_rubric_score(payload["completeness"], label="completeness"),
    )


def anonymous_context_id(index: int) -> str:
    if index < 1:
        raise ValueError("Anonymous context index must be positive")
    return f"C{index}"


def validate_anonymous_context_id(value: str) -> bool:
    return re.fullmatch(r"C[1-9][0-9]*", value) is not None
