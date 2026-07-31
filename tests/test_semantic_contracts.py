import json

import pytest

from rag_eval.semantic.contracts import (
    SEMANTIC_PROMPT_VERSION,
    parse_semantic_judgment,
    render_semantic_user_prompt,
    semantic_prompt_hash,
)


VALID_JUDGMENT = {
    "claims": [
        {
            "claim_id": "claim_1",
            "claim_text": "The method improved accuracy.",
            "cited_context_ids": ["C1"],
            "support_label": "supported",
            "support_context_ids": ["C1"],
            "citation_entailment": "entailed",
            "rationale": "C1 directly reports the improvement.",
        }
    ],
    "semantic_correctness": {
        "score": 4,
        "rationale": "The answer matches the reference and context.",
    },
    "completeness": {
        "score": 3,
        "rationale": "The main result is present but the magnitude is omitted.",
    },
}


def test_semantic_prompt_is_versioned_and_stable():
    assert SEMANTIC_PROMPT_VERSION == "qasper-semantic-judge-v2"
    assert len(semantic_prompt_hash()) == 64
    assert semantic_prompt_hash() == semantic_prompt_hash()


def test_rendered_prompt_contains_only_blinded_payload():
    judge_input = {
        "question": "What improved?",
        "context": [{"context_id": "C1", "text": "Accuracy improved."}],
        "references": ["Accuracy improved."],
        "candidate": {
            "answer": "The method improved accuracy.",
            "abstain": False,
            "citations": ["C1"],
        },
    }

    rendered = render_semantic_user_prompt(judge_input)

    assert "C1" in rendered
    assert "paper::paragraph::0001" not in rendered
    assert "openai/gpt-5.6-luna-pro" not in rendered
    assert "latency" not in rendered
    assert "confidence" not in rendered


def test_parse_semantic_judgment_enforces_claim_and_citation_contract():
    parsed = parse_semantic_judgment(
        json.dumps(VALID_JUDGMENT),
        allowed_context_ids=["C1"],
        candidate_citation_ids=["C1"],
        candidate_answer="The method improved accuracy.",
        candidate_abstain=False,
    )

    assert parsed.claims[0].support_label == "supported"
    assert parsed.claims[0].citation_entailment == "entailed"
    assert parsed.semantic_correctness.score == 4
    assert parsed.completeness.score == 3


def test_parse_semantic_judgment_rejects_nonsequential_claim_ids():
    payload = json.loads(json.dumps(VALID_JUDGMENT))
    payload["claims"][0]["claim_id"] = "claim_2"

    with pytest.raises(ValueError, match="sequential"):
        parse_semantic_judgment(
            payload,
            allowed_context_ids=["C1"],
            candidate_citation_ids=["C1"],
        )


def test_parse_semantic_judgment_rejects_hallucinated_candidate_citation():
    payload = json.loads(json.dumps(VALID_JUDGMENT))
    payload["claims"][0]["cited_context_ids"] = ["C2"]

    with pytest.raises(ValueError, match="unknown context IDs"):
        parse_semantic_judgment(
            payload,
            allowed_context_ids=["C1"],
            candidate_citation_ids=["C1"],
        )


def test_parse_semantic_judgment_requires_claim_for_nonempty_answer():
    payload = json.loads(json.dumps(VALID_JUDGMENT))
    payload["claims"] = []

    with pytest.raises(ValueError, match="at least one claim"):
        parse_semantic_judgment(
            payload,
            allowed_context_ids=["C1"],
            candidate_citation_ids=["C1"],
            candidate_answer="A factual answer.",
            candidate_abstain=False,
        )


def test_parse_semantic_judgment_requires_evidence_for_contradiction():
    payload = json.loads(json.dumps(VALID_JUDGMENT))
    payload["claims"][0]["support_label"] = "contradicted"
    payload["claims"][0]["support_context_ids"] = []

    with pytest.raises(ValueError, match="decision context"):
        parse_semantic_judgment(
            payload,
            allowed_context_ids=["C1"],
            candidate_citation_ids=["C1"],
        )


def test_parse_semantic_judgment_rejects_not_cited_when_candidate_cited():
    payload = json.loads(json.dumps(VALID_JUDGMENT))
    payload["claims"][0]["cited_context_ids"] = []
    payload["claims"][0]["citation_entailment"] = "not_cited"

    with pytest.raises(ValueError, match="candidate supplied citations"):
        parse_semantic_judgment(
            payload,
            allowed_context_ids=["C1"],
            candidate_citation_ids=["C1"],
        )
