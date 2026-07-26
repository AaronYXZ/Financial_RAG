import pytest

from rag_eval.generation_data import PaperPassage
from rag_eval.generation_prompt import (
    SYSTEM_PROMPT,
    parse_generation_response,
    prompt_hash,
    render_user_prompt,
)


PASSAGE = PaperPassage("p1", "paper", "paragraph", "Results", "Evidence.", 0)


def test_prompt_is_deterministic():
    prompt = render_user_prompt("What happened?", [PASSAGE])

    assert "[p1]\nEvidence." in prompt
    assert prompt_hash(SYSTEM_PROMPT, prompt) == prompt_hash(SYSTEM_PROMPT, prompt)


def test_strict_response_contract_accepts_grounded_answer():
    response = parse_generation_response(
        '{"answer":"It happened.","abstain":false,"citations":["p1"],"confidence":0.8}',
        allowed_citation_ids={"p1"},
    )

    assert response.citations == ("p1",)
    assert response.confidence == 0.8


@pytest.mark.parametrize(
    "payload",
    [
        '{"answer":"Unsupported","abstain":false,"citations":["other"],"confidence":0.8}',
        '{"answer":"Text","abstain":true,"citations":[],"confidence":0.2}',
        '{"answer":"Text","abstain":false,"citations":[],"confidence":1.2}',
    ],
)
def test_strict_response_contract_rejects_invalid_payload(payload):
    with pytest.raises(ValueError):
        parse_generation_response(payload, allowed_citation_ids={"p1"})
