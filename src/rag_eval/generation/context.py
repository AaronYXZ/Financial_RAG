"""Fixed-context selection for QASPER generation tracks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from .data import GenerationCase, PaperPassage, select_passages


ContextTrack = Literal["oracle-evidence", "complete-paper", "retrieved-context"]


def build_fixed_context(
    case: GenerationCase,
    track: ContextTrack,
    *,
    retrieved_passage_ids: tuple[str, ...] | None = None,
    passage_lookup: Mapping[str, PaperPassage] | None = None,
) -> tuple[PaperPassage, ...]:
    """Return deterministic passages without invoking retrieval at generation time."""

    if track == "oracle-evidence":
        if case.answerability != "answerable":
            raise ValueError("The primary oracle-evidence track requires an answerable case")
        passages = select_passages(case, case.oracle_passage_ids)
        if not passages:
            raise ValueError(f"Case {case.case_id} has no resolved oracle evidence")
        return passages
    if track == "complete-paper":
        if not case.paper_passages:
            raise ValueError(f"Case {case.case_id} has no paper passages")
        return case.paper_passages
    if track == "retrieved-context":
        if retrieved_passage_ids is None or passage_lookup is None:
            raise ValueError("The retrieved-context track requires a frozen context manifest")
        missing = [item for item in retrieved_passage_ids if item not in passage_lookup]
        if missing:
            raise ValueError(
                f"Case {case.case_id} references unknown retrieved passages: {missing[:3]}"
            )
        passages = tuple(passage_lookup[item] for item in retrieved_passage_ids)
        if not passages:
            raise ValueError(f"Case {case.case_id} has no frozen retrieved passages")
        return passages
    raise ValueError(f"Unknown fixed-context track: {track}")
