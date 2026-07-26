"""Fixed-context selection for QASPER generation tracks."""

from __future__ import annotations

from typing import Literal

from .generation_data import GenerationCase, PaperPassage, select_passages


ContextTrack = Literal["oracle-evidence", "complete-paper"]


def build_fixed_context(
    case: GenerationCase,
    track: ContextTrack,
) -> tuple[PaperPassage, ...]:
    """Return deterministic passages without invoking retrieval."""

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
    raise ValueError(f"Unknown fixed-context track: {track}")
