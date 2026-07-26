"""QASPER normalization for fixed-context generation experiments."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote


QASPER_DATASET = "allenai/qasper"
QASPER_CONFIG = "qasper"
QASPER_VERSION = "0.3.0"
QASPER_PARQUET_REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"
QASPER_PARQUET_URL = (
    "https://huggingface.co/datasets/allenai/qasper/resolve/"
    "{revision}/qasper/{split}/0000.parquet"
)


@dataclass(frozen=True)
class PaperPassage:
    passage_id: str
    paper_id: str
    kind: str
    section_name: str
    text: str
    order: int


@dataclass(frozen=True)
class ReferenceAnswer:
    annotation_id: str
    answer_type: str
    text: str
    unanswerable: bool
    extractive_spans: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_texts: tuple[str, ...]
    highlighted_evidence: tuple[str, ...]
    unresolved_evidence: tuple[str, ...]


@dataclass(frozen=True)
class GenerationCase:
    case_id: str
    split: str
    paper_id: str
    title: str
    question: str
    answerability: str
    paper_passages: tuple[PaperPassage, ...]
    oracle_passage_ids: tuple[str, ...]
    references: tuple[ReferenceAnswer, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _records(value: Any) -> list[dict[str, Any]]:
    """Convert a list-of-structs or HF struct-of-lists into records."""

    if value is None:
        return []
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a sequence structure, received {type(value).__name__}")

    lengths = [len(item) for item in value.values() if isinstance(item, (list, tuple))]
    if not lengths:
        return [dict(value)]
    if len(set(lengths)) != 1:
        raise ValueError("Column-oriented sequence fields have different lengths")

    records: list[dict[str, Any]] = []
    for index in range(lengths[0]):
        records.append(
            {
                key: item[index] if isinstance(item, (list, tuple)) else item
                for key, item in value.items()
            }
        )
    return records


def _passage_id(paper_id: str, kind: str, index: int) -> str:
    return f"{paper_id}::{kind}::{index:04d}"


def _build_passages(row: Mapping[str, Any]) -> tuple[PaperPassage, ...]:
    paper_id = str(row["id"])
    passages: list[PaperPassage] = []

    def add(kind: str, section_name: str, text: Any) -> None:
        normalized = _normalize_text(text)
        if not normalized:
            return
        order = len(passages)
        passages.append(
            PaperPassage(
                passage_id=_passage_id(paper_id, kind, order),
                paper_id=paper_id,
                kind=kind,
                section_name=section_name,
                text=normalized,
                order=order,
            )
        )

    add("title", "Title", row.get("title"))
    add("abstract", "Abstract", row.get("abstract"))

    for section in _records(row.get("full_text")):
        section_name = _normalize_text(section.get("section_name")) or "Untitled section"
        add("section", section_name, section_name)
        paragraphs = section.get("paragraphs") or []
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        for paragraph in paragraphs:
            add("paragraph", section_name, paragraph)

    for item in _records(row.get("figures_and_tables")):
        add("figure_or_table", "Figures and tables", item.get("caption"))

    return tuple(passages)


def _evidence_lookup(passages: Sequence[PaperPassage]) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for passage in passages:
        keys = {_normalize_text(passage.text)}
        if passage.kind == "figure_or_table":
            keys.add(_normalize_text(f"FLOAT SELECTED: {passage.text}"))
        for key in keys:
            if key:
                lookup.setdefault(key, []).append(passage.passage_id)
    return lookup


def _annotation_records(raw_answers: Any) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for item in _records(raw_answers):
        nested = item.get("answer")
        if isinstance(nested, list):
            annotation_ids = item.get("annotation_id") or []
            worker_ids = item.get("worker_id") or []
            for index, answer in enumerate(nested):
                annotations.append(
                    {
                        "answer": answer,
                        "annotation_id": annotation_ids[index] if index < len(annotation_ids) else "",
                        "worker_id": worker_ids[index] if index < len(worker_ids) else "",
                    }
                )
        else:
            annotations.append(item)
    return annotations


def _reference_answer(
    annotation: Mapping[str, Any],
    lookup: Mapping[str, list[str]],
) -> ReferenceAnswer:
    answer = annotation.get("answer") or {}
    if not isinstance(answer, Mapping):
        raise TypeError("QASPER annotation answer must be a mapping")

    unanswerable = bool(answer.get("unanswerable"))
    extractive = tuple(_normalize_text(value) for value in answer.get("extractive_spans") or [])
    extractive = tuple(value for value in extractive if value)
    free_form = _normalize_text(answer.get("free_form_answer"))
    yes_no = answer.get("yes_no")

    if unanswerable:
        answer_type, text = "unanswerable", ""
    elif extractive:
        answer_type, text = "extractive", " ".join(extractive)
    elif free_form:
        answer_type, text = "free_form", free_form
    elif yes_no is not None:
        answer_type, text = "yes_no", "Yes" if bool(yes_no) else "No"
    else:
        answer_type, text = "missing", ""

    evidence_texts = tuple(
        value
        for value in (_normalize_text(item) for item in answer.get("evidence") or [])
        if value
    )
    evidence_ids: list[str] = []
    unresolved: list[str] = []
    for evidence in evidence_texts:
        matches = lookup.get(evidence, [])
        if matches:
            evidence_ids.extend(matches)
        else:
            unresolved.append(evidence)

    return ReferenceAnswer(
        annotation_id=str(annotation.get("annotation_id") or ""),
        answer_type=answer_type,
        text=text,
        unanswerable=unanswerable,
        extractive_spans=extractive,
        evidence_ids=tuple(dict.fromkeys(evidence_ids)),
        evidence_texts=evidence_texts,
        highlighted_evidence=tuple(
            value
            for value in (
                _normalize_text(item) for item in answer.get("highlighted_evidence") or []
            )
            if value
        ),
        unresolved_evidence=tuple(unresolved),
    )


def _answerability(references: Sequence[ReferenceAnswer]) -> str:
    labels = {reference.unanswerable for reference in references}
    if labels == {False}:
        return "answerable"
    if labels == {True}:
        return "unanswerable"
    return "ambiguous"


def normalize_qasper_row(row: Mapping[str, Any], split: str) -> list[GenerationCase]:
    """Flatten one QASPER paper row into one case per question."""

    paper_id = str(row["id"])
    passages = _build_passages(row)
    lookup = _evidence_lookup(passages)
    order = {passage.passage_id: passage.order for passage in passages}
    cases: list[GenerationCase] = []

    for question in _records(row.get("qas")):
        references = tuple(
            _reference_answer(annotation, lookup)
            for annotation in _annotation_records(question.get("answers"))
        )
        oracle_ids = sorted(
            {
                passage_id
                for reference in references
                if not reference.unanswerable
                for passage_id in reference.evidence_ids
            },
            key=order.__getitem__,
        )
        cases.append(
            GenerationCase(
                case_id=str(question.get("question_id") or f"{paper_id}:{len(cases)}"),
                split=split,
                paper_id=paper_id,
                title=_normalize_text(row.get("title")),
                question=_normalize_text(question.get("question")),
                answerability=_answerability(references),
                paper_passages=passages,
                oracle_passage_ids=tuple(oracle_ids),
                references=references,
            )
        )


    return cases
def qasper_parquet_url(split: str, revision: str = QASPER_PARQUET_REVISION) -> str:
    return QASPER_PARQUET_URL.format(
        revision=quote(revision, safe=""),
        split=split,
    )


def load_qasper_cases(
    split: str,
    *,
    cache_dir: str | None = None,
    revision: str = QASPER_PARQUET_REVISION,
    limit_papers: int | None = None,
) -> list[GenerationCase]:
    """Download QASPER through Hugging Face and return normalized cases."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "QASPER loading requires the generation extra: pip install -e '.[generation]'"
        ) from exc

    parquet_url = qasper_parquet_url(split, revision)
    dataset = load_dataset(
        "parquet",
        split=split,
        data_files={split: parquet_url},
        cache_dir=cache_dir,
    )
    if limit_papers is not None:
        if limit_papers < 1:
            raise ValueError("limit_papers must be at least 1")
        dataset = dataset.select(range(min(limit_papers, len(dataset))))

    cases: list[GenerationCase] = []
    for row in dataset:
        cases.extend(normalize_qasper_row(row, split))
    return cases


def select_passages(
    case: GenerationCase,
    passage_ids: Iterable[str],
) -> tuple[PaperPassage, ...]:
    wanted = set(passage_ids)
    return tuple(passage for passage in case.paper_passages if passage.passage_id in wanted)
