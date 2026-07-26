"""Deterministic LangChain chunking for retrieval benchmarks."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

from rag_eval.retrievers import Corpus


TOKEN_SAFETY_RESERVE = 8


class TextSplitter(Protocol):
    def create_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[Any]: ...


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str
    chunk_size: int = 256
    chunk_overlap: int = 32
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        if self.strategy not in {"fixed", "recursive"}:
            raise ValueError("strategy must be 'fixed' or 'recursive'")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    parent_doc_id: str
    chunk_index: int
    start_index: int
    title: str
    text: str
    token_count: int


@dataclass(frozen=True)
class ChunkedCorpus:
    corpus: dict[str, dict[str, str]]
    parent_by_chunk: dict[str, str]
    records: tuple[ChunkRecord, ...]
    manifest_hash: str
    stats: dict[str, float | int]


def chunk_corpus(
    corpus: Corpus,
    config: ChunkingConfig,
    *,
    splitter: TextSplitter | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> ChunkedCorpus:
    """Split each parent document and return a deterministic chunk corpus."""
    if splitter is None or token_counter is None:
        default_splitter, default_counter = _build_langchain_splitter(config)
        splitter = splitter or default_splitter
        token_counter = token_counter or default_counter

    records: list[ChunkRecord] = []
    chunk_documents: dict[str, dict[str, str]] = {}
    parent_by_chunk: dict[str, str] = {}
    chunks_per_document: list[int] = []

    for parent_doc_id in sorted(corpus):
        document = corpus[parent_doc_id]
        title = _normalize_text(document.get("title", ""))
        body = _normalize_text(document.get("text", ""))
        source_text = _join_title_and_body(title, body)
        metadata = {"parent_doc_id": parent_doc_id}
        split_documents = splitter.create_documents([source_text], [metadata])
        if not split_documents:
            split_documents = [_FallbackDocument(source_text, {**metadata, "start_index": 0})]

        chunks_per_document.append(len(split_documents))
        for chunk_index, split_document in enumerate(split_documents):
            text = str(split_document.page_content).strip()
            if not text:
                continue
            start_index = int(split_document.metadata.get("start_index", 0))
            chunk_id = f"{parent_doc_id}::chunk::{chunk_index:04d}"
            token_count = token_counter(text)
            if token_count > config.chunk_size:
                raise ValueError(
                    f"chunk {chunk_id} has {token_count} tokens; limit is {config.chunk_size}"
                )
            record = ChunkRecord(
                chunk_id=chunk_id,
                parent_doc_id=parent_doc_id,
                chunk_index=chunk_index,
                start_index=start_index,
                title=title,
                text=text,
                token_count=token_count,
            )
            records.append(record)
            chunk_documents[chunk_id] = {"title": "", "text": text}
            parent_by_chunk[chunk_id] = parent_doc_id

    manifest_hash = _manifest_hash(records, config)
    return ChunkedCorpus(
        corpus=chunk_documents,
        parent_by_chunk=parent_by_chunk,
        records=tuple(records),
        manifest_hash=manifest_hash,
        stats=_chunk_stats(records, chunks_per_document, len(corpus)),
    )


def write_chunk_manifest(chunked: ChunkedCorpus, path: Any) -> None:
    """Write chunk records as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in chunked.records:
            file.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def _build_langchain_splitter(
    config: ChunkingConfig,
) -> tuple[TextSplitter, Callable[[str], int]]:
    try:
        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,
            SentenceTransformersTokenTextSplitter,
        )
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Chunking requires benchmark dependencies. "
            "Install them with: pip install -e '.[benchmark]'"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # Long unsplit sources are expected here. The benchmark's explicit chunk
    # ceiling, not this tokenizer warning threshold, governs final inputs.
    tokenizer.model_max_length = 1_000_000


    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=True))

    effective_chunk_size = max(
        config.chunk_overlap + 1,
        config.chunk_size - TOKEN_SAFETY_RESERVE,
    )
    if config.strategy == "fixed":
        splitter = SentenceTransformersTokenTextSplitter(
            model_name=config.model_name,
            tokens_per_chunk=effective_chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
        )
    else:
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=effective_chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
            strip_whitespace=True,
        )
    return splitter, count_tokens


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _join_title_and_body(title: str, body: str) -> str:
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def _manifest_hash(records: list[ChunkRecord], config: ChunkingConfig) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(config), sort_keys=True).encode())
    for record in records:
        digest.update(json.dumps(asdict(record), sort_keys=True).encode())
    return digest.hexdigest()


def _chunk_stats(
    records: list[ChunkRecord],
    chunks_per_document: list[int],
    parent_document_count: int,
) -> dict[str, float | int]:
    token_counts = [record.token_count for record in records]
    single_chunk_count = sum(count == 1 for count in chunks_per_document)
    return {
        "parent_documents": parent_document_count,
        "chunks": len(records),
        "single_chunk_documents": single_chunk_count,
        "single_chunk_percent": round(
            100 * single_chunk_count / parent_document_count, 4
        )
        if parent_document_count
        else 0.0,
        "chunks_per_document_mean": _mean(chunks_per_document),
        "chunks_per_document_median": _median(chunks_per_document),
        "chunks_per_document_p95": _percentile(chunks_per_document, 95),
        "chunks_per_document_max": max(chunks_per_document, default=0),
        "tokens_per_chunk_mean": _mean(token_counts),
        "tokens_per_chunk_median": _median(token_counts),
        "tokens_per_chunk_p95": _percentile(token_counts, 95),
        "tokens_per_chunk_max": max(token_counts, default=0),
    }


def _mean(values: list[int]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _median(values: list[int]) -> float:
    return round(float(statistics.median(values)), 4) if values else 0.0


def _percentile(values: list[int] | list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(float(ordered[lower] * (1 - weight) + ordered[upper] * weight), 4)


@dataclass
class _FallbackDocument:
    page_content: str
    metadata: dict[str, Any]
