"""Download, load, and sample datasets in the BEIR JSONL format."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BEIR_DATASET_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
)

BEIR_DATASET_MD5 = {
    "scidocs": "38121350fc3a4d2f48850f6aff52e4a9",
}


@dataclass(frozen=True)
class BeirDataset:
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def download_dataset(dataset: str, data_dir: Path) -> Path:
    """Download and safely extract one public BEIR dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = data_dir / dataset
    if _looks_like_dataset(dataset_dir):
        return dataset_dir

    archive = data_dir / f"{dataset}.zip"
    url = BEIR_DATASET_URL.format(dataset=dataset)
    try:
        with urllib.request.urlopen(url) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target)
        expected_md5 = BEIR_DATASET_MD5.get(dataset)
        actual_md5 = _file_md5(archive)
        if expected_md5 and actual_md5 != expected_md5:
            raise ValueError(f"Checksum mismatch for downloaded BEIR dataset: {dataset}")
        with zipfile.ZipFile(archive) as zip_file:
            _safe_extract(zip_file, data_dir)
    except Exception:
        archive.unlink(missing_ok=True)
        raise
    archive.unlink(missing_ok=True)

    if not _looks_like_dataset(dataset_dir):
        raise ValueError(f"Downloaded archive did not contain a valid BEIR dataset: {dataset_dir}")
    return dataset_dir


def load_dataset(dataset_dir: Path, split: str = "test") -> BeirDataset:
    """Load corpus, queries, and qrels from an extracted BEIR dataset."""
    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"

    missing = [path for path in (corpus_path, queries_path, qrels_path) if not path.is_file()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required BEIR file(s): {joined}")

    corpus: dict[str, dict[str, str]] = {}
    for item in _read_jsonl(corpus_path):
        doc_id = str(item["_id"])
        corpus[doc_id] = {
            "title": str(item.get("title", "")),
            "text": str(item.get("text", "")),
        }

    queries = {
        str(item["_id"]): str(item.get("text", ""))
        for item in _read_jsonl(queries_path)
    }

    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected qrels columns in {qrels_path}")
        for row in reader:
            query_id = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            qrels.setdefault(query_id, {})[doc_id] = int(row["score"])

    valid_queries = {query_id: text for query_id, text in queries.items() if query_id in qrels}
    valid_qrels = {query_id: qrels[query_id] for query_id in valid_queries}
    return BeirDataset(corpus=corpus, queries=valid_queries, qrels=valid_qrels)


def sample_dataset(
    dataset: BeirDataset,
    max_documents: int | None = None,
    max_queries: int | None = None,
    seed: int = 42,
) -> BeirDataset:
    """Create a deterministic sample while retaining every positive document."""
    randomizer = random.Random(seed)
    query_ids = sorted(dataset.queries)
    if max_queries is not None and max_queries < len(query_ids):
        query_ids = sorted(randomizer.sample(query_ids, max_queries))

    queries = {query_id: dataset.queries[query_id] for query_id in query_ids}
    qrels = {query_id: dataset.qrels[query_id] for query_id in query_ids}
    relevant_ids = {
        doc_id
        for judgments in qrels.values()
        for doc_id, relevance in judgments.items()
        if relevance > 0 and doc_id in dataset.corpus
    }

    if max_documents is not None and max_documents < len(relevant_ids):
        raise ValueError(
            f"max_documents={max_documents} is too small to retain "
            f"{len(relevant_ids)} relevant documents"
        )

    all_ids = set(dataset.corpus)
    if max_documents is None or max_documents >= len(all_ids):
        selected_ids = all_ids
    else:
        candidates = sorted(all_ids - relevant_ids)
        filler_count = max_documents - len(relevant_ids)
        selected_ids = relevant_ids | set(randomizer.sample(candidates, filler_count))

    corpus = {doc_id: dataset.corpus[doc_id] for doc_id in sorted(selected_ids)}
    filtered_qrels = {
        query_id: {
            doc_id: relevance
            for doc_id, relevance in judgments.items()
            if doc_id in corpus
        }
        for query_id, judgments in qrels.items()
    }
    return BeirDataset(corpus=corpus, queries=queries, qrels=filtered_qrels)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def _looks_like_dataset(path: Path) -> bool:
    return (path / "corpus.jsonl").is_file() and (path / "queries.jsonl").is_file()


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Unsafe archive member: {member.filename}")
    archive.extractall(destination)


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
