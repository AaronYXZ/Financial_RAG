from dataclasses import dataclass

from rag_eval.chunking import ChunkingConfig, chunk_corpus


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict


class FakeSplitter:
    def create_documents(self, texts, metadatas=None):
        text = texts[0]
        metadata = (metadatas or [{}])[0]
        midpoint = max(1, len(text) // 2)
        return [
            FakeDocument(text[:midpoint], {**metadata, "start_index": 0}),
            FakeDocument(text[midpoint:], {**metadata, "start_index": midpoint}),
        ]


def test_chunk_corpus_creates_deterministic_ids_and_parent_map():
    corpus = {"d1": {"title": "Paper", "text": "An abstract."}}
    config = ChunkingConfig(strategy="fixed", chunk_size=8, chunk_overlap=2)

    first = chunk_corpus(
        corpus,
        config,
        splitter=FakeSplitter(),
        token_counter=lambda text: len(text.split()),
    )
    second = chunk_corpus(
        corpus,
        config,
        splitter=FakeSplitter(),
        token_counter=lambda text: len(text.split()),
    )

    assert list(first.corpus) == ["d1::chunk::0000", "d1::chunk::0001"]
    assert first.parent_by_chunk == {
        "d1::chunk::0000": "d1",
        "d1::chunk::0001": "d1",
    }
    assert first.records[0].text.startswith("Paper")
    assert first.records[1].text.endswith("abstract.")
    assert first.manifest_hash == second.manifest_hash


def test_chunking_config_rejects_invalid_overlap():
    try:
        ChunkingConfig(strategy="recursive", chunk_size=10, chunk_overlap=10)
    except ValueError as error:
        assert "smaller than chunk_size" in str(error)
    else:
        raise AssertionError("expected invalid overlap to be rejected")
