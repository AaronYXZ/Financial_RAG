# SciDocs Retrieval Benchmark Specification

## 1. Purpose

This specification defines Phase 2 of the RAG benchmark: evaluate how chunking
and retrieval choices affect evidence discovery before generation is introduced.
It compares two LangChain chunking strategies across lexical, dense, and hybrid
retrieval on SciDocs.

The benchmark answers:

1. Does the system retrieve at least one relevant document within the top K?
2. Does it rank relevant documents near the top?
3. What quality, latency, memory, and index-size tradeoffs does each configuration
   create?
4. Does recursive chunking improve retrieval over fixed token windows when every
   other component is held constant?

This phase does not evaluate answer generation, faithfulness, or citations.

## 2. Scope and fixed decisions

| Dimension | Decision |
| --- | --- |
| Dataset | BEIR `scidocs` |
| Dataset split | Official `test` split |
| Evaluation unit | Parent SciDocs document |
| Retrieval unit | LangChain-generated chunk |
| Chunkers | Fixed token and recursive |
| Retrievers | BM25, dense cosine, and hybrid RRF |
| Primary metric | Recall@10 |
| Ranking metric | NDCG@10 |
| Secondary metrics | Precision@K, Recall@K, MRR@K, NDCG@K |
| K values | 1, 3, 5, 10, 100 |
| Random seed | 42 |
| Corpus scope | Full corpus |
| Query scope | All official test queries |
| Dense model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dense search | Exact cosine similarity |
| Hybrid method | Equal-weight reciprocal rank fusion, `rrf_k=60` |
| Repeated timed runs | 3 after one warm-up run |

The official BEIR inventory lists SciDocs as a test-only dataset with approximately
1,000 queries, 25,000 documents, and 4.9 relevant documents per query. Record the
actual counts after download rather than treating the rounded inventory values as
validation constants. See the [BEIR dataset inventory](https://github.com/beir-cellar/beir#available-datasets).

## 3. Non-goals

- Do not compare generation models or prompts.
- Do not add a reranker in the primary experiment matrix.
- Do not tune multiple variables in one run.
- Do not compare sampled-corpus scores with full-corpus results.
- Do not report chunk-level relevance metrics because SciDocs provides
  document-level qrels, not passage-level labels.
- Do not claim direct comparability with published whole-document BEIR results
  unless the whole-document control uses the same official evaluation protocol.

## 4. Dataset contract

### 4.1 Source and integrity

Use the BEIR dataset name `scidocs` and the official test split. The benchmark
runner must save:

- download URL
- archive checksum
- corpus, query, and qrels counts
- dataset directory or immutable artifact identifier
- ingestion timestamp in UTC

The BEIR inventory currently publishes MD5
`38121350fc3a4d2f48850f6aff52e4a9` for the SciDocs archive. Treat a checksum
mismatch as a hard failure unless the benchmark specification is deliberately
versioned.

Expected BEIR files:

```text
data/scidocs/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
    └── test.tsv
```

### 4.2 Evaluation population

- Use every query that has an official test qrel.
- Use the complete corpus for reportable results.
- Never remove a positively judged document.
- Do not tune configurations iteratively against the official test results.
- Pre-register the six primary configurations in this document before running
  them.

SciDocs is test-only in BEIR. If iterative tuning becomes necessary, create and
version a deterministic development partition before inspecting candidate
results, then reserve the remaining queries as a local holdout. Such a local
partition is not directly comparable with published full-test scores.

### 4.3 Text normalization

For each parent document:

1. Read `_id`, `title`, and `text` from `corpus.jsonl`.
2. Normalize line endings to `\n`.
3. Strip leading and trailing whitespace.
4. Preserve case and punctuation in stored chunk text.
5. Do not apply stemming, stop-word removal, summarization, or model-generated
   enrichment.
6. Split only within a parent document. Never create a chunk spanning documents.

Split the body in `text`. Prefix every non-empty chunk with the title for indexing:

```text
{title}\n\n{body_chunk}
```

This title policy must be identical for both chunkers and all retrievers.

## 5. LangChain chunking strategies

Use the standalone `langchain-text-splitters` package. Pin the exact package and
tokenizer versions in the run manifest. LangChain documents its current splitter
APIs in the [text splitter guide](https://docs.langchain.com/oss/python/integrations/splitters)
and recommends recursive splitting as a general-purpose starting point.

### 5.1 Shared chunking parameters

| Parameter | Value |
| --- | ---: |
| Target chunk size | 256 tokens |
| Chunk overlap | 32 tokens |
| Tokenizer | Tokenizer for `sentence-transformers/all-MiniLM-L6-v2` |
| Keep empty chunks | No |
| Add start index | Yes |
| Title policy | Prefix title to every body chunk after splitting |
| Chunk order | Original document order |

The overlap is 12.5 percent of the target size. The splitter must count only the
body against the 256-token target. After prefixing the title, validate the final
input against the dense model token limit and record any truncation. The target
is invalid if any indexed chunk is silently truncated.

Because SciDocs records may be shorter than the target size, also report:

- percentage of documents that remain one chunk
- chunks per parent document, mean, median, p95, and maximum
- token count per chunk, mean, median, p95, and maximum
- total chunk count and duplicate-chunk count

If at least 90 percent of documents remain one chunk, fixed versus recursive is
not an informative comparison at 256 tokens. In that case, run a pre-registered
sensitivity experiment at 128 tokens with 16-token overlap. Keep the 256-token
matrix as the primary benchmark and label the 128-token matrix as sensitivity
analysis.

### 5.2 Fixed token strategy

Purpose: create deterministic token windows without respecting paragraph or
sentence boundaries.

Use LangChain `SentenceTransformersTokenTextSplitter` so chunk size is measured
with the same model tokenizer used for dense retrieval:

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    tokens_per_chunk=256,
    chunk_overlap=32,
    add_start_index=True,
)
```

The fixed strategy is the chunking baseline. Record the installed splitter
version because tokenization behavior is part of the experiment.

### 5.3 Recursive strategy

Purpose: preserve semantically coherent boundaries where possible while enforcing
the same token budget.

Use `RecursiveCharacterTextSplitter.from_huggingface_tokenizer` with the dense
model tokenizer and this separator order:

```python
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=256,
    chunk_overlap=32,
    separators=["\n\n", "\n", ". ", " ", ""],
    add_start_index=True,
    strip_whitespace=True,
)
```

The recursive splitter first attempts paragraph, line, sentence-like, and word
boundaries before falling back to characters. This behavior follows LangChain's
[recursive splitting model](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter).

### 5.4 Chunk identity and metadata

Every chunk must have a deterministic identifier:

```text
{parent_doc_id}::chunk::{zero_padded_chunk_index}
```

Example:

```text
12345::chunk::0002
```

Persist this metadata with each chunk:

```json
{
  "chunk_id": "12345::chunk::0002",
  "parent_doc_id": "12345",
  "chunk_index": 2,
  "start_index": 481,
  "title": "Example paper title",
  "chunker": "recursive",
  "chunk_size": 256,
  "chunk_overlap": 32,
  "token_count": 241
}
```

Create a stable chunk-manifest hash from the ordered chunk IDs, content, metadata,
splitter configuration, and splitter version. Reuse indexes only when the hash
matches.

## 6. Retrieval models

### 6.1 Lexical baseline

Use the repository's Okapi BM25 implementation with:

| Parameter | Value |
| --- | ---: |
| `k1` | 1.5 |
| `b` | 0.75 |
| Text | title-prefixed chunk |
| Tokenization | Unicode-aware case-folded word tokens |

Do not tune BM25 parameters during the primary matrix.

### 6.2 Dense baseline

Use:

| Parameter | Value |
| --- | --- |
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Document input | title-prefixed chunk |
| Query input | original SciDocs query text |
| Normalization | L2 normalize document and query embeddings |
| Similarity | cosine similarity via normalized dot product |
| Search | exact, no approximate nearest neighbor index |
| Batch size | 32 |
| Precision | model default, recorded in manifest |

Cache document embeddings by chunk-manifest hash and model revision. Do not cache
query timing results across timed repetitions.

### 6.3 Hybrid baseline

Run BM25 and dense retrieval independently, collapse each result to parent
documents, then combine parent-document ranks with reciprocal rank fusion:

```text
RRF_score(document) = 1 / (60 + lexical_rank)
                    + 1 / (60 + dense_rank)
```

Use equal weights. Each retriever returns at least `max(100, 4 * K)` chunk
candidates before parent collapse. Retrieve more candidates if fewer than K unique
parent documents remain after collapse.

Fuse parent-document ranks, not raw chunk ranks. This prevents documents with more
chunks from receiving repeated fusion credit.

## 7. Chunk-to-document scoring

SciDocs qrels identify relevant parent documents. They do not identify relevant
chunks. Apply this procedure before calculating metrics:

1. Retrieve ranked chunks.
2. Group chunks by `parent_doc_id`.
3. Assign each parent the maximum score among its retrieved chunks.
4. Break score ties by ascending `parent_doc_id`.
5. Rank unique parent documents.
6. Retain the highest-scoring chunk as `winning_chunk_id` for error analysis.
7. Evaluate parent document IDs against the official qrels.

For hybrid retrieval, perform steps 1 through 5 separately for lexical and dense
results before applying RRF.

This aggregation answers whether chunk retrieval finds the relevant document. It
does not prove that the winning chunk contains the relevant evidence. Passage-level
evaluation requires separate passage labels.

## 8. Primary experiment matrix

Run exactly these six reportable configurations first:

| Run ID | Chunker | Retriever | Changed variables |
| --- | --- | --- | --- |
| `scidocs-fixed-bm25` | Fixed token | BM25 | Baseline |
| `scidocs-fixed-dense` | Fixed token | Dense | Retriever only |
| `scidocs-fixed-hybrid` | Fixed token | Hybrid RRF | Retriever only |
| `scidocs-recursive-bm25` | Recursive | BM25 | Chunker only versus fixed BM25 |
| `scidocs-recursive-dense` | Recursive | Dense | Chunker only versus fixed dense |
| `scidocs-recursive-hybrid` | Recursive | Hybrid RRF | Chunker only versus fixed hybrid |

Also run one diagnostic whole-document BM25 control using the unchunked title and
text. Keep it outside the six-run matrix because it changes the retrieval unit.

No other parameter may differ within a paired comparison. Use identical corpus,
queries, qrels, title policy, K values, seed, hardware, and process settings.

## 9. Metrics and statistics

### 9.1 Retrieval quality

Calculate macro-averaged metrics over the same query set:

- Precision@1, @3, @5, @10, @100
- Recall@1, @3, @5, @10, @100
- MRR@1, @3, @5, @10, @100
- NDCG@1, @3, @5, @10, @100

Use Recall@10 as the primary retrieval gate and NDCG@10 as the primary ranking
metric. MRR@10 is the tie-breaker when Recall@10 and NDCG@10 are practically tied.

Save per-query metric contributions. Report paired bootstrap 95 percent confidence
intervals for metric deltas using 10,000 resamples and seed 42. Do not describe a
small delta as an improvement when its interval crosses zero.

### 9.2 Performance and resource metrics

Measure and report:

- chunking wall time
- BM25 index construction time
- embedding construction time
- retrieval latency per query, p50, p95, and p99
- total batch retrieval time and throughput
- peak resident memory
- serialized index size
- embedding cache size
- document and query embedding counts
- estimated embedding cost, recorded as zero for a local model while retaining the
  schema for later hosted models
- errors and retries

Run one untimed warm-up, then three timed repetitions. Report the median of each
aggregate metric. Use a fresh process for cold index-build measurements and a
prebuilt index for warm query-latency measurements.

## 10. Reproducibility manifest

Every run report must include:

```json
{
  "dataset": "scidocs",
  "split": "test",
  "dataset_checksum": "38121350fc3a4d2f48850f6aff52e4a9",
  "chunker": "fixed-or-recursive",
  "chunk_size": 256,
  "chunk_overlap": 32,
  "splitter_package_version": "TBD-at-runtime",
  "tokenizer_name": "sentence-transformers/all-MiniLM-L6-v2",
  "tokenizer_revision": "TBD-at-runtime",
  "retriever": "bm25-or-dense-or-hybrid",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_model_revision": "TBD-at-runtime",
  "similarity": "cosine",
  "rrf_k": 60,
  "k_values": [1, 3, 5, 10, 100],
  "seed": 42,
  "git_commit": "TBD-at-runtime",
  "python_version": "TBD-at-runtime",
  "hardware": "TBD-at-runtime"
}
```

Also record operating system, dependency lock hash, thread counts, device, numeric
precision, environment variables that affect parallelism, and whether cached
embeddings or indexes were used.

## 11. Output artifacts

```text
results/
└── scidocs/
    ├── manifests/
    │   └── <run-id>.json
    ├── chunks/
    │   └── <chunk-manifest-hash>.jsonl
    ├── rankings/
    │   └── <run-id>.jsonl
    ├── metrics/
    │   └── <run-id>.json
    ├── traces/
    │   └── <run-id>.jsonl
    └── comparisons/
        └── phase2-summary.csv
```

Each ranking row must retain query ID, parent document ID, winning chunk ID, rank,
raw score, and retriever. Each trace must retain enough information to reproduce
the parent-document collapse.

## 12. Validation checks

The runner must fail before retrieval if any check fails:

- dataset checksum does not match the registered version
- qrels reference missing queries or documents
- duplicate chunk IDs exist
- any chunk lacks a parent document ID
- chunks cross parent document boundaries
- the same configuration produces different chunk-manifest hashes
- fewer than K unique documents are returned when the corpus has at least K
  documents
- final dense input is silently truncated
- paired runs use different query sets, corpus versions, K values, or title policy
- a reportable run uses a sampled corpus

Add unit tests for fixed chunk overlap, recursive boundaries, deterministic chunk
IDs, title prefixing, parent collapse, tie-breaking, and hybrid fusion after parent
collapse.

## 13. Error analysis

Review at least:

- 20 queries where every retriever misses all relevant documents at K=10
- 20 queries with the largest fixed-versus-recursive Recall@10 difference
- 20 queries with the largest BM25-versus-dense rank difference
- all queries where chunking performs worse than the whole-document BM25 control

Assign one or more causes:

- relevant document text does not match the query vocabulary
- chunk boundary separates useful context
- title dominates or misleads retrieval
- dense semantic mismatch
- query requires citation-graph information absent from indexed text
- several relevant chunks map to one parent document
- document-level qrel cannot identify the supporting passage
- preprocessing or metadata defect

Save reviewed labels beside per-query traces rather than editing raw dataset files.

## 14. Selection rule and exit gate

Do not choose a winner from a single aggregate score.

1. Eliminate configurations that fail validation.
2. Rank remaining configurations by Recall@10.
3. Use NDCG@10, then MRR@10, to resolve practical ties.
4. Prefer the simpler or faster configuration when the paired confidence interval
   shows no reliable quality difference.
5. Publish the quality and resource Pareto frontier instead of hiding tradeoffs in
   one weighted score.

Phase 2 is complete when:

- all six primary runs finish on the full SciDocs corpus and test queries
- results include per-query rankings and traces
- confidence intervals and fixed-versus-recursive paired comparisons are reported
- p50, p95, and p99 warm retrieval latency are reported
- build time, peak memory, and index sizes are reported
- at least the required error-analysis cases are reviewed
- the selected configuration and rejected alternatives have documented reasons

Absolute latency, memory, and storage limits remain product and hardware dependent.
Before this benchmark becomes a release gate, replace that sentence with numeric
limits for the intended deployment environment.

## 15. Implementation action items

- [ ] Add `langchain-text-splitters`, `transformers`, and their pinned versions to
  an optional chunking dependency group.
- [ ] Download and checksum the BEIR SciDocs dataset.
- [ ] Implement a shared chunk record and deterministic chunk IDs.
- [ ] Implement the fixed token splitter.
- [ ] Implement the recursive splitter.
- [ ] Add chunk-distribution validation and reporting.
- [ ] Add chunk-to-parent result collapse with deterministic tie-breaking.
- [ ] Apply hybrid RRF after parent collapse.
- [ ] Persist chunk manifests, rankings, winning chunks, and per-query traces.
- [ ] Add p50, p95, and p99 latency plus build-time instrumentation.
- [ ] Add peak-memory and index-size measurement.
- [ ] Add paired bootstrap confidence intervals.
- [ ] Add the validation and unit tests in Section 12.
- [ ] Run the diagnostic whole-document BM25 control.
- [ ] Run the six pre-registered primary configurations.
- [ ] Run 128-token sensitivity analysis only if the 90 percent single-chunk rule
  is triggered.
- [ ] Complete error analysis and write the Phase 2 comparison report.
