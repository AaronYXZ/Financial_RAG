# RAG Benchmark Specification

## 1. Purpose

This specification covers two diagnostic layers of the RAG benchmark. Sections
1 through 15 define Phase 2 retrieval evaluation. Sections 16 onward define Phase
3 generation evaluation with fixed context.

Phase 2 evaluates how chunking and retrieval choices affect evidence discovery
before generation is introduced. It compares two LangChain chunking strategies
across lexical, dense, and hybrid retrieval on SciDocs.

The benchmark answers:

1. Does the system retrieve at least one relevant document within the top K?
2. Does it rank relevant documents near the top?
3. What quality, latency, memory, and index-size tradeoffs does each configuration
   create?
4. Does recursive chunking improve retrieval over fixed token windows when every
   other component is held constant?

Phase 2 does not evaluate answer generation, faithfulness, or citations.

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

## 3. Phase 2 non-goals

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

Construct one source string, then split the complete source within the token limit:

```text
{title}\n\n{text}
```

This source construction must be identical for both chunkers and all retrievers.

## 5. LangChain chunking strategies

Use the standalone `langchain-text-splitters` package. Pin the exact package and
tokenizer versions in the run manifest. LangChain documents its current splitter
APIs in the [text splitter guide](https://docs.langchain.com/oss/python/integrations/splitters)
and recommends recursive splitting as a general-purpose starting point.

### 5.1 Shared chunking parameters

| Parameter | Value |
| --- | ---: |
| Model input ceiling | 256 tokens |
| Effective splitter budget | 248 tokens |
| Chunk overlap | 32 tokens |
| Tokenizer | Tokenizer for `sentence-transformers/all-MiniLM-L6-v2` |
| Keep empty chunks | No |
| Add start index | Yes |
| Source policy | Split the combined title and body source |
| Chunk order | Original document order |

Reserve eight tokens for special tokens and decode-then-encode variation observed
in sentence-transformer token splitting. Validate every final chunk against the
256-token dense model ceiling. The benchmark is invalid if any indexed chunk is
silently truncated.

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
    tokens_per_chunk=248,
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
    chunk_size=248,
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
| Text | combined-source chunk |
| Tokenization | Unicode-aware case-folded word tokens |

Do not tune BM25 parameters during the primary matrix.

### 6.2 Dense baseline

Use:

| Parameter | Value |
| --- | --- |
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Document input | combined-source chunk |
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
- paired runs use different query sets, corpus versions, K values, or source policy
- a reportable run uses a sampled corpus

Add unit tests for fixed chunk overlap, recursive boundaries, deterministic chunk
IDs, combined-source construction, parent collapse, tie-breaking, and hybrid fusion after parent
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

## 16. Phase 3 purpose and diagnostic boundary

Phase 3 evaluates generation independently from retrieval. The runner must bypass
chunk search, BM25, dense retrieval, hybrid fusion, and reranking. Every generator
receives the same versioned context for a given case and track.

The benchmark answers:

1. Can the generator answer correctly when annotated evidence is supplied?
2. Does it limit material claims to information supported by the context?
3. Are citations valid and do they identify the annotated evidence?
4. Does it abstain when a complete paper does not answer the question?
5. What latency, token, cost, and reliability tradeoffs does each generator create?

A failure in this phase is attributed to generation, prompt handling, citation
construction, output parsing, or evaluation. It is not a retrieval failure because
retrieval is never executed.

## 17. QASPER data contract

### 17.1 Source and version

Use the Hugging Face dataset `allenai/qasper`, configuration `qasper`, as the
Phase 3 corpus and QA source. QASPER version 0.3.0 contains 1,585 NLP papers and
5,049 information-seeking questions. Its license is CC BY 4.0.

Pin and record all of the following in each run manifest:

- dataset name and configuration
- dataset revision or immutable snapshot identifier
- QASPER builder version
- split
- normalized case-file SHA-256
- loader source revision
- schema version

The Hugging Face paper-level split sizes are 888 train, 281 validation, and 416
test rows. Do not repartition papers or allow the same paper to cross splits.

### 17.2 Split policy

- `train`: prompt development, evaluator development, and debugging only
- `validation`: generator selection, threshold calibration, and reported
  development comparisons
- `test`: one held-out final comparison after the protocol is frozen

Do not inspect test outputs while changing prompts, rubrics, context policy,
decoding settings, or evaluator thresholds.

### 17.3 Case normalization

Flatten each paper into one generation case per QASPER question. Preserve all
human answer annotations rather than collapsing them to a single reference.

Each normalized case must contain:

| Field | Requirement |
|---|---|
| `case_id` | QASPER `question_id` |
| `split` | original QASPER split |
| `paper_id` | QASPER paper `id` |
| `title` | paper title |
| `question` | question text |
| `answerability` | `answerable`, `unanswerable`, or `ambiguous` |
| `paper_passages` | ordered title, abstract, section, paragraph, and float records |
| `oracle_passage_ids` | ordered union of resolved evidence IDs from answerable annotations |
| `references` | every answer annotation and its evidence |

Create stable passage IDs from the paper ID, passage kind, and document order.
Normalize whitespace for matching only. Preserve normalized source text in the
case file and never use generated paraphrases as evidence.

QASPER can expose nested sequence fields as either lists of records or records of
lists. The loader must accept both representations and produce identical cases.

### 17.4 Reference-answer normalization

Map each annotation to exactly one answer type using this precedence:

1. `unanswerable` when the annotation marks the question unanswerable
2. `extractive` when one or more extractive spans exist
3. `free_form` when a non-empty free-form answer exists
4. `yes_no` when the yes/no value is not null
5. `missing` otherwise, which is a validation error for reportable cases

For extractive answers, join spans in annotation order for text-based scoring.
Retain the original span list in raw source snapshots when available. Retain
each annotation's paragraph evidence and highlighted evidence separately.

### 17.5 Answerability and disagreement

Assign case answerability from all annotations:

- `answerable` when every annotation is answerable
- `unanswerable` when every annotation is unanswerable
- `ambiguous` when annotations disagree

Use only unanimous cases for the primary abstention metrics. Report ambiguous
cases as a separate disagreement slice. Never force them into either binary class.

### 17.6 Evidence resolution

Resolve annotation evidence against normalized title, abstract, section header,
paragraph, and figure/table caption records using exact whitespace-normalized
matching. Preserve document order and deduplicate passage IDs.

Do not silently discard unmatched evidence. Store unresolved strings on the
reference annotation, report their count in the preparation manifest, and fail a
reportable oracle run if any answerable case has no resolved evidence. Any fuzzy
matching policy must be separately versioned, tested, and audited before use.

## 18. Fixed-context tracks

### 18.1 Track A: oracle evidence

Purpose: isolate answer correctness, groundedness, and citation behavior when the
annotated evidence is present.

- Include `answerable` cases only in the primary Track A score.
- Supply the ordered union of evidence passages from every answerable annotation.
- Use the same passage order and IDs for every generator.
- Do not retrieve, rerank, summarize, paraphrase, or model-select passages.
- Report ambiguous cases separately if they are run.

Track A is not an abstention benchmark. An unanswerable QASPER annotation has no
gold evidence, so supplying an empty context would make abstention artificially
easy.

### 18.2 Track B: complete paper

Purpose: evaluate answering and abstention under QASPER's intended full-paper
setting without retrieval.

- Include unanimous answerable and unanimous unanswerable cases.
- Supply the complete normalized paper in document order.
- Include title, abstract, section headers, paragraphs, and available figure/table
  captions.
- Do not silently truncate, summarize, or omit passages.
- Exclude a paper when its complete serialized prompt exceeds the smallest context
  window among compared generators.
- Freeze the eligible paper list before running any generator and report excluded
  paper and case counts.

The common eligible subset is the primary cross-model Track B comparison. A
secondary per-model coverage result may use each model's full context window, but
it must not be used to rank models because the evaluated cases differ.

### 18.3 Context serialization

Serialize each passage exactly as:

```text
[<passage_id>]
<passage_text>
```

Separate passages with two newline characters. Hash the exact system and user
prompt bytes after serialization. Persist the context passage IDs and prompt hash
with every prediction.

## 19. Prompt and response contract

Freeze one system prompt before the reportable validation run:

```text
Answer the question using only the supplied context.
If the context does not support an answer, abstain.
Every factual answer must cite one or more supplied passage IDs.
Return exactly one JSON object with keys answer, abstain, citations, and confidence.
Do not include markdown or additional text.
```

The response schema is:

```json
{
  "answer": "string",
  "abstain": false,
  "citations": ["paper-id::paragraph::0001"],
  "confidence": 0.0
}
```

Validation rules:

- the object has exactly the four declared keys
- `confidence` is numeric and in `[0, 1]`
- citations contain only passage IDs present in the supplied context
- a non-abstaining answer is non-empty and has at least one citation
- an abstention has an empty answer and no citations
- invalid output is recorded as an invalid response, not silently repaired

One deterministic format-repair retry may be reported separately. Primary quality
metrics use the first response.

## 20. Generator controls

Hold constant across a controlled generator comparison:

- QASPER case file and eligible-case list
- fixed-context track
- prompt text and prompt hash
- passage order and serialization
- tokenizer accounting method
- maximum output tokens
- temperature, top-p, seed, and stop conditions where supported
- response parser and retry policy
- evaluator versions and thresholds

Record provider, exact model identifier, model revision when available, context
window, API or runtime version, hardware for local models, and execution time.
If a provider silently updates a model behind an alias, treat results from the new
date as a different configuration.

Recommended deterministic defaults are temperature `0`, top-p `1`, one sample,
and no hidden conversation history. If a model does not support a parameter,
record that fact rather than emulating unsupported behavior.

## 21. Pre-registered experiment matrix

The primary matrix is:

| Factor | Values |
|---|---|
| Data | QASPER validation, then held-out test once |
| Context track | Track A oracle evidence, Track B complete paper |
| Generator | explicit versioned model IDs, selected before the run |
| Prompt | one frozen prompt hash |
| Decoding | deterministic defaults in Section 20 |
| Repetitions | one primary run, three only for nondeterministic systems |

Compare one factor at a time. Prompt variants and decoding sweeps are development
experiments and must not be mixed into the primary model comparison.

## 22. Metrics

### 22.1 Deterministic answer quality

Normalize candidate and reference text using the published QASPER evaluation
normalization. Score against every reference annotation and take the maximum
reference score per case.

Report:

- token F1 as the primary deterministic answer metric
- exact match as a strict secondary metric
- answer-type slices for extractive, free-form, and yes/no questions

Do not substitute ROUGE or embedding similarity as the primary correctness metric.

### 22.2 Abstention

On unanimous Track B cases, report:

- answerability accuracy
- precision, recall, and F1 for abstention
- false-answer rate on unanswerable cases
- false-abstention rate on answerable cases
- risk-coverage curve and area under the curve using declared confidence
- expected calibration error with bins frozen on validation

Report class counts with every aggregate. Ambiguous cases are excluded from the
primary binary metrics and reported separately.

### 22.3 Citations and evidence

For non-abstaining answers, compare cited passage IDs with each reference
annotation's evidence set and use the best matching reference for that case.

Report:

- citation precision
- citation recall
- citation F1
- citation validity rate
- proportion of answered cases with at least one citation

Evidence overlap alone does not prove that a citation supports the generated
claim. Add a claim-level support evaluator and human audit as described below.

### 22.4 Faithfulness and completeness

Split each candidate answer into material claims. A blinded rubric evaluator must
label every claim as `supported`, `contradicted`, or `not_in_context`, with cited
passage IDs and a short rationale.

Report:

- supported-claim rate
- contradicted-claim rate
- unsupported-claim rate
- case-level fully faithful rate
- rubric correctness and completeness on a `0` to `4` scale

QASPER does not provide atomic required-fact annotations. Do not describe rubric
completeness as exact required-fact recall unless a separately versioned human
annotation layer is created.

### 22.5 Efficiency and reliability

Measure only reportable attempts and disclose retry accounting. Report:

- input, output, and total tokens
- time to first token when available
- end-to-end generation latency p50, p95, and p99
- estimated cost per case and per 1,000 cases
- invalid-response, timeout, provider-error, and retry rates
- peak memory for local inference when measurable

## 23. Evaluator protocol

Use deterministic metrics first. A rubric evaluator supplements them for semantic
correctness, claim support, and completeness.

The evaluator input contains only the question, supplied context, references,
candidate answer, and anonymous citations. It must not contain generator names,
prices, latency, or prior scores. Freeze the evaluator model ID, prompt hash,
temperature, output schema, and parsing policy.

Randomize candidate order when pairwise judging is used and run a position-swap
subset. Report evaluator failures separately. A generator must not judge its own
outputs in the only semantic evaluation path.

## 24. Human audit

Before accepting automated rubric results, manually review at least 100 validation
cases, stratified by track, answerability, answer type, generator, and automated
score band. Oversample disagreements, unsupported-claim flags, invalid citations,
false answers, and false abstentions.

Use two independent reviewers for at least 20 percent of the audit. Report raw
agreement and Cohen's kappa for categorical labels. Reconcile rubric or evaluator
thresholds before the held-out test run, not after seeing test results.

## 25. Statistical reporting

Report macro means over cases and paired bootstrap 95 percent confidence intervals
with at least 10,000 paper-clustered resamples. Cluster by `paper_id` because
questions from the same paper are not independent.

For pairwise model comparisons, bootstrap the paired per-case difference and
report the interval and win probability. Treat overlapping intervals as
inconclusive. Do not select a model from a negligible quality difference without
considering latency, cost, reliability, and context coverage.

## 26. Artifacts and reproducibility

Store Phase 3 artifacts under `results/generation/qasper-v1/`:

```text
results/generation/qasper-v1/
  manifests/
    data.<split>.json
    experiment.<run_id>.json
  cases/
    <split>.cases.jsonl
    <split>.eligible_case_ids.json
  prompts/
    system.txt
    prompt_manifest.json
  predictions/
    <run_id>.jsonl
  judgments/
    <run_id>.jsonl
  metrics/
    <run_id>.json
  audits/
    human_audit.csv
  reports/
    phase3_comparison.md
```

Each prediction row must include run ID, case ID, paper ID, split, track, model
identifier, context passage IDs, prompt hash, raw response, parsed response,
validation errors, token counts, latency, retry count, and estimated cost.

Generated data, provider payloads, and model outputs may contain licensed or
sensitive material. Keep large artifacts out of Git and publish only what the
QASPER license and provider terms permit.

## 27. Required validations

### 27.1 Data tests

- both Hugging Face nested sequence representations normalize identically
- case IDs are unique within a split
- every case maps to exactly one paper and original split
- passage IDs are unique and document ordered
- every cited oracle passage exists in the case paper
- unresolved evidence is counted and never silently discarded
- mixed answerability annotations are labeled `ambiguous`
- yes/no `false` is normalized to `No`, not treated as missing

### 27.2 Context and prompt tests

- Track A includes only annotated evidence IDs
- Track B serializes every normalized paper passage exactly once
- complete-paper eligibility is computed before model execution
- prompt serialization and hash are deterministic
- different generators receive byte-identical prompts for a case and track
- no retrieval or reranking component is called

### 27.3 Response and metric tests

- malformed JSON and extra keys fail validation
- out-of-context citation IDs fail validation
- abstentions cannot contain answer text or citations
- non-abstaining answers require text and citations
- QASPER answer normalization matches official evaluator fixtures
- citation precision and recall handle multiple reference evidence sets
- paper-clustered bootstrap resampling is deterministic under a fixed seed

## 28. Error analysis

Review at least:

- 20 lowest token-F1 Track A cases
- 20 supported-answer failures despite resolved oracle evidence
- 20 citation precision or recall failures
- every false answer on a unanimous unanswerable Track B case, up to 50
- 20 false abstentions on answerable Track B cases
- 20 evaluator-human disagreements
- all invalid responses when there are fewer than 50, otherwise a stratified 50

Use a fixed taxonomy: reference disagreement, answer-type normalization, evidence
resolution, missed evidence use, unsupported inference, contradiction, incomplete
answer, citation mismatch, inappropriate abstention, invalid schema, context-window
exclusion, evaluator error, and other with notes.

## 29. Selection and exit criteria

Choose the generator only after reviewing quality, faithfulness, citations,
abstention, efficiency, reliability, and eligible context coverage together.

Phase 3 is complete when:

- QASPER data and normalized case manifests are versioned and checksummed
- the two context tracks and eligible-case lists are frozen
- prompts and generator configurations are versioned
- Track A and Track B metrics include paper-clustered confidence intervals
- invalid outputs, retries, errors, latency, tokens, and cost are reported
- automated rubric judgments pass the human-agreement audit
- required error analysis is complete
- one generator configuration and rejected alternatives have documented reasons

Numeric release thresholds must be set from product requirements after the first
validation baseline. Do not invent acceptance thresholds after viewing test data.

## 30. Phase 3 implementation action items

- [x] Add the QASPER optional dependency and `rag-generation` CLI entry point.
- [x] Implement QASPER loading for both nested sequence representations.
- [x] Flatten paper rows into versioned question cases.
- [x] Normalize answer types, answerability disagreement, and evidence IDs.
- [x] Preserve unresolved evidence and report it in the data manifest.
- [x] Implement deterministic prompt rendering, hashing, and strict response parsing.
- [x] Add unit tests for loader, context selection, and response validation.
- [x] Pin the QASPER Parquet export to an immutable repository commit.
- [ ] Record or verify the downloaded source Parquet checksum.
- [x] Download and normalize the complete validation split.
- [ ] Download and normalize the train split for prompt development.
- [x] Implement Track A and Track B deterministic case selection.
- [x] Implement shared context-window eligibility before model execution.
- [x] Freeze eligible case IDs as the denominator for each run configuration.
- [x] Port and test the published QASPER answer normalization and token F1.
- [x] Implement the minimum local HTTP adapter and sequential experiment runner.
- [x] Persist raw and parsed prediction JSONL with resume support.
- [x] Implement failure-rate, retry, latency, token, and local-cost metrics.
- [ ] Implement citation, abstention, and calibration metrics.
- [ ] Implement paper-clustered bootstrap confidence intervals.
- [ ] Add the blinded claim-support and completeness evaluator.
- [ ] Complete the stratified human-audit workflow.
- [ ] Run validation baselines and freeze thresholds.
- [ ] Run the held-out test comparison once.
- [ ] Complete error analysis and publish the Phase 3 comparison report.
