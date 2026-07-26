# Local RAG Retrieval Eval

A small evaluation harness inspired by [BEIR](https://github.com/beir-cellar/beir).
It downloads a standard BEIR dataset, runs retrieval, compares ranked documents
with qrels, and records reproducible experiment results.

The initial scope is retrieval evaluation. It deliberately separates retrieval
quality from LLM answer quality so you can diagnose the first stage before adding
generation and judge-based metrics.

## What it measures

- Precision@K
- Recall@K
- MRR@K
- nDCG@K with graded relevance
- Total retrieval latency and latency per query

Supported retrieval methods:

- `bm25`: dependency-free lexical baseline
- `dense`: exact cosine search with a sentence-transformers model
- `hybrid`: BM25 plus dense search using reciprocal rank fusion

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Small, fast BM25 learning run
rag-eval \
  --dataset scifact \
  --retriever bm25 \
  --max-documents 2000 \
  --max-queries 100 \
  --k 1,3,5,10
```

The first run downloads the public BEIR archive into `data/`. Reports are written
to `results/<experiment-id>.json`, and each run is appended to
`results/experiments.csv`.

To compare dense and hybrid retrieval:

```bash
pip install -e ".[dense]"

rag-eval --dataset scifact --retriever dense --max-documents 2000 --max-queries 100
rag-eval --dataset scifact --retriever hybrid --max-documents 2000 --max-queries 100
```

The default dense model is `sentence-transformers/all-MiniLM-L6-v2`. Override it
with `--model`.

## Evaluation design

BEIR datasets have three important pieces:

```text
corpus.jsonl        documents to search
queries.jsonl       evaluation queries
qrels/<split>.tsv   graded query-document relevance judgments
```

When limits are supplied, the sampler selects queries first, retains every
positively judged document for those queries, and fills the remaining document
budget deterministically. This is important. Independently sampling documents
would remove ground-truth positives and make retrieval scores misleading.

Each run records:

- Dataset, split, retriever, model, cutoffs, sample limits, and random seed
- Document, query, and positive-judgment counts
- All retrieval metrics
- Timing and Python version

## Commands

```text
rag-eval [-h]
  [--dataset DATASET]
  [--split SPLIT]
  [--data-dir DATA_DIR]
  [--output-dir OUTPUT_DIR]
  [--retriever {bm25,dense,hybrid}]
  [--model MODEL]
  [--batch-size BATCH_SIZE]
  [--k 1,3,5,10]
  [--max-documents N]
  [--max-queries N]
  [--seed SEED]
```

Use `--max-documents` only as a learning shortcut. Scores from a sampled corpus
are not directly comparable with published full-corpus BEIR results.

## Test

```bash
pip install -e ".[dev]"
pytest
```

## Benchmark project plan

The benchmark should answer three different questions independently:

1. **Retrieval:** Did the system find and rank the necessary evidence?
2. **Generation:** Given sufficient evidence, did the model produce a correct,
   complete, and grounded answer?
3. **End to end:** Does the full pipeline produce a useful answer within production
   latency and cost constraints?

Keep these tracks separate during diagnosis. A poor final answer can come from
missing evidence, poor context construction, or generation failure. A single
end-to-end score cannot identify which component failed.

### Phase 0. Define the benchmark contract

- [ ] Write down the target use cases, users, corpus types, languages, and expected
  question distribution.
- [ ] Define what counts as a correct answer, a valid citation, and an acceptable
  abstention for the product.
- [ ] Choose quality, latency, cost, and storage constraints before comparing
  systems. Include target values and hard limits.
- [ ] Select primary metrics. Use `Recall@K` as the retrieval gate, `NDCG@K` or
  `MRR@K` for ranking, and grounded answer correctness as the primary end-to-end
  outcome.
- [ ] Define a fixed evaluation protocol, including dataset version, split, random
  seed, K values, hardware, dependency versions, and the number of repeated runs.
- [ ] Create a run manifest schema so every result can be reproduced from its
  dataset, configuration, code commit, prompt version, and model version.

**Deliverable:** `benchmark_spec.md` with target workloads, metrics, constraints,
and promotion thresholds.

### Phase 1. Build a representative evaluation dataset

- [ ] Inventory the production corpus and remove secrets, personal information,
  duplicates, and documents that must not enter the benchmark.
- [ ] Create representative queries across common, difficult, ambiguous,
  unanswerable, multi-hop, and freshness-sensitive cases.
- [ ] Split queries by stable IDs into development and held-out test sets. Do not
  tune on the test set.
- [ ] Label every query with its supporting passage IDs and graded relevance
  judgments. Record all valid supporting passages when more than one answer path
  exists.
- [ ] Add a reference answer, answerability label, required facts, and expected
  citations for generation evaluation.
- [ ] Have a second annotator review a sample. Resolve disagreements and document
  the labeling rubric.
- [ ] Version the corpus, queries, qrels, reference answers, and rubric together.
- [ ] Add validation that rejects missing IDs, orphaned qrels, train/test overlap,
  empty references, and answerable queries with no positive passage.

Suggested record:

```json
{
  "query_id": "q-001",
  "question": "What evidence supports the claim?",
  "answerable": true,
  "relevant_passages": {"doc-17#p3": 2, "doc-42#p1": 1},
  "reference_answer": "The claim is supported by ...",
  "required_facts": ["fact A", "fact B"],
  "expected_citations": ["doc-17#p3"],
  "slice": ["multi-hop", "difficult"]
}
```

**Deliverable:** a versioned, validated benchmark dataset with a frozen held-out
test split.

### Phase 2. Establish retrieval baselines

- [x] Implement BM25, dense, and hybrid retrieval.
- [x] Implement Precision@K, Recall@K, MRR@K, and NDCG@K.
- [x] Record total retrieval time, time per query, configuration, sample size, and
  Python version.
- [ ] Run BM25 on a fixed development sample and save it as the lexical baseline.
- [ ] Run dense and hybrid retrieval on the identical queries, corpus, K values,
  and seed.
- [ ] Add warm-up runs and report p50, p95, and p99 latency separately from index
  construction time.
- [ ] Record embedding cost, query cost, peak memory, and on-disk index size.
- [ ] Save ranked passage IDs and scores for every query, not only aggregate
  metrics.
- [ ] Report bootstrap confidence intervals and per-slice results in addition to
  macro averages.
- [ ] Inspect false negatives from the worst queries and classify causes such as
  vocabulary mismatch, bad chunk boundaries, metadata filtering, multi-hop
  evidence, or stale content.
- [ ] Repeat the selected configuration on the full corpus before reporting final
  retrieval results.

**Exit gate:** the selected retriever meets the agreed `Recall@K`, tail-latency,
cost, memory, and index-size thresholds on the held-out set.

### Phase 3. Evaluate generation with fixed context

- [ ] Create a generation-only runner that bypasses retrieval and supplies the
  same gold passages to every model.
- [ ] Freeze the prompt template, context order, token budget, decoding settings,
  model version, and output schema.
- [ ] Require a structured response containing the answer, citations, and an
  abstention indicator.
- [ ] Measure answer correctness against required facts and reference answers.
- [ ] Measure faithfulness by checking that each material claim is supported by
  the supplied context.
- [ ] Measure completeness as the proportion of required facts covered.
- [ ] Measure citation precision, citation recall, and whether each citation
  supports the associated claim.
- [ ] Test appropriate abstention using both answerable and unanswerable
  questions. Report false-answer and false-abstention rates separately.
- [ ] Combine deterministic checks with a rubric-based evaluator. Blind the
  evaluator to system names and randomize candidate order.
- [ ] Manually audit a stratified sample and measure agreement between human and
  automated judgments.
- [ ] Record input tokens, output tokens, cost, p50/p95/p99 latency, errors, and
  retries.

**Exit gate:** with gold context, the generator meets correctness, faithfulness,
completeness, citation, abstention, latency, and cost thresholds. Failures here are
generation failures, not retrieval failures.

### Phase 4. Run the end-to-end benchmark

- [ ] Connect ingestion, chunking, indexing, retrieval, optional reranking,
  context construction, generation, citations, and abstention in one runner.
- [ ] Freeze the corpus snapshot and execute the held-out test set without tuning.
- [ ] Measure final-answer correctness, faithfulness, completeness, citation
  quality, answer relevance, and abstention behavior.
- [ ] Record stage-level and total p50/p95/p99 latency, token usage, monetary cost,
  failure rate, and throughput.
- [ ] Store a trace for every query containing retrieved passages, retrieval
  scores, reranked order, constructed context, model output, citations, metrics,
  and timing.
- [ ] Produce per-slice results so regressions are not hidden by an overall
  average.
- [ ] Classify each failure as dataset, ingestion, retrieval, reranking, context,
  generation, citation, or infrastructure related.
- [ ] Compare end-to-end results with the fixed-context results to estimate the
  quality lost before generation.

**Exit gate:** the complete pipeline meets product quality and production
constraints on the frozen test set.

### Phase 5. Run controlled experiments

- [ ] Create a baseline configuration and change exactly one independent variable
  per experiment.
- [ ] When comparing embedding models, hold the corpus, chunking, indexing,
  similarity function, candidate count, reranker, context builder, generator,
  prompt, and seed constant.
- [ ] Apply the same rule to chunking, top K, filters, hybrid weights, rerankers,
  context ordering, prompts, and generation models.
- [ ] Run paired comparisons on the same queries and include confidence intervals.
- [ ] Reject a candidate that improves an average quality score but violates a
  hard latency, cost, memory, or reliability limit.
- [ ] Promote the best individual components into candidate pipelines, then
  compare those complete pipelines under the same production constraints.
- [ ] Confirm the winner once on the held-out set. Do not repeatedly tune against
  held-out results.

Use a decision table for every experiment:

| Run | Changed variable | Quality delta | p95 latency delta | Cost delta | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| baseline | none | 0 | 0 | 0 | keep |
| candidate | embedding model | TBD | TBD | TBD | promote or reject |

### Phase 6. Make the benchmark a regression suite

- [ ] Add a small deterministic smoke set to continuous integration.
- [ ] Run the full benchmark on scheduled releases or material corpus, model, or
  pipeline changes.
- [ ] Fail the quality gate when a primary metric crosses its allowed regression
  budget or a hard constraint is violated.
- [ ] Keep raw per-query outputs and aggregate reports for comparison over time.
- [ ] Track dataset and judge drift. Periodically refresh production-like queries
  without changing historical benchmark versions.
- [ ] Add online monitoring for retrieval misses, unsupported answers, citation
  failures, abstention rate, latency, cost, and user feedback.
- [ ] Feed reviewed production failures into the next dataset version, not into the
  frozen historical test set.

**Deliverable:** a repeatable benchmark command, a comparison report, and a release
gate that detects quality, latency, cost, and reliability regressions.

## Recommended execution order

1. Finish the benchmark contract and dataset schema.
2. Validate the current harness with a fixed BM25 development run.
3. Complete the retrieval baselines and retrieval error analysis.
4. Build the fixed-context generation evaluator.
5. Add the end-to-end runner and query-level traces.
6. Run controlled component experiments.
7. Select candidate pipelines and test once on the held-out split.
8. Automate the winning benchmark as a regression suite.

Generation scores should not be mixed into the current retrieval baseline until
the fixed-context evaluator is available. This preserves failure attribution and
makes each experiment easier to interpret.
