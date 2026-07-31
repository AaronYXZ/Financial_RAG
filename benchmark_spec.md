# QASPER RAG Benchmark Specification

This contract defines the QASPER retrieval component, fixed-context generation
component, and retrieved-context end-to-end evaluation. Dataset, context,
prompt, model, metric, and judge artifacts must be versioned so comparisons are
reproducible.

## 1. Purpose and diagnostic boundary

Fixed-context generation evaluates generation independently from retrieval. The runner must bypass
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

## 2. QASPER data contract

### 2.1 Source and version

Use the Hugging Face dataset `allenai/qasper`, configuration `qasper`, as the
benchmark corpus and QA source. QASPER version 0.3.0 contains 1,585 NLP papers and
5,049 information-seeking questions. Its license is CC BY 4.0.

Pin and record all of the following in each run manifest:

- dataset name and configuration
- dataset revision or immutable snapshot identifier
- QASPER builder version
- source Parquet SHA-256
- split
- normalized case-file SHA-256
- loader source revision
- schema version

The registered validation source is:

- immutable revision: `06806e4608976fc2fac0a090ac425d5b2b29caf4`
- file: `qasper/validation/0000.parquet`
- byte size: `4,749,127`
- SHA-256:
  `089781b91c337d348dd9e8b57cc8adc100ed2d9cab84a6127402bcccf1559222`

The complete normalized validation file contains 1,005 cases and has SHA-256
`e0172f79d2b17435b5c8c0aaa1ce9db76de0f6619772979a85f5a8c926f38c93`.
The machine-readable validation protocol is `generation_protocol_v1.json`.

The Hugging Face paper-level split sizes are 888 train, 281 validation, and 416
test rows. Do not repartition papers or allow the same paper to cross splits.

### 2.2 Split policy

- `train`: prompt development, evaluator development, and debugging only
- `validation`: generator selection, threshold calibration, and reported
  development comparisons
- `test`: one held-out final comparison after the protocol is frozen

Do not inspect test outputs while changing prompts, rubrics, context policy,
decoding settings, or evaluator thresholds.

### 2.3 Case normalization

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

### 2.4 Reference-answer normalization

Map each annotation to exactly one answer type using this precedence:

1. `unanswerable` when the annotation marks the question unanswerable
2. `extractive` when one or more extractive spans exist
3. `free_form` when a non-empty free-form answer exists
4. `yes_no` when the yes/no value is not null
5. `missing` otherwise, which is a validation error for reportable cases

For extractive answers, join spans in annotation order for text-based scoring.
Retain the original span list in raw source snapshots when available. Retain
each annotation's paragraph evidence and highlighted evidence separately.

### 2.5 Answerability and disagreement

Assign case answerability from all annotations:

- `answerable` when every annotation is answerable
- `unanswerable` when every annotation is unanswerable
- `ambiguous` when annotations disagree

Use only unanimous cases for the primary abstention metrics. Report ambiguous
cases as a separate disagreement slice. Never force them into either binary class.

### 2.6 Evidence resolution

Resolve annotation evidence against normalized title, abstract, section header,
paragraph, and figure/table caption records using exact whitespace-normalized
matching. Preserve document order and deduplicate passage IDs.

Do not silently discard unmatched evidence. Store unresolved strings on the
reference annotation, report their count in the preparation manifest, and fail a
reportable oracle run if any answerable case has no resolved evidence. Any fuzzy
matching policy must be separately versioned, tested, and audited before use.

## 3. Fixed-context tracks

### 3.1 Track A: oracle evidence

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

### 3.2 Track B: complete paper

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

### 3.3 Context serialization

Serialize each passage exactly as:

```text
[<passage_id>]
<passage_text>
```

Separate passages with two newline characters. Hash the exact system and user
prompt bytes after serialization. Persist the context passage IDs and prompt hash
with every prediction.

## 4. Prompt and response contract

Freeze prompt contract `qasper-generation-v2` before the reportable validation run:

```text
Answer the question using only the supplied context.
If the context does not support an answer, abstain.
Every factual answer must cite one or more supplied passage IDs.

Return exactly one JSON object with exactly the keys `answer`, `abstain`,
`citations`, and `confidence`. `answer` is a string of at most 120 words;
`abstain` is an unquoted JSON boolean; `citations` is an array of at most 5 IDs
copied exactly from bracketed context IDs; and `confidence` is a numeric value
from 0.0 to 1.0, never a qualitative string. An abstention must use an empty
answer and citation list. Return no markdown, comments, explanations, or extra
keys. Repeat this contract after the question.
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
- the prediction and eligibility manifest declare `qasper-generation-v2`
- the answer contains at most 120 words and citations contain at most 5 IDs
- `confidence` is numeric and in `[0, 1]`
- citations contain only passage IDs present in the supplied context
- a non-abstaining answer is non-empty and has at least one citation
- an abstention has an empty answer and no citations
- invalid output is recorded as an invalid response, not silently repaired

One deterministic format-repair retry may be reported separately. Primary quality
metrics use the first response.

## 5. Generator controls

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

## 6. Pre-registered experiment matrix

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

## 7. Metrics

### 7.1 Deterministic answer quality

Normalize candidate and reference text using the published QASPER evaluation
normalization. Score against every reference annotation and take the maximum
reference score per case.

Report:

- token F1 as the primary deterministic answer metric
- exact match as a strict secondary metric
- answer-type slices for extractive, free-form, and yes/no questions

Do not substitute ROUGE or embedding similarity as the primary correctness metric.

### 7.2 Abstention

On unanimous Track B cases, report:

- answerability accuracy
- precision, recall, and F1 for abstention
- false-answer rate on unanswerable cases
- false-abstention rate on answerable cases
- risk-coverage curve and area under the curve using declared confidence
- expected calibration error with bins frozen on validation

Use this frozen Stage 5 calibration policy:

- restrict primary calibration to unanimous answerable and unanswerable Track B
  cases
- exclude ambiguous cases and report their count separately
- require a valid response with numeric confidence in `[0.0, 1.0]` for confidence
  evaluation
- report confidence availability over all primary cases so invalid and missing
  responses remain visible
- assign quality `1.0` to correct abstentions and `0.0` to false answers
- assign quality `0.0` to false abstentions
- for answered, answerable cases, use the best-reference QASPER token F1 as the
  continuous quality target
- compute expected calibration error with 10 equal-width bins: `[0.0, 0.1)`,
  through `[0.9, 1.0]`

Build the risk-coverage curve by sorting unique declared-confidence thresholds
from high to low. Include all cases tied at a threshold together. At each
threshold, coverage is the selected count divided by all confidence-evaluable
cases, and risk is `1 - mean(calibration_quality_score)`. Area under the curve is
the discrete sum of risk times each increase in coverage.

Grouping ties makes the result invariant to prediction row order.

Report class counts with every aggregate. Ambiguous cases are excluded from the
primary binary metrics and reported separately.

### 7.3 Citations and evidence

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

### 7.4 Faithfulness and completeness

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

### 7.5 Efficiency and reliability

Measure only reportable attempts and disclose retry accounting. Report:

- input, output, and total tokens
- time to first token when available
- end-to-end generation latency p50, p95, and p99
- estimated cost per case and per 1,000 cases
- invalid-response, timeout, provider-error, and retry rates
- peak memory for local inference when measurable

## 8. Evaluator protocol

Use deterministic metrics first. A rubric evaluator supplements them for semantic
correctness, claim support, and completeness.

The evaluator input contains only the question, supplied context, references,
candidate answer, and anonymous citations. It must not contain generator names,
prices, latency, or prior scores. Freeze the evaluator model ID, prompt hash,
temperature, output schema, and parsing policy.

Randomize candidate order when pairwise judging is used and run a position-swap
subset. Report evaluator failures separately. A generator must not judge its own
outputs in the only semantic evaluation path.

## 9. Human audit

Before accepting automated rubric results, manually review at least 100 validation
cases, stratified by track, answerability, answer type, generator, and automated
score band. Oversample disagreements, unsupported-claim flags, invalid citations,
false answers, and false abstentions.

Use two independent reviewers for at least 20 percent of the audit. Report raw
agreement and Cohen's kappa for categorical labels. Reconcile rubric or evaluator
thresholds before the held-out test run, not after seeing test results.

## 10. Statistical reporting

Report macro means over cases and paired bootstrap 95 percent confidence intervals
with at least 10,000 paper-clustered resamples. Cluster by `paper_id` because
questions from the same paper are not independent.
Use this frozen Stage 6 single-run interval policy:

- resample `paper_id` clusters with replacement, drawing the observed number of
  papers in every replicate
- include every question belonging to each selected paper, including repeated
  copies when a paper is drawn more than once
- use 10,000 replicates and random seed `42`
- calculate two-sided 95 percent percentile intervals
- retain case-macro aggregation inside each replicate
- report the number of valid replicates for metrics that can be undefined in a
  resample, such as citation or abstention precision
- attach intervals to answer quality and citation metrics on both tracks
- additionally attach intervals to abstention and confidence metrics on Track B

Paired model-difference intervals remain part of the comparison-report workflow.


For pairwise model comparisons, bootstrap the paired per-case difference and
report the interval and win probability. Treat overlapping intervals as
inconclusive. Do not select a model from a negligible quality difference without
considering latency, cost, reliability, and context coverage.

## 11. Artifacts and reproducibility

Store benchmark artifacts under `results/generation/qasper-v1/`:

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

## 12. Required validations

### 12.1 Data tests

- both Hugging Face nested sequence representations normalize identically
- case IDs are unique within a split
- every case maps to exactly one paper and original split
- passage IDs are unique and document ordered
- every cited oracle passage exists in the case paper
- unresolved evidence is counted and never silently discarded
- mixed answerability annotations are labeled `ambiguous`
- yes/no `false` is normalized to `No`, not treated as missing

### 12.2 Context and prompt tests

- Track A includes only annotated evidence IDs
- Track B serializes every normalized paper passage exactly once
- complete-paper eligibility is computed before model execution
- retrieved-context uses a checksummed frozen context manifest
- frozen retrieval records the retriever implementation, parameters, top K,
  source case checksum, and source eligibility checksum
- retrieved-context preserves the ordered eligible case IDs from its source track
- prompt serialization and hash are deterministic
- different generators receive byte-identical prompts for a case and track
- no retrieval or reranking component is called during generation

### 12.3 Response and metric tests

- malformed JSON and extra keys fail validation
- out-of-context citation IDs fail validation
- abstentions cannot contain answer text or citations
- non-abstaining answers require text and citations
- QASPER answer normalization matches official evaluator fixtures
- citation precision and recall handle multiple reference evidence sets
- paper-clustered bootstrap resampling is deterministic under a fixed seed

## 13. Error analysis

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

## 14. Selection and exit criteria

Choose the generator only after reviewing quality, faithfulness, citations,
abstention, efficiency, reliability, and eligible context coverage together.

Before generator comparison, rank frozen retrieval candidates against oracle
evidence on identical validation cases. Select one configuration using complete
evidence set, best-reference recall, hit rate, NDCG, MRR, then precision. Do not
use generation metrics to choose retrieval. Run both generators on the selected
context manifest.

For paid API generation, run a pilot first. Record observed input and output
tokens, project expected and conservative full-run cost using a dated official
price table, and require explicit dollar-budget approval before the full
validation or held-out request set.

The benchmark is complete when:

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

## 15. Implementation action items

- [x] Add the QASPER optional dependency and `rag-generation` CLI entry point.
- [x] Implement QASPER loading for both nested sequence representations.
- [x] Flatten paper rows into versioned question cases.
- [x] Normalize answer types, answerability disagreement, and evidence IDs.
- [x] Preserve unresolved evidence and report it in the data manifest.
- [x] Implement deterministic prompt rendering, hashing, and strict response parsing.
- [x] Add unit tests for loader, context selection, and response validation.
- [x] Pin the QASPER Parquet export to an immutable repository commit.
- [x] Record and verify the downloaded validation source Parquet checksum.
- [x] Download and normalize the complete validation split.
- [x] Decide whether the train split is needed. It is deferred because prompt
  contract v3 is frozen and no prompt-development work currently requires it.
- [x] Implement Track A and Track B deterministic case selection.
- [x] Implement shared context-window eligibility before model execution.
- [x] Freeze eligible case IDs as the denominator for each run configuration.
- [x] Port and test the published QASPER answer normalization and token F1.
- [x] Implement the minimum local HTTP adapter and sequential experiment runner.
- [x] Persist raw and parsed prediction JSONL with resume support.
- [x] Implement failure-rate, retry, latency, token, and local-cost metrics.
- [x] Implement citation precision, recall, F1, validity, and coverage metrics.
- [x] Implement basic Track B abstention and answerability metrics.
- [x] Implement risk-coverage and calibration metrics.
- [x] Implement paper-clustered bootstrap confidence intervals.
- [x] Implement paired paper-clustered bootstrap differences for matched
  generator comparisons.
- [x] Implement deterministic evidence availability and primary
  retrieval-versus-generation failure attribution.
- [x] Implement generation-free retrieval comparison against oracle evidence.
- [x] Freeze hybrid MiniLM plus BM25 RRF at paper scope and top 5 as the selected
  validation retrieval configuration.
- [x] Implement pilot-derived OpenAI cost estimates and an optional hard budget
  gate.
- [ ] Run Qwen3-4B and GPT-5 on the selected hybrid retrieval manifest. GPT-5
  requires explicit cost-budget approval before the full run.
- [ ] Add the blinded claim-support and completeness evaluator.
- [ ] Complete the stratified human-audit workflow.
- [ ] Run full validation baselines and freeze product-derived quality,
  latency, and cost thresholds. The existing 25-case runs are smoke baselines
  and must not be used to invent release thresholds.
- [ ] Run the held-out test comparison once.
- [ ] Complete error analysis and publish the benchmark comparison report.
