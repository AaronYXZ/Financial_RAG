# Development History

## 2026-07-29. Validation data provenance and protocol freeze

### Implemented

- Verified the cached immutable QASPER validation Parquet against SHA-256
  `089781b91c337d348dd9e8b57cc8adc100ed2d9cab84a6127402bcccf1559222`.
- Registered the source checksum in code and normalized-data manifests.
- Confirmed that the complete normalized validation split contains 1,005 cases
  with SHA-256
  `e0172f79d2b17435b5c8c0aaa1ce9db76de0f6619772979a85f5a8c926f38c93`.
- Added `generation_protocol_v1.json` to freeze the validation data, prompt,
  context, generation, retrieval, metric, and held-out-test policies.
- Added `--all-cases` to the oracle and lower-level fixed-context commands so
  full validation runs do not depend on an implicit numeric sentinel.
- Deferred the train split because prompt contract v3 is frozen and no active
  prompt-development task requires it.

### Threshold boundary

The existing 25-case results are smoke baselines. They are not sufficient for
numeric release thresholds. Quality, latency, and cost gates remain pending a
full validation run and explicit product requirements. Integrity gates and the
rule against inspecting held-out test outputs before the freeze are mandatory.

## 2026-07-27. OpenAI Responses API generation provider

### Implemented

- Added a separate OpenAI Responses API adapter while retaining the local
  OpenAI-compatible MLX adapter.
- Added structured JSON output, explicit reasoning effort, disabled response
  storage, retry accounting, usage extraction, and model-aware token counting.
- Added `--provider openai`, OpenAI model, reasoning, API-key variable, and
  environment-file options to generation commands.
- Set `gpt-5` as the default and constrained the CLI to `gpt-5`,
  `gpt-5.6-sol`, and `gpt-5.6-luna`.
- Added `.env` protection, a safe `.env.example`, optional dependencies, tests,
  and example generation and metric commands.

## 2026-07-27. Task-oriented generation CLI

### Implemented

- Added `generate-oracle` for controlled generation from annotated evidence.
- Added `generate-retrieved` for generation from an existing frozen retrieval
  manifest without rerunning retrieval.
- Added `generate-end-to-end` to retrieve, persist a checksummed context
  manifest, and generate in one command.
- Kept the lower-level `run` and `freeze-context` commands for compatibility and
  experiment debugging.
- Added task-specific defaults, help text, parser tests, and sample commands.

## 2026-07-27. Matched prompt comparison and frozen retrieved context

### Implemented

- Aligned default Track A prediction and metric paths under one run ID.
- Added matched response-validity comparison with a hard check for identical,
  ordered eligible case IDs.
- Added a dependency-free BM25 context freezer over normalized QASPER passages.
- Recorded cases and eligibility checksums, retriever implementation and
  parameters, top K, ordered case IDs, passage rankings, and scores.
- Added the `retrieved-context` generation track, which requires the frozen
  manifest and records its checksum in eligibility and prediction artifacts.
- Extended answerability, abstention, calibration, and bootstrap metrics to the
  retrieved-context track.
- Added CLI commands, validation, and tests for the comparison and retrieval
  workflow.

### Observed local baseline

- Track A prompt v2 completed 25 of 25 cases with valid responses.
- Track B prompt v1 completed 11 of 25 cases with valid responses.
- Track B prompt v2 completed 16 of 25 cases with valid responses on the same
  ordered case IDs.

## 2026-07-27. Paper-clustered bootstrap intervals, Stage 6

### Implemented

- Added deterministic paper-clustered bootstrap resampling with seed `42`.
- Added 10,000-resample, two-sided 95 percent percentile intervals.
- Resampled observed papers with replacement and retained every question from a
  selected paper, preserving within-paper dependence.
- Added intervals for answer token F1, normalized exact match, citation precision,
  recall, F1, validity, and answered-with-citation rate on both tracks.
- Added Track B intervals for answerability accuracy, abstention precision, recall,
  F1, false-answer rate, false-abstention rate, expected calibration error, and
  area under the risk-coverage curve.
- Reported valid replicate counts for conditionally undefined metrics.
- Persisted method, clustering unit, confidence level, resample count, seed, paper
  count, and case count in `summary.json`.
- Added tests proving fixed-seed determinism and whole-paper cluster behavior.

### Scope boundary

Stage 6 covers confidence intervals for one evaluated prediction file. Paired
bootstrap intervals for direct model differences belong to the later comparison
report workflow, which must join identical cases before resampling.

## 2026-07-27. Confidence calibration and risk-coverage, Stage 5

### Implemented

- Added Track B per-case declared confidence and calibration quality fields.
- Restricted primary calibration to unanimous answerable and unanswerable cases.
  Ambiguous cases remain excluded and explicitly counted.
- Defined a continuous quality target. Correct abstentions score `1.0`, false
  answers and false abstentions score `0.0`, and answered answerable cases use
  best-reference QASPER token F1.
- Added confidence evaluable, unavailable, and availability-rate reporting so
  invalid and missing predictions are not silently removed.
- Added expected calibration error with 10 fixed equal-width validation bins.
- Added a confidence-threshold risk-coverage curve and discrete area under the
  curve.
- Grouped equal-confidence cases at the same threshold, preventing row order from
  changing the curve or area.
- Marked confidence calibration as not applicable to oracle-evidence Track A.
- Added synthetic tests for quality-target construction, invalid-output
  availability, ECE, risk-coverage area, and tied confidence values.

### Frozen policy

Stage 5 bins and quality semantics are frozen in `benchmark_spec.md` before
running reportable comparisons. Confidence calibration applies to valid Track B
responses. Confidence availability is reported over every unanimous Track B
case.

## 2026-07-26. Stronger generation prompt contract v2

### Motivation

The first 25-case complete-paper run produced 14 invalid responses. Ten of those
responses used a word-valued confidence such as `"high"` or omitted another
required field while also using word-valued confidence. The original prompt named
the four keys but did not communicate the validator's complete type constraints.

### Implemented

- Added `PROMPT_VERSION = "qasper-generation-v2"`.
- Specified the exact four-key JSON structure and every required field type.
- Required numeric confidence from `0.0` to `1.0`, never a qualitative string.
- Required a JSON boolean for abstention and empty answer/citations when
  abstaining.
- Required exact copying of bracketed context passage IDs and prohibited
  constructed citation IDs.
- Limited answers to 120 words and citation lists to 5 IDs to reduce truncation
  and over-citation.
- Repeated the compact response contract after the question so it remains close
  to the generation position following a long complete-paper context.
- Added the prompt version to eligibility manifests, prediction rows, metric
  summaries, and evaluation records.
- Added prompt version to resume identity and prediction filtering so legacy and
  v2 outputs cannot be silently mixed.
- Added tests for the contract text, word-valued confidence, answer length,
  citation count, artifact versioning, and evaluator filtering.

### Required comparison

Rerun the same 25 complete-paper cases under a separate v2 output file and compare
valid-response rate, invalid-schema rate, invalid-citation rate, answer token F1,
citation F1, latency, and output tokens against prompt v1.

## 2026-07-26. Basic abstention metrics, Stage 4

### Implemented

- Added Track B answerability accuracy, abstention precision, recall, and F1.
- Added false-answer rates for unanimous unanswerable cases and false-abstention
  rates for unanimous answerable cases.
- Added class counts, valid-decision and no-decision counts, and an explicit
  confusion matrix.
- Counted invalid and missing predictions as `no_decision`. These outcomes reduce
  answerability accuracy and abstention recall without being mislabeled as a
  generated false answer or false abstention.
- Excluded ambiguous QASPER cases from primary binary metrics while reporting
  their answer, abstain, and no-decision outcomes separately.
- Marked abstention evaluation as not applicable for the oracle-evidence track,
  which contains only answerable cases.
- Added synthetic Track B tests covering every primary confusion outcome,
  ambiguous cases, and invalid responses.

### Remaining Stage 4 execution

- Run the complete-paper Track B benchmark to produce reportable abstention
  results. The existing oracle-evidence smoke run cannot measure abstention.

### Next metrics

- Freeze the correctness definition and confidence bins on validation.
- Implement risk-coverage area and expected calibration error.

## 2026-07-26. Citation and evidence metrics, Stage 3

### Implemented

- Added citation precision, recall, and F1 for valid non-abstaining responses.
- Compared predicted passage IDs independently with every human annotation's
  non-empty `evidence_ids` set and retained the best citation-F1 match per case.
- Added per-reference citation scores and the winning citation annotation ID to
  `per_case_metrics.jsonl`.
- Added aggregate citation quality, citation validity, answered-with-citation,
  invalid-citation, abstention, and missing-reference-evidence counts to
  `summary.json`.
- Excluded abstentions and cases without resolved reference evidence from
  citation precision, recall, and F1 averages while reporting their counts.
- Added tests for multiple evidence sets, partial overlap, no overlap, empty
  evidence, abstentions, and invalid citations.

### Smoke-test result

For the current 25-case oracle-evidence Qwen3 smoke run:

- 22 valid answered cases were citation-scorable.
- Citation precision was `0.9303`.
- Citation recall was `0.8447`.
- Citation F1 was `0.8601`.
- Citation validity and answered-with-citation rates were `1.0`.
- Three invalid-schema responses remained outside the citation-quality average
  and visible in reliability reporting.

### Next stage

- Run the complete-paper Track B benchmark.
- Implement answerability accuracy, abstention precision, recall, and F1, plus
  false-answer and false-abstention rates.
- Add risk-coverage and calibration metrics after freezing their correctness and
  confidence-bin definitions on validation.

## 2026-07-26. QASPER generation benchmark, metrics Stage 0 to 2

### Implemented

- Added a local OpenAI-compatible generation adapter for Qwen3 served by
  `mlx_lm.server`, including deterministic decoding controls, retry accounting,
  latency, server token usage, and terminal request errors.
- Added the sequential `rag-generation run` workflow for the QASPER
  `oracle-evidence` and `complete-paper` tracks.
- Added strict response validation for answer text, abstention, citations, and
  confidence, with raw and parsed JSONL persistence and resume support.
- Added deterministic pre-inference eligibility selection using the shared
  context-window limit.
- Added an eligibility sidecar that freezes the exact case IDs used as the metric
  denominator. Resumed runs reject changes to the model, track, limits, or
  eligible case set instead of silently changing the denominator.
- Added `rag-generation metrics` and Stage 0 to 2 evaluation artifacts:
  `evaluation_records.jsonl`, `per_case_metrics.jsonl`, and `summary.json`.
- Added explicit statuses for valid responses, invalid JSON, invalid schema,
  invalid citations, request errors, timeouts, and missing predictions.
- Added reliability and efficiency summaries for status rates, retries, latency
  percentiles, input and output tokens, total tokens, and the local-inference cost
  basis.
- Ported QASPER answer normalization and token F1 semantics. Candidate answers are
  scored independently against every human annotation, and the maximum reference
  score is retained per case.
- Added normalized exact match and answer-type summaries.
- Documented the generation smoke-test execution flow from CLI entry point through
  context construction, Qwen inference, prediction persistence, metric joining,
  and result aggregation.

### QASPER evaluation clarification

- `validation.cases.jsonl` is the ground-truth source. Each row contains the
  question, normalized paper passages, answerability, human reference answers,
  and annotated evidence IDs.
- `paper_passages` form the inference corpus. The oracle-evidence track supplies
  only passages named by `oracle_passage_ids`; the complete-paper track supplies
  all normalized passages from the paper.
- Human reference answers are never sent to Qwen. They are consumed only by the
  evaluator.
- Multiple `annotation_id` values represent independent human annotations. The
  evaluator does not merge their answer text. It scores each annotation
  separately and selects the best score.
- `evidence_texts` contain supporting passages. `extractive_spans` contain the
  shorter answer strings used for extractive answer scoring. `evidence_ids`
  provide stable passage identifiers for context selection and citation metrics.

### Compatibility correction

- Transformers 5.14 returns a `BatchEncoding` from `apply_chat_template()`.
  Calling `len()` on that object counts mapping fields, producing `2`, rather
  than counting prompt token IDs. Token counting now reads `input_ids` from
  mapping-shaped results and handles a single nested token row.
- Existing `qwen3-4b-track-b-v1.jsonl` predictions predate the eligibility sidecar and
  contain the incorrect local token count, although the server token counts are
  intact. After this correction, resume the same generation configuration to
  recompute eligibility and create `qwen3-4b-track-b-v1.eligibility.json` before
  running metrics.

### Remaining work

- Citation precision, recall, and F1.
- Abstention, risk-coverage, and calibration metrics.
- Paper-clustered bootstrap confidence intervals.
- Blinded claim-support and completeness evaluation.
- Human audit and final validation comparison report.
