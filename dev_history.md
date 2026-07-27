# Development History

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
- Existing `qwen3-4b-smoke.jsonl` predictions predate the eligibility sidecar and
  contain the incorrect local token count, although the server token counts are
  intact. After this correction, resume the same generation configuration to
  recompute eligibility and create `qwen3-4b-smoke.eligibility.json` before
  running metrics.

### Remaining work

- Citation precision, recall, and F1.
- Abstention, risk-coverage, and calibration metrics.
- Paper-clustered bootstrap confidence intervals.
- Blinded claim-support and completeness evaluation.
- Human audit and final validation comparison report.
