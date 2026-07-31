# Session Handoff

## Project goal

Build a reproducible RAG benchmark that attributes failures to retrieval,
generation, or the complete pipeline.

The project now uses QASPER across three diagnostic layers:

1. Passage retrieval against annotated evidence.
2. Fixed-context generation with oracle evidence or the complete paper.
3. Retrieved-context generation on a frozen retrieval manifest.

For generation, the current strategic recommendation is:

1. Treat Track A, oracle evidence paragraphs, as the primary controlled generator
   benchmark.
2. Add a frozen retrieved-context track for the closest approximation to
   production RAG generation.
3. Retain Track B, complete paper, as a secondary long-context, answerability,
   abstention, and calibration diagnostic.
4. Use end-to-end retrieval plus generation only after component behavior is
   understood.

Track A answers whether the LLM can produce a correct, grounded, cited answer
when sufficient evidence is supplied. It does not test retrieval noise or missing
evidence, so it should not be the only basis for an end-to-end RAG claim.

## Work completed

### QASPER generation foundation

- Added QASPER loading and normalization.
- Added stable paper passage and evidence IDs.
- Added Track A oracle-evidence context construction.
- Added Track B complete-paper context construction.
- Added deterministic eligibility and eligibility manifests.
- Added a local OpenAI-compatible adapter for Qwen3-4B served through
  `mlx_lm.server`.
- Added strict JSON response validation, prediction persistence, and resume
  support.

Relevant commit:

- `6d12be8 Add QASPER generation benchmark foundation`

### Generation metrics

Implemented stages:

- Stages 0 to 2. Eligibility joins, response status, reliability, efficiency,
  official QASPER answer normalization, token F1, and normalized exact match.
- Stage 3. Citation precision, recall, F1, validity, and citation coverage.
- Stage 4. Track B answerability and abstention metrics.
- Stage 5. Track B confidence calibration, ten-bin ECE, risk-coverage curve, and
  AURC.
- Stage 6. Paper-clustered bootstrap confidence intervals.

Relevant commits:

- `170911e Implement QASPER generation metrics stages 0-2`
- `0b3275e Implement Stage 3 citation metrics`
- `5de29b1 Implement Stage 4 abstention metrics`
- `0be7a00 Implement Stage 5 calibration metrics`

### Prompt contract v2

The first 25-case Track B run completed 11 cases and produced 14 errors. Most
invalid responses used qualitative confidence values such as `"high"` or omitted
required fields.

Prompt contract `qasper-generation-v2` now:

- requires exactly four JSON keys
- requires numeric confidence in `[0.0, 1.0]`
- enforces boolean abstention
- limits answers to 120 words and citations to five IDs
- requires citations copied exactly from supplied passage IDs
- repeats the response contract after long contexts
- versions manifests, predictions, filtering, and resume identity

Relevant commit:

- `cc82b69 Strengthen generation prompt contract v2`

A reportable prompt-v2 Track A or Track B comparison has not yet been run.

## Latest evaluation design decisions, July 29, 2026

### End-to-end RAG evaluation

Do not reduce end-to-end RAG performance to one opaque score. Evaluate the
pipeline in layers so failures can be attributed to retrieval or generation:

1. Retrieval quality:
   - Hit@5
   - best-reference Recall@5
   - complete-evidence-set@5
   - MRR@5
   - Precision@5
   - NDCG@5
2. Evidence availability:
   - whether the retrieved top-K contains enough gold evidence to answer
   - generator correctness conditioned on sufficient evidence,
     `P(correct answer | sufficient evidence retrieved)`
3. Answer quality:
   - normalized exact match
   - token F1
   - claim-level factual precision, recall, and F1
   - answer completeness and relevance
4. Grounding and citations:
   - citation-ID validity
   - citation precision, recall, and F1 against annotated evidence
   - claim-to-citation support
   - citation completeness and redundancy
5. Answerability and calibration:
   - answerability accuracy
   - abstention precision, recall, and F1
   - false-answer and false-abstention rates
   - ECE, AURC, Brier score, and accuracy at coverage
6. Reliability and efficiency:
   - request, format, and validation error rates
   - retries
   - input and output tokens
   - component and total latency percentiles
   - API cost

Use an explicit per-case failure taxonomy:

- retrieval miss
- retrieval noise
- false abstention
- answer failure despite sufficient evidence
- faithfulness failure
- citation failure
- format failure
- request failure
- correct answer
- correct abstention

The first diagnostic split should use deterministic evidence availability and
answer metrics. If required evidence is absent from top-K, attribute the primary
failure to retrieval. If sufficient evidence is present but the answer is
incorrect, attribute the primary failure to generation. Report paired
paper-clustered bootstrap differences when comparing systems on identical
cases.

Keep the custom evaluation harness as the source of truth. `ir-measures` is the
preferred optional package for standard retrieval metrics. Ragas can be added
later as a semantic judge layer. DeepEval is an alternative if CI-oriented test
integration becomes the priority. Do not introduce both Ragas and DeepEval at
the start.

### Where LLM judges enter

The initial retrieval-versus-generation diagnosis does not require an LLM
judge. The following measurements remain deterministic:

- retrieval Hit, Recall, Precision, MRR, and NDCG
- gold-evidence availability in the retrieved context
- normalized exact match and token F1
- citation-ID validity and ID-overlap precision, recall, and F1
- abstention, calibration, latency, cost, retry, and error metrics

LLM judges are introduced only for semantic questions that cannot be answered
reliably through IDs or string overlap:

1. Extracting atomic claims from a generated answer.
2. Deciding whether each claim is entailed, contradicted, or unsupported by the
   supplied context.
3. Deciding whether the specific passage cited for a claim actually supports
   that claim.
4. Measuring citation completeness, meaning whether every externally
   verifiable claim has adequate support.
5. Comparing semantic factual correctness with reference answers when wording
   differs substantially.
6. Judging answer completeness, relevance, and helpfulness.
7. Estimating context relevance or noise when gold evidence labels are absent.

The exact boundary is: citation-ID metrics can verify that a cited ID is valid
and overlaps annotated evidence, but a semantic judge is needed to determine
whether the cited passage truly supports the generated claim.

Recommended rollout:

1. Implement and report deterministic end-to-end attribution first.
2. Freeze the evaluated cases, retrieval outputs, prompts, and model settings.
3. Add LLM-judged claim support, citation entailment, completeness, and semantic
   correctness as a separate, more expensive evaluation layer.
4. Cache judge inputs and outputs, version the judge prompt and model, and
   manually audit a sample before treating judge scores as reportable.

## Progress after the original handoff

The `dev` branch has advanced beyond the snapshot above.

Completed and committed:

- Stage 6 paper-clustered bootstrap intervals in `b2b4f22`.
- Frozen retrieved-context generation support in `4d5b919`.
- Task-oriented generation commands in `09cbfc3`.
- OpenAI Responses API generation support in `fcc6b86`.
- Citation-contract corrections in `39d7b38`.
- Oracle model-comparison documentation in `03b9261`.
- Configurable frozen retrieval contexts in `e7e0bf9`.

The matched 25-case prompt comparison has also been run:

- Track A prompt v2: 25 of 25 valid responses.
- Track B prompt v1: 11 of 25 valid responses.
- Track B prompt v2: 16 of 25 valid responses.

The complete test suite passes:

```text
94 passed
```

Run validation with:

```bash
uv run pytest -q
git diff --check
```

The project keeps `uv.lock` as a tracked reproducibility artifact. `.DS_Store`
and `.idea/` are ignored.

## Immediate next actions

### Stage 1. Repository stabilization

1. Commit this updated handoff, `.gitignore`, and `uv.lock`.
2. Keep generated data and result artifacts outside version control.
3. Re-run `uv run pytest -q` and `git diff --check` before committing.

### Stage 2. Freeze data and experiment protocol

1. [x] Record and verify the downloaded QASPER validation Parquet checksum.
2. [x] Decide against downloading the train split while prompt contract v3 is
   frozen and no active prompt-development work requires it.
3. [x] Freeze the validation data, prompts, model settings, decoding parameters,
   retrieval settings, metrics, and held-out policy in
   `generation_protocol_v1.json`.
4. [ ] Run full validation baselines and combine them with explicit product
   requirements to freeze numeric thresholds before inspecting held-out
   results. The existing 25-case runs are smoke baselines only.

### Stage 3. Complete deterministic comparisons

1. [x] Compare full BM25, dense, and hybrid QASPER retrieval against the oracle
   evidence ceiling before generation. Hybrid MiniLM plus BM25 RRF at paper
   scope and top 5 is selected.
2. [x] Complete Qwen3-4B generation on the selected hybrid retrieval manifest.
   All 930 attempts are persisted. Valid-response rate is `0.9677`, token F1 is
   `0.1938`, citation F1 is `0.3655`, and p95 latency is `4.91` seconds.
3. [ ] Run a GPT-5 pilot on the selected hybrid manifest, estimate full-run cost,
   and obtain explicit budget approval before the full GPT-5 validation run.
   A `$10` budget is approved with a 768-token output cap and zero retries. The
   conservative full-run estimate is `$8.36`. Both 25-case pilot attempts
   returned only `429 insufficient_quota`, so API billing or credits must be
   restored before retrying.
4. [ ] Compare Qwen3-4B and GPT-5 generation on identical hybrid-retrieval cases.
5. [x] Add paired paper-clustered bootstrap differences for matched systems.
6. [x] Report deterministic evidence availability and the complete primary
   retrieval-versus-generation failure taxonomy.

The Stage 3 implementation and current results are summarized in
`stage3_deterministic_report.md`. Previous GPT-5 dense results remain
quota-confounded and non-selectable. The held-out test remains untouched.

### Stage 4. Add and validate semantic judgments

1. [x] Implement blinded claim extraction, claim support, citation entailment,
   faithfulness, semantic correctness, and rubric completeness evaluation.
   `rag-semantic-judge` now prepares anonymous inputs, runs a strict structured
   judge, and aggregates claim-level and case-level semantic metrics. No paid
   semantic judge run has been executed yet.
2. [ ] Cache and version judge inputs, outputs, prompts, and models.
3. [ ] Complete the stratified human audit and measure human-judge agreement.

### Stage 5. Final evaluation and reporting

1. Freeze the complete protocol and acceptance thresholds.
2. Run the held-out test comparison once.
3. Complete retrieval and generation error analyses.
4. Publish the retrieval-component, generation-component, and end-to-end
   report.

## Key files

- `README.md`. User-facing commands and current benchmark status.
- `benchmark_spec.md`. Frozen QASPER evaluation policy and implementation checklist.
- `generation_smoke_test_execution.md`. End-to-end runner and metric flow.
- `dev_history.md`. Chronological implementation decisions.
- `src/rag_eval/generation/runner.py`. Eligibility, execution, persistence, and
  resume.
- `src/rag_eval/generation/prompt.py`. Prompt v2 and response validation.
- `src/rag_eval/generation/metrics.py`. Deterministic generation metrics.
- `src/rag_eval/end_to_end/workflow.py`. Frozen-retrieval and end-to-end
  workflow composition.
- `src/rag_eval/end_to_end/attribution.py`. Deterministic retrieval-versus-
  generation failure attribution.
- `tests/test_generation_metrics.py`. Deterministic metric and bootstrap tests.
